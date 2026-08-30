"""
기존 l5.py + 계단 올라가기 로직 정리 버전.
동작 로직은 원본과 동일하게 유지하고, 구조/안전성만 개선함.

주요 변경점:
  1. try/finally로 카메라·시리얼 자원 정리 보장 (예외 발생 시에도 close 호출)
  2. 프레임 읽기 실패 시 짧은 sleep 후 재시도 (무한 스핀 방지)
  3. 계단(파랑/초록) 감지에 디바운스(연속 N프레임) 적용 → 순간 오검출로 모션 9 전환되는 것 방지
  4. 매직넘버를 상단 CONFIG로 집중
  5. 색상 검출, 라인 중심 계산, 모션 판단을 함수로 분리
"""
import cv2
import numpy as np
import serial
import time
from CsiCamCapture import CsiCamCapture

# ==========================================
# 0. 설정값 (CONFIG)
# ==========================================
PORT = '/dev/ttyUSB0'
BAUDRATE = 57600

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAMERATE = 30

SAMPLE_Y_RATIO = 0.55
MIN_AREA = 200          # 노란 라인 최소 면적
STAIR_MIN_AREA = 500    # 계단(파랑/초록) 최소 면적
STAIR_CONFIRM_FRAMES = 3  # 계단으로 확정하기 위해 연속으로 감지되어야 하는 프레임 수

OFFSET_HARD_TURN = 60   # 이 이상이면 제자리 회전(6/7)
OFFSET_SOFT_TURN = 20   # 이 이상이면 곡선 회전(11/12)

RESEND_INTERVAL = 0.4   # 같은 모션이어도 재전송하는 주기(초)

# 모션 번호 매핑 (참고용)
MOTION_STOP = 1
MOTION_FORWARD = 3
MOTION_BACKWARD = 4
MOTION_TURN_LEFT = 6
MOTION_TURN_RIGHT = 7
MOTION_CURVE_LEFT = 11
MOTION_CURVE_RIGHT = 12
MOTION_STAIR = 9

# HSV 색상 범위
YELLOW_RANGE = (np.array([20, 100, 100]), np.array([35, 255, 255]))
BLUE_RANGE = (np.array([92, 152, 154]), np.array([96, 255, 175]))
GREEN_RANGE = (np.array([43, 85, 195]), np.array([48, 121, 209]))

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ==========================================
# 1. 시리얼 통신
# ==========================================
def setup_serial(port, baudrate):
    """시리얼 포트를 연다. 실패하면 (None, False) 반환하고 비전 테스트만 계속 진행."""
    try:
        s = serial.Serial(port=port,
                           baudrate=baudrate,
                           parity=serial.PARITY_NONE,
                           stopbits=serial.STOPBITS_ONE,
                           bytesize=serial.EIGHTBITS,
                           timeout=0.05)
        print(f"시리얼 포트({port}) 연결 성공")
        time.sleep(2)
        s.reset_input_buffer()
        return s, True
    except serial.SerialException as e:
        print(f"SerialError: {e} (시리얼 없이 카메라/비전 테스트만 진행합니다)")
        return None, False


def send_motion(ser, ser_enabled, motion_index):
    """모션 번호를 6바이트 패킷으로 전송."""
    if not ser_enabled:
        return True
    if motion_index < 0 or motion_index > 65535:
        return False
    try:
        upbit = (motion_index >> 8) & 0xFF
        downbit = motion_index & 0xFF
        ser.write(bytearray([255, 85, downbit, 255 - downbit, upbit, 255 - upbit]))
        return True
    except serial.SerialException:
        return False


# ==========================================
# 2. 비전 처리
# ==========================================
def preprocess(frame):
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=-20)
    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    return frame


def make_mask(hsv, color_range, close=True, open_=False):
    lower, upper = color_range
    mask = cv2.inRange(hsv, lower, upper)
    if close:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    if open_:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    return mask


def detect_stair(hsv):
    """파란색/초록색 계단 여부와, 그린 컨투어(디버그용)를 반환."""
    blue_mask = make_mask(hsv, BLUE_RANGE)
    green_mask = make_mask(hsv, GREEN_RANGE)

    contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blue_detected = any(cv2.contourArea(c) > STAIR_MIN_AREA for c in contours_blue)
    green_detected = any(cv2.contourArea(c) > STAIR_MIN_AREA for c in contours_green)
    return blue_detected or green_detected


def find_line_center(hsv, sample_y):
    """노란 라인의 sample_y 지점에서의 중심 x좌표와 컨투어를 반환. 없으면 (None, None)."""
    yellow_mask = make_mask(hsv, YELLOW_RANGE, close=True, open_=True)
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]
    if not contours:
        return None, None

    c = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(c)
    if not (y <= sample_y <= y + ch):
        return None, None

    mask_c = np.zeros_like(yellow_mask)
    cv2.drawContours(mask_c, [c], -1, 255, thickness=cv2.FILLED)
    row = mask_c[sample_y, :]
    xs = np.where(row > 0)[0]
    if len(xs) == 0:
        return None, None

    return int(xs.mean()), c


# ==========================================
# 3. 모션 판단
# ==========================================
def decide_motion(stair_confirmed, line_cx, screen_center_x, last_seen_direction):
    """현재 프레임 정보로부터 목표 모션과 갱신된 last_seen_direction을 반환."""
    if stair_confirmed:
        return MOTION_STAIR, last_seen_direction

    if line_cx is not None:
        offset = line_cx - screen_center_x
        if offset < -OFFSET_HARD_TURN:
            return MOTION_TURN_LEFT, MOTION_TURN_LEFT
        elif offset < -OFFSET_SOFT_TURN:
            return MOTION_CURVE_LEFT, MOTION_TURN_LEFT
        elif offset > OFFSET_HARD_TURN:
            return MOTION_TURN_RIGHT, MOTION_TURN_RIGHT
        elif offset > OFFSET_SOFT_TURN:
            return MOTION_CURVE_RIGHT, MOTION_TURN_RIGHT
        else:
            return MOTION_FORWARD, last_seen_direction

    # 라인을 잃어버렸을 때: 마지막으로 향했던 방향으로 계속 회전(탐색)
    return last_seen_direction, last_seen_direction


# ==========================================
# 4. 메인 루프
# ==========================================
def main():
    ser, ser_enabled = setup_serial(PORT, BAUDRATE)
    cap = CsiCamCapture(CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT, framerate=FRAMERATE)
    time.sleep(1)

    print("계단 인식 및 라인 트레이싱 시작... 종료하려면 'q'를 누르세요.")

    if send_motion(ser, ser_enabled, MOTION_STOP):
        print(f"{MOTION_STOP} 됬음(ok)")
    else:
        print(f"{MOTION_STOP} (Fail)")

    last_sent_motion = MOTION_STOP
    last_send_time = time.time()
    last_seen_direction = MOTION_TURN_LEFT  # 기존 코드의 기본값(6)과 동일
    stair_streak = 0  # 연속으로 계단이 감지된 프레임 수 (디바운스용)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("프레임을 받아올 수 없습니다.")
                time.sleep(0.05)  # 무한 스핀 방지
                continue

            frame = preprocess(frame)
            h, w = frame.shape[:2]
            sample_y = int(h * SAMPLE_Y_RATIO)
            screen_center_x = w // 2
            base_point = (screen_center_x, h - 10)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            stair_raw = detect_stair(hsv)
            stair_streak = stair_streak + 1 if stair_raw else 0
            stair_confirmed = stair_streak >= STAIR_CONFIRM_FRAMES

            line_cx, line_contour = find_line_center(hsv, sample_y)

            # ---- 디버그 시각화 ----
            dst = frame.copy()
            cv2.circle(dst, base_point, 6, (0, 255, 255), -1)

            if line_cx is not None:
                cv2.circle(dst, (line_cx, sample_y), 8, (0, 255, 255), -1)
                cv2.drawContours(dst, [line_contour], -1, (0, 255, 255), 2)

            target_motion, last_seen_direction = decide_motion(
                stair_confirmed, line_cx, screen_center_x, last_seen_direction
            )

            if stair_confirmed:
                cv2.putText(dst, "Stair Detected! -> Motion 9", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            elif line_cx is not None:
                offset = line_cx - screen_center_x
                cv2.line(dst, base_point, (line_cx, sample_y), (0, 255, 0), 2)
                cv2.putText(dst, f"Offset: {offset:+d}px", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(dst, "Tracking Lost: Spin Search", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # ---- 패킷 전송 ----
            current_time = time.time()
            if (target_motion != last_sent_motion) or (current_time - last_send_time >= RESEND_INTERVAL):
                is_success = send_motion(ser, ser_enabled, target_motion)
                print(f"{target_motion} 됬음(ok)" if is_success else f"{target_motion} (Fail)")
                last_sent_motion = target_motion
                last_send_time = current_time

            cv2.imshow("Yellow Track & Stairs", dst)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.close()
        if ser_enabled:
            ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()