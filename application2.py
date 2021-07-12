import copy
import csv
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from thirdparty import KeyPointClassifier
from thirdparty.utils import (
    calc_landmark_list,
    logging_csv,
    pre_process_landmark,
    pre_process_point_history,
    select_mode,
)


def application():
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)  # ビデオキャプチャの開始
    pencil_coord = []
    brush_coord = []
    pen_coord = []

    colar = (255, 255, 255)
    tool = "pencil"
    _, frame = cap.read()
    Height, Width = frame.shape[:2]

    pallet = cv2.imread("./imgs/pallet.png")
    pallet = cv2.resize(pallet, (100, 600))
    tool_pallet = cv2.imread("./imgs/tools.png")
    tool_pallet = cv2.resize(tool_pallet, (100, 300))

    keypoint_classifier = KeyPointClassifier()

    with open(
        "thirdparty/keypoint_classifier/keypoint_classifier_label.csv",
        encoding="utf-8-sig",
    ) as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    with open(
        "thirdparty/point_history_classifier/point_history_classifier_label.csv",
        encoding="utf-8-sig",
    ) as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    history_length = 16
    point_history = deque(maxlen=history_length)

    with mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()
            key = cv2.waitKey(10)
            mode = 0
            if key == 27:  # ESC
                break
            number, mode = select_mode(key, mode)

            frame = cv2.flip(frame, 1)
            debug_image = copy.deepcopy(frame)

            red = (51, 0, 255)
            yellow = (0, 204, 255)
            white = (255, 255, 255)
            blue = (255, 153, 0)
            black = (0, 0, 0)

            color_pallet = np.zeros_like(frame)
            color_pallet[0:600, 0:100, :] = pallet
            color_pallet[0:300, -100:, :] = tool_pallet
            frame = cv2.addWeighted(frame, 0.7, color_pallet, 0.3, gamma=0)

            results = hands.process(frame)
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    # ランドマークの計算
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # 相対座標・正規化座標への変換
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(
                        debug_image, point_history
                    )
                    # 学習データ保存
                    logging_csv(
                        number,
                        mode,
                        pre_processed_landmark_list,
                        pre_processed_point_history_list,
                    )

                    # ハンドサイン分類
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    if hand_sign_id == 2:  # 指差しサイン
                        point_history.append(landmark_list[8])  # 人差指座標
                    else:
                        point_history.append([0, 0])

            if (
                results.multi_hand_landmarks
                and keypoint_classifier_labels[hand_sign_id] != "Close"
            ):
                hand_landmarks = results.multi_hand_landmarks[0]
                x = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                    * Width
                )
                y = (
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                    * Height
                )
                if tool == "pencil":
                    pencil_coord.append((int(x), int(y), colar))
                elif tool == "brush":
                    brush_coord.append((int(x), int(y), colar))
                elif tool == "pen":
                    pen_coord.append((int(x), int(y), colar))
                if 0 < x <= 100:
                    if 0 <= y <= 100:
                        colar = red
                    elif 100 <= y <= 200:
                        colar = yellow
                    elif 200 < y <= 300:
                        colar = blue
                    elif 300 <= y < 400:
                        colar = white
                    elif 400 <= y < 500:
                        colar = black
                    elif 500 <= y < 600:
                        colar = white
                        pencil_coord = []
                        brush_coord = []
                        pen_coord = []
                if Width - 100 < x < Width:
                    if 0 <= y <= 100:
                        tool = "pencil"
                    elif 100 <= y <= 200:
                        tool = "brush"
                    elif 200 <= y <= 300:
                        tool = "pen"

            else:
                if tool == "pencil":
                    pencil_coord.append((int(x), int(y), -1))
                elif tool == "brush":
                    brush_coord.append((int(x), int(y), -1))
                elif tool == "pen":
                    pen_coord.append((int(x), int(y), -1))

            if len(pencil_coord) >= 2:
                last_pencil_x, last_pencil_y, _ = pencil_coord[0]
                for x, y, c in pencil_coord[1:]:
                    if c == -1:
                        last_pencil_x = None
                        last_pencil_y = None
                    else:
                        if last_pencil_x is not None:
                            frame = cv2.line(
                                frame,
                                (int(x), int(y)),
                                (int(last_pencil_x), int(last_pencil_y)),
                                c,
                                thickness=2,
                                lineType=cv2.LINE_AA,
                            )
                        last_pencil_x = x
                        last_pencil_y = y

            if len(brush_coord) >= 2:
                last_brush_x, last_brush_y, _ = brush_coord[0]
                for x, y, c in brush_coord[1:]:
                    if c == -1:
                        last_brush_x = None
                        last_brush_y = None
                    else:
                        if last_brush_x is not None:
                            frame = cv2.line(
                                frame,
                                (int(x), int(y)),
                                (int(last_brush_x), int(last_brush_y)),
                                (c[0], c[1], c[2], 0.8),
                                thickness=15,
                                lineType=cv2.LINE_AA,
                            )
                        last_brush_x = x
                        last_brush_y = y

            if len(pen_coord) >= 2:
                last_pen_x, last_pen_y, _ = pen_coord[0]
                for x, y, c in pen_coord[1:]:
                    if c == -1:
                        last_pen_x = None
                        last_pen_y = None
                    else:
                        if last_pen_x is not None:
                            frame = cv2.line(
                                frame,
                                (int(x), int(y)),
                                (int(last_pen_x), int(last_pen_y)),
                                c,
                                thickness=6,
                                lineType=cv2.LINE_AA,
                            )
                        last_pen_x = x
                        last_pen_y = y

            cv2.imshow("drawDetectedMarkers", frame)  # マーカが描画された画像を表示

            cv2.waitKey(1)  # キーボード入力の受付

    cap.release()  # ビデオキャプチャのメモリ解放
    cv2.destroyAllWindows()  # すべてのウィンドウを閉じる


if __name__ == "__main__":
    application()
