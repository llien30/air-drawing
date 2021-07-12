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
    coordinate = []
    colar = (255, 255, 255)
    _, frame = cap.read()
    Height, Width = frame.shape[:2]

    pallet = cv2.imread("./imgs/pallet.png")
    pallet = cv2.resize(pallet, (100, 600))

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
                coordinate.append((int(x), int(y), colar))
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
                        coordinate = []
            for x, y, c in coordinate:
                frame = cv2.circle(frame, (int(x), int(y)), 8, c, -1)

            cv2.imshow("drawDetectedMarkers", frame)  # マーカが描画された画像を表示

            cv2.waitKey(1)  # キーボード入力の受付

    cap.release()  # ビデオキャプチャのメモリ解放
    cv2.destroyAllWindows()  # すべてのウィンドウを閉じる


if __name__ == "__main__":
    application()
