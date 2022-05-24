import os
import uuid
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pydirectinput
import win32api
import win32con
from win32api import GetSystemMetrics
import mouse

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
x = mouse.get_position()[0]
y = mouse.get_position()[1]
zr = True
zl = True
Move = False
Util = False
r = False
l = False
VideoGame = True
cap = cv2.VideoCapture(0)

def distance(P1, P2):
    return ((P1.x-P2.x)**2+(P1.y-P2.y)**2)**0.5

def CLOSE(Stat):
    return (Stat<0.12)

def FAR(Stat):
    return (Stat>0.12)

with mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.4) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not results.multi_hand_landmarks:
            zr = True
            zl = True

        # Rendering results
        if results.multi_hand_landmarks:
            # Define Hands
            for num, hand in enumerate(results.multi_hand_landmarks):
                if(len(results.multi_hand_landmarks)==1):
                    if(results.multi_hand_landmarks[0].landmark[2].x-results.multi_hand_landmarks[0].landmark[5].x<0):
                        RNowHand = results.multi_hand_landmarks[0]
                        r = True
                    if(zr == True and r == True):
                        RLastHand = RNowHand
                        zr = False
                    if(results.multi_hand_landmarks[0].landmark[2].x-results.multi_hand_landmarks[0].landmark[5].x>0):
                        LNowHand = results.multi_hand_landmarks[0]
                        l = True
                    if(zl == True and l == True):
                        LLastHand = LNowHand
                        zl = False
                else:
                    if(results.multi_hand_landmarks[0].landmark[2].x>results.multi_hand_landmarks[1].landmark[2].x):
                        RNowHand = results.multi_hand_landmarks[0]
                        r = True
                    else:
                        RNowHand = results.multi_hand_landmarks[1]
                        r = True
                    if(zr == True):
                        RLastHand = RNowHand
                        zr = False
                    if(results.multi_hand_landmarks[0].landmark[2].x<results.multi_hand_landmarks[1].landmark[2].x):
                        LNowHand = results.multi_hand_landmarks[0]
                        l = True
                    else:
                        LNowHand = results.multi_hand_landmarks[1]
                        l = True
                    if(zl == True):
                        LLastHand = LNowHand
                        zl = False
                                            
            if l:
                # Draw Left Hand
                mp_drawing.draw_landmarks(image, LNowHand, mp_hands.HAND_CONNECTIONS,
                                            mp_drawing.DrawingSpec(
                                                color=(84, 224, 105), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(
                                                color=(44, 250, 75), thickness=2, circle_radius=2),
                                            )
                # Left Hand Stats
                LPoint = distance(LNowHand.landmark[4], LNowHand.landmark[8])
                LMiddle = distance(LNowHand.landmark[4], LNowHand.landmark[12])
                LRing = distance(LNowHand.landmark[4], LNowHand.landmark[16])
                LPinky = distance(LNowHand.landmark[4], LNowHand.landmark[20])
                LLPoint = distance(LLastHand.landmark[4], LLastHand.landmark[8])
                LLMiddle = distance(LLastHand.landmark[4], LLastHand.landmark[12])
                LLRing = distance(LLastHand.landmark[4], LLastHand.landmark[16])
                LLPinky = distance(LLastHand.landmark[4], LLastHand.landmark[20])
                
                # Video Game Mouse
                if(VideoGame==True):
                    x = GetSystemMetrics(0) // 2
                    y = GetSystemMetrics(1) // 2 + 1
                if(FAR(LPoint) and FAR(LMiddle) and CLOSE(LRing) and CLOSE(LPinky) and FAR(LLRing) and FAR(LLPinky)):
                    if(VideoGame==True):
                        VideoGame=False
                    else:
                        VideoGame=True

                # Shift
                if(CLOSE(LRing) and FAR(LLRing)):
                    pyautogui.keyDown('shift')
                if(FAR(LRing) and CLOSE(LLRing)):
                    pyautogui.keyUp('shift')
                # Left Pinky release
                if(FAR(LPinky) and CLOSE(LLPinky)):
                    pyautogui.scroll(1)
                # Move
                if(CLOSE(LPoint) and FAR(LLPoint) and FAR(LMiddle) and FAR(LRing) and FAR(LPinky)):
                    Move = True
                if(FAR(LPoint) and CLOSE(LLPoint)):
                    Move = False
                # Util
                if(CLOSE(LMiddle) and FAR(LLMiddle) and FAR(LPoint) and FAR(LRing) and FAR(LPinky)):
                    Util = True
                if(FAR(LMiddle) and CLOSE(LLMiddle)):
                    Util = False
                
            if r:
                # Draw Right Hand
                mp_drawing.draw_landmarks(image, RNowHand, mp_hands.HAND_CONNECTIONS,
                                            mp_drawing.DrawingSpec(
                                                color=(105, 84, 224), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(
                                                color=(76, 44, 250), thickness=2, circle_radius=2),
                                            )

                # Right Hand Stats
                RPoint = distance(RNowHand.landmark[4], RNowHand.landmark[8])
                RMiddle = distance(RNowHand.landmark[4], RNowHand.landmark[12])
                RRing = distance(RNowHand.landmark[4], RNowHand.landmark[16])
                RPinky = distance(RNowHand.landmark[4], RNowHand.landmark[20])
                RLPoint = distance(RLastHand.landmark[4], RLastHand.landmark[8])
                RLMiddle = distance(RLastHand.landmark[4], RLastHand.landmark[12])
                RLRing = distance(RLastHand.landmark[4], RLastHand.landmark[16])
                RLPinky = distance(RLastHand.landmark[4], RLastHand.landmark[20])

                cv2.putText(image, str(round(RPoint, 3)), (10, 40), 2, 1, (255, 0, 0))
                cv2.putText(image, str(round(RMiddle, 3)), (10, 80), 2, 1, (255, 0, 0))
                cv2.putText(image, str(round(RRing, 3)), (10, 120), 2, 1, (255, 0, 0))
                cv2.putText(image, str(round(RPinky, 3)), (10, 160), 2, 1, (255, 0, 0))

                # Hand Mouse Move
                if(abs(RNowHand.landmark[9].x-RLastHand.landmark[9].x) > 0.005 and FAR(RPoint) or abs(RNowHand.landmark[9].x-RLastHand.landmark[9].x) > 0.005 and FAR(RMiddle)):
                    x += (RNowHand.landmark[9].x-RLastHand.landmark[9].x)*5000
                if(abs(RNowHand.landmark[9].y-RLastHand.landmark[9].y) > 0.005 and FAR(RPoint) or abs(RNowHand.landmark[9].y-RLastHand.landmark[9].y) > 0.005 and FAR(RMiddle)):
                    y += (RNowHand.landmark[9].y-RLastHand.landmark[9].y)*5000

                # Hand Mouse Scroll
                if(CLOSE(RRing) and FAR(RPinky) and FAR(RMiddle)):
                    win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, int(x), int(y), int((RLastHand.landmark[9].y-RNowHand.landmark[9].y)*10000), 0)

                # Left Click
                if(CLOSE(RPoint) and FAR(RLPoint) and FAR(RMiddle) and FAR(RRing) and FAR(RPinky) and not Move and not Util):
                    pyautogui.mouseDown(button='left')
                if(FAR(RPoint) and CLOSE(RLPoint) and not Move and not Util):
                    pyautogui.mouseUp(button='left')

                # Right Click
                if(CLOSE(RMiddle) and FAR(RLMiddle) and FAR(RPoint) and FAR(RRing) and FAR(RPinky) and not Move and not Util):
                    pyautogui.mouseDown(button='right')
                if(FAR(RMiddle) and CLOSE(RLMiddle) and not Move and not Util):
                    pyautogui.mouseUp(button='right')

                # Right Pinky release
                if(FAR(RPinky) and CLOSE(RLPinky) and not Move and not Util):
                    pyautogui.scroll(-1)

                # Move Enabled
                if(CLOSE(RPoint) and FAR(RLPoint) and FAR(RMiddle) and FAR(RRing) and FAR(RPinky) and Move) and not Util:
                    pyautogui.keyDown('w')
                if(FAR(RPoint) and CLOSE(RLPoint)):
                    pyautogui.keyUp('w')
                if(CLOSE(RMiddle) and FAR(RLMiddle) and FAR(RPoint) and FAR(RRing) and FAR(RPinky) and Move and not Util):
                    pyautogui.keyDown('s')
                if(FAR(RMiddle) and CLOSE(RLMiddle)):
                    pyautogui.keyUp('s')
                if(CLOSE(RRing) and FAR(RLRing) and FAR(RPoint) and FAR(RMiddle) and FAR(RPinky) and Move and not Util):
                    pyautogui.keyDown('a')
                if(FAR(RRing) and CLOSE(RLRing)):
                    pyautogui.keyUp('a')
                if(CLOSE(RPinky) and FAR(RLPinky) and FAR(RPoint) and FAR(RRing) and FAR(RMiddle) and Move and not Util):
                    pyautogui.keyDown('d')
                if(FAR(RPinky) and CLOSE(RLPinky)):
                    pyautogui.keyUp('d')
                # Util Enabled
                if(CLOSE(RPoint) and FAR(RLPoint) and FAR(RMiddle) and FAR(RRing) and FAR(RPinky) and Util) and not Move:
                    pyautogui.press('e')
                if(CLOSE(RMiddle) and FAR(RLMiddle) and FAR(RPoint) and FAR(RRing) and FAR(RPinky) and Util and not Move):
                    pyautogui.press('space')
                if(CLOSE(RRing) and FAR(RLRing) and FAR(RPoint) and FAR(RMiddle) and FAR(RPinky) and Util and not Move):
                    pyautogui.press('q')
                if(CLOSE(RPinky) and FAR(RLPinky) and FAR(RPoint) and FAR(RRing) and FAR(RMiddle) and Util and not Move):
                    pyautogui.press('r')
            y -= 10
            mouse.move(x, y)
            if(r == True):
                RLastHand = RNowHand
            if(l == True):
                LLastHand = LNowHand
            r = False
            l = False

        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

os.mkdir('Output Images')

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.4) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(
                                              color=(121, 22, 76), thickness=2, circle_radius=4),
                                        
                                          mp_drawing.DrawingSpec(
                                              color=(250, 250, 250), thickness=2, circle_radius=2),
                                          )

        # Save our image
        cv2.imwrite(os.path.join('Output Images',
                                 '{}.jpg'.format(uuid.uuid1())), image)
        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
