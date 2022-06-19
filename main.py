import os
from Hand_detection import HandDetector
import cv2
from tensorflow.keras.models import load_model
import pyautogui
import numpy as np

instance_bonus = 50
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.7, maxHands=1)
model = load_model("D:/VScode/check/program/model/MobileNet_model.h5")
name_gesture = ['Double_Left_mouse',
 'Left_mouse',
 'Long_press_mouse',
 'Move_mouse',
 'Right_mouse',
 'Scroll_mouse']

decode_name_gesture = [1, 2, 3, 4, 5, 6]

current_x = 0
current_y = 0
status_mouse = 4
_, frame = cap.read()
h_cam, w_cam = frame.shape[:2]
w_monitor, h_monitor = pyautogui.size()
frameR = 100

def mouse(target, x=None, y=None):
    global current_x
    global current_y
    global status_mouse
    global smoothening
    global frameR

    if target == 1 and status_mouse != 1:
        pyautogui.doubleClick(button='left')
        
    elif target == 2 and status_mouse != 2:
        pyautogui.click(button='left')
        
    elif target == 3:
        pyautogui.mouseDown(button='left')
        x_old, y_old = pyautogui.position()
        x = x_old + x
        y = y_old + y
        pyautogui.moveTo(x, y, 0.1)
        #pyautogui.mouseUp(button='left')
        
    elif target == 4:
        if frameR < x < w_cam-frameR and frameR < y < h_cam-frameR:
            x3 = np.interp(x, (frameR, w_cam-frameR), (0, w_monitor))
            y3 = np.interp(y, (frameR, h_cam-frameR), (0, h_monitor))
            pyautogui.moveTo(x3, y3, 0.1)

    elif target == 5 and status_mouse != 5:
        pyautogui.click(button='right')
        
    elif target == 6 and status_mouse == 6:
        pyautogui.scroll((current_y - y)*10)
    current_y = y
    status_mouse = target


while True:
    success, img_cam = cap.read()

    img = cv2.flip(img_cam, 1)
    
    hands, lmList, img_draw = detector.findHands(img.copy(), draw=True, flipType=False)

    if hands:
        hands1 = hands[0]
        bbox1 = hands1['bbox']
        img_crop = img[max(0, bbox1[1]-instance_bonus):bbox1[1]+bbox1[3]+instance_bonus, 
                       max(0, bbox1[0]-instance_bonus):bbox1[0]+bbox1[2]+instance_bonus]
        center_point = lmList[0][:2]
        try:
            X = cv2.resize(img_crop, (224, 224))
            X = X.reshape((1,) + X.shape)
            X = X / 255.0
            y = model.predict(X)
            id = y.argmax(axis=1)
            if max(y[0]) > 0.9:
                mouse(decode_name_gesture[id[0]], center_point[0], center_point[1])
            cv2.putText(img_draw, name_gesture[id[0]], (bbox1[0], bbox1[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except Exception as e:
            print(e)
    cv2.imshow('cap', img_draw)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()