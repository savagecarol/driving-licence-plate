from tkinter import *
import tkinter as tk

import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import cv2



window = Tk()


window.title("Welcome to Parking System")
window.geometry('900x500')


lbl = Label(window, text="Hello")
lbl.grid(column=0, row=0)

btn = Button(window, text="Click Me")
btn.grid(column=1, row=0)


def response(x):
    img = cv2.imread(x)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
    status = cv2.imwrite('hi.png', cropped_image)




def click():
    cam = cv2.VideoCapture(1)
    cv2.namedWindow("test")
    img_counter = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            response(img_name)
            print("{} written!".format(img_name))
            img_counter += 1
    cam.release()
    cv2.destroyAllWindows()





btn1 = Button(
    window,
    bd=0,
    relief="groove",
    compound=tk.CENTER,
    bg="white",
    fg="yellow",
    activeforeground="pink",
    activebackground="white",
    font="arial 18",
    text="Click me",
    pady=10,
    command= click    # width=300
)

btn1.grid(column=2, row=0)

window.mainloop()

























