from pickle import FRAME

import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
import os
from PIL import Image
import streamlit as st


st.set_page_config(layout="wide")
st.image('MathGestures.png')

col1,col2 = st.columns([2,1])

with col1:
    run = st.checkbox('Run',value=True)
    FRAME_WINDOW = st.image([])
with col2:
    output_text_area = st.title("Answer")
    output_text_area = st.subheader("")


genai.configure(api_key="AIzaSyC1azJXsFnFtJAZkcjfCW1qgo5l1fKnmVU")
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize the webcam to capture video
# The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
cap = cv2.VideoCapture(0)
cap.set(3,840)
cap.set(4,580)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)


def getHandInfo(img):
    # Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    hands, img = detector.findHands(img, draw=True, flipType=True)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand = hands[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand


        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand)
        print(fingers)
        return fingers , lmList
    else:
        return None



def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None

    # Check if all thumb fingers are up
    if fingers == [1, 0, 0, 0, 0]:
        # Restart the design by clearing the canvas
        canvas = np.zeros_like(canvas)
        prev_pos = None
        return prev_pos, canvas

    # Check if only the index finger is up
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = tuple(lmList[8][0:2])  # Convert the coordinates to a tuple

        if prev_pos is None:
            prev_pos = current_pos

        # Interpolate between previous and current positions for smoother drawing
        for i in range(10):  # Increase the number for smoother drawing
            x = int(prev_pos[0] + (current_pos[0] - prev_pos[0]) * i / 10)
            y = int(prev_pos[1] + (current_pos[1] - prev_pos[1]) * i / 10)
            cv2.circle(canvas, (x, y), 5, (255, 0, 255), -1)  # Draw small circles at the interpolated points

        # Optionally, draw a line between the current and previous positions for better continuity
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 5)

    return current_pos, canvas



def sendToAI(model,canvas,fingers):
    if fingers == [1,1,1,1,0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve the Math Problem", pil_image])
        return response.text





prev_pos = None
canvas = None
image_combined = None
output_text = ""




# Continuously get frames from the webcam
while True:
        # Capture each frame from the webcam
        # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
        success, img = cap.read()
        img = cv2.flip(img,1)


        if canvas is None:
            canvas = np.zeros_like(img)


        info = getHandInfo(img)
        if info:
            fingers, lmList = info
            prev_pos, canvas = draw(info, prev_pos, canvas)  # Correct unpacking of returned values
            output_text = sendToAI(model,canvas,fingers)

        image_combined = cv2.addWeighted(img,0.7,canvas,0.3,0)

        FRAME_WINDOW.image(image_combined,channels="BGR")


        if output_text:
          output_text_area.text(output_text)

        # Display the image in a window
        # cv2.imshow("Image", img)
        # cv2.imshow("Canvas", canvas)
        # cv2.imshow("image_combined", image_combined)

        # Keep the window open and update it for each frame; wait for 1 millisecond between frames
        cv2.waitKey(1)

