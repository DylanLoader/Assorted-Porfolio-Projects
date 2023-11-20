# Imports
import streamlit as st

# Set page configuration
st.set_page_config(layout="wide", page_title="Traffic Analysis Dashboard", page_icon="../images/truck.png")

on = st.toggle('Activate YOLO Overlay')

if on:
    VIDEO_PATH = "../data/result_yolov8_large.mp4"
else:
    VIDEO_PATH = "../data/vehicle-counting.mp4"
    
video_file = open(VIDEO_PATH, 'rb')
video_bytes = video_file.read()

st.video(video_bytes)