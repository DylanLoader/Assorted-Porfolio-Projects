# Imports
import streamlit as st
import pandas as pd
import json
import re

# Set page configuration
st.set_page_config(layout="wide", page_title="Traffic Analysis Dashboard", page_icon="../images/truck.png")

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
st.title("Traffic Analysis Dashboard")
st.markdown("Project Code Repo: [LINK](https://github.com/DylanLoader/Assorted-Porfolio-Projects/tree/main/traffic-detection)")
option = st.selectbox(
    "Select YOLO Model To Display",
    ("None", "Small", "Medium", "Large"),
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
    index=None
)

VIDEO_PATH = "../data/vehicle-counting.mp4"
MODEL_PATH = "../models/yolov8s.pt"
if option=="Small":
    VIDEO_PATH = "../data/vehicle-counting-yolov8s.mp4"
    MODEL_PATH = "../models/yolov8s.pt"
elif option=="Medium":
    VIDEO_PATH = "../data/vehicle-counting-yolov8m.mp4"
    MODEL_PATH = "../models/yolov8m.pt"
elif option=="Large":
    VIDEO_PATH = "../data/vehicle-counting-yolov8l.mp4"
    MODEL_PATH = "../models/yolov8l.pt"
model_name = MODEL_PATH.split('/')[-1]
video_file = open(VIDEO_PATH, 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

JSON_PATH = f"../data/output/{model_name}.json"
df = pd.read_json(JSON_PATH)
df.index = ['Left', 'Right']
df = df.T
if option is not None:
    st.write('Summary Information By Road')
    st.dataframe(
        data = df,
        column_config={
        "name": "App name", 
        },
        hide_index=False,
        use_container_width=True,
        )