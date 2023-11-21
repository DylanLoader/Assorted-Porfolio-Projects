# Imports
import streamlit as st
import pandas as pd

# Set page configuration
st.set_page_config(layout="wide", page_title="Traffic Analysis Dashboard", page_icon="../images/truck.png")

col1, col2 = st.columns(2)

with col1:
    on = st.toggle('Activate YOLO Overlay')

    if on:
        VIDEO_PATH = "../data/vehicle-counting-main.mp4"
    else:
        VIDEO_PATH = "../data/vehicle-counting.mp4"
        
    video_file = open(VIDEO_PATH, 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)
    
with col2:
    st.dataframe(
        data = pd.DataFrame(
            {'Left Lane Summary':[10,1], 'Right Lane Summary':[12,1]}
        ),
        column_config={
        "name": "App name", 
        },
        hide_index=True,
        use_container_width=True, 
        )