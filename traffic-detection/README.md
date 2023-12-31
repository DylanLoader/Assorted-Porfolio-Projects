# Traffic Detection with Supervision

## Introduction

A fun project for detecting and analyzing traffic flow from video using Roboflow's Supervision package and Ultralytics' implementation of YOLOv8.

Primary goal of this project is to be able to count vehicles passing a user defined threshold. This code can easily be extended to new video data by making a few tweaks to the detection criteria based on where the bounding box intersects the polygonzone and defining a new polygon zone. 

Streamlit Web-app Demo:

![](images/demo.gif)

## System Design

![](images/traffic-detection-code-flow.png)

### Implementation

The current implementation of this code reads a on-disk video through OpenCV's Python API. 
From there the video is read frame by frame and detections are made using YOLOv8, by Ultralytics. I chose YOLOv8 as a model for its out-of-the-box performance on vehicles as well as the speed of inference. For tracking I use Bytetrack, which allows us to track individual vehicles, which can be used later to extend this application to cover other tasks such as lane tracking for drive safety. 

## Future Work

Here are some possible avenues for improving this project, either for better performance or functionality. 

### Performance

- We can fine-tune the yolov8 model with a custom dataset that matches how the vehicles are seen in the camera feed. 
- If this system is expected to operate in a real time system we should add more training video data with various weather conditions.

### Functionality

- Change the streamlit app to gradio so we can change the parameters via sliders, but requires an on demand GPU backend. 
- Dockerize the application and add cuda acceleration where possible. Development was done on a M2 Macbook, so cuda isn't accessible. 
- We can add lane polygons for each road so we can monitor lane changes, possibly adding some information about how aggressively people are driving. 
- The tracker ids correspond to initial detections, which in this case include some instances of people (drivers). I have filtered these detections out, so the tracker ids aren't intuitive and seem to not match the number of vehicles detected. 

## References

- This project leans heavily on the great tutorials and Supervision package provided by [Roboflow](https://roboflow.com/)
- Website Icon Attribution: https://www.flaticon.com/free-icon/truck_819873?term=truck&page=1&position=5&origin=search&related_id=819873
