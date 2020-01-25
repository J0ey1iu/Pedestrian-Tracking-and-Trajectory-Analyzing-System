# Pedestrian-Tracking-and-Trajectory-Analyzing-System
This is the source of my bachelor dissertation. Users can use this system to track pedestrians in a video and analyze the trajectories in it. Two tracking methods and two cluster analysis methods are provided in the system.
## Tracking methods
### Feature points detection and optical flow
Extract feature by Shi-Tomasi corner detection and use optical flow to track the movement. This method is faster but less accurate than the method below.
### HOG detection and CSR-DCF
Detect human body by HOG detection and track the movement with CSR-DCF(Discriminative Correlation Filter with Channel and Spatial Reliability). This method is more accurate than the method above, and only track human, while feature points detection detects feature for everything in the scene.
## Analysis methods
The code for analysis is modified from the original code of janbednarik.
### Agglomerative hierarchical cluster
Users need to input how many clusters they want if they use this method.
### spectral cluster
The algorithm auotmatically choose the number of clusters.

## How to use it
You need to have Python3 with OpenCV, Numpy, scipy. Anaconda with OpenCV installed would be perfect for this.
1. Run `GUI.py`
2. Follow the guide
