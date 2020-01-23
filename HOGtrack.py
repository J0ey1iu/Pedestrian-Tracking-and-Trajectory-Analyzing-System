#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:00:40 2019

@author: liujiayu
"""

import cv2
import random
import numpy as np
# from imutils.object_detection import non_max_suppression

class HOGTracker():
    def __init__(self, videopath):
        self.path = videopath
    
    def startTracking(self):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        cap = cv2.VideoCapture(self.path)
        ret, frame = cap.read()
        drawlayer = np.zeros_like(frame)
        trackerlist = []

        frame_index = 0
        ret, firstframe = cap.read()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Stopped at frame %d. Totally %d lines"% (frame_index, len(trackerlist)))
                cap.release()
                cv2.imwrite("before cluster.png", cv2.add(firstframe, drawlayer))
                cv2.destroyAllWindows()
                self.printData(trackerlist)
                return [frame_index, len(trackerlist)]
            frame_index += 1
            if trackerlist:
                print("[INFO] " + str(len(trackerlist)) + " trackers in frame %d"% frame_index)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            for tracker in trackerlist:
                if tracker.flag:
                    tracker.update(gray)
                else:
                    continue
            (rects, weights) = hog.detectMultiScale(gray, winStride=(5, 5), padding=(8, 8), scale=1)
            filtered = []
            if len(trackerlist) == 0:
                for rect in rects:
                    filtered.append(rect)
            else:
                for rect in rects:
                    sameflag = 0
                    for tracker in trackerlist:
                        issame = tracker.isSame(rect)
                        if issame:
                            sameflag = 1
                            break
                    if sameflag == 0:
                        filtered.append(rect)
                    
            for box in filtered:
                trackerlist.append(Tracker(box, gray))
            
            for (x, y, w, h) in rects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            for tracker in trackerlist:
                cv2.circle(drawlayer, (int(tracker.trajectory[-1][0]), int(tracker.trajectory[-1][1])), 2, tracker.color, -1)
                
            show = cv2.add(frame, drawlayer)
            cv2.imshow("test output", show)
            if cv2.waitKey(30) == 27:
                print("[INFO] Stopped at frame %d. Totally %d lines"% (frame_index, len(trackerlist)))
                cap.release()
                cv2.imwrite("before cluster.png", cv2.add(firstframe, drawlayer))
                cv2.destroyAllWindows()
                self.printData(trackerlist)
                return [frame_index, len(trackerlist)]

    def printData(self, trackerlist):
        print("[INFO] Printing trajectories...")
        total = len(trackerlist)
        file = open('trajectories.txt', 'w')
        for i, tracker in enumerate(trackerlist):
            for pos in tracker.trajectory:
                file.write("%f %f,"% (pos[0], pos[1]))
            file.write("\n")
        file.close()


class Tracker():
    def __init__(self, box, frame):
        self.init_box = box
        self.color = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))
        self.flag = 1
        
        self.tracker_type = 'CSRT'
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, tuple(self.init_box))
        center = self.getCenter(self.init_box)
        
        self.trajectory = [center]
    
    def update(self, frame):
        ret, newbox = self.tracker.update(frame)
        if ret:
            center = self.getCenter(newbox)
            if center[0]+2>720 or center[0]-2<0 or center[1]+2>480 or center[1]-2<0:
                print("[INFO] Tracker out of frame")
                self.flag = 0
                return False
            else:
                self.trajectory.append(center)
                return True
        else:
            print("[UPDATE ERROR] Fail to update tracker")
            print(self.trajectory)
            self.flag = 0
            return False
        
    def getCenter(self, box):
        x = box[0]+box[2]/2
        y = box[1]+box[3]/2
        center = (x, y)
        return center
    
    def getColor(self):
        return self.color
    
    def isSame(self, newbox):
        thresh = 70
        newcenter = self.getCenter(newbox)
        distance = np.sqrt((newcenter[0]-self.trajectory[-1][0])**2+(newcenter[1]-self.trajectory[-1][1])**2)
        if distance < thresh:
            return True
        else:
            return False
        
    def getTrajectory(self):
        return self.trajectory
        
if __name__ == "__main__":
    ht = HOGTracker("testvid2.avi")
    ht.startTracking()