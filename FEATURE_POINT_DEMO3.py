#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:44:15 2019

@author: liujiayu
"""

import numpy as np
import cv2
import random
import tkinter.messagebox as messagebox

class FeatureTracker():
    def __init__(self, path, ql):
        # ShiTomasi corner detection的参数
        self.feature_params = dict(maxCorners=100, qualityLevel=ql, minDistance=7, blockSize=7)
        self.path = path
        self.trackercount = [0]

    def trackFeature(self):
        cap = cv2.VideoCapture(self.path)
        ret, oldframe = cap.read()                             # 取出视频的第一帧
        if not ret:
            messagebox.showwarning(title='warning', message="Fail to open the file, please try again.")
        oldgray = cv2.cvtColor(oldframe, cv2.COLOR_BGR2GRAY)  # 灰度化
        oldpts = cv2.goodFeaturesToTrack(oldgray, mask=None, **self.feature_params)
        mask = np.zeros_like(oldframe)                         # 为绘制创建掩码图片
        trackerlist = []
        for pt in oldpts:
            trackerlist.append(Point(pt[0]))
            self.trackercount[0] += 1
        
        frame_index = 1
        ret, firstframe = cap.read()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Finished at frame %d. Totally %d lines"% (frame_index, len(trackerlist)))
                cv2.destroyAllWindows()
                cap.release()
                self.printData(trackerlist)
                cv2.imwrite('before cluster.png', cv2.add(firstframe, mask))
                return [frame_index, len(trackerlist)]
            newgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            newpts = cv2.goodFeaturesToTrack(newgray, mask=None, **self.feature_params)
            
            for tracker in trackerlist:
                if tracker.updateflag:
                    tracker.update(oldgray, newgray, self.trackercount)
                
            for pt in newpts:
                flag = 0
                for tracker in trackerlist:
                    if tracker.updateflag == 1:
                        if tracker.isSame(pt[0]):
                            flag = 1
                            break
                if flag == 0:
                    trackerlist.append(Point(pt[0]))
                    self.trackercount[0] += 1

            print("%d trackers in frame %d"% (self.trackercount[0], frame_index))
            oldgray = newgray.copy()
            frame_index += 1
            
            for tracker in trackerlist:
                cv2.circle(mask, (tracker.trajectory[-1][0], tracker.trajectory[-1][1]), 1, tracker.color, -1)

            cv2.imshow("Feature track", cv2.add(frame, mask))
            if cv2.waitKey(30) == 27:
                cv2.destroyWindow("Feature track")
                cap.release()
                self.printData(trackerlist)
                print("[INFO] Stopped at frame %d. Totally %d lines"% (frame_index, len(trackerlist)))
                cv2.imwrite('before cluster.png', cv2.add(frame, mask))
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

class Point():
    def __init__(self, position):
        self.position = position
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.trajectory = [position]
        self.updateflag = 1
        
    def update(self, oldgray, newgray, trackercount):
        oldpos = np.array([[self.trajectory[-1]]], dtype='float32')
        newpos, st, err = cv2.calcOpticalFlowPyrLK(oldgray, newgray, oldpos, None, **self.lk_params)
        if not len(newpos[st==1]):
            self.updateflag = 0
            trackercount[0] -= 1
        elif newpos[0][0][0]+2>720 or newpos[0][0][0]-2<0 or newpos[0][0][1]-2<0 or newpos[0][0][1]+2>480:
            self.updateflag = 0
            trackercount[0] -= 1
            print('[INFO] One tracker out of frame')
        else:
            self.trajectory.append(newpos[0][0])
            
        if len(self.trajectory) >= 5:
            vec1 = np.array([self.trajectory[-4][0]-self.trajectory[-5][0], self.trajectory[-4][1]-self.trajectory[-4][1]])
            vec2 = np.array([self.trajectory[-1][0]-self.trajectory[-2][0], self.trajectory[-1][1]-self.trajectory[-2][1]])
            theta = (np.arccos(vec1.dot(vec2)/(np.sqrt(vec1.dot(vec1))*np.sqrt(vec2.dot(vec2)))))*360/2/np.pi
            
            # distance = np.sqrt(np.square(self.trajectory[-1][0]-self.trajectory[-5][0])+np.square(self.trajectory[-1][1]-self.trajectory[-5][1]))
            if theta > 90:
                self.updateflag = 0
                trackercount[0] -= 1
                print("[INFO] One trajectory split")
    
    def isSame(self, pt):
        thresh = 20
        distance = np.sqrt(np.square(self.trajectory[-1][0]-pt[0])+np.square(self.trajectory[-1][1]-pt[1]))
        if distance < thresh:
            return True
        else:
            return False

if __name__ == '__main__':
    videopath = "Trimed.avi"
    ft = FeatureTracker(videopath)
    ft.trackFeature()