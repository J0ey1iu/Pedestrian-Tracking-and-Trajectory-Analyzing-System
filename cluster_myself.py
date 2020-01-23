#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:49:44 2019

@author: liujiayu
"""

import numpy as np
import random
import cv2

def d(A, B):
    minab = []
    minba = []
    for pt1 in A:
        temp = None
        for pt2 in B:
            distance = np.sqrt(np.square(pt1[0]-pt2[0])+np.square(pt1[1]-pt2[1]))
            if (temp is None) or (distance < temp):
                temp = distance
        minab.append(temp)
    
    for pt2 in B:
        temp = None
        for pt1 in A:
            distance = np.sqrt(np.square(pt1[0]-pt2[0])+np.square(pt1[1]-pt2[1]))
            if (temp is None) or (distance < temp):
                temp = distance
        minba.append(temp)
                
    Dab = max(minab)
    Dba = max(minba)
    
    if Dab > Dba:
        return Dab
    else:
        return Dba
    
def file2trackerlist(file):
    trackerlist = []
    for line in file:
        tracker = []
        pts = line.split(',')
        for pt in pts:
            if pt != '\n':
                pt = pt.split(' ')
                x = float(pt[0])
                y = float(pt[1])
                tracker.append([x, y])
        trackerlist.append(tracker)
    return trackerlist

def cluster(trackerlist, K=2):
    thresh = 100
    trackercount = len(trackerlist)
    init = [random.randint(0, trackercount-1)]
    for i in range(1, K):
        init.append(random.randint(0, trackercount-1))
        k = i - 1
        while k >= 0:
            while d(trackerlist[init[k]], trackerlist[init[i]]) < thresh:
                init[i] = random.randint(0, trackercount-1)
            k -= 1
    print(init)
    preinit = init
    while True:
        print("another cycle")
        clusters = []
        for i in range(0, K):
            clusters.append([])
#        print(clusters)
        # 给所有tracker分类
        print("sorting trackers")
        for j, tracker in enumerate(trackerlist):
            temp = [None, None]
            for i, number in enumerate(init):
                distance = d(tracker, trackerlist[number])
                if temp[0] is None or distance < temp[0]:
                    temp[0] = distance
                    temp[1] = i
            clusters[temp[1]].append(j)
#        print(clusters)
        # 重新选择中心
        print("renewing center")
        for i, cluster in enumerate(clusters):
            mini = [None, None]
            for j in range(0, len(cluster)):
                temp = 0
                for k in range(0, len(cluster)):
                    distance = d(trackerlist[cluster[j]], trackerlist[cluster[k]])
                    temp += distance
                if mini[1] is None or temp<mini[0]:
                    mini[0] = temp
                    mini[1] = cluster[j]
            init[i] = mini[1]
            print(init)
        if init == preinit:
            return init, clusters
        else:
            print("not match")
            preinit = init

if __name__ == '__main__':
    file = open('trajectories_filtered.txt', 'r')
    trackerlist = file2trackerlist(file)
    centers, clusters = cluster(trackerlist, 6)
    
    cap = cv2.VideoCapture('Trimed.avi')
    ret, frame = cap.read()
    mask = np.zeros_like(frame)
    
    for cluster in clusters:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for tracker in cluster:
            for position in trackerlist[tracker]:
                cv2.circle(mask, (int(position[0]), int(position[1])), 2, color, -1)
    
    cv2.imshow("clustered", cv2.add(frame, mask))
    cv2.waitKey(0)
    
    file.close()
    cap.release()