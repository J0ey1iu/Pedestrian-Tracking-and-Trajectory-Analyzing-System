#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 19:15:58 2019

@author: liujiayu
"""

from FEATURE_POINT_DEMO3 import *
from HOGtrack import *
from clustering_github_edit import *
from filter import Filter
from tkinter import *
from tkinter.filedialog import askopenfilename
import tkinter.messagebox as messagebox
from tkinter.simpledialog import askinteger, askfloat
import cv2
import time
import sys

class Intro():
    def __init__(self, master):
        frame = Frame(master)
        frame.pack(side=TOP)
        
        self.introlabel = Label(frame, text="This is a pedestrian tracking and analyzing system.\n"\
                                + "Click the buttons to use.\n"\
                                + "Hints are on the bottom.")
        self.introlabel.pack(anchor=N)
        
class FuncButtons():
    def __init__(self, master):
        frame = Frame(master)
        frame.pack(side=TOP, pady=10)
        
        Button(frame, text='Choose Video', command=self.chooseVideo).pack()
        Button(frame, text='Start tracking', command=self.trackVideo).pack()
        Button(frame, text='Start Analyzing', command=self.startAnalyzing).pack()
        Button(frame, text='Quit', command=frame.quit, fg="red").pack()
        
        self.method = 0
        
    def chooseVideo(self):
        print("\033[1;32m[INFO] Choosing video...\033[0m")
        global path
        global status
        global logfile
        path = askopenfilename(filetypes=[('mp4', '*.mp4'), ('avi', '*.avi'), ('mov', '*.mov'), ('wmv', '*.wmv')])
        if path == "":
            status.refresh("You didn't choose any video. Try again.")
            logfile.write("[ERROR] Didn't choose any video.\n")
            messagebox.showwarning(title='warning', message="You didn't choose any video. Try again.")
        else:
            logfile.write("[INFO] Video chosen, \"%s\"\n" % path)
            status.refresh("You've chosen a video. Now hit track button.")

    def trackVideo(self):
        global path

        newwindow = Toplevel()
        newwindow.title("Choose track method")
        newwindow.geometry("300x80")

        options = [('HOG track', 1), ('Feature track', 2)]
        v = IntVar()
        v.set(2)
        for optiontext, optionnum in options:
            b = Radiobutton(newwindow, text=optiontext, variable=v, value=optionnum)
            b.pack()
        subbutton = Button(newwindow, text='submit', command=lambda:self.submit1(newwindow, v))
        subbutton.pack()
    
    def submit1(self, window, variable):
        global path
        global status
        global logfile
        if path == "":
            status.refresh("You didn't choose any video. Try again.")
            logfile.write("[ERROR] Didn't choose any video.\n")
            messagebox.showwarning(title='warning', message="You didn't choose any video. Try again.")
            window.destroy()
        else:
            window.destroy()
            print("\033[1;32m[INFO] User chose Method No.%d\033[0m" % variable.get())
            framenum = 0
            trackernum = 0
            if variable.get() == 1:
                logfile.write("[INFO] User chose HOG track.\n")
                ht = HOGTracker(path)
                framenum, trackernum = ht.startTracking()
            elif variable.get() == 2:
                logfile.write("[INFO] User chose Feature track.\n")
                ql = askfloat("Input qualityLevel", "qualityLevel(the higher the value is, the less points there will be): ", initialvalue=0.4)
                ft = FeatureTracker(path, ql)
                framenum, trackernum = ft.trackFeature()
            status.refresh("%d frames were analyzed, %d trajectories detected"% (framenum, trackernum))
            logfile.write("[INFO] %d frames were analyzed, %d trajectories detected.\n"% (framenum, trackernum))
            
    def startAnalyzing(self):
        newwindow = Toplevel()
        newwindow.title("Choose cluster method")
        newwindow.geometry("300x80")

        options = [('Agglomerative Clustering', 1), ('Spectrum Clustering', 2)]
        v = IntVar()
        v.set(2)
        for optiontext, optionnum in options:
            b = Radiobutton(newwindow, text=optiontext, variable=v, value=optionnum)
            b.pack()
        subbutton = Button(newwindow, text='submit', command=lambda:self.submit2(newwindow, v))
        subbutton.pack()

    def submit2(self, window, variable):
        global status
        global path
        global logfile
        if path == "":
            status.refresh("You didn't choose any video. Try again.")
            logfile.write("[ERROR] Didn't choose any video.\n")
            messagebox.showwarning(title='warning', message="You didn't choose any video. Try again.")
            window.destroy()
        else:
            window.destroy()
            status.refresh("Filtering trajectories data")
            filter = Filter('trajectories.txt', 'trajectories_filtered.txt')
            filterlength = askinteger("Input filter length", "filter length: ", initialvalue=60)
            filter.filter(filterlength)
            
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            mask = np.zeros_like(frame)
            colors = []
            for i in range(30):
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                colors.append(color)

            file = open('trajectories_filtered.txt', 'r')
            trajs = []
            for i, line in enumerate(file):
                traj = Trajectory(i)
                pts = line.split(',')[:-1]
                for pt in pts:
                    xy = pt.split(' ')
                    x = float(xy[0])
                    y = float(xy[1])
                    xy = (x, y)
                    traj.addPoint(xy)
                trajs.append(traj)

            print("[INFO] Read in %d trajectories."% len(trajs))

            status.refresh("Clustering")
            clust = Clustering()

            if variable.get() == 1:
                logfile.write("[INFO] User used clusterAgglomerartive.\n")
                num = askinteger("Input number of clusters", "Number of clusters: ", initialvalue=3)
                res = clust.clusterAgglomerartive(trajs, num)
            elif variable.get() == 2:
                logfile.write("[INFO] User used spectrum cluster.\n")
                res = clust.clusterSpectral(trajs)

            status.refresh("Showing result")
            print("printing image")
            for traj in trajs:
                for pt in traj.getPoints():
                    cv2.circle(mask, (int(pt[0]), int(pt[1])), 1, colors[traj.getClusterIdx()-1], -1)

            count = [0]*res
            for traj in trajs:
                count[traj.getClusterIdx()-1] += 1
            print(count)

            logfile.write("[INFO] Got %d clusters, %s.\n" % (res, str(count)))
            
            cv2.imshow("result", cv2.add(frame, mask))
            cv2.imwrite("after cluster.png", cv2.add(frame, mask))
            if cv2.waitKey(0) == 27:
                cap.release()
                cv2.destroyAllWindows()
                file.close()
            
class Status():
    def __init__(self, master):
        frame = Frame(master)
        frame.pack(side=BOTTOM)
        global path
        
        self.statustext = StringVar()
        self.statustext.set("You haven't chosen any video yet.")
        self.status = Label(frame, textvariable=self.statustext, bd=1, relief=SUNKEN, anchor=W)
        self.status.pack(side=BOTTOM, fill=X)
    
    def refresh(self, text):
        self.statustext.set(text)
    
if __name__ == '__main__':
    logfile = open('userlog.log', 'a')
    logfile.write("\n\n\n"+time.asctime()+"\n")
    root = Tk()
    root.title("demo")
    root.geometry('500x500')
    
    path = ""
    top = Intro(root)
    bottom = FuncButtons(root)
    status = Status(root)
    
    try:
        root.mainloop()
    except:
        logfile.write("[ERROR] " + sys.exc_info()[0])
        logfile.close()

    # print(path)
    logfile.close()