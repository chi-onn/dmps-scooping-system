#!/usr/bin/env python

import rospy
import matplotlib.pyplot as plt

class PlotNode:
    def __init__(self):
        # Set up the plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])
        self.ax.set_aspect('equal')
        self.circle = plt.Circle((0, 0), 5, fill=True)
        self.ax.add_artist(self.circle)
        self.counter = 0

        # Handle mouse clicks
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.button == 1:  # left mouse button
            x, y = event.xdata, event.ydata
            if self.counter % 2 == 0:
                rospy.loginfo("Start position selected at x={:.2f}, y={:.2f}".format(x, y))
                self.counter +=1
            else:
                rospy.loginfo("Goal position selected at x={:.2f}, y={:.2f}".format(x, y))
                self.counter +=1

    def run(self):
        rospy.init_node('plot_node')
        plt.show()

if __name__ == '__main__':
    node = PlotNode()
    node.run()

    
