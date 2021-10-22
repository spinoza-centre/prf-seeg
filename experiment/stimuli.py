#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
from psychopy.visual import Line


class FixationLines(object):

    def __init__(self, win, circle_radius, color, pos=[0,0], *args, **kwargs):
        self.color = color
        self.line1 = Line(win, start=(-circle_radius+pos[0], -circle_radius+pos[1]),
                          end=(circle_radius+pos[0], circle_radius+pos[1]), lineColor=self.color, *args, **kwargs)
        self.line2 = Line(win, start=(-circle_radius+pos[0], circle_radius+pos[1]),
                          end=(circle_radius+pos[0], -circle_radius+pos[1]), lineColor=self.color, *args, **kwargs)

    def draw(self):
        self.line1.draw()
        self.line2.draw()

    def setColor(self, color):
        self.line1.color = color
        self.line2.color = color
        self.color = color



        

        
