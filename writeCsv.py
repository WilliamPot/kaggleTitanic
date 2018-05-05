# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:51:07 2018

@author: Chen
"""

from imTools import mhdNpyP,csvP
array = mhdNpyP.loadDataFromNpy('submission.npy')
output = [['PassengerId','Survived']]
pid = 892
for i in range(len(array)):
    output.append([pid,array[i]])
    pid += 1
csvP.writeDataIntoCsv(output)