#!/usr/bin/python
import sys
import re

if(len(sys.argv) < 3):
    print("Usage : tolab [txt] [lab]")
    sys.exit(0)

inputName = sys.argv[1]
outputName = sys.argv[2]

inputFile = open(inputName, 'r')
outputFile = open(outputName, 'w')

r = re.compile('[~:\n]')
for line in inputFile:
    arr = r.split(line)
    print(arr)
    if(len(arr) >= 6):
        h1, m1, s1 = int(arr[0]), int(arr[1]), int(arr[2])
        h2, m2, s2 = int(arr[3]), int(arr[4]), int(arr[5])
        t1 = h1 * 3600 + m1 * 60 + s1
        t2 = h2 * 3600 + m2 * 60 + s2
        print(t1, t2)
        outputFile.write("%s %s sing\n" % (t1, t2))

inputFile.close()
outputFile.close()
