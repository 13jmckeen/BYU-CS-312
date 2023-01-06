from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF, QObject
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF, QObject
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time

# Some global color constants that might be useful
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Global variable that controls the speed of the recursion automation, in seconds
#
PAUSE = 0.25


#
# This is the class you have to complete.
#
class ConvexHullSolver(QObject):

    # Class constructor
    def __init__(self):
        super().__init__()
        self.pause = False

    # Some helper methods that make calls to the GUI, allowing us to send updates
    # to be displayed.

    def showTangent(self, line, color):
        self.view.addLines(line, color)
        if self.pause:
            time.sleep(PAUSE)

    def eraseTangent(self, line):
        self.view.clearLines(line)

    def blinkTangent(self, line, color):
        self.showTangent(line, color)
        self.eraseTangent(line)

    def showHull(self, polygon, color):
        self.view.addLines(polygon, color)
        if self.pause:
            time.sleep(PAUSE)

    def eraseHull(self, polygon):
        self.view.clearLines(polygon)

    def showText(self, text):
        self.view.displayStatusText(text)

    # This function is the recursive function that breaks down into divide & conquer chunks
    # My base cases were 1 & 2 as it was less code to let the recursion sort the direction of the hulls instead
    def convex_hull(self, points):
        if len(points) == 1:
            return points
        if len(points) == 2:
            return points
        leftHalf = self.convex_hull(points[:len(points) // 2])
        rightHalf = self.convex_hull(points[len(points) // 2:])
        return self.combine(leftHalf, rightHalf)

    # Returns the rightmost point of a hull
    def right(self, hull_points):
        rightmost_point = max(hull_points, key=lambda point: point.x())
        return hull_points.index(rightmost_point)

    # This function combines 2 hulls, used in the recursion above
    def combine(self, leftHull, rightHull):
        leftmost = 0
        rightmost = self.right(leftHull)

        topRight, topLeft = self.topLine(leftmost, rightmost, leftHull, rightHull)
        bottomRight, bottomLeft = self.bottomLine(leftmost, rightmost, leftHull, rightHull)
        newHull = []

        for x in range(0, topLeft + 1):
            newHull.append(leftHull[x])
        i = 0
        index = topRight
        while i < len(rightHull):
            newHull.append(rightHull[index])
            if index == bottomRight:
                break
            i += 1
            index = (index + 1) % len(rightHull)

        if bottomLeft != topLeft and bottomLeft != 0:
            for x in range(bottomLeft, len(leftHull)):
                newHull.append(leftHull[x])
        return newHull

    # Function to find the top line between 2 hulls
    def topLine(self, leftIndex, rightIndex, leftHull, rightHull):
        bestLeft = leftIndex  # best leftmost point of right hull
        bestRight = rightIndex  # best rightmost point of left hull
        change = True
        while change:
            nextLeft = self.getSlope(leftHull[bestRight], bestLeft, rightHull, 1)
            nextRight = self.getSlope(rightHull[nextLeft], bestRight, leftHull, -1)
            if nextLeft == bestLeft and nextRight == bestRight:
                change = False
            else:
                bestLeft = nextLeft
                bestRight = nextRight
        return bestLeft, bestRight

    # Function to find the index that
    def getSlope(self, bestPoint, startSpot, hull, increment):
        if len(hull) == 1:
            return 0
        best_index = startSpot
        prev_slope = None
        index = startSpot
        for i in range(len(hull)):
            testSlope = (hull[index].y() - bestPoint.y()) / (hull[index].x() - bestPoint.x())
            if prev_slope is None:
                prev_slope = testSlope
            elif (testSlope > prev_slope and increment == 1) or (testSlope < prev_slope and increment == -1):
                prev_slope = testSlope
                best_index = index
            else:
                best_index = (index - increment) % len(hull)
                break
            index = (index + increment) % len(hull)
        return best_index

    # Function to find the bottom line in 2 hulls
    def bottomLine(self, leftPoint, rightPoint, leftHull, rightHull):
        best_left = leftPoint
        best_right = rightPoint
        flag = True
        while flag:
            next_left = self.getSlope(leftHull[best_right], best_left, rightHull, -1)
            next_right = self.getSlope(rightHull[next_left], best_right, leftHull, 1)
            if next_left == best_left and next_right == best_right:
                flag = False
            else:
                best_left = next_left
                best_right = next_right
        return best_left, best_right

    # This is the method that gets called by the GUI and actually executes
    # the finding of the hull
    def compute_hull(self, points, pause, view):
        self.pause = pause
        self.view = view
        assert (type(points) == list and type(points[0]) == QPointF)
        t1 = time.time()
        points.sort(key=lambda points: points.x())
        t2 = time.time()

        t3 = time.time()

        newPoints = self.convex_hull(points)
        lines = []
        for i in range(0, len(newPoints)):
            if i == len(newPoints) - 1:
                lines.append(QLineF(newPoints[i], newPoints[0]))
            else:
                lines.append(QLineF(newPoints[i], newPoints[i + 1]))
        t4 = time.time()

        # when passing lines to the display, pass a list of QLineF objects.  Each QLineF
        # object can be created with two QPointF objects corresponding to the endpoints
        self.showHull(lines, RED)
        self.showText('Time Elapsed (Convex Hull): {:3.3f} sec'.format(t4 - t3))
