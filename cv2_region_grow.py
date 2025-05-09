import cv2
import itertools
import numpy as np
import random
import sys
from skimage import filters

# Load and prepare the image
img = cv2.imread("/Users/zebpalm/Exjobb 2025/Coding/gaussian_filtering_all_results/Sample 4/ABC-capture-20241209-194134_topo02_13_gaussian_k7_s1.5.png", cv2.IMREAD_GRAYSCALE)

# Apply Sobel filter and convert to proper format
sobel = filters.sobel(img)
sobel = (sobel * 255).astype(np.uint8)  # Convert to uint8
cv2.imshow("Sobel Image", sobel)
cv2.waitKey(1)  # Show the Sobel image but don't wait

# Class for a stack
class Stack():
    def __init__(self):
        self.item = []
        self.obj=[]
    def push(self, value):
        self.item.append(value)

    def pop(self):
        return self.item.pop()

    def size(self):
        return len(self.item)

    def isEmpty(self):
        return self.size() == 0

    def clear(self):
        self.item = []

class regionGrow():
  
    def __init__(self, image, th):
        if isinstance(image, str):
            self.readImage(image)
        else:
            # Ensure the input image is uint8 and 3-channel
            if len(image.shape) == 2:
                self.im = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                self.im = image
            self.im = self.im.astype('int')
        self.h, self.w, _ = self.im.shape
        self.passedBy = np.zeros((self.h,self.w), np.double)
        self.currentRegion = 0
        self.iterations=0
        self.SEGS=np.zeros((self.h,self.w,3), dtype='uint8')
        self.stack = Stack()
        self.thresh=float(th)

    def readImage(self, img_path):
        self.im = cv2.imread(img_path,1).astype('int')

    def getNeighbour(self, x0, y0):
        return [
            (x, y)
            for i, j in itertools.product((-1, 0, 1), repeat=2)
            if (i, j) != (0, 0) and self.boundaries(x := x0 + i, y := y0 + j)
        ]

    def create_seeds(self):
        return [
            [self.h/2,self.w/2],
            [self.h/3,self.w/3],[2*self.h/3,self.w/3],[self.h/3-10,self.w/3],
            [self.h/3,2*self.w/3],[2*self.h/3,2*self.w/3],[self.h/3-10,2*self.w/3],
            [self.h/3,self.w-10],[2*self.h/3,self.w-10],[self.h/3-10,self.w-10]
                    ]
    def ApplyRegionGrow(self, cv_display = True):

        randomseeds = self.create_seeds()
        np.random.shuffle(randomseeds)

        for x0 in range (self.h):
            for y0 in range (self.w):

                if self.passedBy[x0,y0] == 0 : #and (np.all(self.im[x0,y0] > 0)) :  
                    self.currentRegion += 1
                    self.passedBy[x0,y0] = self.currentRegion
                    self.stack.push((x0,y0))
                    self.prev_region_count= 0

                    while not self.stack.isEmpty():
                        x,y = self.stack.pop()
                        self.BFS(x,y)
                        self.iterations+=1

                    if self.PassedAll():
                        break

                    if self.prev_region_count< 8*8 :     
                       x0, y0 = self.reset_region(x0, y0)

        if self.iterations>200000:
            print("Max Iterations")
        print(f"Iterations : {str(self.iterations)}")

        if cv_display:
            [self.color_pixel(i,j) for i, j in itertools.product(range(self.h), range (self.w))]
            self.display()

    def reset_region(self, x0, y0):
        self.passedBy[self.passedBy==self.currentRegion] = 0
        x0=random.randint(x0-4,x0+4)
        y0=random.randint(y0-4,y0+4)
        x0 = np.clip(x0, 0, self.h - 1)
        y0 = np.clip(y0, 0, self.w - 1)
        self.currentRegion-=1
        return x0, y0

    def color_pixel(self, i, j):
        val = self.passedBy[i][j]
        self.SEGS[i][j] = (255, 255, 255) if (val==0) else (val*35, val*90, val*30)

    def display(self):
        cv2.imshow("Region Growing Result", self.SEGS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def BFS(self, x0,y0):
        regionNum = self.passedBy[x0,y0]
        elems = [np.mean(self.im[x0, y0])]
        var=self.thresh
        neighbours=self.getNeighbour(x0,y0)

        for x,y in neighbours:
            if self.passedBy[x,y] == 0 and self.distance(x,y,x0,y0) < var:
                if self.PassedAll():
                    break
                self.passedBy[x,y] = regionNum
                self.stack.push((x,y))
                elems.append(np.mean(self.im[x,y]))
                var=np.var(elems)
                self.prev_region_count+=1
            var=max(var,self.thresh)
    
    def PassedAll(self, max_iteration = 200000):
        return self.iterations > max_iteration or np.all(self.passedBy > 0)

    def boundaries(self, x,y):
        return  0<=x<self.h and 0<=y<self.w
    
    def distance(self,x,y,x0,y0):
        return np.linalg.norm(self.im[x0, y0] - self.im[x, y])

# Example usage - using the Sobel image
exemple = regionGrow(sobel, "20")  # Using a threshold of 20 for uint8 image
exemple.ApplyRegionGrow()