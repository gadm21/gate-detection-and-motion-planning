


import numpy as np 
import cv2 
import argparse  
from matplotlib import pyplot as plt
#from scipy.signal import find_peaks_cwt
from peakutils.peak import indexes
import os 
import imutils
from time import sleep 

def get_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--image', default=None)
    parser.add_argument('--verbose', default=0, type=int)
    args = parser.parse_args() 
    return args 



def read_image(image) :
    return cv2.imread(image) 

def show_image(image):
    cv2.imshow('r',image)
    cv2.waitKey(0) 
    cv2.destroyWindow('r')

#converts the image from RGB to gray scale
def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#converts the image from RGB or gray scale to hsv color space 
def to_hsv(image):
    if len(image.shape)==3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.uint8) 
    else: return None

#binarize the image using an explicit threshold
def to_binary(image, threshold = 110):
    if len(image.shape)==3: image = to_gray(image) 
    light = np.where(image > threshold) 
    dark = np.where(image <= threshold) 
    newimage = image.copy()
    newimage[light] = 255 
    newimage[dark] = 0
    return newimage

#binarize the image using adaptive threshold
def to_binary2(image):
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  cv2.THRESH_BINARY, 15, 5)
    return thresh

#binarize the image using  Otsu’s thresholding
def to_binary3(image):
    if len(image.shape) == 3 : image = to_gray(image)
    _, thresh = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY +  cv2.THRESH_OTSU)
    return thresh

#apply histogram equalization on the image
def equalize(image):
    if len(image.shape) ==3 : image = to_gray(image) 
    newimage = cv2.equalizeHist(image) 
    return newimage

#perform dilation 
def dilate(image, k=5):
    kernel = np.ones((k,k), np.uint8) 
    newimage = cv2.dilate(image, kernel, iterations = 1) 
    return newimage 

#apply median filter on the image
def median(image, k=3):
    return cv2.medianBlur(image, k)

#return the histogram of the image
def to_hist(image):
    d = dict()
    for p in image.ravel():
        if d.get(p): d[p] +=1 
        else : d[p] = 1
    #print(d) 
    #print()
    hist, _ = np.histogram(image.ravel(), 256, [0,256])

    return hist

#plot the histogram  of the image
def plot_hist(hist):
    xs = np.arange(len(hist)) 
    plt.plot(xs, hist) 
    plt.show()

#the return the peaks of the sequence of the sum of column pixels of the image
def get_peaks(image, distance ):
    seq = np.sum(image, axis=0, dtype =int) 
    maxx = np.max(seq) 
    thresh = maxx * 0.25
    #peaks,_ = find_peaks(seq, height=thresh, width = 10)
    peaks = indexes(seq, min_dist = distance) 
    #print("peaks:", len(peaks), " ", peaks)
    return peaks 

#return a bigger image of the input image (not scaling)
def bigger_image(image, part=1):
    h, w = image.shape[:2]
    image = cv2.resize(image, (w//3, h//3))

    if len(image.shape)==3:
        newimage = np.zeros((h, w, 3), dtype = np.uint8)
        newimage[0:h//3, 0:w//3,:] = image.copy()
    else:
        newimage = np.zeros((h, w), dtype = np.uint8)
        newimage[0:h//3, 0:w//3] = image.copy()     
    return newimage

#using connected component analysis to remove noise from the image
def remove_noise(image, min_area_threshold= 1000):
    if len(image.shape) == 3 : image = to_gray(image) 

    _, labels, stats, _= cv2.connectedComponentsWithStats(image, connectivity= 8, ltype= cv2.CV_32S)

    for index, stat in enumerate(stats) : 
        if stat[4] < min_area_threshold:
            noise_indecies= np.where(labels == index)
            image[noise_indecies] = 0

    return image 

#gate detection pipeline
def pipeline(image):
    hsv = to_hsv(image)[:,:,1] 
    hsv = 255 - hsv 
    medianed = cv2.medianBlur(hsv, 3) 
    binary1 = to_binary(medianed, threshold= 100)  
    binary3 = to_binary3(medianed) 
    binary = binary1 | binary3 
    dilated = dilate(binary, k=1) 
    clear = remove_noise(dilated) 
    return clear

#check whether the given contour is for the gate bar 
def is_gate_bar(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True) 
    (x, y, w, h) = cv2.boundingRect(approx) 
    if h < (3*w) : return False,  (x, y, w, h)
    return True, (x, y, w, h)

'''
def show_cnts(image, cnts):
    for c in cnts :
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True) 
        (x, y, w, h) = cv2.boundingRect(approx)
        coords1= (x,y) 
        coords2= (x+w, y+h)
        cv2.rectangle(image, coords1, coords2, 255, 2)
    show_image(image) 
'''

def get_region(image, coords):

    def center(coords):
        x = coords[0][0] + (coords[1][0] - coords[0][0]) // 2
        y = coords[0][1] + (coords[1][1] - coords[0][1]) // 2 
        return x,y
    
    regions = np.arange(1, 10).reshape((3,3))

    x,y = center(coords)
    #print("coords:", x, " ", y)
    h, w = 0, 0
    
    thresh1 = image.shape[0] // 3
    thresh2 = thresh1 *2 
    if y > thresh2 : h= 2
    elif y > thresh1 : h= 1

    thresh1 = image.shape[1] // 3
    thresh2 = thresh1 *2
    if x > thresh2 : w= 2
    elif x > thresh1 : w= 1

    return regions[h][w]
    

#entrance point for detecting gate
def is_gate(original, to_big=True, part=1):
    image = pipeline(original)  
    if to_big: 
        image = bigger_image(image, part)  
        original = bigger_image(original, part)
    #show_image(image) 
    #distance_between_peaks = image.shape[1]//4 if not to_big else image.shape[1]//12
    distance_between_peaks = image.shape[1]//12
    peaks = get_peaks(image, distance = distance_between_peaks) 
    
    if len(peaks) != 2 and len(peaks) != 3 :
        return False, None, None
    
    cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)   
    
    min_00, min_01, max_10, max_11 = peaks[0], np.max(np.nonzero(image)[0]), peaks[-1], np.min(np.nonzero(image)[0]) 
    for c in cnts:
        flag, (x,y,w,h)= is_gate_bar(c)
        if not flag : continue  
        y2 = y + h 
        if min_01 > y : min_01 = y  
        if max_11 < y2 : max_11 = y2 
        
    return True, original, ((min_00, min_01), (max_10, max_11))

def draw_coords(image, coords):
    if len(image.shape)==3:
        cv2.rectangle(image, coords[0], coords[1], (0, 255, 0), 4)
    else : cv2.rectangle(image, coords[0], coords[1], 255, 4)

def check_image(args):
    if args.image is None : return False 
    image = read_image(args.image) 
    if image is None : return False 
    return True 

def update_image(image, directions):
    h, w = image.shape[:2]
    newimage = np.zeros(image.shape) 

    offset_x, offset_y = w//20, h//20

    if 'up' in directions : 
        if 'left' in directions: newimage[offset_y:, offset_x:, :] = image[ :-offset_y, :-offset_x, :]
        elif 'right' in directions: newimage[offset_y:, :-offset_x, :] = image[ :-offset_y, offset_x, :]
        else :newimage[offset_y:, :, :] = image[ :-offset, :, :]
    if 'down' in directions :
        if 'left' in directions: newimage[:-offset_y, offset_x:, :] = image[offset_y:, :-offset_x, :]
        elif 'right' in directions: newimage[:-offset_y, :-offset_x, :] = image[offset_y:, offset_x, :]
        else :newimage[:-offset_y, :, :] = image[offset_y:, :, :]
    else: newimage = image 

    return newimage 


























































import rclpy
from rclpy.node import Node

from std_msgs.msg import Int64, String 





class MinimalPublisher(Node):

    def __init__(self, args): 
        super().__init__('gate_detection')
        self.done = False
        self.image = read_image(args.image)
        self.verbose = args.verbose
        self.publisher = self.create_publisher(Int64, 'gate_location')
        self.subscription = self.create_subscription( String, 'direction', self.listener_callback)
        timer_period = 0.3  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        if self.done : return 
        msg = Int64()
        
        f, newimage, coords = is_gate(self.image.copy(), to_big=False, part=1) 
        if not f :
            region = int(-1)
        else :
            region = int(get_region(newimage, coords) )
            draw_coords(newimage, coords) 
            #show_image(newimage)

        if region==5 : self.done = True 
        msg.data = region
        self.publisher.publish(msg)
        self.get_logger().info('image is at region ' + str( msg.data))
        
    def listener_callback(self, msg):
        if self.done : return 
        
        self.get_logger().info('Planner is telling me to go ' + msg.data)
        #self.image = update_image(self.image, msg.data)
        if self.verbose : show_image(self.image)




def start(args, ros_args=None):
    rclpy.init(args=ros_args)

    minimal_publisher = MinimalPublisher(args)

    rclpy.spin(minimal_publisher)


    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


def main():
    args = get_args() 
    if not check_image(args) :
        print("image wasnot provided or is corrupted")
        return 

    start(args)
    



if __name__ == '__main__':
    main()


