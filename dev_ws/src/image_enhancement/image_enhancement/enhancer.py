


import numpy as np 
import cv2 
import argparse  
import os 

from keras.models import model_from_json

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int64, String 
from sensor_msgs.msg import Image
# from sensor_msgs.msg import UAVImage




def get_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--model', default=None)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--verbose', default=0, type=int)

    parser.add_argument('--method', default="color_correction", type=str)
    parser.add_argument('--use_power', default="no", type=str)
    

    args = parser.parse_args() 
    return args 





class GAN():

    def __init__(self, args):
        
        print()
        print("model:", args.model)
        print()
        self.model = self.get_model(args.model)
        self.model.load_weights(args.weights)
        self.verbose = args.verbose


    def get_model(self, model):
        with open(model, 'r') as file :
            json_model = file.read()
        
        return model_from_json(json_model)


    def preprocess(self, image):
        batch = np.array([image]) # convert to batch
        return (batch/127.5)-1.0


    def operate(self, image):
        original_size = image.shape[0:2]
        image = cv2.resize(image, (256, 256))
        batch = self.preprocess(image)
        prediction = self.model.predict(batch)
        enhanced_image = self.deprocess(prediction)
        return cv2.resize(enhanced_image, original_size)


    def deprocess(self, prediction):
        prediction = prediction[0] # take only the first item in the batch because we only work on individual images
        return np.uint8((prediction+1.0)*127.5) 

class Color_correction:
    def __init__(self, args):
        self.args = args
        
        if self.args.use_power.lower() in ('yes', 'true', 't', 'y', '1') : 
            self.use_power = True
        else: self.use_power = False
        

    # Automatic brightness and contrast optimization with optional histogram clipping
    def Histogram_stretching (self, channel, clip_hist_percent=1.5):
        # Calculate channel histogram
        hist = cv2.calcHist([channel],[0],None,[256],[0,256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_ch = 0
        while accumulator[minimum_ch] < clip_hist_percent:
            minimum_ch += 1

        # Locate right cut
        maximum_ch = hist_size -1
        while accumulator[maximum_ch] >= (maximum - clip_hist_percent):
            maximum_ch -= 1
            
        # Calculate alpha and beta values
        alpha = 255 / (maximum_ch - minimum_ch)
        beta = -minimum_ch * alpha
        
        image_cs = cv2.convertScaleAbs(channel, alpha=alpha, beta=beta)
        return image_cs


    def channels_stretching (self, image):
        R, G, B = cv2.split(image)
        R_s = self.Histogram_stretching(R)
        G_s = self.Histogram_stretching(G)
        B_s = self.Histogram_stretching(B)
        rgb_image = cv2.merge([R_s, G_s, B_s])

        H, S, V = cv2.split(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV))
        S_s = self.Histogram_stretching(S)
        V_s = self.Histogram_stretching(V)
        
        hsv_image = cv2.cvtColor(cv2.merge([H, S_s, V_s]), cv2.COLOR_HSV2RGB)

        return hsv_image


    def power_transformation(self, image):
        info = np.iinfo(image.dtype) # Get the information of the incoming image type
        normalized = image.astype(np.float64) / info.max # normalize the data to 0 - 1

        powered_img = cv2.pow(normalized,1.2)
        powered_img = 255 * powered_img # Now scale by 255
        powered_img = powered_img.astype(np.uint8)
        return powered_img  
    

    def operate(self, image):
        if self.use_power :
            image = self.power_transformation(image)

        return self.channels_stretching(image= image)


class Enhancer(Node):

    def __init__(self, args):
        super().__init__('image_enhancement')

        self.args = args 

        self.z_frame_id = 0
        self.l_frame_id = 0
        self.z_frame = cv2.imread("/home/gad/Desktop/person.png")
        self.l_frame = cv2.imread("/home/gad/Desktop/person.png")
        
        self.method = self.args.method.lower()
        if self.method == "color_correction" : self.operator = Color_correction(args)
        elif self.method == "gan" : self.operator = GAN(args)

        self.l_subscriber = self.create_subscription(Image, 'lowlight_camera', self.l_callback, 10)
        self.z_subscriber = self.create_subscription(Image, '~/left_raw/image_rect_color', self.z_callback, 10)

        self.l_publisher = self.create_publisher(Image, 'enhanced_frame_l', 10)
        self.z_publisher = self.create_publisher(Image, 'enhanced_frmae_z', 10)
        
        # self.timer = self.create_timer(0.5, self.operate)


    def l_callback(self, msg):
        self.l_frame = np.array(msg.data).reshape((msg.height, msg.width, msg.step))
        self.l_frame_id += 1    

        self.operate('l')


    def z_callback(self, msg):
        self.z_frame = np.array(msg.data).reshape((msg.height, msg.width, msg.step))
        self.z_frame_id += 1

        self.operate('z')


    def np2ros(self, image):
        msg = Image()
        msg.height, msg.width, msg.step = image.shape
        msg.data = image.reshape(-1).tolist()
        return msg


    def operate(self, code):

        if code == 'l': image = self.l_frame.copy()
        elif code == 'z': image = self.z_frame.copy()
        
        enhanced_image = self.operator.operate(image)
        # show_image(enhanced_image)
        enhanced_image = self.np2ros(enhanced_image)
        if np.random.randint(0, 50) % 2 :
            self.l_publisher.publish(enhanced_image)
        else: 
            self.z_publisher.publish(enhanced_image)



    def show_image(self, image):
        cv2.imshow('r',image)
        cv2.waitKey(0) 
        cv2.destroyWindow('r')





def main():
    rclpy.init(args=None)

    args = get_args()
    enhancer = Enhancer(args)

    rclpy.spin(enhancer)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    enhancer.destroy_node()
    rclpy.shutdown()




if __name__ == '__main__':
    
    print("start")
    main(args)


