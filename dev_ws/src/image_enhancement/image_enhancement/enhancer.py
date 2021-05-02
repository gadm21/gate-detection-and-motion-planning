


import numpy as np 
import cv2 
import argparse  
import os 

from keras.models import model_from_json

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int64, String 





def get_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--model', default=None)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--verbose', default=0, type=int)

    parser.add_argument('--subscriber_topic_name_1', default="right_raw/image_rect_color", type=str)
    parser.add_argument('--subscriber_topic_name_2', default="left_raw/image_rect_color", type=str)
    parser.add_argument('--subscriber_message_type', default = "string", type=str)

    parser.add_argument('--publisher_topic_name', default="enhanced_frame", type=str)
    parser.add_argument('--publisher_message_type', default = "string", type=str)
    parser.add_argument('--operation_time_period', default = 0.5, type=float)

    args = parser.parse_args() 
    return args 




class Subscriber():

    def __init__(self, args, flag = 0):

        self.message_type = args.subscriber_message_type.lower()
        if not flag : self.topic_name = args.subscriber_topic_name_1.lower()
        else : self.topic_name = args.subscriber_topic_name_2.lower()
        self.message_id = 0 
        self.message = None

        if self.message_type == 'int' :
            self.subscriber_ = self.create_subscription( Int64, self.topic_name, self.listener_callback)
        elif self.message_type == 'string' :
            self.subscriber_ = self.create_subscription( String, self.topic_name, self.listener_callback)
        else: 
            print("unkown message type")

        
        
    def listener_callback(self, msg):
        
        if self.message_type == 'string':
            self.message = str(msg.data) 
            self.message_id += 1    

        elif self.message_type == 'int':
            self.message = int(msg.data) 
            self.message_id += 1    

        else:
            print("unkown message type while listening")
            


class Publisher():

    def __init__(self, args): 
        
        self.message_type = args.publisher_message_type.lower()
        self.topic_name = args.publisher_topic_name.lower()
        self.message_id = 0 
        self.message = None

        if self.message_type == 'string' :
            self.publisher_ = self.create_publisher(String, self.topic_name)
        elif self.message_type == 'int' :
            self.publisher_ = self.create_publisher(Int64, self.topic_name)
        else:
            print("unkown message type")            

        self.subscription = self.create_subscription( String, 'direction', self.listener_callback)
        

    def publish(self, message):

        if self.message_type == 'string':
            msg = String() 
            msg.data = message
            self.message_id += 1
            
        elif self.message_type == 'int':
            msg = Int64()
            msg.data = message
            self.message_id += 1

        else:
            print("unkown message type while sending")

        self.publisher_.publish(msg) 


class GAN():

    def __init__(self, args):
        
        self.model = self.get_model(args.model)
        self.model.load_weights(args.weights)
        self.verbose = args.verbose


    def get_model(self, json_file_path):
        with open(json_model_path, 'r') as file :
            json_model = file.read()
        
        return model_from_json(json_model)


    def preprocess(self, image):
        batch = np.array([image]) # convert to batch
        return (batch/127.5)-1.0


    def predict(self, image):
        batch = self.preprocess(image)
        prediction = self.model.predict(batch)
        return self.deprocess(prediction)


    def deprocess(self, prediction):
        prediction = prediction[0] # take only the first item in the batch because we only work on individual images
        return np.uint8((prediction+1.0)*127.5) 


class Enhancer(Node):

    def __init__(self, args):
        super().__init__('image_enhancement')
        self.args = args 

        self.subscriber_1 = Subscriber(args, flag =0)
        self.subscriber_2 = Subscriber(args, flag =1)
        self.publisher = Publisher(args)
        self.gan = GAN(args)
        self.done = False

        self.timer = self.create_timer(args.operation_time_period, self.timer_callback)



    def timer_callback(self):
        if self.done : return 
    
        self.operate()


    def operate(self):

        image_1 = self.subscriber_1.message
        enhanced_image_1 = self.gan.predict(image_1)
        self.publisher.publish(enhanced_image_1)












def main(args):
    rclpy.init(args=None)

    enhancer = Enhancer(args)

    rclpy.spin(enhancer)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()




if __name__ == '__main__':
    main()


