# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node

from std_msgs.msg import Int64, String


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('motion_planning')
        self.publisher_ = self.create_publisher(String, 'direction')
        self.subscription = self.create_subscription( Int64, 'gate_location', self.listener_callback)

        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.listen_id = 0 
        self.talk_id = 0
        self.location = None

    def listener_callback(self, msg):
        self.location = msg.data 
        self.listen_id += 1 

        self.get_logger().info('I think I heard: ' + str(msg.data))

    def timer_callback(self):
        if self.talk_id == self.listen_id : return
        self.talk_id += 1

        msg = String()
        msg.data = self.find_direction()

        self.publisher_.publish(msg)
        self.get_logger().info('saying: ' + str( msg.data))

    def find_direction(self):
        direction = '' 

        if self.location == 1 or self.location == 2 or self.location == 3 : direction += ' up' 
        if self.location == 1 or self.location == 4 or self.location == 7 : direction += ' left' 
        if self.location == 3 or self.location == 6 or self.location == 9 : direction += ' right' 
        if self.location == 7 or self.location == 8 or self.location == 9 : direction += ' down'
        return direction 

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
