//TODO: reset 'listen_id' and 'talk_id' when they are near overflow

#include <iostream> 
#include <string>
#include <chrono>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/int64.hpp"
using std::placeholders::_1;

using namespace std;

/* This example creates a subclass of Node and uses std::bind() to register a
 * member function as a callback from the timer. */

class MotionPlanner : public rclcpp::Node
{
public:
  MotionPlanner() : Node("MotionPlanner")
  {
    publisher = this->create_publisher<std_msgs::msg::String>("direction");
    subscription = this->create_subscription<std_msgs::msg::Int64>("gate_location", std::bind(&MotionPlanner::topic_callback, this, _1));
    timer_ = this->create_wall_timer( 300ms, bind(&MotionPlanner::timer_callback, this));
  }

private:
  void timer_callback()
  {
    if (done) return;
    if(listen_id == talk_id) return ;
    talk_id += 1;

    auto message = std_msgs::msg::String();
    message.data = find_direction();
    RCLCPP_INFO(this->get_logger(), "Iam Publishing: '%s'", message.data.c_str());
    publisher->publish(message);
  }

  void topic_callback(const std_msgs::msg::Int64::SharedPtr msg)
  {
    if (done) return;
    location = (int)msg->data;
    listen_id += 1;

    if (location==5) done = true;
    RCLCPP_INFO(this->get_logger(), "Iam hearing: '%d'", location);
  }

  string find_direction(){
    string direction = "";
    if(location==1 || location==2 || location==3) direction += " up";
    if(location==1 || location==4 || location==7) direction += " left";
    if(location==3 || location==6 || location==9) direction += " right";
    if(location==7 || location==8 || location==9) direction += " down";
    return direction;
  }

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher;
  rclcpp::Subscription<std_msgs::msg::Int64>::SharedPtr subscription;
  int listen_id = 0 ;
  int talk_id = 0;
  int location = 0;
  bool done = false;

};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(make_shared<MotionPlanner>());
  rclcpp::shutdown();
  return 0;
}
