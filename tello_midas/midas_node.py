#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

import torch
import numpy as np
import cv2
import time

class MiDaSNode(Node):
    def __init__(self):
        super().__init__('midas_node')

        # Parameters
        self.declare_parameter("model_type", "MiDaS_small")  # default to smaller model for CPU
        self.declare_parameter("input_topic", "/tello1/image_raw")
        self.declare_parameter("output_raw_topic", "/tello1/depth/raw")
        self.declare_parameter("output_colormap_topic", "/tello1/depth/colormap")

        model_type = self.get_parameter("model_type").value
        input_topic = self.get_parameter("input_topic").value
        output_raw_topic = self.get_parameter("output_raw_topic").value
        output_colormap_topic = self.get_parameter("output_colormap_topic").value

        self.get_logger().info(f"MiDaS model_type={model_type}, subscribing {input_topic}, publishing {output_raw_topic}")

        self.bridge = CvBridge()

        # ROS interfaces
        self.sub = self.create_subscription(Image, input_topic, self.image_callback, qos_profile_sensor_data)
        self.pub_depth = self.create_publisher(Image, output_raw_topic, qos_profile_sensor_data)
        self.pub_colormap = self.create_publisher(Image, output_colormap_topic, qos_profile_sensor_data)

        # load model once
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        self.get_logger().info(f"Loading MiDaS model: {model_type}")
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.to(self.device)
        self.midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def image_callback(self, msg: Image):
        try:
            # Convert ROS Image to CV image (BGR)
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            input_batch = self.transform(img_rgb).to(self.device)

            with torch.no_grad():
                prediction = self.midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_map = prediction.cpu().numpy().astype(np.float32)  # raw depth (relative)

            # Publish raw depth as 32FC1 (useful for other nodes)
            depth_msg = self.bridge.cv2_to_imgmsg(depth_map, encoding="32FC1")
            depth_msg.header = msg.header
            self.pub_depth.publish(depth_msg)

            # Publish depth colormap
            depth_norm = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX)
            depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            color_msg = self.bridge.cv2_to_imgmsg(depth_color, encoding="bgr8")
            color_msg.header = msg.header
            self.pub_colormap.publish(color_msg)

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = MiDaSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
