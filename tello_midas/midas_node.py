#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
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
        self.declare_parameter("annotated_colormap_topic", "/tello1/depth/colormap_annotated")

        model_type = self.get_parameter("model_type").value
        input_topic = self.get_parameter("input_topic").value
        output_raw_topic = self.get_parameter("output_raw_topic").value
        output_colormap_topic = self.get_parameter("output_colormap_topic").value
        annotated_colormap_topic = self.get_parameter("annotated_colormap_topic").value

        self.get_logger().info(f"MiDaS model_type={model_type}, subscribing {input_topic}, publishing {output_raw_topic}")

        self.bridge = CvBridge()

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ROS interfaces
        self.sub = self.create_subscription(Image, input_topic, self.image_callback, qos_profile)
        self.pub_depth = self.create_publisher(Image, output_raw_topic, qos_profile)
        self.pub_colormap = self.create_publisher(Image, output_colormap_topic, qos_profile)
        self.pub_annotated_colormap = self.create_publisher(Image, annotated_colormap_topic, qos_profile)

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
        # Skip processing if no subscribers to save computation
        if (self.pub_colormap.get_subscription_count() == 0 and 
            self.pub_annotated_colormap.get_subscription_count() == 0 and
            self.pub_depth.get_subscription_count() == 0):
            return
        
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
            self.__publish_raw_depth(depth_map, msg.header)

            # Analyze colors in the depth map and publish colormaps
            self.__process_colormaps(depth_map, msg.header)

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")

    def __process_colormaps(self, depth_map, header):    
        if (self.pub_colormap.get_subscription_count() == 0 and 
            self.pub_annotated_colormap.get_subscription_count() == 0):
            return   
         
        depth_norm = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX)
        depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        height, width = depth_color.shape[:2]

        self.__publish_colormap_data(depth_color, header, height, width)

        # Publish clean colormap
        self.__publish_clean_colormap(depth_color, header)

        # Create copy for annotation to avoid modifying original
        self.__publish_annotated_colormap(depth_color.copy(), header, height, width)

    def __publish_colormap_data(self, depth_colormap, header, h, w):            
        # Define the 3x3 grid regions
        top_left     = depth_colormap[:h//3, :w//3]
        top_center   = depth_colormap[:h//3, w//3:2*w//3]
        top_right    = depth_colormap[:h//3, 2*w//3:]
        
        middle_left  = depth_colormap[h//3:2*h//3, :w//3]
        middle_center= depth_colormap[h//3:2*h//3, w//3:2*w//3]
        middle_right = depth_colormap[h//3:2*h//3, 2*w//3:]
        
        bottom_left  = depth_colormap[2*h//3:, :w//3]
        bottom_center= depth_colormap[2*h//3:, w//3:2*w//3]
        bottom_right = depth_colormap[2*h//3:, 2*w//3:]
        
        # Analyze colors in the middle row regions
        red_middle_left = np.sum((middle_left[:, :, 2] > 150) & (middle_left[:, :, 0] < 50))
        blue_middle_left = np.sum((middle_left[:, :, 0] > 150) & (middle_left[:, :, 2] < 50))
        
        red_middle_center = np.sum((middle_center[:, :, 2] > 150) & (middle_center[:, :, 0] < 50))
        blue_middle_center = np.sum((middle_center[:, :, 0] > 150) & (middle_center[:, :, 2] < 50))
        
        red_middle_right = np.sum((middle_right[:, :, 2] > 150) & (middle_right[:, :, 0] < 50))
        blue_middle_right = np.sum((middle_right[:, :, 0] > 150) & (middle_right[:, :, 2] < 50))
        
        # Further split the middle_center region into 3 columns for detailed analysis
        mc_row_start = h//3
        mc_row_end = 2*h//3
        mc_col_start = w//3
        mc_col_end = 2*w//3
        mc_width = mc_col_end - mc_col_start  # roughly w//3
        sub_width = mc_width // 3             # width of each subdivided region
        
        middle_center_left = depth_colormap[mc_row_start:mc_row_end, mc_col_start: mc_col_start + sub_width]
        middle_center_center = depth_colormap[mc_row_start:mc_row_end, mc_col_start + sub_width: mc_col_start + 2*sub_width]
        middle_center_right = depth_colormap[mc_row_start:mc_row_end, mc_col_start + 2*sub_width: mc_col_end]
        
        # Analyze colors in the subdivided middle_center subregions
        red_mc_left = np.sum((middle_center_left[:, :, 2] > 150) & (middle_center_left[:, :, 0] < 50))
        nonblue_mc_left = np.sum(~((middle_center_left[:, :, 0] > 150) & (middle_center_left[:, :, 2] < 50)))
        blue_mc_left = np.sum((middle_center_left[:, :, 0] > 150) & (middle_center_left[:, :, 2] < 50))
        
        red_mc_center = np.sum((middle_center_center[:, :, 2] > 150) & (middle_center_center[:, :, 0] < 50))
        nonblue_mc_center = np.sum(~((middle_center_center[:, :, 0] > 150) & (middle_center_center[:, :, 2] < 50)))
        blue_mc_center = np.sum((middle_center_center[:, :, 0] > 150) & (middle_center_center[:, :, 2] < 50))
        
        red_mc_right = np.sum((middle_center_right[:, :, 2] > 150) & (middle_center_right[:, :, 0] < 50))
        nonblue_mc_right = np.sum(~((middle_center_right[:, :, 0] > 150) & (middle_center_right[:, :, 2] < 50)))
        blue_mc_right = np.sum((middle_center_right[:, :, 0] > 150) & (middle_center_right[:, :, 2] < 50))
        
        # # Update class attributes with both levels of analysis
        # self.depth_map_colors["middle_row"] = {
        #     "middle_left": {"red": red_middle_left, "blue": blue_middle_left},
        #     "middle_center": {"red": red_middle_center, "blue": blue_middle_center},
        #     "middle_right": {"red": red_middle_right, "blue": blue_middle_right}
        # }
        
        # self.depth_map_colors["middle_center_split"] = {
        #     "left": {"red": red_mc_left, "blue": blue_mc_left, "nonblue": nonblue_mc_left},
        #     "center": {"red": red_mc_center, "blue": blue_mc_center, "nonblue": nonblue_mc_center},
        #     "right": {"red": red_mc_right, "blue": blue_mc_right, "nonblue": nonblue_mc_right}
        # }

    def __publish_raw_depth(self, depth_map, header):
        if self.pub_depth.get_subscription_count() == 0:
            return
        
        depth_msg = self.bridge.cv2_to_imgmsg(depth_map, encoding="32FC1")
        depth_msg.header = header
        self.pub_depth.publish(depth_msg)

    def __publish_clean_colormap(self, depth_color, header):
        if self.pub_colormap.get_subscription_count() == 0:
            return

        colormap_msg = self.bridge.cv2_to_imgmsg(depth_color, encoding="bgr8")
        colormap_msg.header = header
        self.pub_colormap.publish(colormap_msg)

    def __publish_annotated_colormap(self, depth_colormap, header, h, w): 
        if self.pub_annotated_colormap.get_subscription_count() == 0:
            return
                           
        # Draw the outer grid lines (green)
        cv2.line(depth_colormap, (w//3, 0), (w//3, h), (0, 255, 0), 2)
        cv2.line(depth_colormap, (2*w//3, 0), (2*w//3, h), (0, 255, 0), 2)
        cv2.line(depth_colormap, (0, h//3), (w, h//3), (0, 255, 0), 2)
        cv2.line(depth_colormap, (0, 2*h//3), (w, 2*h//3), (0, 255, 0), 2)

        # Further split the middle_center region into 3 columns for detailed analysis
        mc_row_start = h//3
        mc_row_end = 2*h//3
        mc_col_start = w//3
        mc_col_end = 2*w//3
        mc_width = mc_col_end - mc_col_start  # roughly w//3
        sub_width = mc_width // 3             # width of each subdivided region

        # Draw extra vertical lines (blue) within the middle_center region to delineate subdivisions
        cv2.line(depth_colormap, (mc_col_start + sub_width, mc_row_start), (mc_col_start + sub_width, mc_row_end), (255, 0, 0), 2)
        cv2.line(depth_colormap, (mc_col_start + 2*sub_width, mc_row_start), (mc_col_start + 2*sub_width, mc_row_end), (255, 0, 0), 2)

        annotated_msg = self.bridge.cv2_to_imgmsg(depth_colormap, encoding="bgr8")
        annotated_msg.header = header
        self.pub_annotated_colormap.publish(annotated_msg)

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
