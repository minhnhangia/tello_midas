#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge
from midas_msgs.msg import DepthMapAnalysis, ColorCount

import numpy as np
import cv2
import time

class MiDaSAnalysis(Node):
    def __init__(self):
        super().__init__('midas_analysis')

        # Parameters
        self.declare_parameter("input_depth_topic", "/tello1/depth/raw")
        self.declare_parameter("output_colormap_topic", "/tello1/depth/colormap")
        self.declare_parameter("output_annotated_colormap_topic", "/tello1/depth/colormap_annotated")
        self.declare_parameter("output_colormap_analysis_topic", "/tello1/depth/analysis")

        input_depth_topic = self.get_parameter("input_depth_topic").value
        output_colormap_topic = self.get_parameter("output_colormap_topic").value
        output_annotated_colormap_topic = self.get_parameter("output_annotated_colormap_topic").value
        output_colormap_analysis_topic = self.get_parameter("output_colormap_analysis_topic").value

        self.get_logger().info(f"Subscribing {input_depth_topic}, publishing {output_annotated_colormap_topic}, and {output_colormap_analysis_topic}")

        self.bridge = CvBridge()

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ROS interfaces
        self.sub = self.create_subscription(Image, input_depth_topic, self.depth_image_callback, qos_profile)
        self.pub_colormap = self.create_publisher(Image, output_colormap_topic, qos_profile)
        self.pub_annotated_colormap = self.create_publisher(Image, output_annotated_colormap_topic, qos_profile)
        self.pub_analysis = self.create_publisher(DepthMapAnalysis, output_colormap_analysis_topic, qos_profile)

        self.grid_lines = None

    def depth_image_callback(self, msg: Image):
        # Skip processing if no subscribers to save computation
        if (self.pub_colormap.get_subscription_count() == 0 and 
            self.pub_annotated_colormap.get_subscription_count() == 0 and
            self.pub_analysis.get_subscription_count() == 0):
            return
        
        try:
            # Decode 32FC1 image to NumPy float32 array
            depth_map = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

            # Analyze colors in the depth map and publish colormaps
            self.__process_colormaps(depth_map, msg.header)

        except Exception as e:
            self.get_logger().error(f"Error in depth_image_callback: {e}")

    def __process_colormaps(self, depth_map, header):
        depth_norm = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX)
        depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        height, width = depth_color.shape[:2]

        if self.grid_lines is None:
            self.grid_lines = self.__compute_grid_lines(height, width)

        # Analyze and publish colormap analysis
        self.__publish_colormap_analysis(depth_color, header, height, width)

        # Publish clean colormap
        self.__publish_clean_colormap(depth_color, header)

        # Publish annotated colormap
        self.__publish_annotated_colormap(depth_color, header)

    def __publish_colormap_analysis(self, depth_colormap, header, h, w):
        # Define the 3x3 grid regions
        # top_left     = depth_colormap[:h//3, :w//3]
        # top_center   = depth_colormap[:h//3, w//3:2*w//3]
        # top_right    = depth_colormap[:h//3, 2*w//3:]
        
        middle_left  = depth_colormap[h//3:2*h//3, :w//3]
        middle_center= depth_colormap[h//3:2*h//3, w//3:2*w//3]
        middle_right = depth_colormap[h//3:2*h//3, 2*w//3:]
        
        # bottom_left  = depth_colormap[2*h//3:, :w//3]
        # bottom_center= depth_colormap[2*h//3:, w//3:2*w//3]
        # bottom_right = depth_colormap[2*h//3:, 2*w//3:]
        
        # Analyze colors in the middle row regions
        red_middle_left = int(np.sum((middle_left[:, :, 2] > 150) & (middle_left[:, :, 0] < 50)))
        blue_middle_left = int(np.sum((middle_left[:, :, 0] > 150) & (middle_left[:, :, 2] < 50)))

        red_middle_center = int(np.sum((middle_center[:, :, 2] > 150) & (middle_center[:, :, 0] < 50)))
        blue_middle_center = int(np.sum((middle_center[:, :, 0] > 150) & (middle_center[:, :, 2] < 50)))

        red_middle_right = int(np.sum((middle_right[:, :, 2] > 150) & (middle_right[:, :, 0] < 50)))
        blue_middle_right = int(np.sum((middle_right[:, :, 0] > 150) & (middle_right[:, :, 2] < 50)))

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
        red_mc_left = int(np.sum((middle_center_left[:, :, 2] > 150) & (middle_center_left[:, :, 0] < 50)))
        nonblue_mc_left = int(np.sum(~((middle_center_left[:, :, 0] > 150) & (middle_center_left[:, :, 2] < 50))))
        blue_mc_left = int(np.sum((middle_center_left[:, :, 0] > 150) & (middle_center_left[:, :, 2] < 50)))

        red_mc_center = int(np.sum((middle_center_center[:, :, 2] > 150) & (middle_center_center[:, :, 0] < 50)))
        nonblue_mc_center = int(np.sum(~((middle_center_center[:, :, 0] > 150) & (middle_center_center[:, :, 2] < 50))))
        blue_mc_center = int(np.sum((middle_center_center[:, :, 0] > 150) & (middle_center_center[:, :, 2] < 50)))

        red_mc_right = int(np.sum((middle_center_right[:, :, 2] > 150) & (middle_center_right[:, :, 0] < 50)))
        nonblue_mc_right = int(np.sum(~((middle_center_right[:, :, 0] > 150) & (middle_center_right[:, :, 2] < 50))))
        blue_mc_right = int(np.sum((middle_center_right[:, :, 0] > 150) & (middle_center_right[:, :, 2] < 50)))

        msg = DepthMapAnalysis()
        msg.header = header

        # Middle row
        msg.middle_left = ColorCount(red=red_middle_left, blue=blue_middle_left)
        msg.middle_center = ColorCount(red=red_middle_center, blue=blue_middle_center)
        msg.middle_right = ColorCount(red=red_middle_right, blue=blue_middle_right)

        # Middle-center split
        msg.mc_left = ColorCount(red=red_mc_left, blue=blue_mc_left, nonblue=nonblue_mc_left)
        msg.mc_center = ColorCount(red=red_mc_center, blue=blue_mc_center, nonblue=nonblue_mc_center)
        msg.mc_right = ColorCount(red=red_mc_right, blue=blue_mc_right, nonblue=nonblue_mc_right)

        self.pub_analysis.publish(msg)

    def __publish_clean_colormap(self, depth_color, header):
        if self.pub_colormap.get_subscription_count() == 0:
            return

        colormap_msg = self.bridge.cv2_to_imgmsg(depth_color, encoding="bgr8")
        colormap_msg.header = header
        self.pub_colormap.publish(colormap_msg)

    def __publish_annotated_colormap(self, depth_color, header):
        if self.pub_annotated_colormap.get_subscription_count() == 0:
            return

        annotated = depth_color.copy()
        for start, end, color, thickness in self.grid_lines:
            cv2.line(annotated, start, end, color, thickness)
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        annotated_msg.header = header
        self.pub_annotated_colormap.publish(annotated_msg)

    def __compute_grid_lines(self, height, width) -> list[tuple]:
        """Precompute all grid lines for the annotated colormap."""
        lines = []
        # Outer green 3x3 grid
        lines.append(((width//3, 0), (width//3, height), (0, 255, 0), 2))
        lines.append(((2*width//3, 0), (2*width//3, height), (0, 255, 0), 2))
        lines.append(((0, height//3), (width, height//3), (0, 255, 0), 2))
        lines.append(((0, 2*height//3), (width, 2*height//3), (0, 255, 0), 2))

        # Middle-center blue subdivisions
        mc_row_start = height//3
        mc_row_end = 2*height//3
        mc_col_start = width//3
        mc_col_end = 2*width//3
        mc_width = mc_col_end - mc_col_start
        sub_width = mc_width // 3

        lines.append(((mc_col_start + sub_width, mc_row_start), (mc_col_start + sub_width, mc_row_end), (255, 0, 0), 2))
        lines.append(((mc_col_start + 2*sub_width, mc_row_start), (mc_col_start + 2*sub_width, mc_row_end), (255, 0, 0), 2))

        return lines

def main(args=None):
    rclpy.init(args=args)
    node = MiDaSAnalysis()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
