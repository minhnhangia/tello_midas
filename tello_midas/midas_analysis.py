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
        if self.pub_analysis.get_subscription_count() == 0:
            return

        # Precompute masks
        red_mask  = ((depth_colormap[:, :, 2] > 150) & (depth_colormap[:, :, 0] < 50)).astype(np.uint8)
        blue_mask = ((depth_colormap[:, :, 0] > 150) & (depth_colormap[:, :, 2] < 50)).astype(np.uint8)

        # Build integral images using OpenCV (faster than np.cumsum)
        # cv2.integral returns shape (h+1, w+1), so we will adjust indices
        red_integral  = cv2.integral(red_mask, sdepth=cv2.CV_32S)
        blue_integral = cv2.integral(blue_mask, sdepth=cv2.CV_32S)

        def region_sum(integral, r1, r2, c1, c2):
            """Return sum of mask values in rectangle [r1:r2, c1:c2)."""
            # Adjust for integral image extra row/col
            r1_i, r2_i = r1, r2
            c1_i, c2_i = c1, c2
            total = integral[r2_i, c2_i] - integral[r1_i, c2_i] - integral[r2_i, c1_i] + integral[r1_i, c1_i]
            return int(total)

        msg = DepthMapAnalysis()
        msg.header = header

        # Middle row regions (split into thirds)
        row1, row2 = h//3, 2*h//3
        col1, col2 = w//3, 2*w//3

        # left / center / right
        msg.middle_left   = ColorCount(
            red  = region_sum(red_integral, row1, row2, 0, col1),
            blue = region_sum(blue_integral, row1, row2, 0, col1)
        )
        msg.middle_center = ColorCount(
            red  = region_sum(red_integral, row1, row2, col1, col2),
            blue = region_sum(blue_integral, row1, row2, col1, col2)
        )
        msg.middle_right  = ColorCount(
            red  = region_sum(red_integral, row1, row2, col2, w),
            blue = region_sum(blue_integral, row1, row2, col2, w)
        )

        # Further split middle_center into thirds
        mc_width = col2 - col1
        sub_w = mc_width // 3

        for i, field in enumerate(["mc_left", "mc_center", "mc_right"]):
            c1_sub = col1 + i * sub_w
            c2_sub = col1 + (i + 1) * sub_w if i < 2 else col2
            red_val  = region_sum(red_integral,  row1, row2, c1_sub, c2_sub)
            blue_val = region_sum(blue_integral, row1, row2, c1_sub, c2_sub)
            nonblue_val = (row2 - row1) * (c2_sub - c1_sub) - blue_val
            setattr(msg, field, ColorCount(red=red_val, blue=blue_val, nonblue=nonblue_val))

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
