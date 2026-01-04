# tello_midas

A ROS2 package for monocular depth estimation using Intel ISL's [MiDaS](https://github.com/isl-org/MiDaS) model. This package provides nodes for performing inference on video streams (specifically designed for Tello drones) and analyzing the resulting depth maps for obstacle detection.

## Overview

This package contains three main nodes:
1.  **`midas_inference`**: Runs the MiDaS neural network on a single image topic to produce a raw depth map.
2.  **`midas_analysis`**: Consumes raw depth maps, applies colormaps for visualization, and computes "safety" metrics based on the proximity of objects (Red = Close, Blue = Far).
3.  **`multi_midas_inference`**: A resource-efficient node that loads the model once and processes video streams from multiple drones sequentially.

## Dependencies

*   **ROS2** (Jazzy)
*   **Python 3**
*   **PyTorch**: `torch`, `torchvision`
*   **OpenCV**: `opencv-python`
*   **[midas_msgs](midas_msgs)**: Custom message definitions for depth analysis.

## Nodes

### 1. midas_inference

Performs depth estimation on a single camera stream.

**Source:** [tello_midas/midas_inference.py](tello_midas/tello_midas/midas_inference.py)

#### Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_type` | string | `MiDaS_small` | Model variant (`MiDaS_small`, `DPT_Hybrid`, `DPT_Large`). Small is recommended for CPU. |
| `input_topic` | string | `image_raw` | The RGB image topic to subscribe to. |
| `output_raw_topic` | string | `depth/raw` | The topic to publish the float32 depth map to. |

#### Subscribed Topics
*   `image_raw` ([sensor_msgs/Image](http://docs.ros.org/en/api/sensor_msgs/html/msg/Image.html))

#### Published Topics
*   `depth/raw` ([sensor_msgs/Image](http://docs.ros.org/en/api/sensor_msgs/html/msg/Image.html)): Encoding `32FC1`.

---

### 2. midas_analysis

Analyzes the raw depth map to determine obstacle proximity. It divides the image into regions (Left, Center, Right) and counts "Red" (close) vs "Blue" (far) pixels.

**Source:** [tello_midas/midas_analysis.py](tello_midas/tello_midas/midas_analysis.py)

#### Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input_depth_topic` | string | `depth/raw` | Input raw depth map. |
| `output_colormap_topic` | string | `depth/colormap` | Visualization with JET colormap. |
| `output_annotated_colormap_topic` | string | `depth/colormap_annotated` | Colormap with grid lines drawn. |
| `output_colormap_analysis_topic` | string | `depth/analysis` | Analysis data topic. |

#### Published Topics
*   `depth/colormap` ([sensor_msgs/Image](http://docs.ros.org/en/api/sensor_msgs/html/msg/Image.html)): BGR8 visualization.
*   `depth/analysis` ([midas_msgs/DepthMapAnalysis](midas_msgs/msg/DepthMapAnalysis.msg)): Contains pixel counts for red/blue thresholds in specific image regions.

---

### 3. multi_midas_inference

Optimized inference node for swarm applications. Loads the PyTorch model into memory *once* and processes images from multiple drones.

**Source:** [tello_midas/multi_midas_inference.py](tello_midas/tello_midas/multi_midas_inference.py)

#### Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_type` | string | `MiDaS_small` | Model variant. |
| `drone_ids` | string[] | `["tello1", "tello2"]` | List of namespaces/drone IDs to subscribe to. |

#### Subscriptions & Publications
For every `id` in `drone_ids`:
*   Subscribes to: `/{id}/image_raw`
*   Publishes to: `/{id}/depth/raw`

## Usage

**Run single inference:**
```bash
ros2 run tello_midas midas_inference --ros-args -p model_type:=MiDaS_small