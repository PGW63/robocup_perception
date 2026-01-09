from sensor_msgs.msg import Image, CompressedImage
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose, ObjectHypothesis
import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np

class HumanAlignNode(Node):
    def __init__(self):
        super().__init__('human_align_node')
        
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('alignment_threshold', 50)  # pixels

        self.image_topic = self.get_parameter('image_topic').value
        self.image_transport = self.get_parameter('image_transport').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.alignment_threshold = self.get_parameter('alignment_threshold').value

        self.bridge = CvBridge()

        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)
        self.detection_sub = message_filters.Subscriber(self, Detection2DArray, '/detection_node/detections')

        message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.detection_sub],
            queue_size=10,
            slop=0.1
        ).registerCallback(self.process_callback)

        


        self.width, self.height = None
        self.camera_shape_flag = False

        self.kp = 0.005
        self.kd = 0.008

        self.prev_error_x = 0.0

        self.pub_cmd_vel = self.create_publisher(Twist, self.cmd_vel_topic, 10)

    
    def process_callback(self, depth_msg, detection_msg):
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        if self.camera_shape_flag == False:
            self.width, self.height = depth_image.shape[1], depth_image.shape[0]
            self.camera_shape_flag = True
        
        detection_lists = detection_msg.detections
        
        if not detection_lists:
            return

        person_detections = []
        for detection in detection_lists:
            if detection.results and detection.results[0].class_id == 'person':
                person_detections.append(detection)
        
        if not person_detections:
            return
        
        closest_person = None
        min_depth = float('inf')
        
        for detection in person_detections:
            bbox = detection.bbox
            center_x = int(bbox.center.position.x)
            center_y = int(bbox.center.position.y)
            
            x1 = max(0, int(center_x - 20))
            x2 = min(self.width, int(center_x + 20))
            y1 = max(0, int(center_y - 20))
            y2 = min(self.height, int(center_y + 20))
            
            depth_region = depth_image[y1:y2, x1:x2]
            valid_depths = depth_region[depth_region > 0]  
            
            if len(valid_depths) > 0:
                avg_depth = np.median(valid_depths)  
                if avg_depth < min_depth:
                    min_depth = avg_depth
                    closest_person = detection
        
        if closest_person is None:
            return
        
        bbox = closest_person.bbox
        center_x = int(bbox.center.position.x)
        
        error_x = center_x - self.width // 2
        
        gap = error_x - self.prev_error_x
        
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        
        if abs(error_x) > self.alignment_threshold:
            twist_msg.angular.z = - (self.kp * error_x + self.kd * gap)
        else:
            twist_msg.angular.z = 0.0
        
        self.pub_cmd_vel.publish(twist_msg)
        self.prev_error_x = error_x
        
        self.get_logger().info(
            f'가장 가까운 사람 추적 중 - 거리: {min_depth:.2f}mm, '
            f'오차: {error_x}px, 각속도: {twist_msg.angular.z:.3f}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = HumanAlignNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




        
