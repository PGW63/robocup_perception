#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Empty

from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient


class ReturnToSavedPose(Node):
    def __init__(self):
        super().__init__('return_to_saved_pose')

        # Params
        self.declare_parameter('amcl_topic', '/amcl_pose')
        self.declare_parameter('trigger_topic', '/human/go_saved_pose')
        self.declare_parameter('goal_frame', 'map')  # 보통 map

        self.amcl_topic = self.get_parameter('amcl_topic').value
        self.trigger_topic = self.get_parameter('trigger_topic').value
        self.goal_frame = self.get_parameter('goal_frame').value

        # State
        self.saved_pose: PoseStamped | None = None
        self.last_amcl_stamp = None

        # Subs
        self.create_subscription(PoseWithCovarianceStamped, self.amcl_topic, self.on_amcl, 10)
        self.create_subscription(Empty, self.trigger_topic, self.on_trigger, 10)

        # Action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.get_logger().info(
            f"Listening AMCL: {self.amcl_topic} | Trigger: {self.trigger_topic} | Action: /navigate_to_pose"
        )

    def on_amcl(self, msg: PoseWithCovarianceStamped):
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose.pose

        # 프레임이 비어있거나 이상하면 map으로 보정
        if ps.header.frame_id == '':
            ps.header.frame_id = self.goal_frame

        self.saved_pose = ps
        self.last_amcl_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)

    def on_trigger(self, _msg: Empty):
        if self.saved_pose is None:
            self.get_logger().warn("No saved pose yet (haven't received /amcl_pose).")
            return

        # Nav2 action server 대기
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("NavigateToPose action server not available: /navigate_to_pose")
            return

        goal = NavigateToPose.Goal()
        goal.pose = self.saved_pose  # 그대로 goal로 사용 (map 기준)

        # timestamp는 “지금”으로 찍어주는 게 보통 안전함
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        if goal.pose.header.frame_id == '':
            goal.pose.header.frame_id = self.goal_frame

        self.get_logger().info(
            f"Sending goal to saved pose: frame={goal.pose.header.frame_id} "
            f"x={goal.pose.pose.position.x:.3f}, y={goal.pose.pose.position.y:.3f}"
        )

        send_future = self.nav_client.send_goal_async(goal, feedback_callback=self.on_feedback)
        send_future.add_done_callback(self.on_goal_response)

    def on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected.")
            return

        self.get_logger().info("Goal accepted.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_result)

    def on_result(self, future):
        result = future.result().result
        status = future.result().status
        self.get_logger().info(f"Navigation finished. status={status} result={result}")

    def on_feedback(self, feedback_msg):
        # 너무 시끄러우면 주석 처리 가능
        fb = feedback_msg.feedback
        self.get_logger().debug(
            f"Remaining: {fb.distance_remaining:.3f}m"
        )


def main():
    rclpy.init()
    node = ReturnToSavedPose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
