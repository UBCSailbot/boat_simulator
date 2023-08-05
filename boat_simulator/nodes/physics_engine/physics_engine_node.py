#!/usr/bin/env python3

from typing import Optional

import rclpy
from custom_interfaces.action import SimRudderActuation, SimSailTrimTabActuation
from custom_interfaces.msg import (
    GPS,
    DesiredHeading,
    SimWorldState,
    WindSensor,
    WindSensors,
)
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle, Future
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

import boat_simulator.common.constants as Constants
from boat_simulator.common.types import Scalar
from boat_simulator.nodes.physics_engine.decorators import require_all_subs_active

# TODO Setup action server for rudder actuation


def main(args=None):
    rclpy.init(args=args)
    node = PhysicsEngineNode()
    rclpy.spin(node)
    rclpy.shutdown()


class PhysicsEngineNode(Node):
    def __init__(self):
        super().__init__(node_name="physics_engine_node")

        self.get_logger().debug("Initializing node...")

        self.__declare_ros_parameters()
        self.__init_subscriptions()
        self.__init_publishers()
        self.__init_action_clients()
        self.__init_timer_callbacks()

        # TODO Do we need to worry about the counter overflowing?
        self.__publish_counter = 0
        self.__rudder_angle = 0
        self.__sail_trim_tab_angle = 0
        self.__desired_heading = None

        self.get_logger().debug("Node initialization complete. Starting execution...")

    def __declare_ros_parameters(self):
        # TODO Update global YAML file with more configuration parameters and declare them here
        self.declare_parameters(
            namespace="",
            parameters=[
                ("pub_period_sec", rclpy.Parameter.Type.DOUBLE),
            ],
        )

    def __init_subscriptions(self):
        self.get_logger().debug("Initializing subscriptions...")

        self.__desired_heading_sub = self.create_subscription(
            msg_type=DesiredHeading,
            topic=Constants.PHYSICS_ENGINE_SUBSCRIPTIONS.DESIRED_HEADING,
            callback=self.__desired_heading_sub_callback,
            qos_profile=Constants.QOS_DEPTH,
        )

        self.get_logger().debug("Done initializing subscriptions...")

    def __init_publishers(self):
        self.get_logger().debug("Initializing publishers...")

        self.__gps_pub = self.create_publisher(
            msg_type=GPS,
            topic=Constants.PHYSICS_ENGINE_PUBLISHERS.GPS,
            qos_profile=Constants.QOS_DEPTH,
        )
        self.__wind_sensors_pub = self.create_publisher(
            msg_type=WindSensors,
            topic=Constants.PHYSICS_ENGINE_PUBLISHERS.WIND_SENSORS,
            qos_profile=Constants.QOS_DEPTH,
        )
        self.__kinematics_pub = self.create_publisher(
            msg_type=SimWorldState,
            topic=Constants.PHYSICS_ENGINE_PUBLISHERS.KINEMATICS,
            qos_profile=Constants.QOS_DEPTH,
        )

        self.get_logger().debug("Done initializing publishers...")

    def __init_action_clients(self):
        self.__rudder_actuation_action_client = ActionClient(
            node=self,
            action_type=SimRudderActuation,
            action_name=Constants.ACTION_CLIENTS.RUDDER_ACTUATION,
        )
        self.__sail_actuation_action_client = ActionClient(
            node=self,
            action_type=SimSailTrimTabActuation,
            action_name=Constants.ACTION_CLIENTS.SAIL_ACTUATION,
        )

    def __init_timer_callbacks(self):
        self.get_logger().debug("Initializing timer callbacks...")

        self.create_timer(
            timer_period_sec=self.pub_period,
            callback=self.__publish,
        )
        self.create_timer(
            timer_period_sec=Constants.RUDDER_ACTUATION_REQUEST_PERIOD_SEC,
            callback=self.__rudder_action_send_goal,
        )

        self.get_logger().debug("Done initializing timer callbacks...")

    def __publish(self):
        # TODO Get updated boat state and publish
        # TODO Get wind sensor data and publish

        self.get_logger().info("Publishing simulated sensor and kinematics data")
        self.__publish_gps()
        self.__publish_wind_sensors()
        self.__publish_kinematics()

        self.__publish_counter += 1

    def __publish_gps(self):
        # TODO Update to publish real data
        msg = GPS()
        msg.lat_lon.latitude = 0.0
        msg.lat_lon.longitude = 0.0
        msg.speed.speed = 0.0
        msg.heading.heading = 0.0

        self.gps_pub.publish(msg)
        self.get_logger().info(f"Publishing to {self.gps_pub.topic}: {msg}")

    def __publish_wind_sensors(self):
        # TODO Update to publish real data
        windSensor1 = WindSensor()
        windSensor1.speed.speed = 0.0
        windSensor1.direction = 0

        windSensor2 = WindSensor()
        windSensor2.speed.speed = 0.0
        windSensor2.direction = 0

        msg = WindSensors()
        msg.wind_sensors = [windSensor1, windSensor2]

        self.wind_sensors_pub.publish(msg)
        self.get_logger().info(f"Publishing to {self.wind_sensors_pub.topic}: {msg}")

    def __publish_kinematics(self):
        # TODO Update to publish real data
        msg = SimWorldState()

        msg.global_gps.lat_lon.latitude = 0.0
        msg.global_gps.lat_lon.longitude = 0.0
        msg.global_gps.speed.speed = 0.0
        msg.global_gps.heading.heading = 0.0

        msg.global_reference_kinematics.pose.position.x = 0.0
        msg.global_reference_kinematics.pose.position.y = 0.0
        msg.global_reference_kinematics.pose.position.z = 0.0
        msg.global_reference_kinematics.pose.orientation.x = 0.0
        msg.global_reference_kinematics.pose.orientation.y = 0.0
        msg.global_reference_kinematics.pose.orientation.z = 0.0
        msg.global_reference_kinematics.pose.orientation.w = 1.0
        msg.global_reference_kinematics.velocity.linear.x = 0.0
        msg.global_reference_kinematics.velocity.linear.y = 0.0
        msg.global_reference_kinematics.velocity.linear.z = 0.0
        msg.global_reference_kinematics.velocity.angular.x = 0.0
        msg.global_reference_kinematics.velocity.angular.y = 0.0
        msg.global_reference_kinematics.velocity.angular.z = 0.0
        msg.global_reference_kinematics.acceleration.linear.x = 0.0
        msg.global_reference_kinematics.acceleration.linear.y = 0.0
        msg.global_reference_kinematics.acceleration.linear.z = 0.0
        msg.global_reference_kinematics.acceleration.angular.x = 0.0
        msg.global_reference_kinematics.acceleration.angular.y = 0.0
        msg.global_reference_kinematics.acceleration.angular.z = 0.0
        msg.global_reference_kinematics.net_wrench.force.x = 0.0
        msg.global_reference_kinematics.net_wrench.force.y = 0.0
        msg.global_reference_kinematics.net_wrench.force.z = 0.0
        msg.global_reference_kinematics.net_wrench.torque.x = 0.0
        msg.global_reference_kinematics.net_wrench.torque.y = 0.0
        msg.global_reference_kinematics.net_wrench.torque.z = 0.0

        msg.relative_reference_kinematics.pose.position.x = 0.0
        msg.relative_reference_kinematics.pose.position.y = 0.0
        msg.relative_reference_kinematics.pose.position.z = 0.0
        msg.relative_reference_kinematics.pose.orientation.x = 0.0
        msg.relative_reference_kinematics.pose.orientation.y = 0.0
        msg.relative_reference_kinematics.pose.orientation.z = 0.0
        msg.relative_reference_kinematics.pose.orientation.w = 1.0
        msg.relative_reference_kinematics.velocity.linear.x = 0.0
        msg.relative_reference_kinematics.velocity.linear.y = 0.0
        msg.relative_reference_kinematics.velocity.linear.z = 0.0
        msg.relative_reference_kinematics.velocity.angular.x = 0.0
        msg.relative_reference_kinematics.velocity.angular.y = 0.0
        msg.relative_reference_kinematics.velocity.angular.z = 0.0
        msg.relative_reference_kinematics.acceleration.linear.x = 0.0
        msg.relative_reference_kinematics.acceleration.linear.y = 0.0
        msg.relative_reference_kinematics.acceleration.linear.z = 0.0
        msg.relative_reference_kinematics.acceleration.angular.x = 0.0
        msg.relative_reference_kinematics.acceleration.angular.y = 0.0
        msg.relative_reference_kinematics.acceleration.angular.z = 0.0
        msg.relative_reference_kinematics.net_wrench.force.x = 0.0
        msg.relative_reference_kinematics.net_wrench.force.y = 0.0
        msg.relative_reference_kinematics.net_wrench.force.z = 0.0
        msg.relative_reference_kinematics.net_wrench.torque.x = 0.0
        msg.relative_reference_kinematics.net_wrench.torque.y = 0.0
        msg.relative_reference_kinematics.net_wrench.torque.z = 0.0

        sec, nanosec = divmod(self.pub_period * self.publish_counter, 1)
        msg.header.stamp.sec = int(sec)
        msg.header.stamp.nanosec = int(nanosec * 1e9)
        msg.header.frame_id = str(self.publish_counter)

        self.kinematics_pub.publish(msg)

        # TODO Break down this ROS log because it is too large
        # self.get_logger().info(f"Publishing to {self.kinematics_pub.topic}: {msg}")
        self.get_logger().info(f"Publishing to {self.kinematics_pub.topic}")

    def __desired_heading_sub_callback(self, msg: DesiredHeading):
        self.get_logger().info(f"Received data from {self.desired_heading_sub.topic}: {msg}")
        self.__desired_heading = msg

    @require_all_subs_active
    def __rudder_action_send_goal(self):
        self.get_logger().debug("Initiating goal request for rudder actuation action")

        goal_msg = SimRudderActuation.Goal()
        goal_msg.desired_heading = self.desired_heading

        is_timed_out = not self.rudder_actuation_action_client.wait_for_server(
            timeout_sec=Constants.ACTION_SEND_GOAL_TIMEOUT_SEC
        )

        if is_timed_out:
            self.get_logger().warn(
                "Rudder actuation action goal request timed out after "
                + f"{Constants.ACTION_SEND_GOAL_TIMEOUT_SEC} seconds. Aborting..."
            )
        else:
            send_goal_future = self.rudder_actuation_action_client.send_goal_async(
                goal=goal_msg, feedback_callback=self.__rudder_action_feedback_callback
            )
            send_goal_future.add_done_callback(self.__rudder_action_goal_response_callback)
            self.get_logger().debug("Completed goal request for rudder actuation action")

    def __rudder_action_goal_response_callback(self, future: Future):
        goal_handle: Optional[ClientGoalHandle] = future.result()
        if (not goal_handle) or (not goal_handle.accepted):
            self.get_logger().warn("Attempted to send rudder actuation goal, but it was rejected")
            return
        self.get_logger().debug("Rudder actuation goal was accepted. Beginning rudder actuation")
        rudder_get_result_future = goal_handle.get_result_async()
        rudder_get_result_future.add_done_callback(self.__rudder_action_get_result_callback)

    def __rudder_action_get_result_callback(self, future: Future):
        result = future.result().result
        self.get_logger().debug(
            "Rudder actuation action finished with a heading residual of "
            + f"{result.remaining_angular_distance:.2f} radians and final "
            + f"rudder angle of {self.__rudder_angle} radians"
        )

    def __rudder_action_feedback_callback(self, feedback_msg):
        self.__rudder_angle = feedback_msg.feedback.rudder_angle
        self.get_logger().debug(
            f"Rudder actuation action reported a rudder angle of {self.__rudder_angle}"
        )

    @property
    def pub_period(self) -> float:
        return self.get_parameter("pub_period_sec").get_parameter_value().double_value

    @property
    def publish_counter(self) -> int:
        return self.__publish_counter

    @property
    def gps_pub(self) -> Publisher:
        return self.__gps_pub

    @property
    def wind_sensors_pub(self) -> Publisher:
        return self.__wind_sensors_pub

    @property
    def kinematics_pub(self) -> Publisher:
        return self.__kinematics_pub

    @property
    def desired_heading(self) -> Optional[DesiredHeading]:
        return self.__desired_heading

    @property
    def desired_heading_sub(self) -> Subscription:
        return self.__desired_heading_sub

    @property
    def rudder_angle(self) -> Scalar:
        return self.__rudder_angle

    @property
    def sail_trim_tab_angle(self) -> Scalar:
        return self.__sail_trim_tab_angle

    @property
    def rudder_actuation_action_client(self) -> ActionClient:
        return self.__rudder_actuation_action_client

    @property
    def sail_actuation_action_client(self) -> ActionClient:
        return self.__sail_actuation_action_client


if __name__ == "__main__":
    main()
