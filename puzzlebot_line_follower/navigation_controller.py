#!/usr/bin/env python3
"""
navigation_controller.py
------------------------
ROS 2 node that drives a differential-drive robot through a list of waypoints
while obeying traffic light commands published by traffic_light_detector.py.

Controller:
  - Proportional angular velocity  → align heading to goal
  - Proportional linear velocity   → drive forward (scaled by heading error)
  - Speed is multiplied by a factor from the traffic light state:
      GREEN   : factor = 1.0  (full speed)
      YELLOW  : factor = 0.4  (slow)
      RED     : factor = 0.0  (stop, maintain red-lock)
      UNKNOWN : factor = 1.0  (default — continue navigation)

Robustness strategies:
  - Heading normalisation to (-π, π] prevents integral wind-up / wrap issues.
  - Distance deadband avoids oscillation around goal.
  - Velocity clipping ensures actuator limits are respected.
  - State is retained across perception frames (red-lock in detector node).

All goal waypoints are loaded from a ROS parameter list so they can be
changed in the launch file without editing source code.

Libraries: rclpy, geometry_msgs, nav_msgs, std_msgs, NumPy only.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
from rclpy import qos


class NavigationController(Node):
    """
    Point-to-point proportional controller with traffic light awareness.
    """

    # Speed multipliers per traffic state
    SPEED_FACTOR = {
        'GREEN':   1.0,
        'YELLOW':  0.4,
        'RED':     0.0,
        'UNKNOWN': 1.0,
    }

    def __init__(self):
        super().__init__('navigation_controller')

        # ---- Parameters (set via launch file / config YAML) ----------------
        #  waypoints: flat list [x0, y0, x1, y1, ...]
        self.declare_parameter('waypoints',        [2.0, 0.0, 2.0, 2.0, 0.0, 2.0])
        self.declare_parameter('goal_tolerance',   0.15)   # metres
        self.declare_parameter('max_linear_vel',   0.3)    # m/s
        self.declare_parameter('max_angular_vel',  1.2)    # rad/s
        self.declare_parameter('k_linear',         0.4)    # proportional gain linear
        self.declare_parameter('k_angular',        1.5)    # proportional gain angular
        self.declare_parameter('heading_threshold', 0.35)  # rad — drive fwd only when aligned
        self.declare_parameter('cmd_vel_topic',    '/cmd_vel')
        self.declare_parameter('odom_topic',       '/odom')
        self.declare_parameter('state_topic',      '/traffic_light/state')

        flat_wp     	 = self.get_parameter('waypoints').value
        self.goal_tol    = self.get_parameter('goal_tolerance').value
        self.v_max       = self.get_parameter('max_linear_vel').value
        self.w_max       = self.get_parameter('max_angular_vel').value
        self.k_lin       = self.get_parameter('k_linear').value
        self.k_ang       = self.get_parameter('k_angular').value
        self.head_thresh = self.get_parameter('heading_threshold').value
        cmd_topic        = self.get_parameter('cmd_vel_topic').value
        odom_topic       = self.get_parameter('odom_topic').value
        state_topic      = self.get_parameter('state_topic').value

        if len(flat_wp) % 2 != 0:
            self.get_logger().error('waypoints must have an even number of values (x,y pairs)')
            raise ValueError('waypoints parameter length must be even')

        self.waypoints = [
            (float(flat_wp[i]), float(flat_wp[i+1]))
            for i in range(0, len(flat_wp), 2)
        ]
        self.wp_index   = 0

        # ---- State ---------------------------------------------------------
        self.x          = 0.0
        self.y          = 0.0
        self.yaw        = 0.0
        self.tl_state   = 'UNKNOWN'
        self.last_motion_state = ""
        self.current_linear_vel = 0.0
        self.acceleration = 0.01   # rampa de velocidad
        

        # ---- ROS I/O -------------------------------------------------------
        self.pub_vel  = self.create_publisher(Twist, cmd_topic, 10)

        self.sub_odom = self.create_subscription(
            Odometry,
            odom_topic,
            self._odom_callback,
            qos.qos_profile_sensor_data
        )
        
        self.sub_tl   = self.create_subscription(
            String, state_topic, self._tl_callback, 10)

        # Control loop at 20 Hz
        self.timer = self.create_timer(0.05, self._control_loop)

        self.get_logger().info(
            f'NavigationController ready — {len(self.waypoints)} waypoints loaded')

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def _odom_callback(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # Quaternion → yaw (rotation around Z)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw  = math.atan2(siny_cosp, cosy_cosp)

    def _tl_callback(self, msg: String):
        self.tl_state = msg.data

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalise_angle(angle: float) -> float:
        """Wrap angle to (-π, π]."""
        while angle >  math.pi:
            angle -= 2.0 * math.pi
        while angle <= -math.pi:
            angle += 2.0 * math.pi
        return angle

    # -----------------------------------------------------------------------
    # Control loop
    # -----------------------------------------------------------------------

    def _control_loop(self):
        twist = Twist()

        # All waypoints reached
        if self.wp_index >= len(self.waypoints):
            self.pub_vel.publish(twist)  # stop
            return

        gx, gy = self.waypoints[self.wp_index]

        dx        = gx - self.x
        dy        = gy - self.y
        distance  = math.hypot(dx, dy)

        # ---- Goal reached? -------------------------------------------------
        if distance < self.goal_tol:
            self.get_logger().info(
                f'Waypoint {self.wp_index} reached  ({gx:.2f}, {gy:.2f})')
            self.wp_index += 1
            self.pub_vel.publish(twist)  # brief stop between waypoints
            return

        # ---- Heading error -------------------------------------------------
        desired_yaw   = math.atan2(dy, dx)
        heading_error = self._normalise_angle(desired_yaw - self.yaw)

        # ---- Proportional controller ---------------------------------------
        angular_vel = np.clip(
            self.k_ang * heading_error, -self.w_max, self.w_max)

        # Only drive forward when roughly aligned — prevents wide arcs
        target_linear_vel = 0.0

        if abs(heading_error) < self.head_thresh:
            target_linear_vel = np.clip(
                self.k_lin * distance,
                0.0,
                self.v_max
            )

        # -------- Rampa de aceleración ----------
        if self.current_linear_vel < target_linear_vel:
            self.current_linear_vel += self.acceleration
            self.current_linear_vel = min(
                self.current_linear_vel,
                target_linear_vel
            )

        elif self.current_linear_vel > target_linear_vel:
            self.current_linear_vel -= self.acceleration
            self.current_linear_vel = max(
                self.current_linear_vel,
                target_linear_vel
            )

        linear_vel = self.current_linear_vel

        # ---- Traffic light speed factor ------------------------------------
        speed_factor = self.SPEED_FACTOR.get(self.tl_state, 1.0)

        motion_state = ""

        if self.tl_state == "RED":
            motion_state = "🔴 Rojo detectado -> detenido"

        elif self.tl_state == "YELLOW":
            motion_state = "🟡 Amarillo detectado -> reduciendo velocidad"

        elif self.tl_state == "GREEN":
            if linear_vel > 0.01:
                motion_state = f"🟢 Verde detectado -> avanzando ({linear_vel:.2f} m/s)"
            else:
                motion_state = "🟢 Verde detectado -> alineando"

        else:
            motion_state = "⚪ Buscando semaforo"

        # Imprimir solo si cambia el estado
        if motion_state != self.last_motion_state:
            self.get_logger().info(motion_state)
            self.last_motion_state = motion_state
        linear_vel  *= speed_factor
        # Also scale angular velocity so the robot doesn't spin while stopped
        angular_vel *= max(speed_factor, 0.0)

        twist.linear.x  = linear_vel
        twist.angular.z = angular_vel
        self.pub_vel.publish(twist)

        self.get_logger().info(
            f'WP[{self.wp_index}] dist={distance:.2f}m '
            f'head_err={math.degrees(heading_error):.1f}° '
            f'v={linear_vel:.3f} w={angular_vel:.3f} TL={self.tl_state}')


def main(args=None):
    rclpy.init(args=args)
    node = NavigationController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
