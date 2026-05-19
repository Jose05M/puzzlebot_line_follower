#!/usr/bin/env python3
"""
line_follower_controller.py

controller_node
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

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_msgs.msg import Float32
import numpy as np


class LineFollowerController(Node):
    """
    Point-to-point proportional controller with traffic light awareness.
    """

    # Speed multipliers per traffic state
    SPEED_FACTOR = {
        'GREEN':   1.0,
        'YELLOW':  0.5,
        'RED':     0.0,
        'UNKNOWN': 1.0,
    }

    def __init__(self):
        super().__init__('line_follower_controller')

        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('kp', 0.0035)
        self.declare_parameter('kd', 0.0010)
        self.declare_parameter('max_angular_vel', 1.5)
        self.declare_parameter('cmd_vel_topic',    '/cmd_vel')
        self.declare_parameter('line_error_topic', '/line_error')
        self.declare_parameter('state_topic',      '/traffic_light/state')

        self.linear_speed = self.get_parameter('linear_speed').value
        self.kp = self.get_parameter('kp').value
        self.kd = self.get_parameter('kd').value
        self.w_max = self.get_parameter('max_angular_vel').value
        cmd_topic   = self.get_parameter('cmd_vel_topic').value
        state_topic = self.get_parameter('state_topic').value
        line_topic  = self.get_parameter('line_error_topic').value

        # ---- State ---------------------------------------------------------
        self.line_error = 0.0
        self.prev_error = 0.0
        self.tl_state   = 'UNKNOWN'
        self.last_motion_state = ""
        self.current_linear_vel = 0.0
        self.acceleration = 0.01   # rampa de velocidad
        

        # ---- ROS I/O -------------------------------------------------------
        self.pub_vel  = self.create_publisher(Twist, cmd_topic, 10)
        self.sub_line = self.create_subscription(Float32,line_topic,self._line_callback,10)
        self.sub_tl   = self.create_subscription(String, state_topic, self._tl_callback, 10)

        # Control loop at 20 Hz
        self.timer = self.create_timer(0.05, self._control_loop)

        self.get_logger().info('LineFollowerController Started')

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def _line_callback(self, msg: Float32):
        self.line_error = msg.data

    def _tl_callback(self, msg: String):
        self.tl_state = msg.data

    # -----------------------------------------------------------------------
    # Control loop
    # -----------------------------------------------------------------------

    def _control_loop(self):
        twist = Twist()

        # ------------------------------------------------------------
        # PD Controller
        # ------------------------------------------------------------

        derivative = self.line_error - self.prev_error

        angular_vel = -(
            self.kp * self.line_error +
            self.kd * derivative
        )

        self.prev_error = self.line_error

        # Saturation
        angular_vel = np.clip(
            angular_vel,
            -self.w_max,
            self.w_max
        )

        # Constant linear speed
        target_linear_vel = self.linear_speed

        # -------- Acceleration Ramp --------

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
            f'Error={self.line_error:.2f} '
            f'v={linear_vel:.3f} '
            f'w={angular_vel:.3f} '
            f'TL={self.tl_state}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
