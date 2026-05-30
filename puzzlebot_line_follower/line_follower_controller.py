#!/usr/bin/env python3

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

    def __init__(self):
        super().__init__('line_follower_controller')

        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('curve_speed', 0.07)
        self.declare_parameter('curve_threshold', 50.0)

        self.declare_parameter('kp_straight', 0.0015)
        self.declare_parameter('kp_curve', 0.0040)
        
        self.declare_parameter('kd', 0.0030)
        self.declare_parameter('max_angular_vel', 1.5)
        self.declare_parameter('cmd_vel_topic',    '/cmd_vel')
        self.declare_parameter('line_error_topic', '/line_error')
        self.declare_parameter('state_topic',      '/traffic_light/state')

        self.linear_speed = self.get_parameter('linear_speed').value
        self.curve_speed = self.get_parameter('curve_speed').value
        self.curve_threshold = self.get_parameter('curve_threshold').value

        self.kp_straight = self.get_parameter('kp_straight').value
        self.kp_curve = self.get_parameter('kp_curve').value

        self.kd = self.get_parameter('kd').value
        self.w_max = self.get_parameter('max_angular_vel').value
        cmd_topic   = self.get_parameter('cmd_vel_topic').value
        state_topic = self.get_parameter('state_topic').value
        line_topic  = self.get_parameter('line_error_topic').value

        # State
        self.line_error = 0.0
        self.prev_error = 0.0
        self.tl_state   = 'UNKNOWN'
        self.last_motion_state = ""
        self.current_linear_vel = 0.0
        self.acceleration = 0.01   # rampa de velocidad
        self.curve_state = False
        self.line_lost_threshold = 140
        self.filtered_error = 0.0
        

        # ROS I/O
        self.pub_vel  = self.create_publisher(Twist, cmd_topic, 10)
        self.sub_line = self.create_subscription(Float32,line_topic,self._line_callback,1)
        self.sub_tl   = self.create_subscription(String, state_topic, self._tl_callback, 10)

        # Control loop at 20 Hz
        self.timer = self.create_timer(0.05, self._control_loop)
        self.get_logger().info('LineFollowerController Started')

    # Callbacks
    def _line_callback(self, msg: Float32):
        alpha = 0.5
        self.filtered_error = (alpha * self.filtered_error +(1 - alpha) * msg.data)
        self.line_error = self.filtered_error

    def _tl_callback(self, msg: String):
        self.tl_state = msg.data

    # Control loop
    def _control_loop(self):
        twist = Twist()

        curve_detected = (abs(self.line_error) > self.curve_threshold)
        if curve_detected != self.curve_state:
            if curve_detected:
                self.get_logger().info("↩️ Curva detectada")
            else:
                self.get_logger().info("➡️ Recta detectada")
            self.curve_state = curve_detected

        # ADAPTIVE KP
        if curve_detected:
            kp = self.kp_curve
        else:
            kp = self.kp_straight

        # PD Controller
        derivative = self.line_error - self.prev_error
        derivative = np.clip(derivative, -70, 70)

        angular_vel = -(kp * self.line_error + self.kd * derivative)
        self.prev_error = self.line_error

        # Saturation
        angular_vel = np.clip(angular_vel,-self.w_max,self.w_max)
        angular_vel *= 0.85

        # Adaptive Linear Speed
        if curve_detected:
            target_linear_vel = self.curve_speed
        else:
            target_linear_vel = self.linear_speed

        # -------- Acceleration Ramp --------
        if self.current_linear_vel < target_linear_vel:
            self.current_linear_vel += self.acceleration
            self.current_linear_vel = min(self.current_linear_vel, target_linear_vel)

        elif self.current_linear_vel > target_linear_vel:
            self.current_linear_vel -= self.acceleration
            self.current_linear_vel = max(self.current_linear_vel,target_linear_vel)

        linear_vel = self.current_linear_vel
        if abs(self.line_error) > self.line_lost_threshold:
            linear_vel *= 0.4

        # TURN-BASED SPEED REDUCTION
        turn_factor = 1.0 - min(abs(angular_vel) / self.w_max,1.0)
        turn_factor = max(turn_factor,0.35)
        linear_vel *= turn_factor

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

        if self.tl_state == "RED":
            linear_vel = 0.0
        elif self.tl_state == "YELLOW":
            linear_vel *= 0.5

        twist.linear.x  = linear_vel
        twist.angular.z = angular_vel
        self.pub_vel.publish(twist)

        #self.get_logger().info(f'Error={self.line_error:.2f} | v={linear_vel:.3f} | w={angular_vel:.3f} | TL={self.tl_state}')

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