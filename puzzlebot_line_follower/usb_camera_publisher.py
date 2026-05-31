#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image

import cv2
import numpy as np


class UsbCameraPublisher(Node):

    def __init__(self):

        super().__init__('usb_camera_publisher')

        self.publisher_ = self.create_publisher(
            Image,
            '/video_source/raw',
            1
        )

        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self.get_logger().error('Could not open camera')
            raise RuntimeError('Camera not found')

        self.timer = self.create_timer(
            0.033,
            self.timer_callback
        )

        self.get_logger().info(
            'USB Camera Publisher Started'
        )

    def timer_callback(self):
        for _ in range(2):
            self.cap.grab()

        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warning(
                'Failed to capture frame'
            )
            return

        msg = Image()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'

        msg.height = frame.shape[0]
        msg.width = frame.shape[1]

        msg.encoding = 'bgr8'

        msg.is_bigendian = False

        msg.step = frame.shape[1] * 3

        msg.data = np.array(frame).tobytes()

        self.publisher_.publish(msg)

    def destroy_node(self):

        if self.cap.isOpened():
            self.cap.release()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = UsbCameraPublisher()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
