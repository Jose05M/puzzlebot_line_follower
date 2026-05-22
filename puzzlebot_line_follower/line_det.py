#!/usr/bin/env python3
#PRUEBA
import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool


class LineDetector(Node):

    def __init__(self):
        super().__init__('line_det')

        # PARAMETERS
        self.declare_parameter('camera_topic', '/video_source/raw')
        self.declare_parameter('line_error_topic', '/line_error')
        self.declare_parameter('line_detected_topic', '/line_detected')
        self.declare_parameter('debug_view', True)

        camera_topic = self.get_parameter('camera_topic').value
        error_topic  = self.get_parameter('line_error_topic').value
        detected_topic = self.get_parameter('line_detected_topic').value
        self.debug_view = self.get_parameter('debug_view').value

        # ROS
        self.sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            1
        )
        self.pub_error = self.create_publisher(
            Float32,
            error_topic,
            10
        )
        self.pub_detected = self.create_publisher(
            Bool,
            detected_topic,
            10
        )

        # INTERNAL STATE
        self.last_error = 0.0
        self.get_logger().info('Connected Components Line Detector Started')

    # IMAGE CALLBACK
    def image_callback(self, msg):
        # ROS Image -> OpenCV
        frame = np.frombuffer(msg.data,dtype=np.uint8).reshape((msg.height, msg.width, 3))

        # Resize
        target_width  = 640
        target_height = 480

        frame = cv2.resize(frame,(target_width, target_height))

        # ROI
        roi_y = frame[int(target_height * 0.4):, :]
        crop_percent = 0.25
        crop_x = int(target_width * crop_percent)
        roi = roi_y[:, crop_x:target_width - crop_x]
        roi_height, roi_width = roi.shape[:2]

        # Trapezoidal mask
        top_width = int(roi_width * 0.9)

        trapezoid = np.array([[
            ((roi_width - top_width) // 2, 0),
            ((roi_width + top_width) // 2, 0),
            (roi_width, roi_height),
            (0, roi_height)
        ]], dtype=np.int32)

        mask = np.zeros(
            (roi_height, roi_width),
            dtype=np.uint8
        )

        cv2.fillPoly(mask, trapezoid, 255)

        #roi_masked = cv2.bitwise_and(roi,roi,mask=mask)

        # Grayscale
        gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)

        # Gaussian Blur
        blurred = cv2.GaussianBlur(gray,(5, 5),0)

        # Threshold
        _, binary_inv = cv2.threshold(blurred,100,255,cv2.THRESH_BINARY_INV)
        
        binary_inv = cv2.bitwise_and(binary_inv,mask)

        # Morphological Operations
        kernel = np.ones((3, 3), np.uint8)

        morph = cv2.erode(binary_inv,kernel,iterations=3)

        morph = cv2.dilate(morph,kernel,iterations=3)

        # Connected Components
        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(
                morph,
                connectivity=4
            )

        output = roi.copy()

        total_weight = 0.0
        weighted_sum = np.array([0.0, 0.0])

        MIN_AREA = 1000
        MAX_AREA = 100000

        # Process Components
        for i in range(1, num_labels):

            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]

            if MIN_AREA <= area <= MAX_AREA:

                weighted_sum += np.array([cx, cy]) * area
                total_weight += area

                if self.debug_view:

                    cv2.rectangle(output,(x, y),(x + w, y + h),(255, 0, 0),2)
                    cv2.circle(output,(int(cx), int(cy)),4,(0, 0, 255),-1)
                    cv2.putText(output,f"({int(cx)}, {int(cy)})",(int(cx) + 5, int(cy)),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 255, 255),1)

        error = self.last_error
        line_detected = False

        if total_weight > 0:

            avg_cx, avg_cy = (
                weighted_sum / total_weight
            ).astype(int)

            # --------------------------------------------------------
            # Error
            # --------------------------------------------------------

            error = float(avg_cx - (roi_width / 2))

            self.last_error = error
            line_detected = True

            # --------------------------------------------------------
            # Draw Setpoint
            # --------------------------------------------------------

            if self.debug_view:

                cv2.circle(output,(avg_cx, avg_cy),6,(0, 255, 255),-1)

                cv2.putText(output,"Setpoint",(avg_cx + 10, avg_cy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 255),1)

        # ------------------------------------------------------------
        # Publish Error
        # ------------------------------------------------------------

        error_msg = Float32()
        error_msg.data = error

        self.pub_error.publish(error_msg)

        detected_msg = Bool()
        detected_msg.data = line_detected
        self.pub_detected.publish(detected_msg)

        # ------------------------------------------------------------
        # Debug Visualisation
        # ------------------------------------------------------------

        if self.debug_view:

            center_x = roi_width // 2

            cv2.line(output,(center_x, 0),(center_x, roi_height),(0, 255, 0),2)

            cv2.putText(output,f'Error: {error:.2f}',(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255),2)

            cv2.imshow("Connected Components Line Detector",output)

            cv2.imshow("Binary",morph)
            cv2.imshow("Real",frame)
            cv2.waitKey(1)


def main(args=None):

    rclpy.init(args=args)

    node = LineDetector()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
