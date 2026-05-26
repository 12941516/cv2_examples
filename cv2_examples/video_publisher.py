import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher_node')
        
        self.frame_publisher = self.create_publisher(CompressedImage, 'camera', 10)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open camera.")
            return

        self.timer = self.create_timer(0.03, self.publish_frame)

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to grab frame.")
            return

        src = cv2.resize(frame, (640,480))
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = 'jpeg'
        ret, buffer = cv2.imencode('.jpg', src)
        if not ret:
            self.get_logger().warn("Failed to encode frame.")
            return
        msg.data = np.array(buffer).tobytes()
        self.frame_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


