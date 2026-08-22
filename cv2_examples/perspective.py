import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

class Perspective(Node):
    def __init__(self):
        super().__init__('perspective_node')
        self.videoSubscriber = self.create_subscription(
            CompressedImage,
            '/camera',
            self.videoSubscriber_callback,
            10
        )
        self.src_pts = np.float32([[200,200], [440,200], [640,480], [0,480]])
        self.dst_w = 640
        self.dst_h = 480
        self.dst_pts = np.float32([[0,0], [self.dst_w-1,0], [self.dst_w-1,self.dst_h-1], [0,self.dst_h-1]])
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
    
    def perspectiveTransform(self, src):
        warped = cv2.warpPerspective(src, self.M, (self.dst_w, self.dst_h))
        return warped
    
    def videoSubscriber_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            src = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f'Error decoding compressed image: {e}')
            return
            
        perspective = self.perspectiveTransform(src)
        cv2.polylines(src,[self.src_pts.astype(int)], isClosed=True, color=(0,0,255), thickness=2)
        cv2.imshow('video', src)
        cv2.imshow('perspective', perspective)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()

def main():
    rclpy.init()
    node = Perspective()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


