#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Vector3, TwistStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from apriltag_ros.msg import AprilTagDetectionArray
from sensor_msgs.msg import Imu
import tf2_ros

# quaternion --> rotation matrix
def quaternion_to_rotation_matrix(q):
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx**2 - 2*qy**2]
    ])

def apriltag(x, y, z):
    p = PoseStamped()
    p.pose.position.x = x
    p.pose.position.y = y
    p.pose.position.z = z
    return p

apriltags = [apriltag(0, 0, 0),
             apriltag(0.3, 0.3, 0),
             apriltag(-0.3, 0.3, 0),
             apriltag(0.3, -0.3, 0),
             apriltag(-0.3, -0.3, 0),
]

current_velocity = np.zeros(3)
current_acceleration = np.zeros(3)

def process_tag_detection(detection):
    tag_id = detection.id[0]
    position = detection.pose.pose.pose.position
    orientation = detection.pose.pose.pose.orientation
    
    rel_pos_b = np.array([position.x, position.y, position.z])
    q = [orientation.x, orientation.y, orientation.z, orientation.w]
    R = quaternion_to_rotation_matrix(q)
    rel_pos = np.dot(R, rel_pos_b)
    
    if tag_id < len(apriltags):
        tag_pos = np.array([apriltags[tag_id].pose.position.x,
                            apriltags[tag_id].pose.position.y,
                            apriltags[tag_id].pose.position.z])
        measurement = tag_pos - rel_pos
        return tag_id, measurement
    else:
        rospy.logwarn("Detected AprilTag with ID %d is out of range.", tag_id)
        return None, None


def velocity_callback(msg):
    global current_velocity
    current_velocity = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])

    # 發布速度信息給 EKF
    twist_msg = TwistStamped()
    twist_msg.header = msg.header
    twist_msg.twist.linear = msg.twist.linear
    velocity_pub.publish(twist_msg)

def imu_callback(msg):
    global current_acceleration
    gravity = np.array([0, 0, 9.81])
    measured_acc = np.array([msg.linear_acceleration.x, 
                             msg.linear_acceleration.y, 
                             msg.linear_acceleration.z])
    current_acceleration = measured_acc - gravity

    # 發布 IMU 信息給 EKF
    imu_pub.publish(msg)

def tag_detections_callback(msg, pub, tf_broadcaster):
    for detection in msg.detections:
        tag_id, measurement = process_tag_detection(detection)
        if measurement is not None:
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header = msg.header
            pose_msg.header.frame_id = "map"  
            pose_msg.pose.pose.position.x = measurement[0]
            pose_msg.pose.pose.position.y = measurement[1]
            pose_msg.pose.pose.position.z = measurement[2]
            
            pose_msg.pose.covariance = [0.1, 0, 0, 0, 0, 0,
                                        0, 0.1, 0, 0, 0, 0,
                                        0, 0, 0.1, 0, 0, 0,
                                        0, 0, 0, 0.1, 0, 0,
                                        0, 0, 0, 0, 0.1, 0,
                                        0, 0, 0, 0, 0, 0.1]
            
            pub.publish(pose_msg)

            # 廣播 TF
            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "map"
            t.child_frame_id = "base_link"
            
            t.transform.translation.x = measurement[0]
            t.transform.translation.y = measurement[1]
            t.transform.translation.z = measurement[2]
            t.transform.rotation.w = 1.0  
            tf_broadcaster.sendTransform(t)



def main():
    global velocity_pub, imu_pub
    rospy.init_node('apriltag_measure_uav')
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # 創建發布器
    tag_pose_pub = rospy.Publisher('/tag_detections_pose', PoseWithCovarianceStamped, queue_size=10)
    velocity_pub = rospy.Publisher('/mavros/local_position/velocity_body_filtered', TwistStamped, queue_size=10)
    imu_pub = rospy.Publisher('/imu/data_filtered', Imu, queue_size=10)

    # 創建訂閱者
    rospy.Subscriber('/tag_detections', AprilTagDetectionArray, 
                     lambda msg: tag_detections_callback(msg, tag_pose_pub, tf_broadcaster))
    rospy.Subscriber('/mavros/local_position/velocity_body', TwistStamped, velocity_callback)
    rospy.Subscriber('/mavros/imu/data', Imu, imu_callback)
    

    rospy.spin()

if __name__ == '__main__':
    main()