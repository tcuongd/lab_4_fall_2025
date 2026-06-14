import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

import math

np.set_printoptions(precision=3, suppress=True)


def rotation_x(angle: float) -> np.ndarray:
    """angle in radians."""
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )


def rotation_y(angle: float) -> np.ndarray:
    """angle in radians."""
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )


def rotation_z(angle: float) -> np.ndarray:
    """angle in radians."""
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def translation(x: float, y: float, z: float) -> np.ndarray:
    return np.array(
        [
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )


class InverseKinematics(Node):
    def __init__(self):
        super().__init__("inverse_kinematics")
        self.joint_subscription = self.create_subscription(
            JointState, "joint_states", self.listener_callback, 10
        )
        self.joint_subscription  # prevent unused variable warning

        self.command_publisher = self.create_publisher(
            Float64MultiArray, "/forward_command_controller/commands", 10
        )

        self.joint_positions = None
        self.joint_velocities = None
        self.target_joint_positions = None
        self.counter = 0

        # Trotting gate positions, already implemented
        touch_down_position = np.array([0.05, 0.0, -0.14])
        stand_position_1 = np.array([0.025, 0.0, -0.14])
        stand_position_2 = np.array([0.0, 0.0, -0.14])
        stand_position_3 = np.array([-0.025, 0.0, -0.14])
        liftoff_position = np.array([-0.05, 0.0, -0.14])
        mid_swing_position = np.array([0.0, 0.0, -0.05])

        ## trotting
        rf_ee_offset = np.array([0.06, -0.09, 0])
        rf_ee_triangle_positions = (
            np.array(
                [
                    touch_down_position,
                    stand_position_1,
                    stand_position_2,
                    stand_position_3,
                    liftoff_position,
                    mid_swing_position,
                ]
            )
            + rf_ee_offset
        )

        lf_ee_offset = np.array([0.06, 0.09, 0])
        lf_ee_triangle_positions = (
            np.array(
                [
                    liftoff_position,
                    mid_swing_position,
                    touch_down_position,
                    stand_position_1,
                    stand_position_2,
                    stand_position_3,
                ]
            )
            + lf_ee_offset
        )

        rb_ee_offset = np.array([-0.11, -0.09, 0])
        rb_ee_triangle_positions = (
            np.array(
                [
                    liftoff_position,
                    mid_swing_position,
                    touch_down_position,
                    stand_position_1,
                    stand_position_2,
                    stand_position_3,
                ]
            )
            + rb_ee_offset
        )

        lb_ee_offset = np.array([-0.11, 0.09, 0])
        lb_ee_triangle_positions = (
            np.array(
                [
                    touch_down_position,
                    stand_position_1,
                    stand_position_2,
                    stand_position_3,
                    liftoff_position,
                    mid_swing_position,
                ]
            )
            + lb_ee_offset
        )

        self.ee_triangle_positions = [
            rf_ee_triangle_positions,
            lf_ee_triangle_positions,
            rb_ee_triangle_positions,
            lb_ee_triangle_positions,
        ]
        self.fk_functions = [self.fr_leg_fk, self.fl_leg_fk, self.br_leg_fk, self.bl_leg_fk]

        self.target_joint_positions_cache, self.target_ee_cache = (
            self.cache_target_joint_positions()
        )
        print(f"shape of target_joint_positions_cache: {self.target_joint_positions_cache.shape}")
        print(f"shape of target_ee_cache: {self.target_ee_cache.shape}")

        self.pd_timer_period = 1.0 / 200  # 200 Hz
        self.ik_timer_period = 1.0 / 100  # 100 Hz
        self.pd_timer = self.create_timer(self.pd_timer_period, self.pd_timer_callback)
        self.ik_timer = self.create_timer(self.ik_timer_period, self.ik_timer_callback)

    def fr_leg_fk(self, theta) -> np.ndarray:
        # Already implemented in Lab 2
        T_RF_0_1 = translation(0.07500, -0.08350, 0) @ rotation_x(1.57080) @ rotation_z(theta[0])
        T_RF_1_2 = rotation_y(-1.57080) @ rotation_z(theta[1])
        T_RF_2_3 = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(theta[2])
        T_RF_3_ee = translation(0.06231, -0.06216, 0.01800)
        T_RF_0_ee = T_RF_0_1 @ T_RF_1_2 @ T_RF_2_3 @ T_RF_3_ee
        return T_RF_0_ee[:3, 3]

    def fl_leg_fk(self, theta) -> np.ndarray:
        T_0_1 = translation(0.07500, 0.0445, 0) @ rotation_x(1.57080) @ rotation_z(-theta[0])
        T_1_2 = translation(0, 0, -0.039) @ rotation_y(-1.57080) @ rotation_z(+theta[1])
        T_2_3 = translation(0, -0.0494, 0.0685) @ rotation_y(+1.57080) @ rotation_z(-theta[2])
        T_3_ee = translation(0.06231, -0.06216, -0.018)
        T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee
        return T_0_ee[:3, 3]

    def br_leg_fk(self, theta) -> np.ndarray:
        T_0_1 = translation(-0.07500, -0.0335, 0) @ rotation_x(1.57080) @ rotation_z(+theta[0])
        T_1_2 = translation(0, 0, +0.039) @ rotation_y(-1.57080) @ rotation_z(+theta[1])
        T_2_3 = translation(0, -0.0494, 0.0685) @ rotation_y(+1.57080) @ rotation_z(+theta[2])
        T_3_ee = translation(0.06231, -0.06216, +0.018)
        T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee
        return T_0_ee[:3, 3]

    def bl_leg_fk(self, theta) -> np.ndarray:
        T_0_1 = translation(-0.07500, 0.0445, 0) @ rotation_x(1.57080) @ rotation_z(-theta[0])
        T_1_2 = translation(0, 0, -0.039) @ rotation_y(-1.57080) @ rotation_z(+theta[1])
        T_2_3 = translation(0, -0.0494, 0.0685) @ rotation_y(+1.57080) @ rotation_z(-theta[2])
        T_3_ee = translation(0.06231, -0.06216, -0.018)
        T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee
        return T_0_ee[:3, 3]

    def forward_kinematics(self, theta):
        return np.concatenate([self.fk_functions[i](theta[3 * i : 3 * i + 3]) for i in range(4)])

    def listener_callback(self, msg):
        joints_of_interest = [
            "leg_front_r_1",
            "leg_front_r_2",
            "leg_front_r_3",
            "leg_front_l_1",
            "leg_front_l_2",
            "leg_front_l_3",
            "leg_back_r_1",
            "leg_back_r_2",
            "leg_back_r_3",
            "leg_back_l_1",
            "leg_back_l_2",
            "leg_back_l_3",
        ]
        self.joint_positions = np.array(
            [msg.position[msg.name.index(joint)] for joint in joints_of_interest]
        )
        self.joint_velocities = np.array(
            [msg.velocity[msg.name.index(joint)] for joint in joints_of_interest]
        )

    def inverse_kinematics_single_leg(self, target_ee, leg_index, initial_guess=[0.0, 0.0, 0.0]):
        leg_forward_kinematics = self.fk_functions[leg_index]

        def cost_function(theta) -> tuple[float, np.ndarray]:
            """
            Use the forward_kinematics method to get the current end-effector position.
            Calculate the L1 distance between the current and target end-effector positions.
            Return the sum of squared L1 distances as the cost (AKA the squared L2 norm of the error vector).
            """
            current_ee = leg_forward_kinematics(theta)
            l1_errors = np.abs(current_ee - np.array(target_ee))
            return np.sqrt(np.sum(l1_errors**2)), l1_errors

        def gradient(theta: np.ndarray, epsilon=1e-3) -> np.ndarray:
            assert theta.shape == (3,), "expected (3,) array for theta."

            grads = np.zeros(3, dtype=np.float64)

            for i, angle in enumerate(theta):
                theta_back = theta.copy()
                theta_back[i] = angle - epsilon

                theta_forward = theta.copy()
                theta_forward[i] = angle + epsilon

                c_back, _ = cost_function(theta_back)
                c_forward, _ = cost_function(theta_forward)

                grads[i] = (c_forward - c_back) / (2 * epsilon)

            return grads

        theta = np.array(initial_guess, dtype=np.float64)
        learning_rate = 5.0
        max_iterations = 20
        # tolerance in metres
        tolerance = 0.2 / 100

        cost_l = []
        best_theta = None
        best_cost = float("inf")
        for i in range(max_iterations):
            grad = gradient(theta)
            theta -= learning_rate * grad

            # Use mean L1 to check convergence
            _, l1_errors = cost_function(theta)
            cost_l.append(np.mean(l1_errors))

            if np.mean(l1_errors) < best_cost:
                best_cost = np.mean(l1_errors)
                best_theta = theta.copy()

            if cost_l[-1] <= tolerance:
                print(f"Converged after {i} iterations: {cost_l[-1]:.4f}")
                return theta

        print(f"Cost: {cost_l}")

        if best_theta is None:
            print("[WARNING] no change suggested for theta.")
            return theta
        else:
            return best_theta

    def interpolate_triangle(self, t, leg_index) -> np.ndarray:
        positions = self.ee_triangle_positions[leg_index]
        n_positions = len(positions)

        position_idx = t * n_positions
        # 0 -> 0.16, 0.16 -> 0.32, 0.32 -> 0.48, 0.48 -> 0.64, 0.64 -> 0.8, 0.8 -> 1
        # 0 -> 1.  , 1 -> 2.     , 2 -> 3.     , 3 -> 4.     , 4 -> 5.    , 5 -> 0
        idx_prev, idx_next = math.floor(position_idx), math.ceil(position_idx)
        idx_next = idx_next % (n_positions - 1)

        return (position_idx - idx_prev) * (positions[idx_next] - positions[idx_prev])

    def cache_target_joint_positions(self):
        # Calculate and store the target joint positions for a cycle and all 4 legs
        target_joint_positions_cache = []
        target_ee_cache = []
        for leg_index in range(4):
            target_joint_positions_cache.append([])
            target_ee_cache.append([])
            target_joint_positions = [0] * 3
            for t in np.arange(0, 1, 0.02):
                print(t)
                target_ee = self.interpolate_triangle(t, leg_index)
                target_joint_positions = self.inverse_kinematics_single_leg(
                    target_ee, leg_index, initial_guess=target_joint_positions
                )

                target_joint_positions_cache[leg_index].append(target_joint_positions)
                target_ee_cache[leg_index].append(target_ee)

        # (4, 50, 3) -> (50, 12)
        target_joint_positions_cache = np.concatenate(target_joint_positions_cache, axis=1)
        target_ee_cache = np.concatenate(target_ee_cache, axis=1)

        return target_joint_positions_cache, target_ee_cache

    def get_target_joint_positions(self):
        target_joint_positions = self.target_joint_positions_cache[self.counter]
        target_ee = self.target_ee_cache[self.counter]
        self.counter += 1
        if self.counter >= self.target_joint_positions_cache.shape[0]:
            self.counter = 0
        return target_ee, target_joint_positions

    def ik_timer_callback(self):
        if self.joint_positions is not None:
            target_ee, self.target_joint_positions = self.get_target_joint_positions()
            current_ee = self.forward_kinematics(self.joint_positions)

            self.get_logger().info(
                f"Target EE: {target_ee}, \
                Current EE: {current_ee}, \
                Target Angles: {self.target_joint_positions}, \
                Target Angles to EE: {self.forward_kinematics(self.target_joint_positions)}, \
                Current Angles: {self.joint_positions}"
            )

    def pd_timer_callback(self):
        if self.target_joint_positions is not None:
            command_msg = Float64MultiArray()
            command_msg.data = self.target_joint_positions.tolist()
            self.command_publisher.publish(command_msg)


def main():
    rclpy.init()
    inverse_kinematics = InverseKinematics()

    try:
        rclpy.spin(inverse_kinematics)
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        # Send zero torques
        zero_torques = Float64MultiArray()
        zero_torques.data = [0.0] * 12
        inverse_kinematics.command_publisher.publish(zero_torques)

        inverse_kinematics.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
