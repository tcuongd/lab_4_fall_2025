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
        self.initial_joint_positions = None
        self.counter = 0

        # Trotting gate positions, already implemented
        touch_down_position = np.array([0.05, 0.0, -0.14])
        stand_position_1 = np.array([0.025, 0.0, -0.14])
        stand_position_2 = np.array([0.0, 0.0, -0.14])
        stand_position_3 = np.array([-0.025, 0.0, -0.14])
        liftoff_position = np.array([-0.05, 0.0, -0.14])
        mid_swing_position = np.array([0.0, 0.0, -0.05])

        grounded_first = [
            touch_down_position,
            stand_position_1,
            stand_position_2,
            stand_position_3,
            liftoff_position,
            mid_swing_position,
        ]
        liftoff_first = [
            stand_position_3,
            liftoff_position,
            mid_swing_position,
            touch_down_position,
            stand_position_1,
            stand_position_2,
        ]

        ## trotting
        rf_ee_offset = np.array([0.06, -0.09, 0])
        rf_ee_triangle_positions = np.array(grounded_first) + rf_ee_offset

        lf_ee_offset = np.array([0.06, 0.09, 0])
        lf_ee_triangle_positions = np.array(liftoff_first) + lf_ee_offset

        rb_ee_offset = np.array([-0.11, -0.09, 0])
        rb_ee_triangle_positions = np.array(liftoff_first) + rb_ee_offset

        lb_ee_offset = np.array([-0.11, 0.09, 0])
        lb_ee_triangle_positions = np.array(grounded_first) + lb_ee_offset

        self.ee_triangle_positions = [
            rf_ee_triangle_positions,
            lf_ee_triangle_positions,
            rb_ee_triangle_positions,
            lb_ee_triangle_positions,
        ]
        self.fk_functions = [
            self.fr_leg_fk,
            self.fl_leg_fk,
            self.br_leg_fk,
            self.bl_leg_fk,
        ]

        self.target_joint_positions_cache, self.target_ee_cache = (
            self.cache_target_joint_positions()
        )
        print(f"shape of target_joint_positions_cache: {self.target_joint_positions_cache.shape}")
        print(f"shape of target_ee_cache: {self.target_ee_cache.shape}")

        self.pd_timer_period = 1.0 / 200  # 200 Hz
        # Due to `self.counter` incrementing at ik_timer_callback, each full motion lasts 50 / ik_hz
        self.ik_timer_period = 1.0 / 100  # 100 Hz
        self.pd_timer = self.create_timer(self.pd_timer_period, self.pd_timer_callback)
        self.ik_timer = self.create_timer(self.ik_timer_period, self.ik_timer_callback)

        self.standup_counter = 0
        # Enforce 2 seconds for standup. 2 / (ik_period)
        self.standup_steps = 2.0 / (self.ik_timer.timer_period_ns / 1e9)
        self.standup_complete = False

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

        # Capture initial joint positions on first reading
        if self.initial_joint_positions is None:
            self.initial_joint_positions = self.joint_positions.copy()
            self.get_logger().info(
                f"Captured initial joint positions: {self.initial_joint_positions}"
            )

    def inverse_kinematics_single_leg(self, target_ee, leg_index, initial_guess=[0.0, 0.0, 0.0]):
        """
        L-BFGS inverse kinematics solver.
        Approximates the inverse Hessian from recent gradient history,
        with backtracking line search for guaranteed convergence.
        """
        leg_forward_kinematics = self.fk_functions[leg_index]
        target = np.array(target_ee)

        def objective(theta):
            diff = leg_forward_kinematics(theta) - target
            return np.dot(diff, diff)

        def gradient(theta):
            grad = np.zeros(3)
            eps = 1e-7
            for i in range(3):
                e = np.zeros(3)
                e[i] = eps
                grad[i] = (objective(theta + e) - objective(theta - e)) / (2 * eps)
            return grad

        theta = np.array(initial_guess, dtype=np.float64)
        grad = gradient(theta)

        max_iterations = 50
        tol = 1e-10  # squared error tolerance (0.01mm accuracy)
        m = 5  # L-BFGS memory size

        s_hist = []
        y_hist = []
        rho_hist = []

        for _ in range(max_iterations):
            cost = objective(theta)
            if cost < tol:
                return theta

            # L-BFGS two-loop recursion for search direction
            q = grad.copy()
            k = len(s_hist)
            alphas = np.zeros(k)

            for i in range(k - 1, -1, -1):
                alphas[i] = rho_hist[i] * s_hist[i].dot(q)
                q = q - alphas[i] * y_hist[i]

            if k > 0:
                gamma = s_hist[-1].dot(y_hist[-1]) / y_hist[-1].dot(y_hist[-1])
            else:
                gamma = 1.0 / (np.linalg.norm(grad) + 1e-8)
            z = gamma * q

            for i in range(k):
                beta_i = rho_hist[i] * y_hist[i].dot(z)
                z = z + (alphas[i] - beta_i) * s_hist[i]

            direction = -z

            # Fall back to steepest descent if not a descent direction
            if direction.dot(grad) >= 0:
                direction = -grad
                s_hist.clear()
                y_hist.clear()
                rho_hist.clear()

            # Backtracking line search (Armijo condition)
            alpha = 1.0
            dg = grad.dot(direction)
            for _ in range(15):
                if objective(theta + alpha * direction) <= cost + 1e-4 * alpha * dg:
                    break
                alpha *= 0.5

            # Update position and gradient
            s = alpha * direction
            theta_new = theta + s
            grad_new = gradient(theta_new)
            y = grad_new - grad

            # Store correction pair if curvature condition holds
            ys = y.dot(s)
            if ys > 1e-10:
                if len(s_hist) >= m:
                    s_hist.pop(0)
                    y_hist.pop(0)
                    rho_hist.pop(0)
                s_hist.append(s.copy())
                y_hist.append(y.copy())
                rho_hist.append(1.0 / ys)

            theta = theta_new
            grad = grad_new

        return theta

    def interpolate_triangle(self, t, leg_index) -> np.ndarray:
        positions = self.ee_triangle_positions[leg_index]
        n_positions = len(positions)

        position_idx = t * n_positions
        # 0 -> 0.16, 0.16 -> 0.32, 0.32 -> 0.48, 0.48 -> 0.64, 0.64 -> 0.8, 0.8 -> 1
        # 0 -> 1.  , 1 -> 2.     , 2 -> 3.     , 3 -> 4.     , 4 -> 5.    , 5 -> 0
        idx_prev, idx_next = math.floor(position_idx), math.ceil(position_idx)
        idx_next = idx_next % n_positions

        return positions[idx_prev] + (position_idx - idx_prev) * (
            positions[idx_next] - positions[idx_prev]
        )

    def cache_target_joint_positions(self):
        # Calculate and store the target joint positions for a cycle and all 4 legs
        target_joint_positions_cache = []
        target_ee_cache = []
        for leg_index in range(4):
            target_joint_positions_cache.append([])
            target_ee_cache.append([])
            target_joint_positions = [0] * 3
            # This creates 50 steps, so one round of motion will last (ik_timer_period / 50) seconds.
            for t in np.arange(0, 1, 0.02):
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
        if self.joint_positions is None or self.initial_joint_positions is None:
            return

        if not self.standup_complete:
            # Standup phase: interpolate from initial position to first gait target
            alpha = self.standup_counter / self.standup_steps
            # Smooth cubic interpolation (ease in/out)
            alpha = 3 * alpha**2 - 2 * alpha**3
            first_gait_target = self.target_joint_positions_cache[0]
            self.target_joint_positions = (
                self.initial_joint_positions * (1 - alpha) + first_gait_target * alpha
            )
            self.standup_counter += 1

            current_ee = self.forward_kinematics(self.joint_positions)
            target_ee = self.forward_kinematics(self.target_joint_positions)
            self.get_logger().info(
                f"[STANDUP {self.standup_counter}/{self.standup_steps}] "
                f"Target EE: {target_ee}, "
                f"Current EE: {current_ee}, "
                f"Target Angles: {self.target_joint_positions}, "
                f"Current Angles: {self.joint_positions}"
            )

            if self.standup_counter >= self.standup_steps:
                self.standup_complete = True
                self.get_logger().info("Standup complete, starting gait")
        else:
            # Normal gait phase
            target_ee, self.target_joint_positions = self.get_target_joint_positions()
            current_ee = self.forward_kinematics(self.joint_positions)

            self.get_logger().info(
                f"Target EE: {target_ee}, "
                f"Current EE: {current_ee}, "
                f"Target Angles: {self.target_joint_positions}, "
                f"Target Angles to EE: {self.forward_kinematics(self.target_joint_positions)}, "
                f"Current Angles: {self.joint_positions}"
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
