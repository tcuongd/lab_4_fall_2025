"""
Simulate and visualize the motion of all 4 legs of the quadruped robot as an MP4 animation.
Uses the triangle gait waypoints and IK from lab_4.py to compute motor angles,
then runs FK to get the full kinematic chain and animates in the X-Z plane.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import os


def rotation_x(angle):
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )


def rotation_y(angle):
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )


def rotation_z(angle):
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def translation(x, y, z):
    return np.array(
        [
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )


def get_position(T):
    return T[:3, 3]


# --- Forward kinematics: return end-effector position only ---


def fr_leg_fk(theta):
    T_0_1 = (
        translation(0.07500, -0.08350, 0) @ rotation_x(1.57080) @ rotation_z(theta[0])
    )
    T_1_2 = rotation_y(-1.57080) @ rotation_z(theta[1])
    T_2_3 = (
        translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(theta[2])
    )
    T_3_ee = translation(0.06231, -0.06216, 0.01800)
    T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee
    return T_0_ee[:3, 3]


def fl_leg_fk(theta):
    T_0_1 = (
        translation(0.07500, 0.0445, 0) @ rotation_x(1.57080) @ rotation_z(-theta[0])
    )
    T_1_2 = translation(0, 0, -0.039) @ rotation_y(-1.57080) @ rotation_z(theta[1])
    T_2_3 = (
        translation(0, -0.0494, 0.0685) @ rotation_y(1.57080) @ rotation_z(-theta[2])
    )
    T_3_ee = translation(0.06231, -0.06216, -0.018)
    T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee
    return T_0_ee[:3, 3]


def br_leg_fk(theta):
    T_0_1 = (
        translation(-0.07500, -0.0335, 0) @ rotation_x(1.57080) @ rotation_z(theta[0])
    )
    T_1_2 = translation(0, 0, 0.039) @ rotation_y(-1.57080) @ rotation_z(theta[1])
    T_2_3 = translation(0, -0.0494, 0.0685) @ rotation_y(1.57080) @ rotation_z(theta[2])
    T_3_ee = translation(0.06231, -0.06216, 0.018)
    T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee
    return T_0_ee[:3, 3]


def bl_leg_fk(theta):
    T_0_1 = (
        translation(-0.07500, 0.0445, 0) @ rotation_x(1.57080) @ rotation_z(-theta[0])
    )
    T_1_2 = translation(0, 0, -0.039) @ rotation_y(-1.57080) @ rotation_z(theta[1])
    T_2_3 = (
        translation(0, -0.0494, 0.0685) @ rotation_y(1.57080) @ rotation_z(-theta[2])
    )
    T_3_ee = translation(0.06231, -0.06216, -0.018)
    T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee
    return T_0_ee[:3, 3]


# --- Full kinematic chain (return all joint positions for visualization) ---


def fr_leg_chain(theta):
    T_0_1 = (
        translation(0.07500, -0.08350, 0) @ rotation_x(1.57080) @ rotation_z(theta[0])
    )
    T_1_2 = rotation_y(-1.57080) @ rotation_z(theta[1])
    T_2_3 = (
        translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(theta[2])
    )
    T_3_ee = translation(0.06231, -0.06216, 0.01800)
    return np.array(
        [
            get_position(T_0_1),
            get_position(T_0_1 @ T_1_2),
            get_position(T_0_1 @ T_1_2 @ T_2_3),
            get_position(T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee),
        ]
    )


def fl_leg_chain(theta):
    T_0_1 = (
        translation(0.07500, 0.0445, 0) @ rotation_x(1.57080) @ rotation_z(-theta[0])
    )
    T_1_2 = translation(0, 0, -0.039) @ rotation_y(-1.57080) @ rotation_z(theta[1])
    T_2_3 = (
        translation(0, -0.0494, 0.0685) @ rotation_y(1.57080) @ rotation_z(-theta[2])
    )
    T_3_ee = translation(0.06231, -0.06216, -0.018)
    return np.array(
        [
            get_position(T_0_1),
            get_position(T_0_1 @ T_1_2),
            get_position(T_0_1 @ T_1_2 @ T_2_3),
            get_position(T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee),
        ]
    )


def br_leg_chain(theta):
    T_0_1 = (
        translation(-0.07500, -0.0335, 0) @ rotation_x(1.57080) @ rotation_z(theta[0])
    )
    T_1_2 = translation(0, 0, 0.039) @ rotation_y(-1.57080) @ rotation_z(theta[1])
    T_2_3 = translation(0, -0.0494, 0.0685) @ rotation_y(1.57080) @ rotation_z(theta[2])
    T_3_ee = translation(0.06231, -0.06216, 0.018)
    return np.array(
        [
            get_position(T_0_1),
            get_position(T_0_1 @ T_1_2),
            get_position(T_0_1 @ T_1_2 @ T_2_3),
            get_position(T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee),
        ]
    )


def bl_leg_chain(theta):
    T_0_1 = (
        translation(-0.07500, 0.0445, 0) @ rotation_x(1.57080) @ rotation_z(-theta[0])
    )
    T_1_2 = translation(0, 0, -0.039) @ rotation_y(-1.57080) @ rotation_z(theta[1])
    T_2_3 = (
        translation(0, -0.0494, 0.0685) @ rotation_y(1.57080) @ rotation_z(-theta[2])
    )
    T_3_ee = translation(0.06231, -0.06216, -0.018)
    return np.array(
        [
            get_position(T_0_1),
            get_position(T_0_1 @ T_1_2),
            get_position(T_0_1 @ T_1_2 @ T_2_3),
            get_position(T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee),
        ]
    )


# --- IK solver (from lab_4.py) ---


def inverse_kinematics_single_leg(target_ee, fk_fn, initial_guess=None):
    """L-BFGS IK solver matching lab_4.py implementation."""
    if initial_guess is None:
        initial_guess = [0.0, 0.0, 0.0]

    target = np.array(target_ee)

    def objective(theta):
        diff = fk_fn(theta) - target
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
    tol = 1e-10
    m = 5

    s_hist = []
    y_hist = []
    rho_hist = []

    for _ in range(max_iterations):
        cost = objective(theta)
        if cost < tol:
            return theta

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

        if direction.dot(grad) >= 0:
            direction = -grad
            s_hist.clear()
            y_hist.clear()
            rho_hist.clear()

        alpha = 1.0
        dg = grad.dot(direction)
        for _ in range(15):
            if objective(theta + alpha * direction) <= cost + 1e-4 * alpha * dg:
                break
            alpha *= 0.5

        s = alpha * direction
        theta_new = theta + s
        grad_new = gradient(theta_new)
        y = grad_new - grad

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


# --- Triangle gait waypoints (from lab_4.py) ---


def get_triangle_positions():
    """Return the triangle gait waypoints for all 4 legs."""
    touch_down_position = np.array([0.05, 0.0, -0.14])
    stand_position_1 = np.array([0.025, 0.0, -0.14])
    stand_position_2 = np.array([0.0, 0.0, -0.14])
    stand_position_3 = np.array([-0.025, 0.0, -0.14])
    liftoff_position = np.array([-0.05, 0.0, -0.14])
    mid_swing_position = np.array([0.0, 0.0, -0.05])

    rf_ee_offset = np.array([0.06, -0.09, 0])
    rf_positions = (
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
    lf_positions = (
        np.array(
            [
                stand_position_3,
                liftoff_position,
                mid_swing_position,
                touch_down_position,
                stand_position_1,
                stand_position_2,
            ]
        )
        + lf_ee_offset
    )

    rb_ee_offset = np.array([-0.11, -0.09, 0])
    rb_positions = (
        np.array(
            [
                stand_position_3,
                liftoff_position,
                mid_swing_position,
                touch_down_position,
                stand_position_1,
                stand_position_2,
            ]
        )
        + rb_ee_offset
    )

    lb_ee_offset = np.array([-0.11, 0.09, 0])
    lb_positions = (
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

    return [rf_positions, lf_positions, rb_positions, lb_positions]


def interpolate_triangle(positions, t):
    """Replicate the exact interpolate_triangle logic from lab_4.py (fixed version)."""
    n_positions = len(positions)
    position_idx = t * n_positions
    idx_prev = math.floor(position_idx)
    idx_next = math.ceil(position_idx)
    idx_next = idx_next % n_positions
    return positions[idx_prev] + (position_idx - idx_prev) * (
        positions[idx_next] - positions[idx_prev]
    )


def main():
    fk_fns = [fr_leg_fk, fl_leg_fk, br_leg_fk, bl_leg_fk]
    chain_fns = [fr_leg_chain, fl_leg_chain, br_leg_chain, bl_leg_chain]
    leg_names = ["Front-Right", "Front-Left", "Back-Right", "Back-Left"]
    hip_x_offsets = [0.075, 0.075, -0.075, -0.075]

    # Initial joint positions from real robot log (real_run_0.log)
    initial_joint_positions = np.array(
        [0.011, -0.013, 0.673, 0.046, 0.065, -0.683, 0.023, -0.042, 0.693, -0.016, 0.008, -0.664]
    )

    # Get triangle waypoints and use the exact same interpolation as lab_4.py
    triangle_positions = get_triangle_positions()
    t_values = np.arange(0, 1, 0.02)  # Same as lab_4.py: 50 steps

    # Compute gait target angles via IK
    print("Computing IK for gait phase...")
    gait_angles = np.zeros((len(t_values), 4, 3))
    for leg_idx in range(4):
        print(f"  Leg {leg_idx} ({leg_names[leg_idx]})...")
        prev_theta = [0.0, 0.0, 0.0]
        for frame in range(len(t_values)):
            target_ee = interpolate_triangle(triangle_positions[leg_idx], t_values[frame])
            theta = inverse_kinematics_single_leg(
                target_ee, fk_fns[leg_idx], initial_guess=prev_theta
            )
            gait_angles[frame, leg_idx] = theta
            prev_theta = theta.tolist()

    # --- Build full sequence: standup + gait ---
    standup_steps = 40  # 40 frames for standup (matches 2s at 100Hz scaled to animation)
    n_gait_frames = len(t_values)
    n_total_frames = standup_steps + n_gait_frames
    print(f"Standup frames: {standup_steps}, Gait frames: {n_gait_frames}, Total: {n_total_frames}")

    # Compute all angles for the full sequence
    all_angles = np.zeros((n_total_frames, 4, 3))

    # Standup phase: cubic interpolation from rest to first gait target
    first_gait_angles = gait_angles[0]  # (4, 3)
    for frame in range(standup_steps):
        alpha = frame / standup_steps
        # Smooth cubic ease in/out (same as lab_4.py)
        alpha = 3 * alpha**2 - 2 * alpha**3
        for leg_idx in range(4):
            rest_angles = initial_joint_positions[leg_idx * 3 : leg_idx * 3 + 3]
            all_angles[frame, leg_idx] = (
                rest_angles * (1 - alpha) + first_gait_angles[leg_idx] * alpha
            )

    # Gait phase: use precomputed IK angles
    all_angles[standup_steps:] = gait_angles

    # Pre-compute all chain positions using the motor angles
    print("Computing FK chains...")
    all_positions = np.zeros((n_total_frames, 4, 4, 3))
    for frame in range(n_total_frames):
        for leg_idx in range(4):
            all_positions[frame, leg_idx] = chain_fns[leg_idx](
                all_angles[frame, leg_idx]
            )

    # Phase labels for each frame
    phase_labels = []
    for frame in range(n_total_frames):
        if frame < standup_steps:
            phase_labels.append(f"STANDUP {frame + 1}/{standup_steps}")
        else:
            gait_frame = frame - standup_steps
            phase_labels.append(f"GAIT {gait_frame + 1}/{n_gait_frames}")

    # Compute axis limits per leg (over full sequence)
    leg_xlims = []
    leg_zlims = []
    for leg_idx in range(4):
        xs = all_positions[:, leg_idx, :, 0]
        zs = all_positions[:, leg_idx, :, 2]
        margin = 0.02
        leg_xlims.append((xs.min() - margin, xs.max() + margin))
        leg_zlims.append((zs.min() - margin, zs.max() + margin))

    # Set up figure
    fig, axes_arr = plt.subplots(2, 2, figsize=(12, 10))
    axes_arr = axes_arr.flatten()

    # Draw the target triangle waypoints (static) and initialize animated elements
    linkage_lines = []
    joint_markers = []
    ee_dot = []
    ee_trail_lines = []
    ee_trails_x = [[] for _ in range(4)]
    ee_trails_z = [[] for _ in range(4)]
    frame_texts = []

    for leg_idx in range(4):
        ax = axes_arr[leg_idx]
        ax.set_xlim(leg_xlims[leg_idx])
        ax.set_ylim(leg_zlims[leg_idx])
        ax.set_aspect("equal")
        ax.set_title(f"{leg_names[leg_idx]} Leg (X-Z Plane)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)

        # Plot the target triangle path (static reference)
        wp = triangle_positions[leg_idx]
        tri_x = list(wp[:, 0]) + [wp[0, 0]]
        tri_z = list(wp[:, 2]) + [wp[0, 2]]
        ax.plot(tri_x, tri_z, "g--", linewidth=1.0, alpha=0.6, label="Target triangle")
        ax.plot(wp[:, 0], wp[:, 2], "g.", markersize=6, alpha=0.6)

        # Plot initial rest position (static reference)
        rest_theta = initial_joint_positions[leg_idx * 3 : leg_idx * 3 + 3]
        rest_chain = chain_fns[leg_idx](rest_theta)
        ax.plot(
            rest_chain[:, 0], rest_chain[:, 2], "m--",
            linewidth=1.0, alpha=0.5, label="Rest position"
        )
        ax.plot(rest_chain[-1, 0], rest_chain[-1, 2], "m^", markersize=7, alpha=0.5)

        # Hip origin marker
        ax.plot(
            hip_x_offsets[leg_idx], 0, "ks", markersize=10, zorder=5, label="Hip origin"
        )

        (line,) = ax.plot([], [], "b-", linewidth=2.5)
        (markers,) = ax.plot([], [], "ko", markersize=7, zorder=4)
        (ee,) = ax.plot([], [], "ro", markersize=9, zorder=6)
        (trail,) = ax.plot([], [], "r-", linewidth=1.2, alpha=0.6)

        linkage_lines.append(line)
        joint_markers.append(markers)
        ee_dot.append(ee)
        ee_trail_lines.append(trail)

        txt = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        frame_texts.append(txt)
        ax.legend(loc="upper right", fontsize=7)

    plt.suptitle(
        "Quadruped Simulation: Standup + Trotting Gait (2D X-Z View)",
        fontsize=13,
    )
    plt.tight_layout()

    def init():
        for leg_idx in range(4):
            linkage_lines[leg_idx].set_data([], [])
            joint_markers[leg_idx].set_data([], [])
            ee_dot[leg_idx].set_data([], [])
            ee_trail_lines[leg_idx].set_data([], [])
            frame_texts[leg_idx].set_text("")
            ee_trails_x[leg_idx].clear()
            ee_trails_z[leg_idx].clear()
        return linkage_lines + joint_markers + ee_dot + ee_trail_lines + frame_texts

    def update(frame):
        for leg_idx in range(4):
            positions = all_positions[frame, leg_idx]
            xs = positions[:, 0]
            zs = positions[:, 2]

            linkage_lines[leg_idx].set_data(xs, zs)
            joint_markers[leg_idx].set_data(xs, zs)
            ee_dot[leg_idx].set_data([xs[-1]], [zs[-1]])

            # Accumulate end-effector trail
            ee_trails_x[leg_idx].append(xs[-1])
            ee_trails_z[leg_idx].append(zs[-1])
            ee_trail_lines[leg_idx].set_data(ee_trails_x[leg_idx], ee_trails_z[leg_idx])

            # Show motor angles, frame counter, and phase
            theta = all_angles[frame, leg_idx]
            frame_texts[leg_idx].set_text(
                f"{phase_labels[frame]}\n"
                f"\u03b81={theta[0]:+.3f} \u03b82={theta[1]:+.3f} \u03b83={theta[2]:+.3f}"
            )

        return linkage_lines + joint_markers + ee_dot + ee_trail_lines + frame_texts

    n_frames = n_total_frames
    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        blit=True,
        interval=80,
        repeat=True,
    )

    # Save as MP4
    output_path = "leg_simulation.mp4"
    anim.save(output_path, writer="ffmpeg", fps=15, dpi=120)
    print(f"Saved animation to {output_path}")

    # Save sample frames for verification
    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)
    for sf in range(n_frames):
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
        axes2 = axes2.flatten()
        for leg_idx in range(4):
            ax = axes2[leg_idx]
            ax.set_xlim(leg_xlims[leg_idx])
            ax.set_ylim(leg_zlims[leg_idx])
            ax.set_aspect("equal")
            ax.set_title(f"{leg_names[leg_idx]} Leg (X-Z Plane)")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)

            # Target triangle
            wp = triangle_positions[leg_idx]
            tri_x = list(wp[:, 0]) + [wp[0, 0]]
            tri_z = list(wp[:, 2]) + [wp[0, 2]]
            ax.plot(tri_x, tri_z, "g--", linewidth=1.0, alpha=0.6)
            ax.plot(wp[:, 0], wp[:, 2], "g.", markersize=6, alpha=0.6)

            # Rest position reference
            rest_theta = initial_joint_positions[leg_idx * 3 : leg_idx * 3 + 3]
            rest_chain = chain_fns[leg_idx](rest_theta)
            ax.plot(
                rest_chain[:, 0], rest_chain[:, 2], "m--",
                linewidth=1.0, alpha=0.5
            )
            ax.plot(rest_chain[-1, 0], rest_chain[-1, 2], "m^", markersize=7, alpha=0.5)

            # Hip
            ax.plot(hip_x_offsets[leg_idx], 0, "ks", markersize=10, zorder=5)

            # Linkage at this frame
            positions = all_positions[sf, leg_idx]
            xs = positions[:, 0]
            zs = positions[:, 2]
            ax.plot(xs, zs, "b-", linewidth=2.5)
            ax.plot(xs, zs, "ko", markersize=7, zorder=4)
            ax.plot(xs[-1], zs[-1], "ro", markersize=9, zorder=6)

            # EE trail up to this frame
            trail_x = all_positions[: sf + 1, leg_idx, -1, 0]
            trail_z = all_positions[: sf + 1, leg_idx, -1, 2]
            ax.plot(trail_x, trail_z, "r-", linewidth=1.2, alpha=0.6)

            theta = all_angles[sf, leg_idx]
            ax.text(
                0.02,
                0.95,
                f"{phase_labels[sf]}\n"
                f"\u03b81={theta[0]:+.3f} \u03b82={theta[1]:+.3f} \u03b83={theta[2]:+.3f}",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.suptitle(
            "Quadruped Simulation: Standup + Trotting Gait (2D X-Z View)",
            fontsize=13,
        )
        plt.tight_layout()
        frame_path = os.path.join(frames_dir, f"frame_{sf:03d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved frame {sf} to {frame_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
