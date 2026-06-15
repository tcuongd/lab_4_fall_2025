You might notice that the fr_leg_fk method pre-implemented in lab_4.py looks different than your implementation from lab 3. Are they functionally different? If so, why do we need to make these changes here? If not, how are they empirically the same?

An underactuated system that has more degrees of freedom that can be controlled, than the number of independently controlled actuators. How many degrees of freedom does Pupper have? Is it an underactuated system?

Why are under-actuated systems more challenging to control?

Think about why we are giving you six reference positions for each leg, instead of just three as in lab 3.

What are some other gaits that Pupper could exhibit, and why/when would they be useful? List 3 alternative gaits.

What are some potential setbacks that may prevent Pupper from exhibiting these gaits you listed above?


The controller implemented is a “heuristic” controller. That means it follows a pre-programmed trajectory, and doesn’t use online (real-time) sensor feedback outside the motor to optimize its trajectory. What are some potential pitfalls of this approach? How will Pupper react if you push it?

Many commercial quadrupeds once used model-based controllers that solve an optimization problem online (they all shift to reinforcement learning-based controllers now for locomotion). Why would it be challenging to deploy MBC/MPC on Pupper, which has a lower cost hardware and runs computation on a Raspberry Pi 5?


Adjust the ik_timer_period to find the best balance between performance and computational load.

As described in lecture, the center of mass of the robot influences how the robot can walk, whether forward or backward. Play around with the offset values in the ee_positions, and see how that affects performance.

Implement two gaits for Pupper. Make Pupper walk fast, and walk slow. Include videos of Pupper walking fast and walking slow with your submission to Gradescope

In your lab document, report on:

The effects of different trajectory shapes on the trotting gait

How timer periods affect the system’s performance

How does the center of mass affect performance?

Think about ways you can make Pupper walk/run even faster (you can change the timer frequencies, stride lengths, end-effector positions, etc to make Pupper even faster). HINT The positions defined after the init() function in the InverseKinematics class, define each of the stances.

Report on what you tried to make Pupper go faster. What worked and what didn’t?

The positions defined after the init() function in the InverseKinematics class, define each of the stances. Play around with these values and you can discover some new gaits!

The cache_target_joint_positions method pre-calculates joint positions for a full gait cycle. Understand how this affects the system’s performance.


Claude suggestions:

3. Add joint limit clamping in the IK solver to prevent solutions outside the servo range (typically ±1.5 rad for hobby
servos).
4. Increase IK iterations or switch to a better solver (e.g., damped least-squares / Levenberg-Marquardt) for more reliable
convergence at workspace edges.
5. Add velocity smoothing — clamp the maximum angle change per timestep, or filter the IK output with a low-pass filter before
sending to the PD controller.

## Analysis

Analysis: real_run_1.log vs Simulation

Standup Phase — Worked Well

The standup phase (200 steps, 2 seconds) performed as designed:
- Started from rest: [0.029, -0.013, 0.612, -0.009, 0.02, -0.604, ...] (knees at ~0.61 rad)
- Final tracking error at step 200: 0.026 rad (1.5 deg) — the robot successfully stood up
- The cubic interpolation gave the PD controller enough time to track

Gait Phase — Significant Tracking Error
┌───────────────────────────────────────────┬──────────────────────┐
│                  Metric                   │        Value         │
├───────────────────────────────────────────┼──────────────────────┤
│ Mean tracking error                       │ 0.188 rad (10.8 deg) │
├───────────────────────────────────────────┼──────────────────────┤
│ Max tracking error                        │ 1.291 rad (74.0 deg) │
├───────────────────────────────────────────┼──────────────────────┤
│ Steps with all joints < 0.1 rad of target │ 0%                   │
├───────────────────────────────────────────┼──────────────────────┤
│ Gait cycles completed                     │ ~17.8                │
└───────────────────────────────────────────┴──────────────────────┘
The error is stable over time (doesn't diverge), so no backflip — but the robot never actually tracks the commanded trajectory
closely.

Root Causes of Difference from Simulation

1. Lag (~60ms / 6 control steps)
The robot's current position at time T best matches the target from ~60ms earlier. The simulation assumes instantaneous tracking (target = actual), but real motors have inertia, communication delay, and PD bandwidth limits.

2. The gait advances open-loop regardless of tracking
The counter increments every 10ms no matter how far behind the joints are. By the time the robot reaches one target, the controller is already commanding a position 6 steps ahead. This is the fundamental mismatch — the sim shows "ideal" motion, but on the real robot the legs are always chasing a target they can never catch.

3. High commanded joint velocities
The IK-solved trajectory demands up to 37.6 rad/s (2154 deg/s) peak joint velocity. The real motors can't change that fast, creating the persistent lag. In contrast, the simulation implicitly assumes infinite actuator bandwidth.

4. Per-leg breakdown
All legs have similar errors (~0.18-0.20 rad mean), indicating this is a systemic timing issue, not a problem with a specific leg's kinematics.

Recommendations

1. Slow down the gait — reduce t_values resolution or increase the gait period so commanded velocity stays within motor
bandwidth
2. Add feedforward or tracking-aware counter — don't advance the gait counter until the tracking error drops below a threshold
3. Increase PD gains (if motor torque allows) — tighter tracking of the commanded trajectory
4. Reduce the swing amplitude — smaller mid_swing_position z-height reduces the extreme velocity demands during
liftoff/touchdown transitions

Here's the visual summary of the findings:

Key Plots

1. Joint Tracking (diagnosis_1_tracking.png): The dashed red lines (target) and solid blue lines (current) show the standup is
smooth and well-tracked, but once gait starts, the current consistently lags behind the target — the blue curve is a delayed,
amplitude-reduced copy of the red.

2. Error & Velocity (diagnosis_1_errors.png):
- Top: Peak joint error spikes to 60-70 deg during swing phases, drops near zero during stance
- Middle: EE position error peaks at 60-70mm during swing
- Bottom: Joint velocities spike to 1500-2000 deg/s — this exceeds what the motors can physically track

3. Lag Analysis (diagnosis_1_lag.png):
- Top: FR knee target vs actual clearly shows the current is a time-delayed version of the target
- Bottom: Optimal lag is 6 steps (60ms) — shifting the current signal back by 60ms reduces mean error from 10.6 deg to just
2.8 deg. This proves the robot is following the trajectory, just 60ms late.

4. Sim vs Real EE Trajectory (diagnosis_1_sim_vs_real.png): The green triangle is what the simulation shows (perfect
tracking). The red traces are what the robot actually does — a rounded, smaller, lagging version of the triangle. The swing
phase never reaches full height, and the sharp corners of the triangle are smoothed by the motor dynamics.

Summary
┌──────────────────────┬───────────────┬───────────────────────────────────┐
│                      │  Simulation   │            Real Robot             │
├──────────────────────┼───────────────┼───────────────────────────────────┤
│ Tracking delay       │ 0 ms          │ ~60 ms                            │
├──────────────────────┼───────────────┼───────────────────────────────────┤
│ Mean joint error     │ 0 deg         │ 10.6 deg                          │
├──────────────────────┼───────────────┼───────────────────────────────────┤
│ Swing height reached │ 100%          │ ~70%                              │
├──────────────────────┼───────────────┼───────────────────────────────────┤
│ Triangle shape       │ Sharp corners │ Rounded/blurred                   │
├──────────────────────┼───────────────┼───────────────────────────────────┤
│ Standup              │ Instant       │ Worked well (1.5 deg final error) │
└──────────────────────┴───────────────┴───────────────────────────────────┘
The standup fix worked perfectly. The remaining issue is purely that the gait is too fast for the motor bandwidth — the
trajectory commands change faster than the PD controller + motor dynamics can follow.
