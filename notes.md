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
