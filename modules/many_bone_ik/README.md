# Many Bone IK

A custom inverse kinematics system solver for multi-chain skeletons and with constraints.

Please use Godot Engine fork at **https://github.com/V-Sekai/godot/tree/vsk-many-bone-ik-4.3**

Lyuma mentioned that there are multiple plausible solutions, especially for dance, where people move internal joints in unusual ways. However, for everything other than dance, there is typically just one right answer - the energy minimizing one.

Lyuma also agreed that energy minimization is a good concept, as it results in the optimal solution and aligns with what your mind would choose without specific intent.

Eron explained that with the latest ewb-ik version, energy minimization is handled by defining 0 cost regions in kusudama constraints along with hard boundary (high cost regions).

Joints always negotiate between moving toward their own 0 cost region and toward the goal, with the former being less important than the latter. The "goal" here refers to whatever orientation allows descendant joints to reach their target.
