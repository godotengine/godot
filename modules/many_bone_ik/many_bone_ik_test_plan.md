# many_bone_ik Test Plan Summary

The conversation revolves around the development and testing of an Inverse Kinematics (IK) solver. The user, `fire`, has developed a custom IK system solver for multi-chain skeletons with constraints and is facing robustness issues. They are considering using the `.obj` method for unit testing and have an export to blender blend method.

Here's the summarized plan:

1. **Test Method for IK Solvers:** The user wants to find a good test method for IK solvers. They are considering the `.obj` method of unit testing.

2. **Comparing Code Versions:** The user wants to compare different versions of their code, like current revision vs past revisions.

3. **Loss Calculation:** The user suggests calculating loss per vertex position. However, the vertices need to be in the same order for this to work.

4. **No Vertex Order Modification in IK:** The user mentions that they don't modify the vertex order in their IK implementation.

5. **Existing Tools:** `bqqbarbhg` mentions that there might already exist a tool for comparing `.obj` files, but they don't have any generic `.obj` diff tool. They suggest that parsing `v` and `f` lines should work fine as `.obj` is a simple enough format.

6. **Creating a Separate Tool:** If no existing tools meet the requirements, `bqqbarbhg` suggests creating a separate tool for comparing `.obj` files.

The GitHub link provided by the user leads to their project - a custom inverse kinematics system solver for multi-chain skeletons and with constraints.

```python
# Link to the user's project
"https://github.com/V-Sekai/many_bone_ik#"
```

This plan aims to improve the robustness of the IK solver and ensure its correctness across different code revisions.
