# Open problems

Our system attempts to infer some sane default transforms and parameters when the user hasn't specified anything. If the user has manually specified something, our system should prefer to use those user specified values instead of the inferred values. What is the best way to accomplish this?

Add handles.

Kusuduama cone radius is bugged.

Overlapping limit cones don't visualize.

1. for limit cone directions: intersect camera mouse ray with the kusudama sphere and pick the closest intersection.
2. for limit cone radius, same as direction, except take the arccosine of the dot product with the direction to get the radius
3. for axial angle: intersect with xz plane, take arcosine of dot product with the z axis

Mirror button for kusudama cones.

Make changing the skeleton not lose the set parameters. Don't rebuild the skeleton from scratch. Reuse the the parameters already set.

Leave stabilizing pass counts off when developing. Add a ui for it.

We don't like painfullness but it seems to work in the reference implementation.
