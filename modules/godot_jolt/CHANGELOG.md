# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Breaking changes are denoted with ⚠️.

## [Unreleased]

### Changed

- ⚠️ Changed the way that the `body_test_motion` method of `PhysicsServer3D` discards contacts,
  which should fix issues related to jitter/ping-ponging. Note that this can also result in *more*
  ghost collisions under certain conditions. This affects `move_and_slide`, `move_and_collide` and
  `test_move`.
- ⚠️ Changed the inertia of shapeless bodies to be `(1, 1, 1)`, to match Godot Physics.
- Changed `SeparationRayShape3D` to not treat other convex shapes as solid, meaning it will now only
  ever collide with the hull of other convex shapes, which better matches Godot Physics.
- Mirrored the way in which angular limits are visualized for `JoltHingeJoint3D`,
  `JoltConeTwistJoint3D` and `JoltGeneric6DOFJoint`.

### Added

- Added support for `SoftBody3D`.
- Added support for double-precision.
- ⚠️ Added new project setting, "Use Enhanced Internal Edge Detection", which can help alleviate
  collisions with internal edges of `ConcavePolygonShape3D` and `HeightMapShape3D` shapes, also
  known as ghost collisions. This setting is enabled by default and may change the behavior of
  character controllers relying things like `move_and_slide`.
- Added support for partial custom inertia, where leaving one or two components at zero will use the
  automatically calculated values for those specific components.
- Added error-handling for invalid scaling of bodies/shapes.
- Added project settings "Body Pair Cache Enabled", "Body Pair Cache Distance Threshold" and "Body
  Pair Cache Angle Threshold" to allow fine-tuning the scale by which collision results are reused
  inbetween physics ticks.

### Fixed

- ⚠️ Fixed issue with shape queries not returning the full contact manifold. This applies to the
  `collide_shape` method of `PhysicsDirectSpaceState3D` as well as the `body_test_motion` method of
  `PhysicsServer3D`, which subsequently affects the `test_move` and `move_and_collide` methods of
  `PhysicsBody3D` as well as the `move_and_slide` method of `CharacterBody3D`.
- ⚠️ Fixed issue with the `body_get_direct_state` method of `PhysicsServer3D` returning a non-null
  reference when the body has no space.
- Fixed issue with not being able to pass a physics space `RID` to `area_get_param`,
  `area_attach_object_instance_id` and `area_get_object_instance_id`.
- Fixed issue where the `inverse_inertia` property of `PhysicsDirectBodyState3D` would have some of
  its components swapped.
- Fixed issue where shapeless bodies wouldn't have custom center-of-mass applied to them.
- Fixed issue with high (>1) damping values producing different results across different update
  frequencies.
- Fixed issue where physics queries performed within the editor (e.g. editor plugins or tool
  scripts) would end up ignoring potentially large swathes of bodies in scenes with many physics
  bodies.

## [0.12.0] - 2024-01-07

### Changed

- ⚠️ Changed so that single-body joints now implicitly sets `node_a` to be the "world node" rather
  than `node_b`. This diverges from how Godot Physics behaves, but matches how Bullet behaves in
  Godot 3, and yields more intuitive outcomes for the 6DOF joints.
- ⚠️ Changed `Generic6DOFJoint3D` and `ConeTwistJointImpl3D`, as well as their substitute joints, to
  use pyramid-shaped angular limits instead of cone-shaped limits, to better match Godot Physics.
- ⚠️ Reversed the direction of the `equilibrium_point` properties for `Generic6DOFJoint3D` and
  `JoltGeneric6DOFJoint3D`, to match the direction of the angular limits.
- ⚠️ Changed the rotation order of the `equilibrium_point` properties for `Generic6DOFJoint3D` and
  `JoltGeneric6DOFJoint3D`, from ZXY to XYZ, to match the rotation order of the angular limits.
- Mirrored the way in which linear limits are visualized for `JoltSliderJoint3D` and
  `JoltGeneric6DOFJoint3D`.

### Added

- Added new project setting, "World Node", for controlling which of the two nodes in a single-body
  joint becomes the "world node" when omitting one of the nodes. This allows for reverting back to
  the behavior of Godot Physics if needed, effectively undoing the breaking change mentioned above.
- Added new project setting, "Report All Kinematic Contacts", for allowing `RigidBody3D` frozen with
  `FREEZE_MODE_KINEMATIC` to report contacts/collisions with other kinematic/static bodies, at a
  potentially heavy performance/memory cost.
- Added support for using NaN to indicate holes in `HeightMapShape3D`.
- Added support for holes in a non-square `HeightMapShape3D`.

### Fixed

- ⚠️ Fixed issue with non-square `HeightMapShape3D` not using back-face collision.
- Fixed issue where contact shape indices would sometimes always be the same index across all
  contacts with a particular body.
- Fixed runtime crash when setting the `max_contacts_reported` property to a lower value.
- Fixed issue where `Generic6DOFJoint3D` and `JoltGeneric6DOFJoint3D` would yield odd limit shapes
  when using both linear and angular asymmetrical limits.
- Fixed issue where the equilibrium point for `Generic6DOFJoint3D` and `JoltGeneric6DOFJoint3D`
  would be moved when using asymmetrical limits.
- Fixed crash that could occur under rare circumstances when shutting down the editor after having
  added/removed collision shapes.
- Fixed issue where a `RigidBody3D` with locked axes colliding with a `StaticBody3D` (or another
  frozen `RigidBody3D` using `FREEZE_MODE_STATIC`) would result in NaNs.
- Fixed issue where `HingeJoint3D` and `JoltHingeJoint3D` would sometimes dull forces applied to
  either of its bodies when at either of its limits.
- Fixed issue with iOS `Info.plist` missing the `MinimumOSVersion` key.

## [0.11.0] - 2023-12-01

### Changed

- ⚠️ Changed `HeightMapShape3D` to always use back-face collision, to match Godot Physics.

### Fixed

- Fixed issue with project randomly freezing up when having many active physics spaces.
- Fixed issue with static and kinematic bodies not correctly incorporating surface velocities, also
  known as "constant velocities", as part of their reported velocities. This also makes it so
  `move_and_slide` will respect such velocities.
- Fixed issue with global transform not being preserved when reparenting a `RigidBody3D`.
- Fixed issue where the callback passed to `body_set_force_integration_callback` could be called
  even when the body is sleeping.

## [0.10.0] - 2023-11-12

### Changed

- ⚠️ Changed `gravity_point_unit_distance` for `Area3D` to result in a constant gravity when set to
  zero, rather than resulting in a zero gravity, to match Godot Physics.
- ⚠️ Changed so that ray-casts using the `hit_from_inside` parameter report a zero normal when
  hitting a convex shape from inside, to match Godot Physics.
- Changed the `space_get_contacts` method of `PhysicsServer3D` (and thus also the "Visible Collision
  Shapes" debug rendering) to no longer include contacts generated by overlaps with `Area3D`.

### Added

- Added support for Android (ARM64, ARM32, x86-64 and x86).
- Added support for iOS.

### Fixed

- Fixed issue where an error saying `Parameter "body" is null` would be emitted after freeing
  certain bodies while they were in contact with a `CharacterBody3D`.
- Fixed issue where a `RigidBody3D` could sometimes still be moved by another `RigidBody3D` despite
  the first body not having the second body in its collision mask.

## [0.9.0] - 2023-10-12

### Changed

- Changed `ConvexPolygonShape3D` to no longer emit errors about failing to build the shape when
  adding one to the scene tree with 0 points.

### Added

- Added new project setting, "Active Edge Threshold", for tuning the cut-off angle for Jolt's active
  edge detection, which can help balance trade-offs related to triangle edge collisions.

### Fixed

- ⚠️ Fixed issue where `Generic6DOFJoint` and `JoltGeneric6DOFJoint` would lock up any axis that
  used a spring stiffness/frequency of 0.
- Greatly reduced creation/modification/loading times for `ConcavePolygonShape3D`.

## [0.8.0] - 2023-09-28

### Changed

- ⚠️ Changed `apply_force` and `apply_impulse` to be applied at an offset relative to the body's
  origin rather than at an offset relative to the body's center-of-mass, to match Godot Physics.
- ⚠️ Changed collision layers and masks for `Area3D` to behave like they do in Godot Physics,
  allowing for asymmetrical setups, where overlaps are only reported if the mask of an `Area3D`
  contains the layer of the overlapping object.
- ⚠️ Changed the `body_set_force_integration_callback` method of `PhysicsServer3D` to behave like it
  does with Godot Physics, where omitting the binding of `userdata` requires that the callback also
  doesn't take any `userdata`. It also will no longer be called when the body is sleeping.

### Added

- Added timings of Jolt's various jobs to the "Physics 3D" profiler category.
- Added registering of `JoltPhysicsServer3D` as an actual singleton, which makes Jolt-specific
  server methods (e.g. `pin_joint_get_applied_force`) easier to deal with from dynamic scripting
  languages like GDScript.
- Added `space_dump_debug_snapshot` to `JoltPhysicsServer3D`, for dumping a binary debug snapshot of
  a particular physics space.
- Added `dump_debug_snapshots` to `JoltPhysicsServer3D`, for dumping binary debug snapshots of all
  currently active physics spaces.
- Added a "Dump Debug Snapshots" menu option to "Project / Tools / Jolt Physics", for dumping binary
  debug snapshots of all the editor's physics spaces.

### Fixed

- Fixed issue with `move_and_slide`, where under certain conditions you could get stuck on internal
  edges of a `ConcavePolygonShape3D` if the floor was within 5-ish degrees of `floor_max_angle`.
- Fixed issue with `move_and_slide`, where under certain conditions, while using a `BoxShape3D` or
  `CylinderShape3D` shape, you could get stuck on internal edges of a `ConcavePolygonShape3D`.
- Fixed issue where collision with `ConvexPolygonShape3D` could yield a flipped contact normal.
- Fixed issue where an `Area3D` with `monitoring` disabled wouldn't emit any entered events for
  already overlapping bodies once `monitoring` was enabled.
- Fixed issue where changing the center-of-mass of a `RigidBody3D` attached to a joint would shift
  its transform relative to the joint.
- Fixed issue where the `total_gravity` property on `PhysicsDirectBodyState3D` would always return a
  zero vector for kinematic bodies.
- Fixed issue with `Area3D` detecting overlaps slightly outside of its collision shapes.

## [0.7.0] - 2023-08-29

### Removed

- ⚠️ Disabled the `JoltDebugGeometry3D` node for all distributed builds. If you still need it, build
  from source using the `*-development` or `*-debug` configurations.

### Changed

- ⚠️ Ray-casts will no longer hit the back-faces of `ConcavePolygonShape3D` if its `hit_back_faces`
  parameter is set to `false`, regardless of what the `backface_collision` property of the
  `ConcavePolygonShape3D` is set to.
- ⚠️ Changed the triangulation of `HeightMapShape3D` to match Godot Physics.

### Fixed

- ⚠️ Fixed regression where a motored `HingeJoint3D` (or `JoltHingeJoint3D`) would rotate
  counter-clockwise instead of clockwise.
- Fixed issue where ray-casting a `ConcavePolygonShape3D` that had `backface_collision` enabled, you
  would sometimes end up with a flipped normal.
- Fixed issue where a `CharacterBody3D` using `move_and_slide` could sometimes get stuck when
  sliding along a wall.
- Fixed issue where attaching a `RigidBody3D` with locked axes to a joint could result in NaN
  velocities/position and subsequently a lot of random errors being emitted from within Godot.

## [0.6.0] - 2023-08-17

### Changed

- Changed the editor gizmo for `JoltPinJoint3D`.
- Changed the editor gizmo for `JoltHingeJoint3D`.
- Changed the editor gizmo for `JoltSliderJoint3D`.
- Changed the editor gizmo for `JoltConeTwistJoint3D`.
- Changed the editor gizmo for `JoltGeneric6DOFJoint3D`.

### Added

- Added support for `HeightMapShape3D` with non-power-of-two dimensions.
- Added support for `HeightMapShape3D` with non-square dimensions.
- Added support for `HeightMapShape3D` with no heights.

### Fixed

- Fixed issue where bodies would catch on internal edges of `ConcavePolygonShape3D`.

## [0.5.0] - 2023-08-08

### Removed

- ⚠️ Removed the ability to lock all six axes of a `RigidBody3D`. Consider freezing the body as
  static instead.

### Added

- Added substitutes for all the joint nodes, to better align with the interface that Jolt offers,
  which consist of `JoltPinJoint3D`, `JoltHingeJoint3D`, `JoltSliderJoint3D`, `JoltConeTwistJoint3D`
  and `JoltGeneric6DOFJoint3D`. These differ in the following ways:
  - You can enable/disable the limits on all joints.
  - You can enable/disable the joint itself using its `enabled` property.
  - You can fetch the magnitude of the force/torque that was last applied to keep the joint
    together, using the `get_applied_force` and `get_applied_torque` methods. These coupled with the
    `enabled` property allows for creating breakable joints.
  - You can increase the joint's solver iterations, to improve stability, using its
    `solver_velocity_iterations` and `solver_position_iterations` properties.
  - Springs use frequency and damping instead of stiffness and damping.
  - Soft limits are achieved with limit springs.
  - `JoltConeTwistJoint3D` can be configured with a motor.
  - Angular motor velocities are set in radians per second, but displayed in degrees per second.
  - Any motion parameters like bias, damping and relaxation are omitted.
  - Any angular motion parameters for the slider joint are omitted.

### Fixed

- Fixed issue where linear axis locks could be budged a bit if enough force was applied.
- Fixed issue where `CharacterBody3D` and other kinematic bodies wouldn't respect locked axes.
- Fixed issue where passing `null` to the `result` parameter (or omitting it entirely) of the
  `body_test_motion` method in `PhysicsServer3D` would cause a crash.
- Fixed issue where the `body_is_omitting_force_integration` method in `PhysicsServer3D` would
  always return `false`.

## [0.4.1] - 2023-07-08

### Fixed

- Fixed issue where colliding with certain types of degenerate triangles in `ConcavePolygonShape3D`
  would cause the application to hang or emit a vast amount of errors.

## [0.4.0] - 2023-07-08

### Changed

- ⚠️ Changed the `cast_motion` method in `PhysicsDirectSpaceState3D` to return `[1.0, 1.0]` instead
  of `[]` when no collision was detected, to match Godot Physics.
- ⚠️ Changed contact positions to be absolute global positions instead of relative global positions,
  to match the new behavior in Godot Physics.

### Added

- Added support for springs in `Generic6DOFJoint3D`.

### Fixed

- Fixed issue where angular surface velocities (like `constant_angular_velocity` on `StaticBody3D`)
  wouldn't be applied as expected if the imparted upon body was placed across the imparting body's
  center of mass.
- Fixed issue where going from `CENTER_OF_MASS_MODE_CUSTOM` to `CENTER_OF_MASS_MODE_AUTO` wouldn't
  actually reset the body's center-of-mass.
- Fixed issue where any usage of `PhysicsServer3D`, `PhysicsDirectBodyState3D` or
  `PhysicsDirectSpaceState3D` in C# scripts would trigger an exception.
- Fixed issue where the `recovery_as_collision` parameter in the `move_and_collide` and `test_move`
  methods on bodies would always be `true`.
- Fixed issue where the `input_ray_pickable` property on bodies and areas would always be `true`.

## [0.3.0] - 2023-06-28

### Changed

- ⚠️ Changed collision layers and masks to behave as they do in Godot Physics, allowing for
  asymmetrical collisions, where the body whose mask does not contain the layer of the other body
  effectively gets infinite mass and inertia in the context of that collision.

### Added

- Added new project setting, "Use Shape Margins", which when disabled leads to all shape margins
  being ignored and instead set to 0, at a slight performance cost.
- Added new project setting, "Areas Detect Static Bodies", to allow `Area3D` to detect overlaps with
  static bodies (including `RigidBody3D` using `FREEZE_MODE_STATIC`) at a potentially heavy
  performance/memory cost.

### Fixed

- Fixed issue where a `RigidBody3D` using `FREEZE_MODE_KINEMATIC` wouldn't have its
  `_integrate_forces` method called when monitoring contacts.
- Fixed issue where scaling bodies/shapes with negative values would break them in various ways.
- Fixed issue where `CharacterBody3D` platform velocities would always be zero.
- Fixed issue where the velocity of kinematic colliders would always be zero in `_physics_process`.

## [0.2.3] - 2023-06-16

### Fixed

- Fixed issue where bodies would transform in unintuitive ways when attached to a rotated joint.
- Fixed issue where bodies would sometimes transform in unintuitive ways when attached to a
  `Generic6DOFJoint` that used both linear and angular limits.
- Fixed issue where setting the limits of a `SliderJoint3D` to the same value would make it free
  instead of fixed.
- Fixed issue where you could still rotate a `RigidBody3D` slightly while using `lock_rotation`.
- Fixed issue where friction would be applied more on one axis than the other.

## [0.2.2] - 2023-06-09

### Fixed

- Fixed issue where `AnimatableBody3D` would de-sync from its underlying body when moved.
- Fixed issue where `CharacterBody3D` and other kinematic bodies would sometimes maintain a velocity
  after having moved.

## [0.2.1] - 2023-06-06

### Fixed

- Fixed issue where having scaled bodies attached to a joint would result in the bodies being
  displaced from their starting position.

## [0.2.0] - 2023-06-06

### Changed

- ⚠️ Changed friction values to be combined in the same way as in Godot Physics.
- ⚠️ Changed bounce values to be combined in the same way as in Godot Physics.
- ⚠️ Changed the direction of `Generic6DOFJoint` angular motors to match Godot Physics.
- ⚠️ Changed how linear/angular velocities are applied to a frozen `RigidBody3D`, to better match
  Godot Physics. They now apply a surface velocity, also known as a "constant velocity", instead of
  actually moving the body.
- ⚠️ Changed shape margins to be interpreted as an upper bound. They are now scaled according to the
  shape's extents, which removes the ability to have an incorrect margin, thereby removing any
  warnings about that.
- Changed warning/error messages to provide more context, such as the names of any bodies
  related/connected to that particular thing.

### Added

- Added new project setting "Bounce Velocity Threshold".
- Added support for the `custom_integrator` property on `RigidBody3D` and `PhysicalBone3D`.
- Added support for the `integrate_forces` method on `PhysicsDirectBodyState3D`.
- Added support for the `rough` and `absorbent` properties on `PhysicsMaterial`.
- Added support for surface velocities, also known as "constant velocities", for both static and
  kinematic bodies.
- Added support for more flexible limits for `HingeJoint3D` and `SliderJoint3D`.

### Fixed

- Fixed issue where `CharacterBody3D` and other kinematic bodies wouldn't elicit a proper collision
  response from dynamic bodies.
- Fixed issue where setting friction on an already entered body would instead set bounce.
- Fixed issue where disabling or removing a body connected to a joint would error or crash.
- Fixed issue where `RigidBody3D` would de-sync from its underlying body after freezing with the
  "Static" freeze mode.
- Fixed issue where bodies connected to a `Generic6DOFJoint` would lose their relative pose when
  changing any of the joint's limits

## [0.1.0] - 2023-05-24

Initial release.

[Unreleased]: https://github.com/godot-jolt/godot-jolt/compare/v0.12.0-stable...HEAD
[0.12.0]: https://github.com/godot-jolt/godot-jolt/compare/v0.11.0-stable...v0.12.0-stable
[0.11.0]: https://github.com/godot-jolt/godot-jolt/compare/v0.10.0-stable...v0.11.0-stable
[0.10.0]: https://github.com/godot-jolt/godot-jolt/compare/v0.9.0-stable...v0.10.0-stable
[0.9.0]: https://github.com/godot-jolt/godot-jolt/compare/v0.8.0-stable...v0.9.0-stable
[0.8.0]: https://github.com/godot-jolt/godot-jolt/compare/v0.7.0-stable...v0.8.0-stable
[0.7.0]: https://github.com/godot-jolt/godot-jolt/compare/v0.6.0-stable...v0.7.0-stable
[0.6.0]: https://github.com/godot-jolt/godot-jolt/compare/v0.5.0-stable...v0.6.0-stable
[0.5.0]: https://github.com/godot-jolt/godot-jolt/compare/v0.4.1-stable...v0.5.0-stable
[0.4.1]: https://github.com/godot-jolt/godot-jolt/compare/v0.4.0-stable...v0.4.1-stable
[0.4.0]: https://github.com/godot-jolt/godot-jolt/compare/v0.3.0-stable...v0.4.0-stable
[0.3.0]: https://github.com/godot-jolt/godot-jolt/compare/v0.2.3-stable...v0.3.0-stable
[0.2.3]: https://github.com/godot-jolt/godot-jolt/compare/v0.2.2-stable...v0.2.3-stable
[0.2.2]: https://github.com/godot-jolt/godot-jolt/compare/v0.2.1-stable...v0.2.2-stable
[0.2.1]: https://github.com/godot-jolt/godot-jolt/compare/v0.2.0-stable...v0.2.1-stable
[0.2.0]: https://github.com/godot-jolt/godot-jolt/compare/v0.1.0-stable...v0.2.0-stable
[0.1.0]: https://github.com/godot-jolt/godot-jolt/releases/tag/v0.1.0-stable
