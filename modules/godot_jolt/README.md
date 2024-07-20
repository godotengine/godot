<p align="center">
  <a href="https://github.com/godot-jolt/godot-jolt">
    <img alt="Godot Jolt" src="docs/logo.svg">
  </a>
</p>

Godot Jolt is a native extension for the [Godot game engine][god] that allows you to use the [Jolt
physics engine][jlt] to power Godot's 3D physics.

It functions as a drop-in replacement for Godot Physics, by implementing the same nodes that you
would use normally, like `RigidBody3D` or `CharacterBody3D`.

## Table of Contents

- [What features are there?](#what-features-are-there)
- [What about determinism?](#what-about-determinism)
- [What's not supported?](#whats-not-supported)
- [What else is different?](#what-else-is-different)
- [What versions of Godot are supported?](#what-versions-of-godot-are-supported)
- [What platforms are supported?](#what-platforms-are-supported)
- [How do I get started?](#how-do-i-get-started)
- [What settings are there?](#what-settings-are-there)
- [How do I build from source?](#how-do-i-build-from-source)
- [What's the license?](#whats-the-license)

## What features are there?

Better performance, mainly, but also generally having a more stable simulation compared to Godot
Physics.

There are also (completely optional) substitute nodes available for all the joints, which line up
better with the interface that Jolt offers than what the default joints do. This allows for things
like breakable joints, soft limits and the ability to override solver iterations per-joint.

## What about determinism?

While Jolt itself offers deterministic simulations, Godot Jolt is not able to make such guarantees.
Simulations in Godot Jolt may look deterministic, and may even happen to be deterministic, but this
should not be relied upon if determinism is a hard requirement.

## What's not supported?

- `WorldBoundaryShape3D` is not supported
- The physics server is not thread-safe (yet)
- Memory usage is not reflected in Godot's performance monitors (yet)
- Ray-casts do not support `face_index`
- `SoftBody3D` does not support any interactions with `Area3D`

## What else is different?

- `Area3D` detecting static bodies is opt-in, at a potentially [heavy performance/memory cost][jst]
- Joints only support soft limits through their substitutes (`JoltHingeJoint3D`, etc.)
- Springs and linear motors are actually implemented in `Generic6DOFJoint3D`
- Single-body joints will make `node_a` be the "world node" rather than `node_b`
- Ray-casts using `hit_back_faces` will hit the back/inside of all shapes, not only concave ones
- Ray-casts are not affected by the `backface_collision` property of `ConcavePolygonShape3D`
- Shape-casts should be more accurate, but their cost also scale with the cast distance
- Shape margins are used, but are treated as an upper bound and scale with the shape's extents
- Manipulating a body's shape(s) after it has entered a scene tree can be costly
- Contact impulses are estimations and won't be accurate when colliding with multiple bodies
- Contact reporting for kinematic bodies is partially opt-in, at a potentially [heavy
  performance/memory cost][jst]

Also consider this note from Jolt's [documentation][jdc]:

> In order for the simulation to be accurate, dynamic objects should be in the order of 0.1 to 10 m
> long, have speeds in the order of 0 to 500 m/s and have gravity in the order of 0 to 10 m/s^2.
> Static object should be in the order of 0.1 to 2000 m long.

## What versions of Godot are supported?

Currently the **only** supported version is **Godot 4.3** (including 4.3.x).

## What platforms are supported?

- Windows (x86-64, x86)
- Linux (x86-64, x86)
- macOS (x86-64 + Apple Silicon)
- iOS
- Android (ARM64, ARM32, x86-64, x86)

Note that Linux support is limited to glibc 2.31 or newer, which for Ubuntu means 20.04 (Focal
Fossa) or newer.

## How do I get started?

1. Download it from [GitHub][rls] or from [Godot Asset Library][ast]
2. Extract the files to your project directory
3. Start (or restart) Godot
4. Open your project settings
5. Make sure "Advanced Settings" is enabled
6. Go to "Physics" and then "3D"
7. Change "Physics Engine" to "JoltPhysics3D"
8. Restart Godot

## What settings are there?

See [`docs/settings.md`][set] for information about the project settings available in Godot Jolt.

## How do I build from source?

See [`docs/building.md`][bld] for information about how to build Godot Jolt from source.

## What's the license?

Godot Jolt is distributed under the MIT license. See [`LICENSE.txt`][lic] for more details and
[`THIRDPARTY.txt`][trd] for third-party licenses.

[god]: https://godotengine.org/
[jlt]: https://github.com/jrouwe/JoltPhysics
[jst]: docs/settings.md#jolt-3d
[jdc]: https://jrouwe.github.io/JoltPhysics/
[rls]: https://github.com/godot-jolt/godot-jolt/releases/latest
[ast]: https://godotengine.org/asset-library/asset/1918
[set]: docs/settings.md
[bld]: docs/building.md
[lic]: LICENSE.txt
[trd]: THIRDPARTY.txt
