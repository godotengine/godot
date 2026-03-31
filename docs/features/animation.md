# Animation

## Purpose

Animate per-splat properties (position, color, opacity, scale, rotation) over time using `GaussianAnimationStateMachine`. The animation system provides clip-based playback, keyframe interpolation with multiple curve types, and weighted blending between clips.

## Animation system overview

`GaussianAnimationStateMachine` is a `Resource`-derived class that holds animation clips, tracks, and keyframes. Each clip contains one or more tracks, and each track targets a specific `AnimationProperty`. Keyframes store timed values with configurable interpolation.

| Concept | Description | Implementation reference |
| --- | --- | --- |
| AnimationClip | Named container with a duration, loop flag, and a set of tracks. | `modules/gaussian_splatting/animation/animation_state_machine.h:44` |
| AnimationTrack | Targets one `AnimationProperty` and holds a sorted list of keyframes. | `modules/gaussian_splatting/animation/animation_state_machine.h:34` |
| AnimationProperty | Enum: `POSITION`, `COLOR`, `OPACITY`, `SCALE`, `ROTATION`. | `modules/gaussian_splatting/animation/animation_state_machine.h:17` |
| AnimationState | Enum: `STOPPED`, `PLAYING`, `PAUSED`, `SEEKING`. | `modules/gaussian_splatting/animation/animation_state_machine.h:25` |
| KeyframeInterpolator | Handles value interpolation between keyframes using linear, constant, cubic Bezier, smooth-step, and smoother-step curves. | `modules/gaussian_splatting/animation/keyframe_interpolator.h:13` |

## Setting up animation clips

### Create a clip and add tracks

```gdscript
var anim := GaussianAnimationStateMachine.new()
anim.set_splat_count(data.get_count())

# Create a 2-second clip named "wobble"
var clip_idx := anim.add_clip("wobble", 2.0)
anim.set_clip_looping(clip_idx, true)

# Add a position track and an opacity track
anim.add_track_to_clip(clip_idx, GaussianAnimationStateMachine.ANIMATION_PROPERTY_POSITION)
anim.add_track_to_clip(clip_idx, GaussianAnimationStateMachine.ANIMATION_PROPERTY_OPACITY)
```

| Method | Description | Implementation reference |
| --- | --- | --- |
| `add_clip(name, duration)` | Creates a new clip. Returns the clip index. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:199` |
| `set_clip_duration(index, duration)` | Changes the clip duration. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:205` |
| `set_clip_looping(index, looping)` | Enables or disables looping. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:207` |
| `add_track_to_clip(clip_index, property)` | Adds a track for the given property. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:210` |
| `remove_track_from_clip(clip_index, property)` | Removes the track for the given property. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:211` |
| `has_track(clip_index, property)` | Returns whether the clip has a track for the property. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:212` |

### Add keyframes

```gdscript
# Linear keyframes
anim.add_keyframe(clip_idx, GaussianAnimationStateMachine.ANIMATION_PROPERTY_POSITION,
    0.0, Vector3(0, 0, 0))
anim.add_keyframe(clip_idx, GaussianAnimationStateMachine.ANIMATION_PROPERTY_POSITION,
    1.0, Vector3(0, 1, 0))
anim.add_keyframe(clip_idx, GaussianAnimationStateMachine.ANIMATION_PROPERTY_POSITION,
    2.0, Vector3(0, 0, 0))

# Bezier keyframes for smoother motion
anim.add_keyframe_bezier(clip_idx, GaussianAnimationStateMachine.ANIMATION_PROPERTY_OPACITY,
    0.0, 1.0, Vector2(0.0, 0.0), Vector2(0.3, 0.0))
anim.add_keyframe_bezier(clip_idx, GaussianAnimationStateMachine.ANIMATION_PROPERTY_OPACITY,
    1.0, 0.5, Vector2(-0.3, 0.0), Vector2(0.3, 0.0))
anim.add_keyframe_bezier(clip_idx, GaussianAnimationStateMachine.ANIMATION_PROPERTY_OPACITY,
    2.0, 1.0, Vector2(-0.3, 0.0), Vector2(0.0, 0.0))
```

| Method | Description | Implementation reference |
| --- | --- | --- |
| `add_keyframe(clip_index, property, time, value)` | Inserts a keyframe with default linear interpolation. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:215` |
| `add_keyframe_bezier(clip_index, property, time, value, in_handle, out_handle)` | Inserts a keyframe with cubic Bezier interpolation handles. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:216` |
| `remove_keyframe(clip_index, property, keyframe_index)` | Removes a keyframe by index. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:217` |
| `get_keyframe_count(clip_index, property)` | Returns the number of keyframes on a track. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:218` |
| `get_keyframe_time(clip_index, property, keyframe_index)` | Returns the time of a specific keyframe. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:219` |
| `get_keyframe_value(clip_index, property, keyframe_index)` | Returns the value stored in a keyframe. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:220` |

### Interpolation types

Keyframe interpolation is determined by the `InterpolationType` stored with each keyframe.

| Type | Behavior | Implementation reference |
| --- | --- | --- |
| `CONSTANT` | Holds the keyframe value until the next keyframe. | `modules/gaussian_splatting/animation/keyframe_interpolator.h:13` |
| `LINEAR` | Linear interpolation (`lerp` for scalars/vectors, `slerp` for quaternions). | `modules/gaussian_splatting/animation/keyframe_interpolator.cpp:88` |
| `CUBIC_BEZIER` | Cubic Bezier curve using in/out handles. | `modules/gaussian_splatting/animation/keyframe_interpolator.cpp:148` |
| `SMOOTH_STEP` | Hermite smooth-step: `t*t*(3-2t)`. | `modules/gaussian_splatting/animation/keyframe_interpolator.cpp:204` |
| `SMOOTHER_STEP` | Ken Perlin smoother-step: `t*t*t*(t*(6t-15)+10)`. | `modules/gaussian_splatting/animation/keyframe_interpolator.cpp:208` |

## Playback and blending

### Playback control

```gdscript
func _ready() -> void:
    anim.play(clip_idx)

func _process(delta: float) -> void:
    anim.update(delta)

    # Sample properties for rendering
    var pos := anim.sample_position(splat_index)
    var col := anim.sample_color(splat_index)
    var opa := anim.sample_opacity(splat_index)
```

| Method | Description | Implementation reference |
| --- | --- | --- |
| `play(clip_index)` | Starts playback. Pass `-1` to resume the current clip. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:223` |
| `pause()` | Pauses playback at the current time. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:224` |
| `stop()` | Stops playback and resets to the beginning. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:225` |
| `seek(time)` | Jumps to a specific time in the current clip. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:226` |
| `set_playback_speed(speed)` | Changes the playback rate multiplier. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:227` |
| `update(delta)` | Advances the animation by `delta` seconds. Call this every frame. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:242` |
| `is_playing()` | Returns `true` when the state machine is in the `PLAYING` state. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:234` |
| `get_current_time()` | Returns the current playback position in seconds. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:232` |

### Blending between clips

The blending system allows smooth transitions between clips over a specified duration. Each clip has a weight that can be controlled directly or driven by `blend_to_clip`.

```gdscript
# Create a second clip
var idle_idx := anim.add_clip("idle", 3.0)
anim.set_clip_looping(idle_idx, true)
anim.add_track_to_clip(idle_idx, GaussianAnimationStateMachine.ANIMATION_PROPERTY_POSITION)
# ... add keyframes ...

# Blend from current clip to "idle" over 0.5 seconds
anim.blend_to_clip(idle_idx, 0.5)
```

| Method | Description | Implementation reference |
| --- | --- | --- |
| `blend_to_clip(clip_index, blend_duration)` | Smoothly transitions to the target clip. Default duration is 0.3 seconds. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:237` |
| `set_clip_weight(clip_index, weight)` | Manually sets the blend weight for a clip. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:238` |
| `get_clip_weight(clip_index)` | Returns the current blend weight. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:239` |

### Sampling

Sample animated property values at the current playback time (or at an explicit time).

| Method | Returns | Description | Implementation reference |
| --- | --- | --- | --- |
| `sample_position(splat_index, time)` | `Vector3` | Position at the given time (default: current time). | `modules/gaussian_splatting/animation/animation_state_machine.cpp:249` |
| `sample_color(splat_index, time)` | `Color` | Color at the given time. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:250` |
| `sample_opacity(splat_index, time)` | `float` | Opacity at the given time. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:251` |
| `sample_scale(splat_index, time)` | `Vector3` | Scale at the given time. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:252` |
| `sample_rotation(splat_index, time)` | `Quaternion` | Rotation at the given time. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:253` |

### Serialization

Animation data can be saved and restored through dictionary serialization.

```gdscript
# Save animation state
var dict := anim.to_dict()
var file := FileAccess.open("user://animation.json", FileAccess.WRITE)
file.store_string(JSON.stringify(dict))
file.close()

# Restore animation state
var loaded := JSON.parse_string(FileAccess.get_file_as_string("user://animation.json"))
anim.from_dict(loaded)
```

## Troubleshooting

| Symptom | Cause | Fix | Implementation reference |
| --- | --- | --- | --- |
| `add_clip` returns an error for duplicate name | A clip with that name already exists. | Use a unique name or remove the existing clip with `remove_clip_by_name()`. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:274` |
| Sampled values do not change over time | `update(delta)` is not being called each frame. | Add `anim.update(delta)` to your `_process` callback. | `modules/gaussian_splatting/animation/animation_state_machine.cpp:242` |
| Rotation interpolation produces unexpected results | Keyframe values are not normalized quaternions. | Provide normalized `Quaternion` values; the interpolator uses `slerp`. | `modules/gaussian_splatting/animation/keyframe_interpolator.cpp:116` |
| Blend transition has no visible effect | Both clips animate different properties or one clip has no keyframes. | Verify that both clips have tracks for the same `AnimationProperty` with at least two keyframes each. | `modules/gaussian_splatting/animation/animation_state_machine.h:74` |
| `sample_position` returns a default value | No position track exists or the track has no keyframes. | Add a position track with `add_track_to_clip` and insert keyframes. | `modules/gaussian_splatting/animation/animation_state_machine.h:159` |
