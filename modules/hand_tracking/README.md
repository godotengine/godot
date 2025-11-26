# Hand Tracking Module for Godot

This module provides Apple Vision Pro hand tracking integration for Godot Engine.

## Overview

The hand tracking module enables Godot applications running on visionOS to access ARKit hand tracking data. It provides a bridge between the native visionOS platform layer and Godot's XR system, allowing developers to create hand-tracking-based interactions.

## Architecture

The module consists of three main layers:

### 1. C Bridge Layer (`hand_tracking.h`, `hand_tracking_bridge.cpp`)

- Provides a C-compatible API that can be called from Swift/Objective-C
- Defines data structures for hand joint positions and orientations
- Thread-safe storage of the latest hand tracking frame

### 2. HandTrackingServer Singleton (`hand_tracking_server.h/cpp`)

- Godot singleton that manages hand tracking state
- Creates and updates `XRHandTracker` instances for left and right hands
- Integrates with Godot's existing XR system (`XRServer`)
- Maps ARKit joint IDs to Godot's XRHandTracker joint enumeration

### 3. Swift/visionOS Integration (to be implemented in platform layer)

- Uses ARKit's `HandTrackingProvider` to get hand anchor data
- Converts hand joint transforms to the C bridge format
- Calls `godot_visionos_set_hand_frame()` each frame

## Usage in GDScript

```gdscript
extends Node3D

func _ready():
    var hand_tracking = HandTrackingServer.get_singleton()

    if hand_tracking.is_hand_tracking_available():
        var left_hand = hand_tracking.get_left_hand_tracker()
        var right_hand = hand_tracking.get_right_hand_tracker()
        print("Hand tracking initialized!")

func _process(delta):
    # Update hand tracking data (called automatically by the engine)
    var hand_tracking = HandTrackingServer.get_singleton()
    hand_tracking.update_hand_tracking()

    # Access hand tracker data through XRServer
    var left_hand = hand_tracking.get_left_hand_tracker()
    if left_hand and left_hand.get_has_tracking_data():
        var wrist_transform = left_hand.get_hand_joint_transform(XRHandTracker.HAND_JOINT_WRIST)
        print("Left wrist position: ", wrist_transform.origin)
```

## Hand Joint Mapping

The module maps ARKit hand joints to Godot's XRHandTracker joints:

| ARKit Joint | Godot XRHandTracker Joint |
|------------|---------------------------|
| Wrist | HAND_JOINT_WRIST |
| Thumb Knuckle | HAND_JOINT_THUMB_METACARPAL |
| Thumb Intermediate | HAND_JOINT_THUMB_PHALANX_PROXIMAL |
| Thumb Tip | HAND_JOINT_THUMB_TIP |
| Index Knuckle | HAND_JOINT_INDEX_FINGER_METACARPAL |
| Index Intermediate | HAND_JOINT_INDEX_FINGER_PHALANX_PROXIMAL |
| Index Distal | HAND_JOINT_INDEX_FINGER_PHALANX_INTERMEDIATE |
| Index Tip | HAND_JOINT_INDEX_FINGER_TIP |
| ... (similar for middle, ring, and little fingers) |

## Building

The module is built automatically when compiling Godot for macOS or visionOS:

```bash
scons platform=macos arch=arm64
```

## Future Work

- Implement visionOS platform layer integration with ARKit
- Add gesture recognition helpers
- Create demo scenes showing hand tracking interactions
- Add XRHandModifier3D integration for automatic skeleton updates
- Performance optimizations for high-frequency updates
