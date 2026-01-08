# Input: Add native Virtual Device Input System for on-screen controls

## Overview

This PR introduces a native **Virtual Device Input System** to Godot Engine. It provides a robust, high-performance architecture for on-screen controls (joysticks, buttons, touchpads) by treating them as first-class input devices. This implementation directly addresses the long-standing community need for standardized mobile input, as outlined in **Godot Proposals [#13943](https://github.com/godotengine/godot-proposals/issues/13943)** and **[#11192](https://github.com/godotengine/godot-proposals/issues/11192)**.

## Intentions & Philosophy

The core intention of this system is to provide a **"Hardware Abstraction Layer for UI"**.

Currently, developers must choose between limited `TouchScreenButton` nodes or complex, fragmented GDScript-based joysticks. This PR changes that by:

1. **Unifying Input Logic**: Game code should not care if an `InputEvent` comes from a physical Sony DualSense or a `VirtualJoystick` on an iPad. By emitting native `InputEventVirtual*` events, this system allows the `InputMap` to handle on-screen controls exactly like hardware.
2. **Performance (C++ Core)**: By moving touch tracking, vector calculations, and event dispatching to the engine core, we achieve the lowest possible latency and eliminate the frame-budget cost of running complex input logic in GDScript.
3. **Developer Experience (DX)**: Providing a suite of `Control`-based nodes that respect the layout system, the Inspector, and the Theme system allows artists and designers to iterate on mobile controls without writing boilerplate code.

## Technical Breakdown

### 1. Core Input Expansion

The `InputEvent` hierarchy has been expanded with two primary types:

- **`InputEventVirtualButton`**: Represents digital states (On/Off) from virtual devices. Supports `device_id` to allow multiple virtual controllers.
- **`InputEventVirtualMotion`**: Carries analog data (X/Y axes) for joysticks, sliders, or touchpads.
- Integration into `Input.parse_input_event()` ensures that `is_action_pressed()` and `get_vector()` work seamlessly with both physical and virtual inputs.

### 2. Architecture: `VirtualDevice` base class

A new abstract base class `VirtualDevice` (inheriting from `Control`) serves as the bridge between UI interactions and the Input pipeline. It manages:

- **Touch ID Tracking**: Automatically handles multi-touch state and focus capture across different UI layers.
- **Event Dispatching**: Logic to translate raw `InputEventScreenTouch/Drag` into high-level virtual events.

### 3. New UI Nodes (`scene/gui/`)

The system includes specialized nodes based on the requirements of modern games:

- **`VirtualButton`**: A touch-optimized button supporting various press behaviors and full theme integration.
- **`VirtualJoystick` & `VirtualJoystickDynamic`**: High-performance analog sticks with support for Fixed and Dynamic modes. `VirtualJoystickDynamic` introduces a capture area that spawns the joystick at the touch point, with a **"Visible By Default"** option to keep the joystick at a pre-defined position when idle.
- **`VirtualDPad`**: Standard 4-way directional pad for retro and UI navigation.
- **`VirtualTouchPad`**: A relative-motion area specifically designed for 3D camera controls and look-around logic.

### 4. Integration with ThemeDB

To ensure visual consistency, the system is fully integrated with Godot's **Theme System**:

- New nodes use `BIND_THEME_ITEM` to expose StyleBoxes, Colors, and Fonts.
- Default aesthetics have been added to `default_theme.cpp`, ensuring an "out-of-the-box" professional look while remaining fully customizable via "Theme Overrides".

## Why in Core?

While many third-party addons exist, they lack the deep integration required for a "first-class" experience. By bringing this to core:

- We standardize mobile development across the engine.
- We enable the official demo projects to work flawlessly on touch devices.
- We provide a performance-optimized path that is difficult to achieve in GDScript alone.

## Documentation

- Full XML reference updated for all new classes and events in `doc/classes/`.
- Updated descriptions for `Input` and `InputMap` to include virtual device routing logic.
