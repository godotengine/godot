# Patch Note - Virtual Device Input System

## 1. Overview

The **Virtual Input** system introduces a new layer of hardware abstraction in Godot Engine, treating UI controls as first-class input devices. This implementation deeply integrates tactile interactions into the native input pipeline, allowing the engine to process virtual buttons and joysticks with the same priority and structure as a physical gamepad or keyboard.

## 2. Motivation

The implementation of a native virtual input system in Godot Engine is supported by four fundamental pillars to facilitate the **porting of PC and console games to touch devices**:

- **Deterministic Performance**: By processing touch detection and event dispatching in C++ core, we ensure an immediate and fluid response, essential for maintaining the fidelity of the original experience on touchscreens.
- **Input Unification**: Through the new native events, developers can map actions in the `InputMap` that simultaneously accept physical (gamepad/keyboard) and virtual inputs. This allows a game designed for consoles to work on mobile devices with minimal changes to gameplay logic.
- **Development Acceleration (DX)**: Integration with the Inspector and the Theme system allows artists and designers to customize the control interface without touching code, using the engine's standard workflow.
- **Multi-touch Robustness**: Touch state management (ID tracking and focus capture) is handled natively, preventing common conflicts in manual GUI implementations.

## 3. Core/Input

The core of the change lies in the expansion of the `InputEvent` class to support specific data types for virtual devices:

- **`InputEventVirtualButton`**: Represents binary states (pressed/released) coming from virtual devices, including `device_id` support for multiple controls on the same screen.
- **`InputEventVirtualMotion`**: Carries relative or absolute movement data (X/Y axes), ideal for analog sticks and sliding areas (touchpads).
- **Integration with the `Input` Singleton**: The `Input.parse_input_event()` method has been optimized to route these events, ensuring that the `is_action_pressed()` system works transparently with the new virtual nodes.

## 4. Scene/GUI

The visual layer was structured in a flexible class hierarchy under the interface module:

- **`VirtualDevice` (Base)**: Abstract class inheriting from `Control`. Manages the tactile focus state and provides the common interface for event dispatching.
- **`VirtualButton`**: An optimized control for triggering discrete actions, supporting icons, text, and full visual states (Pressed, Hover, Disabled).
- **`VirtualJoystick`**: Analog stick implementation with configurable deadzone and clamp areas.
- **`VirtualJoystickDynamic`**: A capture area that makes the dynamic joystick appear exactly where the user touched within the region.
- **`VirtualTouchPad`**: Area focused on relative movement (delta), functioning similarly to a laptop trackpad, essential for 3D camera control.

## 5. Doc/*.xml

The API reference documentation has been fully integrated into Godot's internal help system. Each new node and event has detailed descriptions of properties, methods, and signals, located in:

- Each node in `scene/gui/` has its own dedicated XML documentation file.
- Core files such as `Input` and `InputMap` have been updated to support the new event types and routing logic.
- Synchronized documentation for: `VirtualButton`, `VirtualJoystick`, `VirtualJoystickDynamic`, `VirtualTouchPad`, `InputEventVirtualButton`, and `InputEventVirtualMotion`.

This ensures that the developer has immediate access to technical documentation directly through the editor (F1) or via the official online documentation.

## 6. Scene/Theme

To ensure the virtual system adapts to any project, it has been fully integrated into the **ThemeDB**:

- **Customization via Inspector**: Using the `BIND_THEME_ITEM` macro, all visual properties (StyleBoxes, Colors, Fonts) are exposed in the "Theme Overrides" section of the Inspector.
- **Default Themes**: Consistent initial definitions were added to `default_theme.cpp`, providing a functional and aesthetically pleasing interface out-of-the-box, maintaining visual coherence with other Godot UI controls.
