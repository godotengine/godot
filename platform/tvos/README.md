# tvOS platform port

This folder contains the C++, Objective-C and Objective-C++ code for the tvOS
platform port.

This platform derives from the Apple Embedded abstract platform ([`drivers/apple_embedded`](/drivers/apple_embedded)).

This platform uses shared Apple code ([`drivers/apple`](/drivers/apple)).

See also [`misc/dist/apple_embedded_xcode`](/misc/dist/apple_embedded_xcode) folder for the Xcode
project template used for packaging the tvOS export templates.

## Documentation

The compiling and exporting process is the same as on iOS, but replacing the `ios` parameter by `tvos`.

- [Compiling for iOS](https://docs.godotengine.org/en/latest/engine_details/development/compiling/compiling_for_ios.html)
  - Instructions on building this platform port from source.
- [Exporting for iOS](https://docs.godotengine.org/en/latest/tutorials/export/exporting_for_ios.html)
  - Instructions on using the compiled export templates to export a project.

## tvOS specifics

- There is no touchscreen or clipboard; the corresponding `DisplayServer`
  features report as unsupported. Text entry uses the system tvOS keyboard
  through the standard virtual keyboard API.
- Games are controlled with the Siri Remote and MFi game controllers, which
  arrive through the standard Godot joypad API (remote clicks, trackpad
  swipes, and buttons included). Note the default `ui_accept`/`ui_cancel`
  bindings are keyboard-only, so games must add joypad buttons to them for
  remote-driven menus. The remote Menu button also sends
  `WINDOW_EVENT_GO_BACK_REQUEST` (or quits if `quit_on_go_back` is set). In
  the Simulator, where no remote exists, remote presses are synthesized as
  key events instead.
- Process spawning (`OS.execute`, `OS.create_process`) is prohibited by the
  tvOS sandbox and fails with `ERR_UNAVAILABLE`.
- Both the Metal (Forward+/Mobile) and OpenGL ES (Compatibility) rendering
  drivers are supported on device. Simulator builds use OpenGL ES only.
