# visionOS platform port

This folder contains the C++, Objective-C and Objective-C++ code for the visionOS
platform port.

This platform derives from the Apple Embedded abstract platform ([`drivers/apple_embedded`](drivers/apple_embedded)).

This platform uses shared Apple code ([`drivers/apple`](drivers/apple)).

See also [`misc/dist/ios_xcode`](/misc/dist/ios_xcode) folder for the Xcode
project template used for packaging the iOS export templates.

## Documentation

The compiling and exporting process is the same as on iOS, but replacing the `ios` parameter by `visionos`.

- [Compiling for iOS](https://docs.godotengine.org/en/latest/contributing/development/compiling/compiling_for_ios.html)
  - Instructions on building this platform port from source.
- [Exporting for iOS](https://docs.godotengine.org/en/latest/tutorials/export/exporting_for_ios.html)
  - Instructions on using the compiled export templates to export a project.
