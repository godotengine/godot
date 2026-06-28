# Windows installer

`godot.iss` is an [Inno Setup](https://jrsoftware.org/isinfo.php) installer file
that can be used to build a Windows installer. The generated installer is able
to run without Administrator privileges and can optionally add Godot to the
user's `PATH` environment variable.

To use Inno Setup on Linux, use [innoextract](https://constexpr.org/innoextract/)
to extract the Inno Setup installer then run `ISCC.exe` using
[WINE](https://www.winehq.org/).

## Building

- Place a Godot editor executable in this folder and rename it to `godot.exe`.
- Run the Inno Setup Compiler (part of the Inno Setup suite) on the `godot.iss` file.

If everything succeeds, an installer will be generated in this folder.
