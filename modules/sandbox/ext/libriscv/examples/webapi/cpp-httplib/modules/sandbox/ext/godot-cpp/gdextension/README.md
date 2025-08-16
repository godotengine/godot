# GDExtension header and API

This repository contains the C header and API JSON for
[**Godot Engine**](https://github.com/godotengine/godot)'s *GDExtensions* API.

## Updating header and API

If the current branch is not up-to-date for your needs, or if you want to sync
the header and API JSON with your own modified version of Godot, here is the
update procedure used to sync this repository with upstream releases:

- Compile [Godot Engine](https://github.com/godotengine/godot) at the specific
  version/commit which you are using.
  * Or if you use an official release, download that version of the Godot editor.
- Use the compiled or downloaded executable to generate the `extension_api.json`
  and `gdextension_interface.h` files with:

```
godot --dump-extension-api --dump-gdextension-interface
```
