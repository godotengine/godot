Mobile & Web Support
=========================

This page documents the specific setups needed for non-desktop platforms. All platforms are experimental at best, and need Godot 4.2 or later.

**Table of Contents**
* [Android](#android)
* [IOS](#ios)
* [Steam Deck](#steam-deck)
* [WebGL & Compatibility Renderer](#webgl)

## Android

As of Terrain3D 0.9.1 and Godot 4.2, Android is reported to work. 

* Use textures that Godot imports (converts) such as PNG or TGA, not DDS.
* Enable `Project Settings/Rendering/Textures/VRAM Compression/Import ETC2 ASTC`.

The release builds include binaries for arm32 and arm64.

There is a [texture artifact](https://github.com/TokisanGames/Terrain3D/issues/137) affecting some systems using the demo DDS textures. This may be alleviated by using PNGs as noted above, but isn't confirmed.

Further reading:

* [Issue 197](https://github.com/TokisanGames/Terrain3D/issues/197)

## IOS

As of Terrain3D 0.9.1 and Godot 4.2, iOS is reported to work with the following setup:

* Use textures that Godot imports (converts) such as PNG or TGA, not DDS.
* Enable `Project Settings/Rendering/Textures/VRAM Compression/Import ETC2 ASTC`.
* Set `Project Settings/Application/Config/Icon` to a valid file (eg `res://icon.png` or svg).
* The Terrain3D release includes iOS builds, however they aren't signed and may not work.
* If needed, build the iOS library and make sure the binaries are placed where identified in `terrain.gdextension`:
```
     scons platform=ios target=template_debug
     scons platform=ios target=template_release
```

* Select `Project/Export`, Add the iOS export preset and configure with `App Store Team ID` and `Bundle Identifier`, then export.

```{image} images/ios_export.png
:target: ../_images/ios_export.png
```

Once it has been exported, you can open it in XCode, run locally, or on your device. Providing Apple support is out of scope for us.

Further reading:
* [Issue 218](https://github.com/TokisanGames/Terrain3D/issues/218)
* [PR 219](https://github.com/TokisanGames/Terrain3D/pull/219)
* [PR 295](https://github.com/TokisanGames/Terrain3D/pull/295)


## Steam Deck

As of Terrain3D v0.9.1 and Godot 4.2, the first generation Steam Deck is reported working, running the demo at 200+ fps.

The user got it working with the following:
* Use SteamOS 3.5.7
* Install `glibc` and `linux-api-headers` in addition to the standard Godot dependencies
* [Build from source](building_from_source.md)


Further reading:

* [Issue 220](https://github.com/TokisanGames/Terrain3D/issues/220#issuecomment-1837552459)


## WebGL

WebGL and the Compatibility Renderer are not fully supported yet. The terrain mesh builds and is reported to work fine. 

The remaining issue is that the shader does not work because Godot does not fully support Texture2DArrays in the compatibility renderer. However it seems to work for some texture arrays, just not albedo. The exact problem isn't fully tested or identified.

As an alternative, you can write your own custom shader. You can add texture samplers in the Terrain3DMaterial shader and in textures not in a TextureArray.

Further reading:

* [Issue 217](https://github.com/TokisanGames/Terrain3D/issues/217)

