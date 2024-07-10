Control Map Format
=====================

The control map defines how the terrain is textured using unsigned, 32-bit integer data. Godot doesn't fully support integer Image or Texture formats, so we use FORMAT_RF. See further below for details.

We process each uint32 pixel as a bit field with the following definition, starting with the left most bits:

| Description | Range | # Bits | Bit #s | Encode | Decode
|-|-|-|-|-|-|
| Base texture id | 0-31 | 5 | 32-28 | `(x & 0x1F) <<27` | `x >>27 & 0x1F`
| Overlay texture id | 0-31 | 5 | 27-23 | `(x & 0x1F) <<22` | `x >>22 & 0x1F`
| Texture blend | 0-255 | 8 | 22-15 | `(x & 0xFF) <<14` | `x >>14 & 0xFF`
| UV angle | 0-15 | 4 | 14-11 | `(x & 0xF) <<10` | `x >>10 & 0xF`
| UV scale | 0-8 | 3 | 10-8 | `(x & 0x7) <<6` | `x >>6 & 0x7`
| ... reserved ... | | | 
| Hole: 0 terrain, 1 hole | 0-1 | 1 | 3 | `(x & 0x1) <<2` | `x >>2 & 0x1`
| Navigation: 0 no, 1 yes | 0-1 | 1 | 2 | `(x & 0x1) <<1` | `x >>1 & 0x1`
| Autoshader: 0 manual, 1 auto | 0-1 | 1 | 1 | `x & 0x1` | `x & 0x1`

* The encode/decode formulas work in both C++ or GLSL, though may need a `u` at the end of literals when working with an unsigned integer. e.g. `x >> 14u & 0xFFu`.
* We use a FORMAT_RF 32-bit float Image or Texture to allocate the memory. Then in C++, we read or write each uint32 pixel directly into the "float" memory. The values are meaningless when interpreted as floats. We don't convert the integer values to float, so there is no precision loss. Godot shaders support usamplers so we can interpret the memory directly as uint32, without requiring any conversion.
* Gamedevs can use the conversion and testing functions found in Terrain3DUtil defined in [C++](https://github.com/TokisanGames/Terrain3D/blob/main/src/terrain_3d_util.h) and [GDScript](../api/class_terrain3dutil.rst).
* Possible future plans for reserved bits:
  * 5 bits - 32 paintable particles
  * 3 bits - paintable slope array index+
  * 2 bits - 4 layers (including Hole above, eg water, non-destructive, hole, normal mesh) 
  * 1 bit - future use (maybe added to particles)
  * 2 bits - Texture blend weight array index+
  * Array for all + marked 3-bit indices above: `{ 0.0f, .125f, .25f, .334f, .5f, .667f, .8f, 1.0f }`;

The 3 bit indices above are used to select a value from an 8-index array of values between 0-1, such as `{ 0.0f, .125f, .25f, .334f, .5f, .667f, .8f, 1.0f };` This allows us to store full range 0-1 values that would normally require 8 bits (256 values) in only 3 bits (8 values), since the fine gradations are not important. This idea came from the Witcher 3 presentation.

