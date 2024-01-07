Control Map Format
=====================

Godot doesn't fully support integer Image formats. So, we store the data as a 32-bit float Image and Texture. However, we read or write pixels as a 32-bit unsigned integer. We do not convert int/float values so there is no precision loss. The values are meaningless interpreted as floats. We read or write 32-bit uints directly from/into the memory block, both in C++ and the shader.

We process the uint as a bit field with the following definition, starting with the left most bits:

| Description | Range | # Bits | Bit #s | Encode | Decode
|-|-|-|-|-|-|
| Base texture id | 0-31 | 5 | 32-28 | `(x & 0x1F) <<27` | `x >>27 & 0x1F`
| Overlay texture id | 0-31 | 5 | 27-23 | `(x & 0x1F) <<22` | `x >>22 & 0x1F`
| Texture blend | 0-255 | 8 | 22-15 | `(x & 0xFF) <<14` | `x >>14 & 0xFF`
| ... reserved ... | | | 
| Hole: 0 terrain, 1 hole | 0-1 | 1 | 3 | `(x & 0x1) <<2` | `x >>2 & 0x1`
| Navigation: 0 no, 1 yes | 0-1 | 1 | 2 | `(x & 0x1) <<1` | `x >>1 & 0x1`
| Autoshader: 0 manual, 1 auto | 0-1 | 1 | 1 | `x & 0x1` | `x & 0x1`

* Encode/decode work in C++ or GLSL and may need a `u` at the end of literals when working with an unsigned integer. e.g. `x >> 14u & 0xFFu`.
* Possible future plans for reserved bits:
  * 5 bits - 32 paintable particles
  * 3 bits - paintable uv scale array index+
  * 3 bits - paintable slope array index+
  * 3 bits - paintable rotation array index+
  * 2 bits - 4 layers (including Hole above, eg water, non-destructive, hole, normal mesh) 
  * 1 bit - future use (maybe added to particles)
  * 2 bits - Texture blend weight array index+
  * Array for all + marked 3-bit indices above: `{ 0.0f, .125f, .25f, .334f, .5f, .667f, .8f, 1.0f }`;

The 3 bit indices above are used to select a value from an 8-index array of values between 0-1, such as `{ 0.0f, .125f, .25f, .334f, .5f, .667f, .8f, 1.0f };` This allows us to store full range 0-1 values that would normally require 8 bits (256 values) in only 3 bits (8 values), since the fine gradations are not important. This idea came from the Witcher 3 presentation.



