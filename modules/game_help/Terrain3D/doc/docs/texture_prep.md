Preparing Textures
=========================

Terrain3D supports up to 32 texture sets using albedo, height, normal, and roughness textures. This page describes everything you need to know to prepare your texture files. Continue on to [Texture Painting](texture_painting.md) to learn how to use them.

TLDR: Just read channel pack textures with Gimp.

**Table of Contents**
* [Required Texture Format](#required-texture-format)
* [Channel pack textures with Gimp](#channel-pack-textures-with-gimp)
* [Where to get textures](#where-to-get-textures)
* [Frequently Asked Questions](#faq)


## Required Texture Format

### Texture Files
You need two files per texture set, channel packed as follows:

| Name | Format |
| - | - |
| albedo_texture | RGB: Albedo texture, A: Height texture
| normal_texture| RGB: Normal map texture ([OpenGL +Y](#normal-map-format)), A: Roughness texture

### Texture Sizes
All albedo textures and all normal textures must be the same size.

All texture sizes should be a power of 2 (128, 256, 512, 1024, 2048, 4096).

The albedo textures can be a different size than the normal textures. This is because all of the albedo+height textures are combined into a TextureArray and the normal+roughness textures are combined into another.

### Normal Map Format
Normal maps come in two formats: DirectX with -Y, and OpenGL with +Y. You can convert DirectX to OpenGL by inverting the green channel in a photo editing app. You can visually tell which type of normal map you have by whether the bumps are sticking out towards you (OpenGL) or pushed in away from you (DirectX). The left shapes are the clearest examples.

```{image} images/tex_normalmap.png
:target: ../_images/tex_normalmap.png
```

### Compression Format

| Type | Supports | Format |
| - | - | - |
| **DDS** | Desktop | BC3 / DXT5, linear (not srgb), Color + alpha, Generate Mipmaps. These are accepted instantly by Godot with no import settings. **Highly recommended** |
| **PNG** | Desktop, Mobile | Standard RGBA. In Godot you must go to the Import tab and select: `Mode: VRAM Compressed`, `Normal Map: Disabled`, `Mipmaps Generate: On`, then reimport each file. For mobile, enable `Project Settings: rendering/textures/vram_compression/import_etc2_astc`.
| **Others** | | Other formats like KTX and TGA are probably supported as long as you match similar settings above.

### Other Notes

* Make sure you have seamless textures that can be repeated without an obvious seam.

* Textures can be channel packed in in tools like Photoshop, Krita, [Gimp](https://www.gimp.org/), or many similar tools. Working with alpha channels in Photoshop and Krita can be a bit challenging, so we recommend Gimp. See [Channel Pack Textures with Gimp](#channel-pack-textures-with-gimp) below.

* Some "roughness" textures are actually smoothness or gloss textures. You can convert between them by inverting it. You can tell which is which just by looking at distinctive textures. If it's glass it should be mostly black, which is near 0 roughness. If it's dry rock or dirt, it should be mostly white, which is near 1 roughness. Smoothness would be the opposite. 

* You can create DDS files by exporting them directly from Gimp, exporting from Photoshop with [Intel's DDS plugin](https://www.intel.com/content/www/us/en/developer/articles/tool/intel-texture-works-plugin.html), or converting RGBA PNGs using [NVidia's Texture Tools](https://developer.nvidia.com/nvidia-texture-tools-exporter).

* Once you've created the textures externally, place them in your Godot project folder. Then create a new texture slot in the Textures panel and drag your files from the FileSystem panel into the appropriate texture slot.

* It's best to remove unused texture slots to save memory. You can only remove them from the end of the list. You can reorder textures by changing their ID.


## Channel Pack Textures with Gimp

1. Open up your RGB Albedo and greyscale Height files (or Normal and Roughness).

2. On the RGB file select `Colors/Components/Decompose`. Select `RGB`. Keep `Decompose to layers` checked. On the resulting image you have three greyscale layers for RGB. 

3. Copy the greyscale Height (or Roughness) file and paste it as a new layer into this decomposed file. Name the new layer `alpha`.

This would be a good time to invert the green channel if you need to convert a Normalmap from DirectX to OpenGL, or to invert the alpha channel if you need to convert a smoothness texture to a roughness texture.

4. Select `Colors/Components/Compose`. Select `RGBA` and ensure each named layer connects to the correct channel.

5. Now export the file with the following settings. DDS is highly recommended. 

Also recommended is to export directly into your Godot project folder. Then drag the files from the FileSystem panel into the appropriate texture slots. With this setup, you can make adjustments in Gimp and export again, and Godot will automatically update with any file changes.

### Exporting As DDS
* Change `Compression` to `BC3 / DXT5`
* `Mipmaps` to `Generate Mipmaps`. 
* Insert into Godot and you're done.

```{image} images/io_gimp_dds_export.png
:target: ../_images/io_gimp_dds_export.png
```

### Exporting As PNG
* Change `automatic pixel format` to `8bpc RGBA`. 
* In Godot you must go to the Import tab and select: `Mode: VRAM Compressed`, `Normal Map: Disabled`, `Mipmaps Generate: On`, then click `Reimport`.

```{image} images/io_gimp_png_export.png
:target: ../_images/io_gimp_png_export.png
```

## Where to get textures

### Tools

You can make textures in dedicated texture tools, such as those below. There are many other tools and ai texture generators to be found online. You can also paint textures in applications like krita/gimp/photoshop.
 
* [Materialize](http://boundingboxsoftware.com/materialize/) - Great free tool for generating missing maps. e.g. you only have an albedo texture that you love and want to generate a normal and height map
* [Material Maker](https://www.materialmaker.org/) - Free, open source material maker made in Godot
* [ArmorLab](https://armorpaint.org/) - Free & open source texture creator
* Substance Designer - Commercial, "industry standard" texture maker


### Download Sites

There are numerous websites where you can download high quality, royalty free textures for free or pay. These textures come as individual maps, with the expectation that you will download only the maps you need and then channel pack them. Here are just a few:

* [PolyHaven](https://polyhaven.com/textures) - many free textures
* [AmbientCG](https://ambientcg.com/) - many free textures
* [Poliigon](https://www.poliigon.com/textures/free) - free and commercial
* [GameTextures](https://gametextures.com/) - commercial
* [Textures](https://www.textures.com/) - commercial

## FAQ

### Why can't we just use regular textures? Why is this so difficult / impossible to do?

We provide easy, [5-step instructions](#channel-pack-textures-with-gimp) that take less than 2 minutes once you're familiar with the process. 

Channel packing is a very common task done by professional game developers. Every pro asset pack you've used has channel packed textures. When you download texture packs from websites, they provide individual textures so you can pack them how you want. They are not intended to be used individually!

We want high performance games, right? Then, we need to optimize our systems for the graphics hardware. The shader can retrieve four channels RGBA from a texture at once. Albedo and normal textures only have RGB. Thus, reading Alpha is free, and a waste if not used. So, we put height / roughness in the Alpha channel.

Efficiency is also why we want power of 2 textures.

We could have the system channel pack for you, however that would mean processing up to 128 images every time any scene with Terrain3D loads, both in the editor and running games. Exported games may not even work since Godot's image compression libraries only exist in the editor. The most reasonable path is for gamedevs to learn a simple process that they'll use for their entire career and use it to set up terrain textures one time. In the future we may add a [channel packing tool](https://github.com/TokisanGames/Terrain3D/issues/125) to facilitate file creation within Godot.

### What about AO, Emissive, Metal, and other texture maps?

Most terrain textures like grass, rock, and dirt do not need these. 

The only one that might be useful generally is AO, however that is debatable. We do include height, which can double for AO. Or, if you have no height texture, you can substitute an AO texture. These two are nearly interchangeable depending on the specific texture, and it's not worth allocating another texture array just for AO.

Occasional textures do need additional texture maps. Lava rock might need emissive, or rock with gold veins might need metallic, or some unique texture might need both height and AO. These are most likely only 1-2 textures out of the possible 32, so setting up these additional options for all textures is a waste of memory. For this you can use a [custom shader](tips.md#add-a-custom-texture-map) to add the individual texture map.

### Why not use Standard Godot materials?

All materials in Godot are just shaders. The standard shader is both overly complex, yet inadequate for our needs. Dirt does not need SSS, refraction, or backlighting for instance. See [a more thorough explanation](https://github.com/TokisanGames/Terrain3D/issues/199).

### What about displacement?

Godot doesn't support any sort of texture displacement or tessellation in the renderer. It does have depth parallax (called height), which is quite unattractive and is only usable on certain textures like brick. There are [alternatives](https://github.com/TokisanGames/Terrain3D/issues/175) that might prove useful in the future.

### What about...

We provide a base texture with the most commonly needed terrain options. Then we provide the option for a custom shader so you can explore `what about` on your own. Any of the options in the Godot StandardMaterial can be converted to a shader, and then you can insert that code into a custom shader. You could experiment with Godot's standard depth parallax technique, or any of the alternatives above. Or anything else you can imagine, like a sinewave that ripples the vertices outward for a VFX ground ripple effect, or ripples on a puddle ground texture.


