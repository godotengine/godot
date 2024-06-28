Preparing Textures
=========================

Terrain3D supports up to 32 texture sets using albedo, height, normal, and roughness textures. This page describes everything you need to know to prepare your texture files. Continue on to [Texture Painting](texture_painting.md) to learn how to use them.

**Table of Contents**
* [Texture Requirements](#texture-requirements)
* [Texture Content](#texture-content)
* [Channel Pack Textures in Terrain3D](#channel-pack-textures-in-terrain3d)
* [Channel Pack Textures with Gimp](#channel-pack-textures-with-gimp)
* [Where to Get Textures](#where-to-get-textures)
* [Frequently Asked Questions](#faq)


## Texture Requirements

### Texture Files
You need two files per texture set. Terrain3D is designed for textures that are channel packed as follows:

| Name | Format |
| - | - |
| albedo_texture | RGB: Albedo texture, A: Height texture
| normal_texture| RGB: Normal map texture ([OpenGL +Y](#normal-map-format)), A: Roughness texture

The terrain can work without the alpha channels, however it won't have height blending or roughness. That may be fine for a low-poly or stylized terrain, but not for a realistic one.

Textures can be channel packed using the Pack Textures option in the Terrain3D Tools menu at the top of the viewport, or in tools like [Gimp](https://www.gimp.org/). Photoshop or [Krita](https://krita.org/) are possible, but working with alpha channels can be a bit challenging. See [Channel Pack Textures with Gimp](#channel-pack-textures-with-gimp) below.

### Texture Sizes
All albedo textures must be the same size, and all normal textures must be the same size. Each type gets combined into separate Texture2DArrays, so their sizes can differ.

For GPU efficiency, it is recommended that all textures have dimensions that are a power of 2 (128, 256, 512, 1024, 2048, 4096, 8192), but this isn't required.

### Compression Format

All albedo textures must be the same format, and all normal textures must be the same format. They are combined into separate Texture2DArrays, so the two can have different formats.

| Type | Supports | Format |
| - | - | - |
| **DDS** | Desktop | BC3 / DXT5, linear (intel plugin), Color + alpha, mipmaps generated. These files are used directly by Godot and are not converted, so there are no import settings. **Recommended for desktop only.** |
| **PNG** | Desktop, Mobile | Standard RGBA. In Godot you must go to the Import tab and select: `Mode: VRAM Compressed`, `Normal Map: Disabled`, `Mipmaps Generate: On`, optionally check `High Quality`, then reimport each file. 
| **Others** | | Other [Godot supported formats](https://docs.godotengine.org/en/stable/tutorials/assets_pipeline/importing_images.html#supported-image-formats) like KTX, TGA, JPG, WEBP should work as long as you match similar settings to PNG.
| **EXR** | | While EXRs can work, they store color data as 16/32-bit float, not 8-bit integer. Don't use them unless you know what you're doing.

For mobile/web platforms, see [mobile & web support](mobile_web.md).

Godot converts some texture files on import, so it is entirely possible that DDS, PNG, and TGA textures will all end up converted to DXT5 RGBA8. In this case, different file types can all be used in the same albedo or normal texture array because the compression format is the same. Double-clicking a texture in the FileSystem panel will display it in the Inspector with the current format of the file. They also need to match whether they have mipmaps or not.

DDS (BC3/DXT5) is recommended for desktop platforms as it tends to be higher quality than the default PNG to BC3/DXT5 conversion in Godot. When creating DDS files in Gimp you have a lot more conversion options, such as different mipmaps filter algorithms which can be helpful to remove artifacts in reflections (eg try Mitchell). Alternatively, you can mark PNGs as high quality which will convert them to BC6/BPTC and makes them on par with Gimp BC3/DXT5. Godot does not currently support anything higher than BC3/DXT5 in DDS files.

You can create DDS files by:
  * Exporting directly from Gimp
  * Exporting from Photoshop with [Intel's DDS plugin](https://www.intel.com/content/www/us/en/developer/articles/tool/intel-texture-works-plugin.html)
  * Converting RGBA PNGs using [NVidia's Texture Tools](https://developer.nvidia.com/nvidia-texture-tools-exporter)

You can create KTX files with [Khronos' KTX tools](https://github.com/KhronosGroup/KTX-Software/releases).

## Texture Content

### Seamless Textures

Make sure you have seamless textures that can be repeated without an obvious seam.

### Height Textures

If creating your own height textures, aim for a central point of grey (0.5) with bumps and divots above and below that value. Adjust the contrast so that troughs and peaks reach just about 0 and 1.

### Normal Map Format

Normal maps come in two formats: DirectX with -Y, and OpenGL with +Y. 

DirectX can be converted to OpenGL and vice versa by inverting the green channel in a photo editing app.

They can often be identified visually by whether bumps appear to stick out (OpenGL) or appear pushed in (DirectX). The sphere and pyramid on the left in the image below are the clearest examples. 

```{image} images/tex_normalmap.png
:target: ../_images/tex_normalmap.png
```

Natural textures like rock or grass can be very difficult to tell. However if you get assets made for a certain engine like Unreal or Unity, you can generally assume their format and convert as needed. On occasion artists get it wrong though, so if the lighting looks off on your object, try inverting the normal map.

| Software | DirectX | OpenGL |
|----------|---------|--------|
| 3DS Max				| ✓ | |
| Blender				| | ✓ |
| Cinema 4D				| | ✓ |
| CryEngine				| ✓ | |
| Godot Engine			| | ✓ |
| Houdini				| | ✓ |
| Marmoset Toolbag		| | ✓ |
| Maya					| | ✓ |
| Substance Painter		| ✓ | |
| Unity					| | ✓ |
| Unreal Engine			| ✓ | |
| Zbrush				| | ✓ |

### Roughness vs Smoothness

Some "roughness" textures are actually smoothness or gloss textures. You can convert between them by inverting the image.

You can tell which is which just by looking at distinctive textures and thinking about the material. If it's glass it should be glossy, so on a roughness texture values are near 0 and should appear mostly black. If it's dry rock or dirt, it should be mostly white, which is near 1 roughness. A smoothness texture would show the opposite. 


## Channel Pack Textures in Terrain3D

You can use our built in tool to pack textures for you.

1. At the top of your viewport, click the `Terrain3D Tools` menu, then `Pack Textures`.
2. Select your textures for albedo and height.
3. Optionally, also select textures for normal and roughness.
4. Optionally, convert a DirectX normal map to OpenGL.
5. Click `Pack Textures As...` and save the resulting PNG files to disk.
6. Go to the Import tab and one at a time, select your new PNG files, specify the following settings and click `reimport`. 
	* `Mode: VRAM Compressed`
	* Optional: `High Quality: On` if you wish BPTC instead of DXT5.
	* `Normal Map: Disabled`
	* `Mipmaps Generate: On`

Make sure to reimport both files. Double click each file in the filesystem and ensure the inspector reveals the expected format and size. All of the files you put in your texture list must match.


## Channel Pack Textures with Gimp

1. Open your RGB Albedo and greyscale Height files (or Normal and Roughness).

2. On the RGB file select `Colors/Components/Decompose`. Select `RGB`. Keep `Decompose to layers` checked. On the resulting image you have three greyscale layers for RGB. 

3. Copy the greyscale Height (or Roughness) file and paste it as a new layer into this decomposed file. Name the new layer `alpha`.

This would be a good time to invert the green channel if you need to convert a Normalmap from DirectX to OpenGL, or to invert the alpha channel if you need to convert a smoothness texture to a roughness texture.

4. Select `Colors/Components/Compose`. Select `RGBA` and ensure each named layer connects to the correct channel.

5. Now export the file with the following settings. DDS is highly recommended. 

Also recommended is to export directly into your Godot project folder. Then drag the files from the FileSystem panel into the appropriate texture slots. With this setup, you can make adjustments in Gimp and export again, and Godot will automatically update with any file changes.

### Exporting As DDS
* Change `Compression` to `BC3 / DXT5`
* `Mipmaps` to `Generate Mipmaps`. 
* Optionally, change the `Mipmap Options` `Filter`, such as to Mitchell if you get reflection artifacts in your normal+roughness texture.
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

## Where to Get Textures

### Tools

You can make textures in dedicated texture tools, such as those below. There are many other tools and ai texture generators to be found online. You can also paint textures in applications like krita/gimp/photoshop.
 
* [Materialize](http://boundingboxsoftware.com/materialize/) - Great free tool for generating missing maps. e.g. you only have an albedo texture that you love and want to generate a normal and height map
* [Material Maker](https://www.materialmaker.org/) - Free, open source material maker made in Godot
* [ArmorLab](https://armorpaint.org/) - Free & open source texture creator
* Substance Designer - Commercial, "industry standard" texture maker


### Download Sites

There are numerous websites where you can download high quality, royalty free textures for free or pay. These textures come as individual maps, with the expectation that you will download only the maps you need and then channel pack them. Here are just a few:

* [PolyHaven](https://polyhaven.com/textures) - many free textures (Download PNG, not EXR)
* [AmbientCG](https://ambientcg.com/) - many free textures
* [Poliigon](https://www.poliigon.com/textures/free) - free and commercial
* [GameTextures](https://gametextures.com/) - commercial
* [Textures](https://www.textures.com/) - commercial

## FAQ

### Why do we have to channel pack textures? Why is this so difficult?

Regular textures are now supported from v0.9.1. However if you want a realistic terrain with height blending and roughness, you need to channel pack textures.

Channel packing is a very common task done by professional game developers. Every pro asset pack you've used has channel packed textures. When you download texture packs from websites, they provide individual textures so you can pack them how you want. They are not intended to be used individually!

We offer a built in `Pack Textures` tool, found in the Terrain3D Tools menu at the top of the viewport that facilitates the texture creation process within Godot.

Finally, we provide easy, 5-step instructions for packing textures with Gimp, which takes less than 2 minutes once you're familiar with the process. 

We want high performance games, so we need to optimize our systems for the graphics hardware. A shader can retrieve four channels RGBA from a texture at once. Albedo and normal textures only have RGB. Thus, reading Alpha is free, and a waste if not used. So, we put height / roughness in the Alpha channel.

We could have the system channel pack for you at startup, however that would mean processing up to 128 images every time any scene with Terrain3D loads, both in the editor and running games. Exported games may not even work since Godot's image compression libraries only exist in the editor. The most reasonable path is for gamedevs to learn a simple process that they'll use for their entire career and use it to set up terrain textures one time.

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


