Importing & Exporting Data
===========================

Currently importing and exporting is possible via code or our import tool. We will [make a UI](https://github.com/TokisanGames/Terrain3D/issues/81)  eventually. In the meantime, we have written a script that uses the Godot Inspector as a makeshift UI. You can use it to make a data file for your other scenes.

## Importing Data

1) Open `addons/terrain_3d/tools/importer.tscn`.

2) Click Importer in the scene tree.

```{image} images/io_importer.png
:target: ../_images/io_importer.png
```

3) In the inspector, select a file for height, control, and/or color maps. See [formats](#supported-import-formats) below. File type is determined by extension.

4) Specify the `import_position` of where in the world you want to import. Values are rounded to the nearest `region_size` (defaults to 1024). So a location of (-2000, 1000) will be imported at (-2048, 1024).

     Notes:
     * You can import multiple times into the greater 16k^2 world map by specifying different positions. So you could import multiple maps as separate islands or combined regions.
     * It will slice and pad odd sized images into region sized chunks ([default is 1024x1024](https://github.com/TokisanGames/Terrain3D/issues/77)). e.g. You could import a 4k x 2k, several 1k x 1ks, and a 5123 x 3769 and position them so they are adjacent.
     * You can also reimport to the same location to overwrite anything there using individual maps or a complete set of height, control, and/or color.

5) Specify any desired `height_offset` or `import_scale`. The scale gets applied first. (eg. 100, -100 would scale the terrain by 100, then lower the whole terrain by 100).

     * Note that we store full range values. If you sculpt a hill to a height of 50, that's what goes into the data file. Your heightmap values (esp w/ EXR) may be normalized to the range of 0-1. If you import and the terrain is still flat, try scaling the height up by 300-500.

6) If you have a RAW or R16 file (same thing), it should have an extension of `r16` or `raw`. You can specify the height range and dimensions next. These are not stored in the file so you must know them. I prefer to place them in the filename.

7) Click `Run Import` and wait 10-30 seconds. Look at the console for activity or errors. If the `Terrain3D.debug_level` is set to `debug`, you'll also see progress.

8) When you are happy with the import, scroll down in the inspector (half of it is hidden by the `Textures` panel) until you see `Terrain3D, Storage`.

9) Click the right arrow next to the Terrain3DStorage file and save the file wherever you wish. Make sure to save it as `.res` which is a binary Godot resource file. 

```{image} images/io_save_storage.png
:target: ../_images/io_save_storage.png
```

You can now load this `res` file into a Terrain3D in any of your scenes. You can also preload an existing storage file in the importer, then import more data into it and save it again.

## Supported Import Formats

We can import any supported image format Godot can read. These include:
* [Image formats for external files](https://docs.godotengine.org/en/stable/tutorials/assets_pipeline/importing_images.html#supported-image-formats): `bmp`, `dds`, `exr`, `hdr`, `jpg`, `jpeg`, `png`, `tga`, `svg`, `webp`
* [Image formats stored in a Godot resource file](https://docs.godotengine.org/en/stable/classes/class_image.html#enum-image-format): `tres`, `res`
* R16 Height map aka RAW. See below.

### Height map
* Only `exr` or `r16/raw` are recommended for heightmaps. Godot PNG only supports 8-bit per channel, so don't use it for heightmaps. It is fine for external editing of the color map which is RGBA. See [Terrain3DStorage](../api/class_terrain3dstorage.rst) for details on internal storage.
* R16: For 16-bit heightmaps read/writable by World Machine, Unity, Krita, etc. The extension should be `r16` or `raw`. Min/Max heights and image size are not stored in the file, so you must keep track of them elsewhere (such as in the name).
* `Photoshop Raw` is not R16. Use [exr](https://www.exr-io.com/) for photoshop.
* [Zylann's HTerrain](https://github.com/Zylann/godot_heightmap_plugin/) stores height data in a `res` file which we can import directly. No need to export it, but his tool also exports `exr` and `r16`.

### Control map
* Control maps use a proprietary format. We only import our own format. Use `exr` to export or reimport only from this tool.

### Color map
* Any regular color format is fine, `png` is recommended. The alpha channel is interpretted as a [roughness modifier](../api/class_terrain3dstorage.rst#class-terrain3dstorage-property-color-maps) for wetness.


## Exporting Data

1) Open `addons/terrain_3d/tools/importer.tscn`.

2) Click Importer in the scene tree.

3) Scroll the inspector down to Terrain3D, Storage. Click the right arrow, and load the storage file you wish to export from.

```{image} images/io_load_storage.png
:target: ../_images/io_load_storage.png
```

4) Scroll the inspector to `Export File`.

```{image} images/io_exporter.png
:target: ../_images/io_exporter.png
```

5) Select the type of map you wish to extract: Height (32-bit floats), Color (rgba), Control (proprietary).

6) Specify the full path and file name to save. The file type is determined based upon the extension. You can enter any location on your hard drive, or preface the file name with `res://` to save it in your Godot project folder. See [export formats](#supported-export-formats) for recommendations.

7) Click `Run Export` and wait. 10-30s is normal. Look at your file system or the console for status.

Notes:

* The exporter takes the smallest rectangle that will fit around all active regions in the 16k^2 world and export that as an image. So, if you have a 1k x 1k island in the NW corner, and a 2k x 3k island in the center, with a 1k strait between them, the resulting export image will be something like 4k x 5k. You'll need to specify the location (rounded to `region_size`) when reimporting to have a perfect round trip.

* The exporter tool does not offer region by region export, but there is an API where you can retrieve any given region, then you can use `Image` to save it externally yourself.

## Supported Export Formats

We can export any supported image format Godot can write. These include:
* [Image save functions for external files](https://docs.godotengine.org/en/stable/classes/class_image.html#class-image-method-save-exr): `exr`, `png`, `jpg`, `webp`
* Images stored in a Godot resource file: `tres` for text, `res` for binary (with `ResourceSaver::FLAG_COMPRESS` enabled)
* R16 Height map aka RAW. See below.

### Height map
* Use `exr` or `r16/raw` for external tools, or `res` for Godot only use. Godot PNG only supports 8-bit per channel, so it will give you blocky heightmaps.
* R16: For 16-bit heightmaps read/writable by World Machine, Unity, Krita, etc. Save with the extension `r16` or `raw`. Min/Max heights and image size are not stored in the file, so you must keep track of them elsewhere. See below to acquire the dimensions. 
* `Photoshop Raw` is not raw, don't use it. Use [exr](https://www.exr-io.com/) for photoshop.

### Control map
* Control maps use a proprietary format. We only import our own. Use `exr`. It won't give you a valid image editable in other tools. This is only for transferring the image to another Terrain3D Storage file. See [Controlmap Format](controlmap_format.md).

### Color map
* Use `png` or `webp`, as they are lossless rgba formats that external tools can edit. Use `res` for Godot only use. The alpha channel is interpretted as a [roughness modifier](../api/class_terrain3dstorage.rst#class-terrain3dstorage-property-color-maps) for wetness. 


## Exported Image Dimensions

Upon export, the console reports the image size.

You can get the height of the data by clicking `Update Height Range`, then looking in the read only data of the storage file.

```{image} images/io_height_range.png
:target: ../_images/io_height_range.png
```