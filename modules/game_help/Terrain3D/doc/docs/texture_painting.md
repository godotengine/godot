Texturing the Terrain
=========================

## Texture List

Terrain3D supports up to 32 texture sets.

### Installing a Texture
1. Once you've [created your textures](texture_prep.md), place them in your Godot project folder.
2. Set the appropriate Import settings for them as defined in [compression formats](texture_prep.md#compression-format).
3. Make a new texture slot in the `Textures` panel by clicking `Add New`. 
4. Drag your texture file for albedo+height from the `FileSystem` panel into the albedo slot. Drag your normal+roughness texture into the normal slot. 
5. In the inspector, name the texture and adjust the other settings as needed.


### Managing the Texture List
* Unused texture slots take up memory with the default generated textures. Remove unused slots.
* Right-click any texture slot in the panel to bring it into edit mode.
* Middle-click any texture slot to clear or delete it. You can only delete the last texture in the list.
* Reorder textures by changing the texture id. This will change the texture rendered in the viewport as it does not change the values painted on the control map. In the future we'll add image processing tools that will allow changing texture ids painted on the terrain.


## Texture Painting

The idea for texturing the terrain is to enable the autoshader to automatically texture the terrain in most areas, then manually paint textures only where you need it.

There are a handful of tools to familiarize yourself with:
* The toolbar has a `Paint Base Texture`, `Spray Overlay Texture`, and `Autoshader` tools to paint where the terrain is manually or automatically textured.
* The material has an option to enable or disable the autoshader for the whole terrain.
* The `material/debug view/autoshader` displays where the terrain is automatically or manually textured.


### Manual Painting Technique

In an area where autoshading has been disabled, each vertex has a base texture, an overlay texture, and a blending value. If the autoshader is enabled, painting textures will disable the autoshader in that area. 

The following technique will allow you to achieve a natural result.

* Use the `Paint Base Texture` tool to cover large sections with the base texture. This tool also clears the blend value. You can and should paint similar but different textures in an area for a natural variety. e.g. gravel and dirt; mud, dirt, and rocks.
* Use the `Spray Overlay Texture` tool to blend the edges of the base textures to give it a natural look. This sets the overlay texture, replacing whatever was there before. It also increases the blend value, meaning "show more of the overlay texture". If you spray the same texture as the base, it will decrease the blend value instead.
* Example: Use the Paint tool for both a grass field and a dirt pathway. Then use the Spray tool and repeatedly switch between grass and dirt to blend the edges of the path randomly until it looks realistic.
* Use the [control texture](../api/class_terrain3dmaterial.rst#class-terrain3dmaterial-property-show-control-texture) and [control blend](../api/class_terrain3dmaterial.rst#class-terrain3dmaterial-property-show-control-blend) debug views to understand how your textures are painted and blended. 


### Autoshading
New regions are set to enable the autoshader by default. If you started using Terrain3D before the autoshader, all of your regions are set to manual shading. To enable it:
* In the material, enable the autoshader.
* Specify the base and overlay texture IDs the autoshader should use.
* Use the `Autoshader` tool and paint over the areas you want to be autoshaded.

Enabling the autoshader will not change your manually painted textures. In autoshaded regions, the shader will ignore your manual painting, but it's still there and will be visible any time you disable the autoshader in the material or by painting.


### Mixing Manual Painting & the Autoshader
Since the paint brushes will disable the autoshader, it's easy to make artifacts appear without the right process. This technique will allow you to achieve seamless painting using both the autoshader and manual painting. 

Let's say we want to paint a pathway through grass, surrounded by autoshaded hills:

* Find a flat or evenly sloped area with a uniform texture. Avoid corners where the autoshader transitions between textures. In our example, find a flat grassy area.
* Use the `Paint Base Texture` tool to paint the same grass texture the autoshader is using in the flat area. The changes won't be visible unless you disable the autoshader in the material, or enable the autoshader debug view. But you will be simultanously disabling the painted autoshader, and painting the same texture.
* Use the same tool to paint a pathway texture.
* Use the `Spray Overlay Texture` tool to blend the edges.

You can see the manual painting technique is the same as above. The key step here is invisibly carving out a manually painted section by laying down a base texture the same as the autoshader. If you go too far out with the manual painting, you can use the `Autoshader` tool to bring it back in.


## Painting Angle & Scale

Both the `Paint Base Texture` and `Spray Overlay Texture` tools have Angle and Scale modifiers, which allow painting textures at different angles and scales. This can be useful for creating paths, water embankments, and generally having textures more closely follow terrain features.

These tools have pickers allowing you to select current values from the terrain. Angle also has a `Dynamic` mode which causes the angle to change based upon mouse movement while painting. This ignores any value set by the slider.

Paint/Spray brushes have toggleable options for Texture, Angle, and Scale, meaning they can be independently applied. This allows you to disable Texture and Scale in order to repaint an area using only Angle modification. This will change texture rotation without affecting the other parameters.

You can paint Angle and Scale on top of the autoshader.


## Color Painting

In addition to painting textures, you can also paint colors on the terrain. There are two primary uses for the colormap.


### 1. GIS Applications

You can import a full image such as a satellite photo, and enable the color map debug view for GIS visualization.

```{image} images/gis.png
:target: ../_images/gis.png
```


### 2. Color Variation

You can use the `Paint Color` tool to paint colors on the terrain. This is useful to add variation. Colors are multiplied on to painted textures, which is a blend mode that only darkens.

Try painting your terrain with subtle light grays, greens and browns to add depth, contours, and variation. Subtlety is key.

Paint white to reset.

You can paint with the colormap on top of the autoshader. You can also use the picker to select a colormap value from the terrain.


## Painting Wetness

Use the `Paint Wetness` tool to modify the roughness of the textures. Reduce the roughness percentage to say -30% and wherever you paint the textures will become more glossy.

If you wish to turn dirt into mud, try painting a light/medium grey on the colormap to darken it, then paint a -30% on the wetness.

Paint 0 to reset.

You can paint wetness on top of the autoshader. You can also use the picker to select a wetness value from the terrain.
