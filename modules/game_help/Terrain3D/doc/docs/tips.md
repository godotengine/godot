Tips
======

## General

* Always run Godot with the [console](troubleshooting.md#use-the-console) open so you can see errors and messages from the engine. The output panel is slow and inadequate.

* When another mesh intersects with Terrain3D far away from the camera, such as in the case of water on a beach, the two meshes can flicker as the renderer can't decide which mesh should be displayed in front. This is also called Z-fighting. You can greatly reduce it by increasing `Camera3D.near` to 0.25. You can also set it for the editor camera in the main viewport by adjusting `View/Settings/View Z-Near`.

*  Many of the brush settings can be manually entered by clicking the number next to the sliders. Some can extend beyond the maximum slider value.

## Performance
* The Terrain3DMaterial shader has some advanced features that look nice but consume some performance. You can get better performance by disabling them:
    * Set `WorldNoise` to `Flat` or `None`
	* Disable `Auto Shader`
	* Disable `Dual Scaling`
* `WorldNoise` exposes additional shader settings, such as octaves and LOD. You can adjust these settings for performance. However this world generating noise is expensive. Consider not using it at all in a commercial game, and instead obscure your background with meshes, or use an HDR skybox.
* Reduce the size of the mesh and levels of detail by reducing `Mesh/Size` (`mesh_size`) or `Mesh/Lods` (`mesh_lods`) in the `Terrain3D` node.

## Shaders

### Make a region smaller than 1024^2
Make a custom shader, then look in `vertex()` where it sets the vertex to an invalid number `VERTEX.x = 0./0.;`. Edit the conditional above it to filter out vertices that are < 0 or > 256 for instance. It will still build collision and consume memory for 1024 x 1024 maps, but this will allow you to control the visual aspect until alternate region sizes are supported.

### Regarding day/night cycles
The terrain shader is set to `cull_back`, meaning back faces are not rendered. Nor do they block light. If you have a day/night cycle and the sun sets below the horizon, it will shine through the terrain. Enable the shader override and change the second line to `cull_disabled` and the horizon will block sunlight. This does cost performance.

### Add a custom texture map

Here's an example of using a custom texture map for one texture, such as adding an emissive texture for lava. Add in this code and add an emissive texture, then adjust the emissive ID to match the lava texture, and adjust the strength.

Add the uniforms at the top of the file:
```glsl
uniform int emissive_id : hint_range(0, 31) = 0;
uniform float emissive_strength = 1.0;
uniform sampler2D emissive_tex : source_color, filter_linear_mipmap_anisotropic;
```

Modify the return struct to house the emissive texture.

```glsl
struct Material {
	...
	vec3 emissive;
};
```

Modify `get_material()` to read the emissive texture.
```glsl
// Add the initial value for emissive, adding the last vec3
out_mat = Material(vec4(0.), vec4(0.), 0, 0, 0.0, vec3(0.));

// Immediately after albedo_ht and normal_rg get assigned:
// albedo_ht = ...
// normal_rg = ...
vec4 emissive = vec4(0.);
if(out_mat.base == emissive_id) {
	emissive = texture(emissive_tex, matUV);
}

// Immediately after albedo_ht2 and normal_rg2 get assigned:
// albedo_ht2 = ...
// normal_rg2 = ...
vec4 emissive2 = vec4(0.);
emissive2 = texture(emissive_tex, matUV2) * float(out_mat.over == emissive_id);

// Immediately after the calls to height_blend:
// albedo_ht = height_blend(...
// normal_rg = height_blend(...
emissive = height_blend(emissive, albedo_ht.a, emissive2, albedo_ht2.a, out_mat.blend);

// At the bottom of the function, just before `return`.
out_mat.emissive = emissive.rgb;
```

// Then at the very bottom of `fragment()`, before the final }, apply the weighting and send it to the GPU.
```glsl
vec3 emissive = weight_inv * (
	mat[0].emissive * weights.x +
	mat[1].emissive * weights.y +
	mat[2].emissive * weights.z +
	mat[3].emissive * weights.w );
EMISSION = emissive * emissive_strength;
```

Note: Avoid sub branches: an if statement within an if statement, and enable your FPS counter so you can test as you build your code. Some branch configurations may be free, some may be very expensive, or even more performant than you expect.
