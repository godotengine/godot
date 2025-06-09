#[compute]

// NOTE Right now most lighting code is a direct copy from scene_forward_clustered.glsl but we'll never have:
// - SHADOWS_DISABLED
// - USE_LIGHTMAP
// - USE_VERTEX_LIGHTING
// - LIGHT_TRANSMITTANCE_USED
// - DEBUG_DRAW_PSSM_SPLITS
// - LIGHT_BACKLIGHT_USED
// - LIGHT_RIM_USED
// - LIGHT_CLEARCOAT_USED
// - LIGHT_ANISOTROPY_USED
// Most of these can't be implemented because they are instance based,
// some of these we may need to look into supporting.
// We should cull out all code that can't work in a deferred renderer
//
// Also all code related to `instance_index` has been commented out
// (where it isn't already excluded due to the above), this we may
// with to re-enabled by writing this value into a buffer.

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#define M_PI 3.14159265359
#define MAX_VIEWS 2

// TODO (re)introduce specialization constants for this
bool sc_use_directional_soft_shadows() {
	return false;
}
bool sc_use_lightmap_bicubic_filter() {
	return false;
}
bool sc_use_light_soft_shadows() {
	return false;
}
bool sc_use_light_projector() {
	return false;
}
bool sc_projector_use_mipmaps() {
	return false;
}
uint sc_soft_shadow_samples() {
	return 1;
}
uint sc_directional_soft_shadow_samples() {
	return 1;
}
uint sc_penumbra_shadow_samples() {
	return 1;
}
uint sc_directional_penumbra_shadow_samples() {
	return 1;
}

// Mimic raster shader variables for our lighting code
#define gl_FragCoord mimic_gl_FragCoord
vec4 mimic_gl_FragCoord;

#include "../light_data_inc.glsl"
#include "../scene_data_inc.glsl"

// Inputs
layout(set = 0, binding = 0) uniform sampler shadow_sampler;

layout(set = 0, binding = 1, std430) restrict readonly buffer OmniLights {
	LightData data[];
}
omni_lights;

layout(set = 0, binding = 2, std430) restrict readonly buffer SpotLights {
	LightData data[];
}
spot_lights;

layout(set = 0, binding = 3, std140) uniform DirectionalLights {
	DirectionalLightData data[MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS];
}
directional_lights;

layout(set = 0, binding = 4, std140) uniform SceneDataBlock {
	SceneData data;
	SceneData prev_data;
}
scene_data_block;

#define scene_data scene_data_block.data

layout(set = 0, binding = 5) uniform texture2D shadow_atlas;

layout(set = 0, binding = 6) uniform texture2D directional_shadow_atlas;

layout(set = 0, binding = 7) uniform texture2D decal_atlas;
layout(set = 0, binding = 8) uniform texture2D decal_atlas_srgb;

layout(set = 0, binding = 9) uniform sampler light_projector_sampler;

#ifdef USE_MULTIVIEW
layout(rgba16f, set = 0, binding = 10) uniform restrict readonly image2DArray albedo_buffer;
layout(rgba8, set = 0, binding = 11) uniform restrict readonly image2DArray normal_buffer;
layout(rgba8, set = 0, binding = 12) uniform restrict readonly image2DArray orm_buffer;
layout(rgba16f, set = 0, binding = 13) uniform restrict readonly image2DArray position_buffer;
#else // USE_MULTIVIEW
layout(rgba16f, set = 0, binding = 10) uniform restrict readonly image2D albedo_buffer;
layout(rgba8, set = 0, binding = 11) uniform restrict readonly image2D normal_buffer;
layout(rgba8, set = 0, binding = 12) uniform restrict readonly image2D orm_buffer;
layout(rgba16f, set = 0, binding = 13) uniform restrict readonly image2D position_buffer;
#endif

#include "../samplers_inc.glsl"

#define NO_RELECTION_PROCESS
#include "../scene_forward_lights_inc.glsl"

layout(rgba16f, set = 1, binding = 0) uniform image2D color_image;

layout(push_constant, std430) uniform Params {
	ivec2 raster_size;
	uint view;
	float pad;
}
params;

void main() {
	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = params.raster_size;

	if (uv.x >= size.x || uv.y >= size.y) {
		return;
	}

#ifdef USE_MULTIVIEW
#define ViewIndex params.view
	ivec3 uv2 = ivec3(uv, ViewIndex);
#else
#define ViewIndex 0
	ivec2 uv2 = uv;
#endif

	vec4 albedo_org = imageLoad(albedo_buffer, uv2);

	if (albedo_org.a > 0.0) { // If alpha is zero, this is an unshaded fragment, ignore.
		vec4 frag_color = imageLoad(color_image, uv);

		vec2 screen_uv = vec2(uv);
		gl_FragCoord = vec4(screen_uv / vec2(size), 0.0, 1.0);

		vec4 normal_org = imageLoad(normal_buffer, uv2);
		vec4 orm = imageLoad(orm_buffer, uv2);
		vec4 position = imageLoad(position_buffer, uv2);

		// Convert data to what we expect
		vec3 albedo = albedo_org.rgb;
		float alpha = albedo_org.a;
		vec3 vertex = position.xyz;
		vec3 normal = normalize(normal_org.xyz * 2.0 - 1.0);
		uint orms = packUnorm4x8(orm);

		float ao = orm.x;
		float roughness = orm.y;
		float metallic = orm.z;
		float specular = orm.w;
		vec3 f0 = F0(metallic, specular, albedo);

#ifdef USE_MULTIVIEW
		vec3 eye_offset = scene_data.eye_offset[ViewIndex].xyz;
		vec3 view = -normalize(vertex - eye_offset);
#else
		vec3 eye_offset = vec3(0.0, 0.0, 0.0);
		vec3 view = -normalize(vertex);
#endif //USE_MULTIVIEW

		vec3 geo_normal = normal; // Q this seems to be the normal before normal map is applied, we don't have this.
		vec3 energy_compensation = vec3(1.0); // Need to properly calculate this..

		// Our light values
		vec3 specular_light = vec3(0.0, 0.0, 0.0);
		vec3 diffuse_light = vec3(0.0, 0.0, 0.0);
		// vec3 ambient_light = vec3(0.0, 0.0, 0.0);

		{ // Directional light.

			// Do shadow and lighting in two passes to reduce register pressure.
#ifndef SHADOWS_DISABLED
			uint shadow0 = 0;
			uint shadow1 = 0;

			float shadowmask = 1.0;

#ifdef USE_LIGHTMAP
			uint shadowmask_mode = LIGHTMAP_SHADOWMASK_MODE_NONE;

			if (bool(instances.data[instance_index].flags & INSTANCE_FLAGS_USE_LIGHTMAP)) {
				const uint ofs = instances.data[instance_index].gi_offset & 0xFFFF;
				shadowmask_mode = lightmaps.data[ofs].flags;

				if (shadowmask_mode != LIGHTMAP_SHADOWMASK_MODE_NONE) {
					const uint slice = instances.data[instance_index].gi_offset >> 16;
					const vec2 scaled_uv = uv2 * instances.data[instance_index].lightmap_uv_scale.zw + instances.data[instance_index].lightmap_uv_scale.xy;
					const vec3 uvw = vec3(scaled_uv, float(slice));

					if (sc_use_lightmap_bicubic_filter()) {
						shadowmask = textureArray_bicubic(lightmap_textures[MAX_LIGHTMAP_TEXTURES + ofs], uvw, lightmaps.data[ofs].light_texture_size).x;
					} else {
						shadowmask = textureLod(sampler2DArray(lightmap_textures[MAX_LIGHTMAP_TEXTURES + ofs], SAMPLER_LINEAR_CLAMP), uvw, 0.0).x;
					}
				}
			}

			if (shadowmask_mode != LIGHTMAP_SHADOWMASK_MODE_ONLY) {
#endif // USE_LIGHTMAP

#ifdef USE_VERTEX_LIGHTING
				// Only process the first light's shadow for vertex lighting.
				for (uint i = 0; i < 1; i++) {
#else
			for (uint i = 0; i < 8; i++) {
				if (i >= scene_data.directional_light_count) {
					break;
				}
#endif

					/* TODO Disabled for now, we're doing this global so we don't have instance data here
					if (!bool(directional_lights.data[i].mask & instances.data[instance_index].layer_mask)) {
						continue; //not masked
					}

					if (directional_lights.data[i].bake_mode == LIGHT_BAKE_STATIC && bool(instances.data[instance_index].flags & INSTANCE_FLAGS_USE_LIGHTMAP)) {
						continue; // Statically baked light and object uses lightmap, skip
					}
					*/

					float shadow = 1.0;

					if (directional_lights.data[i].shadow_opacity > 0.001) {
						float depth_z = -vertex.z;
						vec3 light_dir = directional_lights.data[i].direction;
						vec3 base_normal_bias = geo_normal * (1.0 - max(0.0, dot(light_dir, -geo_normal)));

#define BIAS_FUNC(m_var, m_idx)                                                                 \
	m_var.xyz += light_dir * directional_lights.data[i].shadow_bias[m_idx];                     \
	vec3 normal_bias = base_normal_bias * directional_lights.data[i].shadow_normal_bias[m_idx]; \
	normal_bias -= light_dir * dot(light_dir, normal_bias);                                     \
	m_var.xyz += normal_bias;

						//version with soft shadows, more expensive
						if (sc_use_directional_soft_shadows() && directional_lights.data[i].softshadow_angle > 0) {
							uint blend_count = 0;
							const uint blend_max = directional_lights.data[i].blend_splits ? 2 : 1;

							if (depth_z < directional_lights.data[i].shadow_split_offsets.x) {
								vec4 v = vec4(vertex, 1.0);

								BIAS_FUNC(v, 0)

								vec4 pssm_coord = (directional_lights.data[i].shadow_matrix1 * v);
								pssm_coord /= pssm_coord.w;

								float range_pos = dot(directional_lights.data[i].direction, v.xyz);
								float range_begin = directional_lights.data[i].shadow_range_begin.x;
								float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
								vec2 tex_scale = directional_lights.data[i].uv_scale1 * test_radius;
								shadow = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale, scene_data.taa_frame_count);
								blend_count++;
							}

							if (blend_count < blend_max && depth_z < directional_lights.data[i].shadow_split_offsets.y) {
								vec4 v = vec4(vertex, 1.0);

								BIAS_FUNC(v, 1)

								vec4 pssm_coord = (directional_lights.data[i].shadow_matrix2 * v);
								pssm_coord /= pssm_coord.w;

								float range_pos = dot(directional_lights.data[i].direction, v.xyz);
								float range_begin = directional_lights.data[i].shadow_range_begin.y;
								float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
								vec2 tex_scale = directional_lights.data[i].uv_scale2 * test_radius;
								float s = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale, scene_data.taa_frame_count);

								if (blend_count == 0) {
									shadow = s;
								} else {
									//blend
									float blend = smoothstep(0.0, directional_lights.data[i].shadow_split_offsets.x, depth_z);
									shadow = mix(shadow, s, blend);
								}

								blend_count++;
							}

							if (blend_count < blend_max && depth_z < directional_lights.data[i].shadow_split_offsets.z) {
								vec4 v = vec4(vertex, 1.0);

								BIAS_FUNC(v, 2)

								vec4 pssm_coord = (directional_lights.data[i].shadow_matrix3 * v);
								pssm_coord /= pssm_coord.w;

								float range_pos = dot(directional_lights.data[i].direction, v.xyz);
								float range_begin = directional_lights.data[i].shadow_range_begin.z;
								float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
								vec2 tex_scale = directional_lights.data[i].uv_scale3 * test_radius;
								float s = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale, scene_data.taa_frame_count);

								if (blend_count == 0) {
									shadow = s;
								} else {
									//blend
									float blend = smoothstep(directional_lights.data[i].shadow_split_offsets.x, directional_lights.data[i].shadow_split_offsets.y, depth_z);
									shadow = mix(shadow, s, blend);
								}

								blend_count++;
							}

							if (blend_count < blend_max) {
								vec4 v = vec4(vertex, 1.0);

								BIAS_FUNC(v, 3)

								vec4 pssm_coord = (directional_lights.data[i].shadow_matrix4 * v);
								pssm_coord /= pssm_coord.w;

								float range_pos = dot(directional_lights.data[i].direction, v.xyz);
								float range_begin = directional_lights.data[i].shadow_range_begin.w;
								float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
								vec2 tex_scale = directional_lights.data[i].uv_scale4 * test_radius;
								float s = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale, scene_data.taa_frame_count);

								if (blend_count == 0) {
									shadow = s;
								} else {
									//blend
									float blend = smoothstep(directional_lights.data[i].shadow_split_offsets.y, directional_lights.data[i].shadow_split_offsets.z, depth_z);
									shadow = mix(shadow, s, blend);
								}
							}

						} else { //no soft shadows

							vec4 pssm_coord;
							float blur_factor;

							if (depth_z < directional_lights.data[i].shadow_split_offsets.x) {
								vec4 v = vec4(vertex, 1.0);

								BIAS_FUNC(v, 0)

								pssm_coord = (directional_lights.data[i].shadow_matrix1 * v);
								blur_factor = 1.0;
							} else if (depth_z < directional_lights.data[i].shadow_split_offsets.y) {
								vec4 v = vec4(vertex, 1.0);

								BIAS_FUNC(v, 1)

								pssm_coord = (directional_lights.data[i].shadow_matrix2 * v);
								// Adjust shadow blur with reference to the first split to reduce discrepancy between shadow splits.
								blur_factor = directional_lights.data[i].shadow_split_offsets.x / directional_lights.data[i].shadow_split_offsets.y;
							} else if (depth_z < directional_lights.data[i].shadow_split_offsets.z) {
								vec4 v = vec4(vertex, 1.0);

								BIAS_FUNC(v, 2)

								pssm_coord = (directional_lights.data[i].shadow_matrix3 * v);
								// Adjust shadow blur with reference to the first split to reduce discrepancy between shadow splits.
								blur_factor = directional_lights.data[i].shadow_split_offsets.x / directional_lights.data[i].shadow_split_offsets.z;
							} else {
								vec4 v = vec4(vertex, 1.0);

								BIAS_FUNC(v, 3)

								pssm_coord = (directional_lights.data[i].shadow_matrix4 * v);
								// Adjust shadow blur with reference to the first split to reduce discrepancy between shadow splits.
								blur_factor = directional_lights.data[i].shadow_split_offsets.x / directional_lights.data[i].shadow_split_offsets.w;
							}

							pssm_coord /= pssm_coord.w;

							shadow = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale * (blur_factor + (1.0 - blur_factor) * float(directional_lights.data[i].blend_splits)), pssm_coord, scene_data.taa_frame_count);

							if (directional_lights.data[i].blend_splits) {
								float pssm_blend;
								float blur_factor2;

								if (depth_z < directional_lights.data[i].shadow_split_offsets.x) {
									vec4 v = vec4(vertex, 1.0);
									BIAS_FUNC(v, 1)
									pssm_coord = (directional_lights.data[i].shadow_matrix2 * v);
									pssm_blend = smoothstep(directional_lights.data[i].shadow_split_offsets.x - directional_lights.data[i].shadow_split_offsets.x * 0.1, directional_lights.data[i].shadow_split_offsets.x, depth_z);
									// Adjust shadow blur with reference to the first split to reduce discrepancy between shadow splits.
									blur_factor2 = directional_lights.data[i].shadow_split_offsets.x / directional_lights.data[i].shadow_split_offsets.y;
								} else if (depth_z < directional_lights.data[i].shadow_split_offsets.y) {
									vec4 v = vec4(vertex, 1.0);
									BIAS_FUNC(v, 2)
									pssm_coord = (directional_lights.data[i].shadow_matrix3 * v);
									pssm_blend = smoothstep(directional_lights.data[i].shadow_split_offsets.y - directional_lights.data[i].shadow_split_offsets.y * 0.1, directional_lights.data[i].shadow_split_offsets.y, depth_z);
									// Adjust shadow blur with reference to the first split to reduce discrepancy between shadow splits.
									blur_factor2 = directional_lights.data[i].shadow_split_offsets.x / directional_lights.data[i].shadow_split_offsets.z;
								} else if (depth_z < directional_lights.data[i].shadow_split_offsets.z) {
									vec4 v = vec4(vertex, 1.0);
									BIAS_FUNC(v, 3)
									pssm_coord = (directional_lights.data[i].shadow_matrix4 * v);
									pssm_blend = smoothstep(directional_lights.data[i].shadow_split_offsets.z - directional_lights.data[i].shadow_split_offsets.z * 0.1, directional_lights.data[i].shadow_split_offsets.z, depth_z);
									// Adjust shadow blur with reference to the first split to reduce discrepancy between shadow splits.
									blur_factor2 = directional_lights.data[i].shadow_split_offsets.x / directional_lights.data[i].shadow_split_offsets.w;
								} else {
									pssm_blend = 0.0; //if no blend, same coord will be used (divide by z will result in same value, and already cached)
									blur_factor2 = 1.0;
								}

								pssm_coord /= pssm_coord.w;

								float shadow2 = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale * (blur_factor2 + (1.0 - blur_factor2) * float(directional_lights.data[i].blend_splits)), pssm_coord, scene_data.taa_frame_count);
								shadow = mix(shadow, shadow2, pssm_blend);
							}
						}

#ifdef USE_LIGHTMAP
						if (shadowmask_mode == LIGHTMAP_SHADOWMASK_MODE_REPLACE) {
							shadow = mix(shadow, shadowmask, smoothstep(directional_lights.data[i].fade_from, directional_lights.data[i].fade_to, vertex.z)); //done with negative values for performance
						} else if (shadowmask_mode == LIGHTMAP_SHADOWMASK_MODE_OVERLAY) {
							shadow = shadowmask * mix(shadow, 1.0, smoothstep(directional_lights.data[i].fade_from, directional_lights.data[i].fade_to, vertex.z)); //done with negative values for performance
						} else {
#endif
							shadow = mix(shadow, 1.0, smoothstep(directional_lights.data[i].fade_from, directional_lights.data[i].fade_to, vertex.z)); //done with negative values for performance
#ifdef USE_LIGHTMAP
						}
#endif

#ifdef USE_VERTEX_LIGHTING
						diffuse_light_interp *= mix(1.0, shadow, diffuse_light_interp.a);
						specular_light *= mix(1.0, shadow, specular_light_interp.a);
#endif

#undef BIAS_FUNC
					} // shadows

					if (i < 4) {
						shadow0 |= uint(clamp(shadow * 255.0, 0.0, 255.0)) << (i * 8);
					} else {
						shadow1 |= uint(clamp(shadow * 255.0, 0.0, 255.0)) << ((i - 4) * 8);
					}
				}

#ifdef USE_LIGHTMAP
			} else { // shadowmask_mode == LIGHTMAP_SHADOWMASK_MODE_ONLY

#ifdef USE_VERTEX_LIGHTING
				diffuse_light *= mix(1.0, shadowmask, diffuse_light_interp.a);
				specular_light *= mix(1.0, shadowmask, specular_light_interp.a);
#endif

				shadow0 |= uint(clamp(shadowmask * 255.0, 0.0, 255.0));
			}
#endif // USE_LIGHTMAP

#endif // SHADOWS_DISABLED

#ifndef USE_VERTEX_LIGHTING

			for (uint i = 0; i < 8; i++) {
				if (i >= scene_data.directional_light_count) {
					break;
				}

				/* TODO Disabled for now, we're doing this global so we don't have instance data here
				if (!bool(directional_lights.data[i].mask & instances.data[instance_index].layer_mask)) {
					continue; //not masked
				}

				if (directional_lights.data[i].bake_mode == LIGHT_BAKE_STATIC && bool(instances.data[instance_index].flags & INSTANCE_FLAGS_USE_LIGHTMAP)) {
					continue; // Statically baked light and object uses lightmap, skip
				}
				*/

#ifdef LIGHT_TRANSMITTANCE_USED
				float transmittance_z = transmittance_depth;
#ifndef SHADOWS_DISABLED
				if (directional_lights.data[i].shadow_opacity > 0.001) {
					float depth_z = -vertex.z;

					if (depth_z < directional_lights.data[i].shadow_split_offsets.x) {
						vec4 trans_vertex = vec4(vertex - geo_normal * directional_lights.data[i].shadow_transmittance_bias.x, 1.0);
						vec4 trans_coord = directional_lights.data[i].shadow_matrix1 * trans_vertex;
						trans_coord /= trans_coord.w;

						float shadow_z = textureLod(sampler2D(directional_shadow_atlas, SAMPLER_LINEAR_CLAMP), trans_coord.xy, 0.0).r;
						shadow_z *= directional_lights.data[i].shadow_z_range.x;
						float z = trans_coord.z * directional_lights.data[i].shadow_z_range.x;

						transmittance_z = z - shadow_z;
					} else if (depth_z < directional_lights.data[i].shadow_split_offsets.y) {
						vec4 trans_vertex = vec4(vertex - geo_normal * directional_lights.data[i].shadow_transmittance_bias.y, 1.0);
						vec4 trans_coord = directional_lights.data[i].shadow_matrix2 * trans_vertex;
						trans_coord /= trans_coord.w;

						float shadow_z = textureLod(sampler2D(directional_shadow_atlas, SAMPLER_LINEAR_CLAMP), trans_coord.xy, 0.0).r;
						shadow_z *= directional_lights.data[i].shadow_z_range.y;
						float z = trans_coord.z * directional_lights.data[i].shadow_z_range.y;

						transmittance_z = z - shadow_z;
					} else if (depth_z < directional_lights.data[i].shadow_split_offsets.z) {
						vec4 trans_vertex = vec4(vertex - geo_normal * directional_lights.data[i].shadow_transmittance_bias.z, 1.0);
						vec4 trans_coord = directional_lights.data[i].shadow_matrix3 * trans_vertex;
						trans_coord /= trans_coord.w;

						float shadow_z = textureLod(sampler2D(directional_shadow_atlas, SAMPLER_LINEAR_CLAMP), trans_coord.xy, 0.0).r;
						shadow_z *= directional_lights.data[i].shadow_z_range.z;
						float z = trans_coord.z * directional_lights.data[i].shadow_z_range.z;

						transmittance_z = z - shadow_z;
					} else {
						vec4 trans_vertex = vec4(vertex - geo_normal * directional_lights.data[i].shadow_transmittance_bias.w, 1.0);
						vec4 trans_coord = directional_lights.data[i].shadow_matrix4 * trans_vertex;
						trans_coord /= trans_coord.w;

						float shadow_z = textureLod(sampler2D(directional_shadow_atlas, SAMPLER_LINEAR_CLAMP), trans_coord.xy, 0.0).r;
						shadow_z *= directional_lights.data[i].shadow_z_range.w;
						float z = trans_coord.z * directional_lights.data[i].shadow_z_range.w;

						transmittance_z = z - shadow_z;
					}
				}
#endif // !SHADOWS_DISABLED
#endif // LIGHT_TRANSMITTANCE_USED

				float shadow = 1.0;
#ifndef SHADOWS_DISABLED
				if (i < 4) {
					shadow = float(shadow0 >> (i * 8u) & 0xFFu) / 255.0;
				} else {
					shadow = float(shadow1 >> ((i - 4u) * 8u) & 0xFFu) / 255.0;
				}

				shadow = mix(1.0, shadow, directional_lights.data[i].shadow_opacity);
#endif

				blur_shadow(shadow);

#ifdef DEBUG_DRAW_PSSM_SPLITS
				vec3 tint = vec3(1.0);
				if (-vertex.z < directional_lights.data[i].shadow_split_offsets.x) {
					tint = vec3(1.0, 0.0, 0.0);
				} else if (-vertex.z < directional_lights.data[i].shadow_split_offsets.y) {
					tint = vec3(0.0, 1.0, 0.0);
				} else if (-vertex.z < directional_lights.data[i].shadow_split_offsets.z) {
					tint = vec3(0.0, 0.0, 1.0);
				} else {
					tint = vec3(1.0, 1.0, 0.0);
				}
				tint = mix(tint, vec3(1.0), shadow);
				shadow = 1.0;
#endif

				float size_A = sc_use_directional_soft_shadows() ? directional_lights.data[i].size : 0.0;

				light_compute(normal, directional_lights.data[i].direction, normalize(view), size_A,
#ifndef DEBUG_DRAW_PSSM_SPLITS
						directional_lights.data[i].color * directional_lights.data[i].energy,
#else
						directional_lights.data[i].color * directional_lights.data[i].energy * tint,
#endif
						true, shadow, f0, orms, directional_lights.data[i].specular, albedo, alpha, screen_uv, energy_compensation,
#ifdef LIGHT_BACKLIGHT_USED
						backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
						transmittance_color,
						transmittance_depth,
						transmittance_boost,
						transmittance_z,
#endif
#ifdef LIGHT_RIM_USED
						rim, rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
						clearcoat, clearcoat_roughness, geo_normal,
#endif // LIGHT_CLEARCOAT_USED
#ifdef LIGHT_ANISOTROPY_USED
						binormal,
						tangent, anisotropy,
#endif
						diffuse_light,
						specular_light);
			}
#endif // USE_VERTEX_LIGHTING
		}

		// multiply by albedo
		diffuse_light *= albedo; // ambient must be multiplied by albedo at the end

		// apply direct light AO
		ao = unpackUnorm4x8(orms).x;
		specular_light *= ao;
		diffuse_light *= ao;

		// apply metallic
		metallic = unpackUnorm4x8(orms).z;
		diffuse_light *= 1.0 - metallic;
		// ambient_light *= 1.0 - metallic;

		// Add our new colors in...
		// frag_color.rgb += ambient_light + diffuse_light + specular_light;
		frag_color.rgb += diffuse_light + specular_light;

		// Write back
		imageStore(color_image, uv, frag_color);
	}
}
