#[vertex]

#version 450

#VERSION_DEFINES

#include "scene_forward_clustered_inc.glsl"

/* INPUT ATTRIBS */

layout(location = 0) in vec3 vertex_attrib;

//only for pure render depth when normal is not used

#ifdef NORMAL_USED
layout(location = 1) in vec3 normal_attrib;
#endif

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)
layout(location = 2) in vec4 tangent_attrib;
#endif

#if defined(COLOR_USED)
layout(location = 3) in vec4 color_attrib;
#endif

#ifdef UV_USED
layout(location = 4) in vec2 uv_attrib;
#endif

#if defined(UV2_USED) || defined(USE_LIGHTMAP) || defined(MODE_RENDER_MATERIAL)
layout(location = 5) in vec2 uv2_attrib;
#endif

#if defined(CUSTOM0_USED)
layout(location = 6) in vec4 custom0_attrib;
#endif

#if defined(CUSTOM1_USED)
layout(location = 7) in vec4 custom1_attrib;
#endif

#if defined(CUSTOM2_USED)
layout(location = 8) in vec4 custom2_attrib;
#endif

#if defined(CUSTOM3_USED)
layout(location = 9) in vec4 custom3_attrib;
#endif

#if defined(BONES_USED) || defined(USE_PARTICLE_TRAILS)
layout(location = 10) in uvec4 bone_attrib;
#endif

#if defined(WEIGHTS_USED) || defined(USE_PARTICLE_TRAILS)
layout(location = 11) in vec4 weight_attrib;
#endif

/* Varyings */

layout(location = 0) out vec3 vertex_interp;

#ifdef NORMAL_USED
layout(location = 1) out vec3 normal_interp;
#endif

#if defined(COLOR_USED)
layout(location = 2) out vec4 color_interp;
#endif

#ifdef UV_USED
layout(location = 3) out vec2 uv_interp;
#endif

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
layout(location = 4) out vec2 uv2_interp;
#endif

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)
layout(location = 5) out vec3 tangent_interp;
layout(location = 6) out vec3 binormal_interp;
#endif

#ifdef MATERIAL_UNIFORMS_USED
layout(set = MATERIAL_UNIFORM_SET, binding = 0, std140) uniform MaterialUniforms{

#MATERIAL_UNIFORMS

} material;
#endif

#ifdef MODE_DUAL_PARABOLOID

layout(location = 8) out float dp_clip;

#endif

layout(location = 9) out flat uint instance_index_interp;

invariant gl_Position;

#GLOBALS

void main() {
	vec4 instance_custom = vec4(0.0);
#if defined(COLOR_USED)
	color_interp = color_attrib;
#endif

	uint instance_index = draw_call.instance_index;
	instance_index_interp = instance_index;

	bool is_multimesh = bool(instances.data[instance_index].flags & INSTANCE_FLAGS_MULTIMESH);
	if (!is_multimesh) {
		instance_index += gl_InstanceIndex;
	}

	mat4 world_matrix = instances.data[instance_index].transform;

	mat3 world_normal_matrix;
	if (bool(instances.data[instance_index].flags & INSTANCE_FLAGS_NON_UNIFORM_SCALE)) {
		world_normal_matrix = transpose(inverse(mat3(world_matrix)));
	} else {
		world_normal_matrix = mat3(world_matrix);
	}

	if (is_multimesh) {
		//multimesh, instances are for it

		mat4 matrix;

#ifdef USE_PARTICLE_TRAILS
		uint trail_size = (instances.data[instance_index].flags >> INSTANCE_FLAGS_PARTICLE_TRAIL_SHIFT) & INSTANCE_FLAGS_PARTICLE_TRAIL_MASK;
		uint stride = 3 + 1 + 1; //particles always uses this format

		uint offset = trail_size * stride * gl_InstanceIndex;

#ifdef COLOR_USED
		vec4 pcolor;
#endif
		{
			uint boffset = offset + bone_attrib.x * stride;
			matrix = mat4(transforms.data[boffset + 0], transforms.data[boffset + 1], transforms.data[boffset + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weight_attrib.x;
#ifdef COLOR_USED
			pcolor = transforms.data[boffset + 3] * weight_attrib.x;
#endif
		}
		if (weight_attrib.y > 0.001) {
			uint boffset = offset + bone_attrib.y * stride;
			matrix += mat4(transforms.data[boffset + 0], transforms.data[boffset + 1], transforms.data[boffset + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weight_attrib.y;
#ifdef COLOR_USED
			pcolor += transforms.data[boffset + 3] * weight_attrib.y;
#endif
		}
		if (weight_attrib.z > 0.001) {
			uint boffset = offset + bone_attrib.z * stride;
			matrix += mat4(transforms.data[boffset + 0], transforms.data[boffset + 1], transforms.data[boffset + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weight_attrib.z;
#ifdef COLOR_USED
			pcolor += transforms.data[boffset + 3] * weight_attrib.z;
#endif
		}
		if (weight_attrib.w > 0.001) {
			uint boffset = offset + bone_attrib.w * stride;
			matrix += mat4(transforms.data[boffset + 0], transforms.data[boffset + 1], transforms.data[boffset + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weight_attrib.w;
#ifdef COLOR_USED
			pcolor += transforms.data[boffset + 3] * weight_attrib.w;
#endif
		}

		instance_custom = transforms.data[offset + 4];

#ifdef COLOR_USED
		color_interp *= pcolor;
#endif

#else
		uint stride = 0;
		{
			//TODO implement a small lookup table for the stride
			if (bool(instances.data[instance_index].flags & INSTANCE_FLAGS_MULTIMESH_FORMAT_2D)) {
				stride += 2;
			} else {
				stride += 3;
			}
			if (bool(instances.data[instance_index].flags & INSTANCE_FLAGS_MULTIMESH_HAS_COLOR)) {
				stride += 1;
			}
			if (bool(instances.data[instance_index].flags & INSTANCE_FLAGS_MULTIMESH_HAS_CUSTOM_DATA)) {
				stride += 1;
			}
		}

		uint offset = stride * gl_InstanceIndex;

		if (bool(instances.data[instance_index].flags & INSTANCE_FLAGS_MULTIMESH_FORMAT_2D)) {
			matrix = mat4(transforms.data[offset + 0], transforms.data[offset + 1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0));
			offset += 2;
		} else {
			matrix = mat4(transforms.data[offset + 0], transforms.data[offset + 1], transforms.data[offset + 2], vec4(0.0, 0.0, 0.0, 1.0));
			offset += 3;
		}

		if (bool(instances.data[instance_index].flags & INSTANCE_FLAGS_MULTIMESH_HAS_COLOR)) {
#ifdef COLOR_USED
			color_interp *= transforms.data[offset];
#endif
			offset += 1;
		}

		if (bool(instances.data[instance_index].flags & INSTANCE_FLAGS_MULTIMESH_HAS_CUSTOM_DATA)) {
			instance_custom = transforms.data[offset];
		}

#endif
		//transpose
		matrix = transpose(matrix);
		world_matrix = world_matrix * matrix;
		world_normal_matrix = world_normal_matrix * mat3(matrix);
	}

	vec3 vertex = vertex_attrib;
#ifdef NORMAL_USED
	vec3 normal = normal_attrib * 2.0 - 1.0;
#endif

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)
	vec3 tangent = tangent_attrib.xyz * 2.0 - 1.0;
	float binormalf = tangent_attrib.a * 2.0 - 1.0;
	vec3 binormal = normalize(cross(normal, tangent) * binormalf);
#endif

#ifdef UV_USED
	uv_interp = uv_attrib;
#endif

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
	uv2_interp = uv2_attrib;
#endif

#ifdef OVERRIDE_POSITION
	vec4 position;
#endif

	mat4 projection_matrix = scene_data.projection_matrix;

//using world coordinates
#if !defined(SKIP_TRANSFORM_USED) && defined(VERTEX_WORLD_COORDS_USED)

	vertex = (world_matrix * vec4(vertex, 1.0)).xyz;

	normal = world_normal_matrix * normal;

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)

	tangent = world_normal_matrix * tangent;
	binormal = world_normal_matrix * binormal;

#endif
#endif

	float roughness = 1.0;

	mat4 modelview = scene_data.inv_camera_matrix * world_matrix;
	mat3 modelview_normal = mat3(scene_data.inv_camera_matrix) * world_normal_matrix;

	{
#CODE : VERTEX
	}

// using local coordinates (default)
#if !defined(SKIP_TRANSFORM_USED) && !defined(VERTEX_WORLD_COORDS_USED)

	vertex = (modelview * vec4(vertex, 1.0)).xyz;
#ifdef NORMAL_USED
	normal = modelview_normal * normal;
#endif

#endif

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)

	binormal = modelview_normal * binormal;
	tangent = modelview_normal * tangent;
#endif

//using world coordinates
#if !defined(SKIP_TRANSFORM_USED) && defined(VERTEX_WORLD_COORDS_USED)

	vertex = (scene_data.inv_camera_matrix * vec4(vertex, 1.0)).xyz;
	normal = mat3(scene_data.inverse_normal_matrix) * normal;

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)

	binormal = mat3(scene_data.camera_inverse_binormal_matrix) * binormal;
	tangent = mat3(scene_data.camera_inverse_tangent_matrix) * tangent;
#endif
#endif

	vertex_interp = vertex;
#ifdef NORMAL_USED
	normal_interp = normal;
#endif

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)
	tangent_interp = tangent;
	binormal_interp = binormal;
#endif

#ifdef MODE_RENDER_DEPTH

#ifdef MODE_DUAL_PARABOLOID

	vertex_interp.z *= scene_data.dual_paraboloid_side;

	dp_clip = vertex_interp.z; //this attempts to avoid noise caused by objects sent to the other parabolloid side due to bias

	//for dual paraboloid shadow mapping, this is the fastest but least correct way, as it curves straight edges

	vec3 vtx = vertex_interp;
	float distance = length(vtx);
	vtx = normalize(vtx);
	vtx.xy /= 1.0 - vtx.z;
	vtx.z = (distance / scene_data.z_far);
	vtx.z = vtx.z * 2.0 - 1.0;
	vertex_interp = vtx;

#endif

#endif //MODE_RENDER_DEPTH

#ifdef OVERRIDE_POSITION
	gl_Position = position;
#else
	gl_Position = projection_matrix * vec4(vertex_interp, 1.0);
#endif

#ifdef MODE_RENDER_DEPTH
	if (scene_data.pancake_shadows) {
		if (gl_Position.z <= 0.00001) {
			gl_Position.z = 0.00001;
		}
	}
#endif
#ifdef MODE_RENDER_MATERIAL
	if (scene_data.material_uv2_mode) {
		vec2 uv_offset = unpackHalf2x16(draw_call.uv_offset);
		gl_Position.xy = (uv2_attrib.xy + uv_offset) * 2.0 - 1.0;
		gl_Position.z = 0.00001;
		gl_Position.w = 1.0;
	}
#endif
}

#[fragment]

#version 450

#VERSION_DEFINES

/* Specialization Constants (Toggles) */

layout(constant_id = 0) const bool sc_use_forward_gi = false;
layout(constant_id = 1) const bool sc_use_light_projector = false;
layout(constant_id = 2) const bool sc_use_light_soft_shadows = false;
layout(constant_id = 3) const bool sc_use_directional_soft_shadows = false;

/* Specialization Constants (Values) */

layout(constant_id = 6) const uint sc_soft_shadow_samples = 4;
layout(constant_id = 7) const uint sc_penumbra_shadow_samples = 4;

layout(constant_id = 8) const uint sc_directional_soft_shadow_samples = 4;
layout(constant_id = 9) const uint sc_directional_penumbra_shadow_samples = 4;

layout(constant_id = 10) const bool sc_decal_use_mipmaps = true;
layout(constant_id = 11) const bool sc_projector_use_mipmaps = true;

// not used in clustered renderer but we share some code with the mobile renderer that requires this.
const float sc_luminance_multiplier = 1.0;

#include "scene_forward_clustered_inc.glsl"

/* Varyings */

layout(location = 0) in vec3 vertex_interp;

#ifdef NORMAL_USED
layout(location = 1) in vec3 normal_interp;
#endif

#if defined(COLOR_USED)
layout(location = 2) in vec4 color_interp;
#endif

#ifdef UV_USED
layout(location = 3) in vec2 uv_interp;
#endif

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
layout(location = 4) in vec2 uv2_interp;
#endif

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)
layout(location = 5) in vec3 tangent_interp;
layout(location = 6) in vec3 binormal_interp;
#endif

#ifdef MODE_DUAL_PARABOLOID

layout(location = 8) in float dp_clip;

#endif

layout(location = 9) in flat uint instance_index_interp;

//defines to keep compatibility with vertex

#define world_matrix instances.data[instance_index].transform
#define projection_matrix scene_data.projection_matrix

#if defined(ENABLE_SSS) && defined(ENABLE_TRANSMITTANCE)
//both required for transmittance to be enabled
#define LIGHT_TRANSMITTANCE_USED
#endif

#ifdef MATERIAL_UNIFORMS_USED
layout(set = MATERIAL_UNIFORM_SET, binding = 0, std140) uniform MaterialUniforms{

#MATERIAL_UNIFORMS

} material;
#endif

#GLOBALS

#ifdef MODE_RENDER_DEPTH

#ifdef MODE_RENDER_MATERIAL

layout(location = 0) out vec4 albedo_output_buffer;
layout(location = 1) out vec4 normal_output_buffer;
layout(location = 2) out vec4 orm_output_buffer;
layout(location = 3) out vec4 emission_output_buffer;
layout(location = 4) out float depth_output_buffer;

#endif // MODE_RENDER_MATERIAL

#ifdef MODE_RENDER_NORMAL_ROUGHNESS
layout(location = 0) out vec4 normal_roughness_output_buffer;

#ifdef MODE_RENDER_VOXEL_GI
layout(location = 1) out uvec2 voxel_gi_buffer;
#endif

#endif //MODE_RENDER_NORMAL
#else // RENDER DEPTH

#ifdef MODE_MULTIPLE_RENDER_TARGETS

layout(location = 0) out vec4 diffuse_buffer; //diffuse (rgb) and roughness
layout(location = 1) out vec4 specular_buffer; //specular and SSS (subsurface scatter)
#else

layout(location = 0) out vec4 frag_color;
#endif // MODE_MULTIPLE_RENDER_TARGETS

#endif // RENDER DEPTH

#include "scene_forward_aa_inc.glsl"

#if !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

/* Make a default specular mode SPECULAR_SCHLICK_GGX. */
#if !defined(SPECULAR_DISABLED) && !defined(SPECULAR_SCHLICK_GGX) && !defined(SPECULAR_BLINN) && !defined(SPECULAR_PHONG) && !defined(SPECULAR_TOON)
#define SPECULAR_SCHLICK_GGX
#endif

#include "scene_forward_lights_inc.glsl"

#include "scene_forward_gi_inc.glsl"

#endif //!defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

#ifndef MODE_RENDER_DEPTH

vec4 volumetric_fog_process(vec2 screen_uv, float z) {
	vec3 fog_pos = vec3(screen_uv, z * scene_data.volumetric_fog_inv_length);
	if (fog_pos.z < 0.0) {
		return vec4(0.0);
	} else if (fog_pos.z < 1.0) {
		fog_pos.z = pow(fog_pos.z, scene_data.volumetric_fog_detail_spread);
	}

	return texture(sampler3D(volumetric_fog_texture, material_samplers[SAMPLER_LINEAR_CLAMP]), fog_pos);
}

vec4 fog_process(vec3 vertex) {
	vec3 fog_color = scene_data.fog_light_color;

	if (scene_data.fog_aerial_perspective > 0.0) {
		vec3 sky_fog_color = vec3(0.0);
		vec3 cube_view = scene_data.radiance_inverse_xform * vertex;
		// mip_level always reads from the second mipmap and higher so the fog is always slightly blurred
		float mip_level = mix(1.0 / MAX_ROUGHNESS_LOD, 1.0, 1.0 - (abs(vertex.z) - scene_data.z_near) / (scene_data.z_far - scene_data.z_near));
#ifdef USE_RADIANCE_CUBEMAP_ARRAY
		float lod, blend;
		blend = modf(mip_level * MAX_ROUGHNESS_LOD, lod);
		sky_fog_color = texture(samplerCubeArray(radiance_cubemap, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), vec4(cube_view, lod)).rgb;
		sky_fog_color = mix(sky_fog_color, texture(samplerCubeArray(radiance_cubemap, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), vec4(cube_view, lod + 1)).rgb, blend);
#else
		sky_fog_color = textureLod(samplerCube(radiance_cubemap, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), cube_view, mip_level * MAX_ROUGHNESS_LOD).rgb;
#endif //USE_RADIANCE_CUBEMAP_ARRAY
		fog_color = mix(fog_color, sky_fog_color, scene_data.fog_aerial_perspective);
	}

	if (scene_data.fog_sun_scatter > 0.001) {
		vec4 sun_scatter = vec4(0.0);
		float sun_total = 0.0;
		vec3 view = normalize(vertex);

		for (uint i = 0; i < scene_data.directional_light_count; i++) {
			vec3 light_color = directional_lights.data[i].color * directional_lights.data[i].energy;
			float light_amount = pow(max(dot(view, directional_lights.data[i].direction), 0.0), 8.0);
			fog_color += light_color * light_amount * scene_data.fog_sun_scatter;
		}
	}

	float fog_amount = 1.0 - exp(min(0.0, -length(vertex) * scene_data.fog_density));

	if (abs(scene_data.fog_height_density) >= 0.0001) {
		float y = (scene_data.camera_matrix * vec4(vertex, 1.0)).y;

		float y_dist = y - scene_data.fog_height;

		float vfog_amount = 1.0 - exp(min(0.0, y_dist * scene_data.fog_height_density));

		fog_amount = max(vfog_amount, fog_amount);
	}

	return vec4(fog_color, fog_amount);
}

void cluster_get_item_range(uint p_offset, out uint item_min, out uint item_max, out uint item_from, out uint item_to) {
	uint item_min_max = cluster_buffer.data[p_offset];
	item_min = item_min_max & 0xFFFF;
	item_max = item_min_max >> 16;
	;

	item_from = item_min >> 5;
	item_to = (item_max == 0) ? 0 : ((item_max - 1) >> 5) + 1; //side effect of how it is stored, as item_max 0 means no elements
}

uint cluster_get_range_clip_mask(uint i, uint z_min, uint z_max) {
	int local_min = clamp(int(z_min) - int(i) * 32, 0, 31);
	int mask_width = min(int(z_max) - int(z_min), 32 - local_min);
	return bitfieldInsert(uint(0), uint(0xFFFFFFFF), local_min, mask_width);
}

#endif //!MODE_RENDER DEPTH

void main() {
#ifdef MODE_DUAL_PARABOLOID

	if (dp_clip > 0.0)
		discard;
#endif

#ifdef USE_SUBGROUPS
	//ensures instance_index is in sgpr
	uint instance_index = subgroupBroadcastFirst(instance_index_interp);
#else
	uint instance_index = instance_index_interp;
#endif
	//lay out everything, whathever is unused is optimized away anyway
	vec3 vertex = vertex_interp;
	vec3 view = -normalize(vertex_interp);
	vec3 albedo = vec3(1.0);
	vec3 backlight = vec3(0.0);
	vec4 transmittance_color = vec4(0.0, 0.0, 0.0, 1.0);
	float transmittance_depth = 0.0;
	float transmittance_boost = 0.0;
	float metallic = 0.0;
	float specular = 0.5;
	vec3 emission = vec3(0.0);
	float roughness = 1.0;
	float rim = 0.0;
	float rim_tint = 0.0;
	float clearcoat = 0.0;
	float clearcoat_gloss = 0.0;
	float anisotropy = 0.0;
	vec2 anisotropy_flow = vec2(1.0, 0.0);
	vec4 fog = vec4(0.0);
#if defined(CUSTOM_RADIANCE_USED)
	vec4 custom_radiance = vec4(0.0);
#endif
#if defined(CUSTOM_IRRADIANCE_USED)
	vec4 custom_irradiance = vec4(0.0);
#endif

	float ao = 1.0;
	float ao_light_affect = 0.0;

	float alpha = float(instances.data[instance_index].flags >> INSTANCE_FLAGS_FADE_SHIFT) / float(255.0);

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)
	vec3 binormal = normalize(binormal_interp);
	vec3 tangent = normalize(tangent_interp);
#else
	vec3 binormal = vec3(0.0);
	vec3 tangent = vec3(0.0);
#endif

#ifdef NORMAL_USED
	vec3 normal = normalize(normal_interp);

#if defined(DO_SIDE_CHECK)
	if (!gl_FrontFacing) {
		normal = -normal;
	}
#endif

#endif //NORMAL_USED

#ifdef UV_USED
	vec2 uv = uv_interp;
#endif

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
	vec2 uv2 = uv2_interp;
#endif

#if defined(COLOR_USED)
	vec4 color = color_interp;
#endif

#if defined(NORMAL_MAP_USED)

	vec3 normal_map = vec3(0.5);
#endif

	float normal_map_depth = 1.0;

	vec2 screen_uv = gl_FragCoord.xy * scene_data.screen_pixel_size + scene_data.screen_pixel_size * 0.5; //account for center

	float sss_strength = 0.0;

#ifdef ALPHA_SCISSOR_USED
	float alpha_scissor_threshold = 1.0;
#endif // ALPHA_SCISSOR_USED

#ifdef ALPHA_HASH_USED
	float alpha_hash_scale = 1.0;
#endif // ALPHA_HASH_USED

#ifdef ALPHA_ANTIALIASING_EDGE_USED
	float alpha_antialiasing_edge = 0.0;
	vec2 alpha_texture_coordinate = vec2(0.0, 0.0);
#endif // ALPHA_ANTIALIASING_EDGE_USED

	{
#CODE : FRAGMENT
	}

#ifdef LIGHT_TRANSMITTANCE_USED
	transmittance_color.a *= sss_strength;
#endif

#ifndef USE_SHADOW_TO_OPACITY

#ifdef ALPHA_SCISSOR_USED
	if (alpha < alpha_scissor_threshold) {
		discard;
	}
#endif // ALPHA_SCISSOR_USED

// alpha hash can be used in unison with alpha antialiasing
#ifdef ALPHA_HASH_USED
	if (alpha < compute_alpha_hash_threshold(vertex, alpha_hash_scale)) {
		discard;
	}
#endif // ALPHA_HASH_USED

// If we are not edge antialiasing, we need to remove the output alpha channel from scissor and hash
#if (defined(ALPHA_SCISSOR_USED) || defined(ALPHA_HASH_USED)) && !defined(ALPHA_ANTIALIASING_EDGE_USED)
	alpha = 1.0;
#endif

#ifdef ALPHA_ANTIALIASING_EDGE_USED
// If alpha scissor is used, we must further the edge threshold, otherwise we won't get any edge feather
#ifdef ALPHA_SCISSOR_USED
	alpha_antialiasing_edge = clamp(alpha_scissor_threshold + alpha_antialiasing_edge, 0.0, 1.0);
#endif
	alpha = compute_alpha_antialiasing_edge(alpha, alpha_texture_coordinate, alpha_antialiasing_edge);
#endif // ALPHA_ANTIALIASING_EDGE_USED

#ifdef USE_OPAQUE_PREPASS
	if (alpha < opaque_prepass_threshold) {
		discard;
	}
#endif // USE_OPAQUE_PREPASS

#endif // !USE_SHADOW_TO_OPACITY

#ifdef NORMAL_MAP_USED

	normal_map.xy = normal_map.xy * 2.0 - 1.0;
	normal_map.z = sqrt(max(0.0, 1.0 - dot(normal_map.xy, normal_map.xy))); //always ignore Z, as it can be RG packed, Z may be pos/neg, etc.

	normal = normalize(mix(normal, tangent * normal_map.x + binormal * normal_map.y + normal * normal_map.z, normal_map_depth));

#endif

#ifdef LIGHT_ANISOTROPY_USED

	if (anisotropy > 0.01) {
		//rotation matrix
		mat3 rot = mat3(tangent, binormal, normal);
		//make local to space
		tangent = normalize(rot * vec3(anisotropy_flow.x, anisotropy_flow.y, 0.0));
		binormal = normalize(rot * vec3(-anisotropy_flow.y, anisotropy_flow.x, 0.0));
	}

#endif

#ifdef ENABLE_CLIP_ALPHA
	if (albedo.a < 0.99) {
		//used for doublepass and shadowmapping
		discard;
	}
#endif

	/////////////////////// FOG //////////////////////
#ifndef MODE_RENDER_DEPTH

#ifndef CUSTOM_FOG_USED
	// fog must be processed as early as possible and then packed.
	// to maximize VGPR usage
	// Draw "fixed" fog before volumetric fog to ensure volumetric fog can appear in front of the sky.

	if (scene_data.fog_enabled) {
		fog = fog_process(vertex);
	}

	if (scene_data.volumetric_fog_enabled) {
		vec4 volumetric_fog = volumetric_fog_process(screen_uv, -vertex.z);
		if (scene_data.fog_enabled) {
			//must use the full blending equation here to blend fogs
			vec4 res;
			float sa = 1.0 - volumetric_fog.a;
			res.a = fog.a * sa + volumetric_fog.a;
			if (res.a == 0.0) {
				res.rgb = vec3(0.0);
			} else {
				res.rgb = (fog.rgb * fog.a * sa + volumetric_fog.rgb * volumetric_fog.a) / res.a;
			}
			fog = res;
		} else {
			fog = volumetric_fog;
		}
	}
#endif //!CUSTOM_FOG_USED

	uint fog_rg = packHalf2x16(fog.rg);
	uint fog_ba = packHalf2x16(fog.ba);

#endif //!MODE_RENDER_DEPTH

	/////////////////////// DECALS ////////////////////////////////

#ifndef MODE_RENDER_DEPTH

	uvec2 cluster_pos = uvec2(gl_FragCoord.xy) >> scene_data.cluster_shift;
	uint cluster_offset = (scene_data.cluster_width * cluster_pos.y + cluster_pos.x) * (scene_data.max_cluster_element_count_div_32 + 32);

	uint cluster_z = uint(clamp((-vertex.z / scene_data.z_far) * 32.0, 0.0, 31.0));

	//used for interpolating anything cluster related
	vec3 vertex_ddx = dFdx(vertex);
	vec3 vertex_ddy = dFdy(vertex);

	{ // process decals

		uint cluster_decal_offset = cluster_offset + scene_data.cluster_type_size * 2;

		uint item_min;
		uint item_max;
		uint item_from;
		uint item_to;

		cluster_get_item_range(cluster_decal_offset + scene_data.max_cluster_element_count_div_32 + cluster_z, item_min, item_max, item_from, item_to);

#ifdef USE_SUBGROUPS
		item_from = subgroupBroadcastFirst(subgroupMin(item_from));
		item_to = subgroupBroadcastFirst(subgroupMax(item_to));
#endif

		for (uint i = item_from; i < item_to; i++) {
			uint mask = cluster_buffer.data[cluster_decal_offset + i];
			mask &= cluster_get_range_clip_mask(i, item_min, item_max);
#ifdef USE_SUBGROUPS
			uint merged_mask = subgroupBroadcastFirst(subgroupOr(mask));
#else
			uint merged_mask = mask;
#endif

			while (merged_mask != 0) {
				uint bit = findMSB(merged_mask);
				merged_mask &= ~(1 << bit);
#ifdef USE_SUBGROUPS
				if (((1 << bit) & mask) == 0) { //do not process if not originally here
					continue;
				}
#endif
				uint decal_index = 32 * i + bit;

				if (!bool(decals.data[decal_index].mask & instances.data[instance_index].layer_mask)) {
					continue; //not masked
				}

				vec3 uv_local = (decals.data[decal_index].xform * vec4(vertex, 1.0)).xyz;
				if (any(lessThan(uv_local, vec3(0.0, -1.0, 0.0))) || any(greaterThan(uv_local, vec3(1.0)))) {
					continue; //out of decal
				}

				float fade = pow(1.0 - (uv_local.y > 0.0 ? uv_local.y : -uv_local.y), uv_local.y > 0.0 ? decals.data[decal_index].upper_fade : decals.data[decal_index].lower_fade);

				if (decals.data[decal_index].normal_fade > 0.0) {
					fade *= smoothstep(decals.data[decal_index].normal_fade, 1.0, dot(normal_interp, decals.data[decal_index].normal) * 0.5 + 0.5);
				}

				//we need ddx/ddy for mipmaps, so simulate them
				vec2 ddx = (decals.data[decal_index].xform * vec4(vertex_ddx, 0.0)).xz;
				vec2 ddy = (decals.data[decal_index].xform * vec4(vertex_ddy, 0.0)).xz;

				if (decals.data[decal_index].albedo_rect != vec4(0.0)) {
					//has albedo
					vec4 decal_albedo;
					if (sc_decal_use_mipmaps) {
						decal_albedo = textureGrad(sampler2D(decal_atlas_srgb, decal_sampler), uv_local.xz * decals.data[decal_index].albedo_rect.zw + decals.data[decal_index].albedo_rect.xy, ddx * decals.data[decal_index].albedo_rect.zw, ddy * decals.data[decal_index].albedo_rect.zw);
					} else {
						decal_albedo = textureLod(sampler2D(decal_atlas_srgb, decal_sampler), uv_local.xz * decals.data[decal_index].albedo_rect.zw + decals.data[decal_index].albedo_rect.xy, 0.0);
					}
					decal_albedo *= decals.data[decal_index].modulate;
					decal_albedo.a *= fade;
					albedo = mix(albedo, decal_albedo.rgb, decal_albedo.a * decals.data[decal_index].albedo_mix);

					if (decals.data[decal_index].normal_rect != vec4(0.0)) {
						vec3 decal_normal;
						if (sc_decal_use_mipmaps) {
							decal_normal = textureGrad(sampler2D(decal_atlas, decal_sampler), uv_local.xz * decals.data[decal_index].normal_rect.zw + decals.data[decal_index].normal_rect.xy, ddx * decals.data[decal_index].normal_rect.zw, ddy * decals.data[decal_index].normal_rect.zw).xyz;
						} else {
							decal_normal = textureLod(sampler2D(decal_atlas, decal_sampler), uv_local.xz * decals.data[decal_index].normal_rect.zw + decals.data[decal_index].normal_rect.xy, 0.0).xyz;
						}
						decal_normal.xy = decal_normal.xy * vec2(2.0, -2.0) - vec2(1.0, -1.0); //users prefer flipped y normal maps in most authoring software
						decal_normal.z = sqrt(max(0.0, 1.0 - dot(decal_normal.xy, decal_normal.xy)));
						//convert to view space, use xzy because y is up
						decal_normal = (decals.data[decal_index].normal_xform * decal_normal.xzy).xyz;

						normal = normalize(mix(normal, decal_normal, decal_albedo.a));
					}

					if (decals.data[decal_index].orm_rect != vec4(0.0)) {
						vec3 decal_orm;
						if (sc_decal_use_mipmaps) {
							decal_orm = textureGrad(sampler2D(decal_atlas, decal_sampler), uv_local.xz * decals.data[decal_index].orm_rect.zw + decals.data[decal_index].orm_rect.xy, ddx * decals.data[decal_index].orm_rect.zw, ddy * decals.data[decal_index].orm_rect.zw).xyz;
						} else {
							decal_orm = textureLod(sampler2D(decal_atlas, decal_sampler), uv_local.xz * decals.data[decal_index].orm_rect.zw + decals.data[decal_index].orm_rect.xy, 0.0).xyz;
						}
						ao = mix(ao, decal_orm.r, decal_albedo.a);
						roughness = mix(roughness, decal_orm.g, decal_albedo.a);
						metallic = mix(metallic, decal_orm.b, decal_albedo.a);
					}
				}

				if (decals.data[decal_index].emission_rect != vec4(0.0)) {
					//emission is additive, so its independent from albedo
					if (sc_decal_use_mipmaps) {
						emission += textureGrad(sampler2D(decal_atlas_srgb, decal_sampler), uv_local.xz * decals.data[decal_index].emission_rect.zw + decals.data[decal_index].emission_rect.xy, ddx * decals.data[decal_index].emission_rect.zw, ddy * decals.data[decal_index].emission_rect.zw).xyz * decals.data[decal_index].emission_energy * fade;
					} else {
						emission += textureLod(sampler2D(decal_atlas_srgb, decal_sampler), uv_local.xz * decals.data[decal_index].emission_rect.zw + decals.data[decal_index].emission_rect.xy, 0.0).xyz * decals.data[decal_index].emission_energy * fade;
					}
				}
			}
		}
	}

	//pack albedo until needed again, saves 2 VGPRs in the meantime

#endif //not render depth
	/////////////////////// LIGHTING //////////////////////////////

#ifdef NORMAL_USED
	if (scene_data.roughness_limiter_enabled) {
		//https://www.jp.square-enix.com/tech/library/pdf/ImprovedGeometricSpecularAA.pdf
		float roughness2 = roughness * roughness;
		vec3 dndu = dFdx(normal), dndv = dFdy(normal);
		float variance = scene_data.roughness_limiter_amount * (dot(dndu, dndu) + dot(dndv, dndv));
		float kernelRoughness2 = min(2.0 * variance, scene_data.roughness_limiter_limit); //limit effect
		float filteredRoughness2 = min(1.0, roughness2 + kernelRoughness2);
		roughness = sqrt(filteredRoughness2);
	}
#endif
	//apply energy conservation

	vec3 specular_light = vec3(0.0, 0.0, 0.0);
	vec3 diffuse_light = vec3(0.0, 0.0, 0.0);
	vec3 ambient_light = vec3(0.0, 0.0, 0.0);

#if !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

	if (scene_data.use_reflection_cubemap) {
		vec3 ref_vec = reflect(-view, normal);
		float horizon = min(1.0 + dot(ref_vec, normal), 1.0);
		ref_vec = scene_data.radiance_inverse_xform * ref_vec;
#ifdef USE_RADIANCE_CUBEMAP_ARRAY

		float lod, blend;
		blend = modf(roughness * MAX_ROUGHNESS_LOD, lod);
		specular_light = texture(samplerCubeArray(radiance_cubemap, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), vec4(ref_vec, lod)).rgb;
		specular_light = mix(specular_light, texture(samplerCubeArray(radiance_cubemap, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), vec4(ref_vec, lod + 1)).rgb, blend);

#else
		specular_light = textureLod(samplerCube(radiance_cubemap, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), ref_vec, roughness * MAX_ROUGHNESS_LOD).rgb;

#endif //USE_RADIANCE_CUBEMAP_ARRAY
		specular_light *= horizon * horizon;
		specular_light *= scene_data.ambient_light_color_energy.a;
	}

#if defined(CUSTOM_RADIANCE_USED)
	specular_light = mix(specular_light, custom_radiance.rgb, custom_radiance.a);
#endif

#ifndef USE_LIGHTMAP
	//lightmap overrides everything
	if (scene_data.use_ambient_light) {
		ambient_light = scene_data.ambient_light_color_energy.rgb;

		if (scene_data.use_ambient_cubemap) {
			vec3 ambient_dir = scene_data.radiance_inverse_xform * normal;
#ifdef USE_RADIANCE_CUBEMAP_ARRAY
			vec3 cubemap_ambient = texture(samplerCubeArray(radiance_cubemap, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), vec4(ambient_dir, MAX_ROUGHNESS_LOD)).rgb;
#else
			vec3 cubemap_ambient = textureLod(samplerCube(radiance_cubemap, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), ambient_dir, MAX_ROUGHNESS_LOD).rgb;
#endif //USE_RADIANCE_CUBEMAP_ARRAY

			ambient_light = mix(ambient_light, cubemap_ambient * scene_data.ambient_light_color_energy.a, scene_data.ambient_color_sky_mix);
		}
	}
#endif // USE_LIGHTMAP
#if defined(CUSTOM_IRRADIANCE_USED)
	ambient_light = mix(ambient_light, custom_irradiance.rgb, custom_irradiance.a);
#endif
#endif //!defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

	//radiance

/// GI ///
#if !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

#ifdef USE_LIGHTMAP

	//lightmap
	if (bool(instances.data[instance_index].flags & INSTANCE_FLAGS_USE_LIGHTMAP_CAPTURE)) { //has lightmap capture
		uint index = instances.data[instance_index].gi_offset;

		vec3 wnormal = mat3(scene_data.camera_matrix) * normal;
		const float c1 = 0.429043;
		const float c2 = 0.511664;
		const float c3 = 0.743125;
		const float c4 = 0.886227;
		const float c5 = 0.247708;
		ambient_light += (c1 * lightmap_captures.data[index].sh[8].rgb * (wnormal.x * wnormal.x - wnormal.y * wnormal.y) +
				c3 * lightmap_captures.data[index].sh[6].rgb * wnormal.z * wnormal.z +
				c4 * lightmap_captures.data[index].sh[0].rgb -
				c5 * lightmap_captures.data[index].sh[6].rgb +
				2.0 * c1 * lightmap_captures.data[index].sh[4].rgb * wnormal.x * wnormal.y +
				2.0 * c1 * lightmap_captures.data[index].sh[7].rgb * wnormal.x * wnormal.z +
				2.0 * c1 * lightmap_captures.data[index].sh[5].rgb * wnormal.y * wnormal.z +
				2.0 * c2 * lightmap_captures.data[index].sh[3].rgb * wnormal.x +
				2.0 * c2 * lightmap_captures.data[index].sh[1].rgb * wnormal.y +
				2.0 * c2 * lightmap_captures.data[index].sh[2].rgb * wnormal.z);

	} else if (bool(instances.data[instance_index].flags & INSTANCE_FLAGS_USE_LIGHTMAP)) { // has actual lightmap
		bool uses_sh = bool(instances.data[instance_index].flags & INSTANCE_FLAGS_USE_SH_LIGHTMAP);
		uint ofs = instances.data[instance_index].gi_offset & 0xFFFF;
		vec3 uvw;
		uvw.xy = uv2 * instances.data[instance_index].lightmap_uv_scale.zw + instances.data[instance_index].lightmap_uv_scale.xy;
		uvw.z = float((instances.data[instance_index].gi_offset >> 16) & 0xFFFF);

		if (uses_sh) {
			uvw.z *= 4.0; //SH textures use 4 times more data
			vec3 lm_light_l0 = textureLod(sampler2DArray(lightmap_textures[ofs], material_samplers[SAMPLER_LINEAR_CLAMP]), uvw + vec3(0.0, 0.0, 0.0), 0.0).rgb;
			vec3 lm_light_l1n1 = textureLod(sampler2DArray(lightmap_textures[ofs], material_samplers[SAMPLER_LINEAR_CLAMP]), uvw + vec3(0.0, 0.0, 1.0), 0.0).rgb;
			vec3 lm_light_l1_0 = textureLod(sampler2DArray(lightmap_textures[ofs], material_samplers[SAMPLER_LINEAR_CLAMP]), uvw + vec3(0.0, 0.0, 2.0), 0.0).rgb;
			vec3 lm_light_l1p1 = textureLod(sampler2DArray(lightmap_textures[ofs], material_samplers[SAMPLER_LINEAR_CLAMP]), uvw + vec3(0.0, 0.0, 3.0), 0.0).rgb;

			uint idx = instances.data[instance_index].gi_offset >> 20;
			vec3 n = normalize(lightmaps.data[idx].normal_xform * normal);

			ambient_light += lm_light_l0 * 0.282095f;
			ambient_light += lm_light_l1n1 * 0.32573 * n.y;
			ambient_light += lm_light_l1_0 * 0.32573 * n.z;
			ambient_light += lm_light_l1p1 * 0.32573 * n.x;
			if (metallic > 0.01) { // since the more direct bounced light is lost, we can kind of fake it with this trick
				vec3 r = reflect(normalize(-vertex), normal);
				specular_light += lm_light_l1n1 * 0.32573 * r.y;
				specular_light += lm_light_l1_0 * 0.32573 * r.z;
				specular_light += lm_light_l1p1 * 0.32573 * r.x;
			}

		} else {
			ambient_light += textureLod(sampler2DArray(lightmap_textures[ofs], material_samplers[SAMPLER_LINEAR_CLAMP]), uvw, 0.0).rgb;
		}
	}
#else

	if (sc_use_forward_gi && bool(instances.data[instance_index].flags & INSTANCE_FLAGS_USE_SDFGI)) { //has lightmap capture

		//make vertex orientation the world one, but still align to camera
		vec3 cam_pos = mat3(scene_data.camera_matrix) * vertex;
		vec3 cam_normal = mat3(scene_data.camera_matrix) * normal;
		vec3 cam_reflection = mat3(scene_data.camera_matrix) * reflect(-view, normal);

		//apply y-mult
		cam_pos.y *= sdfgi.y_mult;
		cam_normal.y *= sdfgi.y_mult;
		cam_normal = normalize(cam_normal);
		cam_reflection.y *= sdfgi.y_mult;
		cam_normal = normalize(cam_normal);
		cam_reflection = normalize(cam_reflection);

		vec4 light_accum = vec4(0.0);
		float weight_accum = 0.0;

		vec4 light_blend_accum = vec4(0.0);
		float weight_blend_accum = 0.0;

		float blend = -1.0;

		// helper constants, compute once

		uint cascade = 0xFFFFFFFF;
		vec3 cascade_pos;
		vec3 cascade_normal;

		for (uint i = 0; i < sdfgi.max_cascades; i++) {
			cascade_pos = (cam_pos - sdfgi.cascades[i].position) * sdfgi.cascades[i].to_probe;

			if (any(lessThan(cascade_pos, vec3(0.0))) || any(greaterThanEqual(cascade_pos, sdfgi.cascade_probe_size))) {
				continue; //skip cascade
			}

			cascade = i;
			break;
		}

		if (cascade < SDFGI_MAX_CASCADES) {
			bool use_specular = true;
			float blend;
			vec3 diffuse, specular;
			sdfgi_process(cascade, cascade_pos, cam_pos, cam_normal, cam_reflection, use_specular, roughness, diffuse, specular, blend);

			if (blend > 0.0) {
				//blend
				if (cascade == sdfgi.max_cascades - 1) {
					diffuse = mix(diffuse, ambient_light, blend);
					if (use_specular) {
						specular = mix(specular, specular_light, blend);
					}
				} else {
					vec3 diffuse2, specular2;
					float blend2;
					cascade_pos = (cam_pos - sdfgi.cascades[cascade + 1].position) * sdfgi.cascades[cascade + 1].to_probe;
					sdfgi_process(cascade + 1, cascade_pos, cam_pos, cam_normal, cam_reflection, use_specular, roughness, diffuse2, specular2, blend2);
					diffuse = mix(diffuse, diffuse2, blend);
					if (use_specular) {
						specular = mix(specular, specular2, blend);
					}
				}
			}

			ambient_light = diffuse;
			if (use_specular) {
				specular_light = specular;
			}
		}
	}

	if (sc_use_forward_gi && bool(instances.data[instance_index].flags & INSTANCE_FLAGS_USE_VOXEL_GI)) { // process voxel_gi_instances

		uint index1 = instances.data[instance_index].gi_offset & 0xFFFF;
		vec3 ref_vec = normalize(reflect(normalize(vertex), normal));
		//find arbitrary tangent and bitangent, then build a matrix
		vec3 v0 = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
		vec3 tangent = normalize(cross(v0, normal));
		vec3 bitangent = normalize(cross(tangent, normal));
		mat3 normal_mat = mat3(tangent, bitangent, normal);

		vec4 amb_accum = vec4(0.0);
		vec4 spec_accum = vec4(0.0);
		voxel_gi_compute(index1, vertex, normal, ref_vec, normal_mat, roughness * roughness, ambient_light, specular_light, spec_accum, amb_accum);

		uint index2 = instances.data[instance_index].gi_offset >> 16;

		if (index2 != 0xFFFF) {
			voxel_gi_compute(index2, vertex, normal, ref_vec, normal_mat, roughness * roughness, ambient_light, specular_light, spec_accum, amb_accum);
		}

		if (amb_accum.a > 0.0) {
			amb_accum.rgb /= amb_accum.a;
		}

		if (spec_accum.a > 0.0) {
			spec_accum.rgb /= spec_accum.a;
		}

		specular_light = spec_accum.rgb;
		ambient_light = amb_accum.rgb;
	}

	if (!sc_use_forward_gi && bool(instances.data[instance_index].flags & INSTANCE_FLAGS_USE_GI_BUFFERS)) { //use GI buffers

		vec2 coord;

		if (scene_data.gi_upscale_for_msaa) {
			vec2 base_coord = screen_uv;
			vec2 closest_coord = base_coord;
			float closest_ang = dot(normal, textureLod(sampler2D(normal_roughness_buffer, material_samplers[SAMPLER_LINEAR_CLAMP]), base_coord, 0.0).xyz * 2.0 - 1.0);

			for (int i = 0; i < 4; i++) {
				const vec2 neighbours[4] = vec2[](vec2(-1, 0), vec2(1, 0), vec2(0, -1), vec2(0, 1));
				vec2 neighbour_coord = base_coord + neighbours[i] * scene_data.screen_pixel_size;
				float neighbour_ang = dot(normal, textureLod(sampler2D(normal_roughness_buffer, material_samplers[SAMPLER_LINEAR_CLAMP]), neighbour_coord, 0.0).xyz * 2.0 - 1.0);
				if (neighbour_ang > closest_ang) {
					closest_ang = neighbour_ang;
					closest_coord = neighbour_coord;
				}
			}

			coord = closest_coord;

		} else {
			coord = screen_uv;
		}

		vec4 buffer_ambient = textureLod(sampler2D(ambient_buffer, material_samplers[SAMPLER_LINEAR_CLAMP]), coord, 0.0);
		vec4 buffer_reflection = textureLod(sampler2D(reflection_buffer, material_samplers[SAMPLER_LINEAR_CLAMP]), coord, 0.0);

		ambient_light = mix(ambient_light, buffer_ambient.rgb, buffer_ambient.a);
		specular_light = mix(specular_light, buffer_reflection.rgb, buffer_reflection.a);
	}
#endif // !USE_LIGHTMAP

	if (scene_data.ssao_enabled) {
		float ssao = texture(sampler2D(ao_buffer, material_samplers[SAMPLER_LINEAR_CLAMP]), screen_uv).r;
		ao = min(ao, ssao);
		ao_light_affect = mix(ao_light_affect, max(ao_light_affect, scene_data.ssao_light_affect), scene_data.ssao_ao_affect);
	}

	{ // process reflections

		vec4 reflection_accum = vec4(0.0, 0.0, 0.0, 0.0);
		vec4 ambient_accum = vec4(0.0, 0.0, 0.0, 0.0);

		uint cluster_reflection_offset = cluster_offset + scene_data.cluster_type_size * 3;

		uint item_min;
		uint item_max;
		uint item_from;
		uint item_to;

		cluster_get_item_range(cluster_reflection_offset + scene_data.max_cluster_element_count_div_32 + cluster_z, item_min, item_max, item_from, item_to);

#ifdef USE_SUBGROUPS
		item_from = subgroupBroadcastFirst(subgroupMin(item_from));
		item_to = subgroupBroadcastFirst(subgroupMax(item_to));
#endif

		for (uint i = item_from; i < item_to; i++) {
			uint mask = cluster_buffer.data[cluster_reflection_offset + i];
			mask &= cluster_get_range_clip_mask(i, item_min, item_max);
#ifdef USE_SUBGROUPS
			uint merged_mask = subgroupBroadcastFirst(subgroupOr(mask));
#else
			uint merged_mask = mask;
#endif

			while (merged_mask != 0) {
				uint bit = findMSB(merged_mask);
				merged_mask &= ~(1 << bit);
#ifdef USE_SUBGROUPS
				if (((1 << bit) & mask) == 0) { //do not process if not originally here
					continue;
				}
#endif
				uint reflection_index = 32 * i + bit;

				if (!bool(reflections.data[reflection_index].mask & instances.data[instance_index].layer_mask)) {
					continue; //not masked
				}

				reflection_process(reflection_index, vertex, normal, roughness, ambient_light, specular_light, ambient_accum, reflection_accum);
			}
		}

		if (reflection_accum.a > 0.0) {
			specular_light = reflection_accum.rgb / reflection_accum.a;
		}

#if !defined(USE_LIGHTMAP)
		if (ambient_accum.a > 0.0) {
			ambient_light = ambient_accum.rgb / ambient_accum.a;
		}
#endif
	}

	//finalize ambient light here
	ambient_light *= albedo.rgb;
	ambient_light *= ao;

	// convert ao to direct light ao
	ao = mix(1.0, ao, ao_light_affect);

	//this saves some VGPRs
	vec3 f0 = F0(metallic, specular, albedo);

	{
#if defined(DIFFUSE_TOON)
		//simplify for toon, as
		specular_light *= specular * metallic * albedo * 2.0;
#else

		// scales the specular reflections, needs to be computed before lighting happens,
		// but after environment, GI, and reflection probes are added
		// Environment brdf approximation (Lazarov 2013)
		// see https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
		const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
		const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
		vec4 r = roughness * c0 + c1;
		float ndotv = clamp(dot(normal, view), 0.0, 1.0);
		float a004 = min(r.x * r.x, exp2(-9.28 * ndotv)) * r.x + r.y;
		vec2 env = vec2(-1.04, 1.04) * a004 + r.zw;

		specular_light *= env.x * f0 + env.y;
#endif
	}

#endif //GI !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

#if !defined(MODE_RENDER_DEPTH)
	//this saves some VGPRs
	uint orms = packUnorm4x8(vec4(ao, roughness, metallic, specular));
#endif

// LIGHTING
#if !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

	{ // Directional light.

		// Do shadow and lighting in two passes to reduce register pressure.
#ifndef SHADOWS_DISABLED
		uint shadow0 = 0;
		uint shadow1 = 0;

		for (uint i = 0; i < 8; i++) {
			if (i >= scene_data.directional_light_count) {
				break;
			}

			if (!bool(directional_lights.data[i].mask & instances.data[instance_index].layer_mask)) {
				continue; //not masked
			}

			if (directional_lights.data[i].bake_mode == LIGHT_BAKE_STATIC && bool(instances.data[instance_index].flags & INSTANCE_FLAGS_USE_LIGHTMAP)) {
				continue; // Statically baked light and object uses lightmap, skip
			}

			float shadow = 1.0;

			if (directional_lights.data[i].shadow_enabled) {
				float depth_z = -vertex.z;
				vec3 light_dir = directional_lights.data[i].direction;
				vec3 base_normal_bias = normalize(normal_interp) * (1.0 - max(0.0, dot(light_dir, -normalize(normal_interp))));

#define BIAS_FUNC(m_var, m_idx)                                                                 \
	m_var.xyz += light_dir * directional_lights.data[i].shadow_bias[m_idx];                     \
	vec3 normal_bias = base_normal_bias * directional_lights.data[i].shadow_normal_bias[m_idx]; \
	normal_bias -= light_dir * dot(light_dir, normal_bias);                                     \
	m_var.xyz += normal_bias;

				//version with soft shadows, more expensive
				if (sc_use_directional_soft_shadows && directional_lights.data[i].softshadow_angle > 0) {
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
						shadow = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale);
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
						float s = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale);

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
						float s = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale);

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
						float s = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale);

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
					if (depth_z < directional_lights.data[i].shadow_split_offsets.x) {
						vec4 v = vec4(vertex, 1.0);

						BIAS_FUNC(v, 0)

						pssm_coord = (directional_lights.data[i].shadow_matrix1 * v);
					} else if (depth_z < directional_lights.data[i].shadow_split_offsets.y) {
						vec4 v = vec4(vertex, 1.0);

						BIAS_FUNC(v, 1)

						pssm_coord = (directional_lights.data[i].shadow_matrix2 * v);
					} else if (depth_z < directional_lights.data[i].shadow_split_offsets.z) {
						vec4 v = vec4(vertex, 1.0);

						BIAS_FUNC(v, 2)

						pssm_coord = (directional_lights.data[i].shadow_matrix3 * v);

					} else {
						vec4 v = vec4(vertex, 1.0);

						BIAS_FUNC(v, 3)

						pssm_coord = (directional_lights.data[i].shadow_matrix4 * v);
					}

					pssm_coord /= pssm_coord.w;

					shadow = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale, pssm_coord);

					if (directional_lights.data[i].blend_splits) {
						float pssm_blend;

						if (depth_z < directional_lights.data[i].shadow_split_offsets.x) {
							vec4 v = vec4(vertex, 1.0);
							BIAS_FUNC(v, 1)
							pssm_coord = (directional_lights.data[i].shadow_matrix2 * v);
							pssm_blend = smoothstep(0.0, directional_lights.data[i].shadow_split_offsets.x, depth_z);
						} else if (depth_z < directional_lights.data[i].shadow_split_offsets.y) {
							vec4 v = vec4(vertex, 1.0);
							BIAS_FUNC(v, 2)
							pssm_coord = (directional_lights.data[i].shadow_matrix3 * v);
							pssm_blend = smoothstep(directional_lights.data[i].shadow_split_offsets.x, directional_lights.data[i].shadow_split_offsets.y, depth_z);
						} else if (depth_z < directional_lights.data[i].shadow_split_offsets.z) {
							vec4 v = vec4(vertex, 1.0);
							BIAS_FUNC(v, 3)
							pssm_coord = (directional_lights.data[i].shadow_matrix4 * v);
							pssm_blend = smoothstep(directional_lights.data[i].shadow_split_offsets.y, directional_lights.data[i].shadow_split_offsets.z, depth_z);
						} else {
							pssm_blend = 0.0; //if no blend, same coord will be used (divide by z will result in same value, and already cached)
						}

						pssm_coord /= pssm_coord.w;

						float shadow2 = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale, pssm_coord);
						shadow = mix(shadow, shadow2, pssm_blend);
					}
				}

				shadow = mix(shadow, 1.0, smoothstep(directional_lights.data[i].fade_from, directional_lights.data[i].fade_to, vertex.z)); //done with negative values for performance

#undef BIAS_FUNC
			} // shadows

			if (i < 4) {
				shadow0 |= uint(clamp(shadow * 255.0, 0.0, 255.0)) << (i * 8);
			} else {
				shadow1 |= uint(clamp(shadow * 255.0, 0.0, 255.0)) << ((i - 4) * 8);
			}
		}
#endif // SHADOWS_DISABLED

		for (uint i = 0; i < 8; i++) {
			if (i >= scene_data.directional_light_count) {
				break;
			}

			if (!bool(directional_lights.data[i].mask & instances.data[instance_index].layer_mask)) {
				continue; //not masked
			}

#ifdef LIGHT_TRANSMITTANCE_USED
			float transmittance_z = transmittance_depth;

			if (directional_lights.data[i].shadow_enabled) {
				float depth_z = -vertex.z;

				if (depth_z < directional_lights.data[i].shadow_split_offsets.x) {
					vec4 trans_vertex = vec4(vertex - normalize(normal_interp) * directional_lights.data[i].shadow_transmittance_bias.x, 1.0);
					vec4 trans_coord = directional_lights.data[i].shadow_matrix1 * trans_vertex;
					trans_coord /= trans_coord.w;

					float shadow_z = textureLod(sampler2D(directional_shadow_atlas, material_samplers[SAMPLER_LINEAR_CLAMP]), trans_coord.xy, 0.0).r;
					shadow_z *= directional_lights.data[i].shadow_z_range.x;
					float z = trans_coord.z * directional_lights.data[i].shadow_z_range.x;

					transmittance_z = z - shadow_z;
				} else if (depth_z < directional_lights.data[i].shadow_split_offsets.y) {
					vec4 trans_vertex = vec4(vertex - normalize(normal_interp) * directional_lights.data[i].shadow_transmittance_bias.y, 1.0);
					vec4 trans_coord = directional_lights.data[i].shadow_matrix2 * trans_vertex;
					trans_coord /= trans_coord.w;

					float shadow_z = textureLod(sampler2D(directional_shadow_atlas, material_samplers[SAMPLER_LINEAR_CLAMP]), trans_coord.xy, 0.0).r;
					shadow_z *= directional_lights.data[i].shadow_z_range.y;
					float z = trans_coord.z * directional_lights.data[i].shadow_z_range.y;

					transmittance_z = z - shadow_z;
				} else if (depth_z < directional_lights.data[i].shadow_split_offsets.z) {
					vec4 trans_vertex = vec4(vertex - normalize(normal_interp) * directional_lights.data[i].shadow_transmittance_bias.z, 1.0);
					vec4 trans_coord = directional_lights.data[i].shadow_matrix3 * trans_vertex;
					trans_coord /= trans_coord.w;

					float shadow_z = textureLod(sampler2D(directional_shadow_atlas, material_samplers[SAMPLER_LINEAR_CLAMP]), trans_coord.xy, 0.0).r;
					shadow_z *= directional_lights.data[i].shadow_z_range.z;
					float z = trans_coord.z * directional_lights.data[i].shadow_z_range.z;

					transmittance_z = z - shadow_z;

				} else {
					vec4 trans_vertex = vec4(vertex - normalize(normal_interp) * directional_lights.data[i].shadow_transmittance_bias.w, 1.0);
					vec4 trans_coord = directional_lights.data[i].shadow_matrix4 * trans_vertex;
					trans_coord /= trans_coord.w;

					float shadow_z = textureLod(sampler2D(directional_shadow_atlas, material_samplers[SAMPLER_LINEAR_CLAMP]), trans_coord.xy, 0.0).r;
					shadow_z *= directional_lights.data[i].shadow_z_range.w;
					float z = trans_coord.z * directional_lights.data[i].shadow_z_range.w;

					transmittance_z = z - shadow_z;
				}
			}
#endif

			float shadow = 1.0;
#ifndef SHADOWS_DISABLED
			if (i < 4) {
				shadow = float(shadow0 >> (i * 8) & 0xFF) / 255.0;
			} else {
				shadow = float(shadow1 >> ((i - 4) * 8) & 0xFF) / 255.0;
			}
#endif

			blur_shadow(shadow);

			float size_A = sc_use_light_soft_shadows ? directional_lights.data[i].size : 0.0;

			light_compute(normal, directional_lights.data[i].direction, normalize(view), size_A, directional_lights.data[i].color * directional_lights.data[i].energy, shadow, f0, orms, 1.0,
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
					rim, rim_tint, albedo,
#endif
#ifdef LIGHT_CLEARCOAT_USED
					clearcoat, clearcoat_gloss,
#endif
#ifdef LIGHT_ANISOTROPY_USED
					binormal, tangent, anisotropy,
#endif
#ifdef USE_SHADOW_TO_OPACITY
					alpha,
#endif
					diffuse_light,
					specular_light);
		}
	}

	{ //omni lights

		uint cluster_omni_offset = cluster_offset;

		uint item_min;
		uint item_max;
		uint item_from;
		uint item_to;

		cluster_get_item_range(cluster_omni_offset + scene_data.max_cluster_element_count_div_32 + cluster_z, item_min, item_max, item_from, item_to);

#ifdef USE_SUBGROUPS
		item_from = subgroupBroadcastFirst(subgroupMin(item_from));
		item_to = subgroupBroadcastFirst(subgroupMax(item_to));
#endif

		for (uint i = item_from; i < item_to; i++) {
			uint mask = cluster_buffer.data[cluster_omni_offset + i];
			mask &= cluster_get_range_clip_mask(i, item_min, item_max);
#ifdef USE_SUBGROUPS
			uint merged_mask = subgroupBroadcastFirst(subgroupOr(mask));
#else
			uint merged_mask = mask;
#endif

			while (merged_mask != 0) {
				uint bit = findMSB(merged_mask);
				merged_mask &= ~(1 << bit);
#ifdef USE_SUBGROUPS
				if (((1 << bit) & mask) == 0) { //do not process if not originally here
					continue;
				}
#endif
				uint light_index = 32 * i + bit;

				if (!bool(omni_lights.data[light_index].mask & instances.data[instance_index].layer_mask)) {
					continue; //not masked
				}

				if (omni_lights.data[light_index].bake_mode == LIGHT_BAKE_STATIC && bool(instances.data[instance_index].flags & INSTANCE_FLAGS_USE_LIGHTMAP)) {
					continue; // Statically baked light and object uses lightmap, skip
				}

				float shadow = light_process_omni_shadow(light_index, vertex, normal);

				shadow = blur_shadow(shadow);

				light_process_omni(light_index, vertex, view, normal, vertex_ddx, vertex_ddy, f0, orms, shadow,
#ifdef LIGHT_BACKLIGHT_USED
						backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
						transmittance_color,
						transmittance_depth,
						transmittance_boost,
#endif
#ifdef LIGHT_RIM_USED
						rim,
						rim_tint,
						albedo,
#endif
#ifdef LIGHT_CLEARCOAT_USED
						clearcoat, clearcoat_gloss,
#endif
#ifdef LIGHT_ANISOTROPY_USED
						tangent, binormal, anisotropy,
#endif
#ifdef USE_SHADOW_TO_OPACITY
						alpha,
#endif
						diffuse_light, specular_light);
			}
		}
	}

	{ //spot lights

		uint cluster_spot_offset = cluster_offset + scene_data.cluster_type_size;

		uint item_min;
		uint item_max;
		uint item_from;
		uint item_to;

		cluster_get_item_range(cluster_spot_offset + scene_data.max_cluster_element_count_div_32 + cluster_z, item_min, item_max, item_from, item_to);

#ifdef USE_SUBGROUPS
		item_from = subgroupBroadcastFirst(subgroupMin(item_from));
		item_to = subgroupBroadcastFirst(subgroupMax(item_to));
#endif

		for (uint i = item_from; i < item_to; i++) {
			uint mask = cluster_buffer.data[cluster_spot_offset + i];
			mask &= cluster_get_range_clip_mask(i, item_min, item_max);
#ifdef USE_SUBGROUPS
			uint merged_mask = subgroupBroadcastFirst(subgroupOr(mask));
#else
			uint merged_mask = mask;
#endif

			while (merged_mask != 0) {
				uint bit = findMSB(merged_mask);
				merged_mask &= ~(1 << bit);
#ifdef USE_SUBGROUPS
				if (((1 << bit) & mask) == 0) { //do not process if not originally here
					continue;
				}
#endif

				uint light_index = 32 * i + bit;

				if (!bool(spot_lights.data[light_index].mask & instances.data[instance_index].layer_mask)) {
					continue; //not masked
				}

				if (spot_lights.data[light_index].bake_mode == LIGHT_BAKE_STATIC && bool(instances.data[instance_index].flags & INSTANCE_FLAGS_USE_LIGHTMAP)) {
					continue; // Statically baked light and object uses lightmap, skip
				}

				float shadow = light_process_spot_shadow(light_index, vertex, normal);

				shadow = blur_shadow(shadow);

				light_process_spot(light_index, vertex, view, normal, vertex_ddx, vertex_ddy, f0, orms, shadow,
#ifdef LIGHT_BACKLIGHT_USED
						backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
						transmittance_color,
						transmittance_depth,
						transmittance_boost,
#endif
#ifdef LIGHT_RIM_USED
						rim,
						rim_tint,
						albedo,
#endif
#ifdef LIGHT_CLEARCOAT_USED
						clearcoat, clearcoat_gloss,
#endif
#ifdef LIGHT_ANISOTROPY_USED
						tangent, binormal, anisotropy,
#endif
#ifdef USE_SHADOW_TO_OPACITY
						alpha,
#endif
						diffuse_light, specular_light);
			}
		}
	}

#ifdef USE_SHADOW_TO_OPACITY
	alpha = min(alpha, clamp(length(ambient_light), 0.0, 1.0));

#if defined(ALPHA_SCISSOR_USED)
	if (alpha < alpha_scissor) {
		discard;
	}
#endif // ALPHA_SCISSOR_USED

#ifdef USE_OPAQUE_PREPASS

	if (alpha < opaque_prepass_threshold) {
		discard;
	}

#endif // USE_OPAQUE_PREPASS

#endif // USE_SHADOW_TO_OPACITY

#endif //!defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

#ifdef MODE_RENDER_DEPTH

#ifdef MODE_RENDER_SDF

	{
		vec3 local_pos = (scene_data.sdf_to_bounds * vec4(vertex, 1.0)).xyz;
		ivec3 grid_pos = scene_data.sdf_offset + ivec3(local_pos * vec3(scene_data.sdf_size));

		uint albedo16 = 0x1; //solid flag
		albedo16 |= clamp(uint(albedo.r * 31.0), 0, 31) << 11;
		albedo16 |= clamp(uint(albedo.g * 31.0), 0, 31) << 6;
		albedo16 |= clamp(uint(albedo.b * 31.0), 0, 31) << 1;

		imageStore(albedo_volume_grid, grid_pos, uvec4(albedo16));

		uint facing_bits = 0;
		const vec3 aniso_dir[6] = vec3[](
				vec3(1, 0, 0),
				vec3(0, 1, 0),
				vec3(0, 0, 1),
				vec3(-1, 0, 0),
				vec3(0, -1, 0),
				vec3(0, 0, -1));

		vec3 cam_normal = mat3(scene_data.camera_matrix) * normalize(normal_interp);

		float closest_dist = -1e20;

		for (uint i = 0; i < 6; i++) {
			float d = dot(cam_normal, aniso_dir[i]);
			if (d > closest_dist) {
				closest_dist = d;
				facing_bits = (1 << i);
			}
		}

		imageAtomicOr(geom_facing_grid, grid_pos, facing_bits); //store facing bits

		if (length(emission) > 0.001) {
			float lumas[6];
			vec3 light_total = vec3(0);

			for (int i = 0; i < 6; i++) {
				float strength = max(0.0, dot(cam_normal, aniso_dir[i]));
				vec3 light = emission * strength;
				light_total += light;
				lumas[i] = max(light.r, max(light.g, light.b));
			}

			float luma_total = max(light_total.r, max(light_total.g, light_total.b));

			uint light_aniso = 0;

			for (int i = 0; i < 6; i++) {
				light_aniso |= min(31, uint((lumas[i] / luma_total) * 31.0)) << (i * 5);
			}

			//compress to RGBE9995 to save space

			const float pow2to9 = 512.0f;
			const float B = 15.0f;
			const float N = 9.0f;
			const float LN2 = 0.6931471805599453094172321215;

			float cRed = clamp(light_total.r, 0.0, 65408.0);
			float cGreen = clamp(light_total.g, 0.0, 65408.0);
			float cBlue = clamp(light_total.b, 0.0, 65408.0);

			float cMax = max(cRed, max(cGreen, cBlue));

			float expp = max(-B - 1.0f, floor(log(cMax) / LN2)) + 1.0f + B;

			float sMax = floor((cMax / pow(2.0f, expp - B - N)) + 0.5f);

			float exps = expp + 1.0f;

			if (0.0 <= sMax && sMax < pow2to9) {
				exps = expp;
			}

			float sRed = floor((cRed / pow(2.0f, exps - B - N)) + 0.5f);
			float sGreen = floor((cGreen / pow(2.0f, exps - B - N)) + 0.5f);
			float sBlue = floor((cBlue / pow(2.0f, exps - B - N)) + 0.5f);
			//store as 8985 to have 2 extra neighbour bits
			uint light_rgbe = ((uint(sRed) & 0x1FF) >> 1) | ((uint(sGreen) & 0x1FF) << 8) | (((uint(sBlue) & 0x1FF) >> 1) << 17) | ((uint(exps) & 0x1F) << 25);

			imageStore(emission_grid, grid_pos, uvec4(light_rgbe));
			imageStore(emission_aniso_grid, grid_pos, uvec4(light_aniso));
		}
	}

#endif

#ifdef MODE_RENDER_MATERIAL

	albedo_output_buffer.rgb = albedo;
	albedo_output_buffer.a = alpha;

	normal_output_buffer.rgb = normal * 0.5 + 0.5;
	normal_output_buffer.a = 0.0;
	depth_output_buffer.r = -vertex.z;

	orm_output_buffer.r = ao;
	orm_output_buffer.g = roughness;
	orm_output_buffer.b = metallic;
	orm_output_buffer.a = sss_strength;

	emission_output_buffer.rgb = emission;
	emission_output_buffer.a = 0.0;
#endif

#ifdef MODE_RENDER_NORMAL_ROUGHNESS
	normal_roughness_output_buffer = vec4(normal * 0.5 + 0.5, roughness);

#ifdef MODE_RENDER_VOXEL_GI
	if (bool(instances.data[instance_index].flags & INSTANCE_FLAGS_USE_VOXEL_GI)) { // process voxel_gi_instances
		uint index1 = instances.data[instance_index].gi_offset & 0xFFFF;
		uint index2 = instances.data[instance_index].gi_offset >> 16;
		voxel_gi_buffer.x = index1 & 0xFF;
		voxel_gi_buffer.y = index2 & 0xFF;
	} else {
		voxel_gi_buffer.x = 0xFF;
		voxel_gi_buffer.y = 0xFF;
	}
#endif

#endif //MODE_RENDER_NORMAL_ROUGHNESS

//nothing happens, so a tree-ssa optimizer will result in no fragment shader :)
#else

	// multiply by albedo
	diffuse_light *= albedo; // ambient must be multiplied by albedo at the end

	// apply direct light AO
	ao = unpackUnorm4x8(orms).x;
	specular_light *= ao;
	diffuse_light *= ao;

	// apply metallic
	metallic = unpackUnorm4x8(orms).z;
	diffuse_light *= 1.0 - metallic;
	ambient_light *= 1.0 - metallic;

	//restore fog
	fog = vec4(unpackHalf2x16(fog_rg), unpackHalf2x16(fog_ba));

#ifdef MODE_MULTIPLE_RENDER_TARGETS

#ifdef MODE_UNSHADED
	diffuse_buffer = vec4(albedo.rgb, 0.0);
	specular_buffer = vec4(0.0);

#else

#ifdef SSS_MODE_SKIN
	sss_strength = -sss_strength;
#endif
	diffuse_buffer = vec4(emission + diffuse_light + ambient_light, sss_strength);
	specular_buffer = vec4(specular_light, metallic);
#endif

	diffuse_buffer.rgb = mix(diffuse_buffer.rgb, fog.rgb, fog.a);
	specular_buffer.rgb = mix(specular_buffer.rgb, vec3(0.0), fog.a);

#else //MODE_MULTIPLE_RENDER_TARGETS

#ifdef MODE_UNSHADED
	frag_color = vec4(albedo, alpha);
#else
	frag_color = vec4(emission + ambient_light + diffuse_light + specular_light, alpha);
	//frag_color = vec4(1.0);
#endif //USE_NO_SHADING

	// Draw "fixed" fog before volumetric fog to ensure volumetric fog can appear in front of the sky.
	frag_color.rgb = mix(frag_color.rgb, fog.rgb, fog.a);

#endif //MODE_MULTIPLE_RENDER_TARGETS

#endif //MODE_RENDER_DEPTH
}
