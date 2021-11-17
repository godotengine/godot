#[vertex]

#version 450

#VERSION_DEFINES

/* Include our forward mobile UBOs definitions etc. */
#include "scene_forward_mobile_inc.glsl"

#define SHADER_IS_SRGB false

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
#endif // MODE_RENDER_MATERIAL

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

layout(location = 0) highp out vec3 vertex_interp;

#ifdef NORMAL_USED
layout(location = 1) mediump out vec3 normal_interp;
#endif

#if defined(COLOR_USED)
layout(location = 2) mediump out vec4 color_interp;
#endif

#ifdef UV_USED
layout(location = 3) mediump out vec2 uv_interp;
#endif

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
layout(location = 4) mediump out vec2 uv2_interp;
#endif

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)
layout(location = 5) mediump out vec3 tangent_interp;
layout(location = 6) mediump out vec3 binormal_interp;
#endif

#ifdef MATERIAL_UNIFORMS_USED
layout(set = MATERIAL_UNIFORM_SET, binding = 0, std140) uniform MaterialUniforms{

#MATERIAL_UNIFORMS

} material;
#endif

#ifdef MODE_DUAL_PARABOLOID

layout(location = 8) out highp float dp_clip;

#endif

#ifdef USE_MULTIVIEW
#ifdef has_VK_KHR_multiview
#define ViewIndex gl_ViewIndex
#else
// !BAS! This needs to become an input once we implement our fallback!
#define ViewIndex 0
#endif
#else
// Set to zero, not supported in non stereo
#define ViewIndex 0
#endif //USE_MULTIVIEW

invariant gl_Position;

#GLOBALS

void main() {
	vec4 instance_custom = vec4(0.0);
#if defined(COLOR_USED)
	color_interp = color_attrib;
#endif

	bool is_multimesh = bool(draw_call.flags & INSTANCE_FLAGS_MULTIMESH);

	mat4 world_matrix = draw_call.transform;

	mat3 world_normal_matrix;
	if (bool(draw_call.flags & INSTANCE_FLAGS_NON_UNIFORM_SCALE)) {
		world_normal_matrix = transpose(inverse(mat3(world_matrix)));
	} else {
		world_normal_matrix = mat3(world_matrix);
	}

	if (is_multimesh) {
		//multimesh, instances are for it

		mat4 matrix;

#ifdef USE_PARTICLE_TRAILS
		uint trail_size = (draw_call.flags >> INSTANCE_FLAGS_PARTICLE_TRAIL_SHIFT) & INSTANCE_FLAGS_PARTICLE_TRAIL_MASK;
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
			if (bool(draw_call.flags & INSTANCE_FLAGS_MULTIMESH_FORMAT_2D)) {
				stride += 2;
			} else {
				stride += 3;
			}
			if (bool(draw_call.flags & INSTANCE_FLAGS_MULTIMESH_HAS_COLOR)) {
				stride += 1;
			}
			if (bool(draw_call.flags & INSTANCE_FLAGS_MULTIMESH_HAS_CUSTOM_DATA)) {
				stride += 1;
			}
		}

		uint offset = stride * gl_InstanceIndex;

		if (bool(draw_call.flags & INSTANCE_FLAGS_MULTIMESH_FORMAT_2D)) {
			matrix = mat4(transforms.data[offset + 0], transforms.data[offset + 1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0));
			offset += 2;
		} else {
			matrix = mat4(transforms.data[offset + 0], transforms.data[offset + 1], transforms.data[offset + 2], vec4(0.0, 0.0, 0.0, 1.0));
			offset += 3;
		}

		if (bool(draw_call.flags & INSTANCE_FLAGS_MULTIMESH_HAS_COLOR)) {
#ifdef COLOR_USED
			color_interp *= transforms.data[offset];
#endif
			offset += 1;
		}

		if (bool(draw_call.flags & INSTANCE_FLAGS_MULTIMESH_HAS_CUSTOM_DATA)) {
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

#ifdef USE_MULTIVIEW
	mat4 projection_matrix = scene_data.projection_matrix_view[ViewIndex];
	mat4 inv_projection_matrix = scene_data.inv_projection_matrix_view[ViewIndex];
#else
	mat4 projection_matrix = scene_data.projection_matrix;
	mat4 inv_projection_matrix = scene_data.inv_projection_matrix;
#endif //USE_MULTIVIEW

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

	/* output */

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
#endif // OVERRIDE_POSITION

#ifdef MODE_RENDER_DEPTH
	if (scene_data.pancake_shadows) {
		if (gl_Position.z <= 0.00001) {
			gl_Position.z = 0.00001;
		}
	}
#endif // MODE_RENDER_DEPTH
#ifdef MODE_RENDER_MATERIAL
	if (scene_data.material_uv2_mode) {
		vec2 uv_offset = draw_call.lightmap_uv_scale.xy; // we are abusing lightmap_uv_scale here, we shouldn't have a lightmap during a depth pass...
		gl_Position.xy = (uv2_attrib.xy + uv_offset) * 2.0 - 1.0;
		gl_Position.z = 0.00001;
		gl_Position.w = 1.0;
	}
#endif // MODE_RENDER_MATERIAL
}

#[fragment]

#version 450

#VERSION_DEFINES

#define SHADER_IS_SRGB false

/* Specialization Constants */

#if !defined(MODE_RENDER_DEPTH)

#if !defined(MODE_UNSHADED)

layout(constant_id = 0) const bool sc_use_light_projector = false;
layout(constant_id = 1) const bool sc_use_light_soft_shadows = false;
layout(constant_id = 2) const bool sc_use_directional_soft_shadows = false;

layout(constant_id = 3) const uint sc_soft_shadow_samples = 4;
layout(constant_id = 4) const uint sc_penumbra_shadow_samples = 4;

layout(constant_id = 5) const uint sc_directional_soft_shadow_samples = 4;
layout(constant_id = 6) const uint sc_directional_penumbra_shadow_samples = 4;

layout(constant_id = 8) const bool sc_projector_use_mipmaps = true;

layout(constant_id = 9) const bool sc_disable_omni_lights = false;
layout(constant_id = 10) const bool sc_disable_spot_lights = false;
layout(constant_id = 11) const bool sc_disable_reflection_probes = false;
layout(constant_id = 12) const bool sc_disable_directional_lights = false;

#endif //!MODE_UNSHADED

layout(constant_id = 7) const bool sc_decal_use_mipmaps = true;
layout(constant_id = 13) const bool sc_disable_decals = false;
layout(constant_id = 14) const bool sc_disable_fog = false;

#endif //!MODE_RENDER_DEPTH

layout(constant_id = 15) const float sc_luminance_multiplier = 2.0;

/* Include our forward mobile UBOs definitions etc. */
#include "scene_forward_mobile_inc.glsl"

/* Varyings */

layout(location = 0) highp in vec3 vertex_interp;

#ifdef NORMAL_USED
layout(location = 1) mediump in vec3 normal_interp;
#endif

#if defined(COLOR_USED)
layout(location = 2) mediump in vec4 color_interp;
#endif

#ifdef UV_USED
layout(location = 3) mediump in vec2 uv_interp;
#endif

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
layout(location = 4) mediump in vec2 uv2_interp;
#endif

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)
layout(location = 5) mediump in vec3 tangent_interp;
layout(location = 6) mediump in vec3 binormal_interp;
#endif

#ifdef MODE_DUAL_PARABOLOID

layout(location = 8) highp in float dp_clip;

#endif

#ifdef USE_MULTIVIEW
#ifdef has_VK_KHR_multiview
#define ViewIndex gl_ViewIndex
#else
// !BAS! This needs to become an input once we implement our fallback!
#define ViewIndex 0
#endif
#else
// Set to zero, not supported in non stereo
#define ViewIndex 0
#endif //USE_MULTIVIEW

//defines to keep compatibility with vertex

#define world_matrix draw_call.transform
#ifdef USE_MULTIVIEW
#define projection_matrix scene_data.projection_matrix_view[ViewIndex]
#else
#define projection_matrix scene_data.projection_matrix
#endif

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

/* clang-format on */

#ifdef MODE_RENDER_DEPTH

#ifdef MODE_RENDER_MATERIAL

layout(location = 0) out vec4 albedo_output_buffer;
layout(location = 1) out vec4 normal_output_buffer;
layout(location = 2) out vec4 orm_output_buffer;
layout(location = 3) out vec4 emission_output_buffer;
layout(location = 4) out float depth_output_buffer;

#endif // MODE_RENDER_MATERIAL

#else // RENDER DEPTH

#ifdef MODE_MULTIPLE_RENDER_TARGETS

layout(location = 0) out vec4 diffuse_buffer; //diffuse (rgb) and roughness
layout(location = 1) out vec4 specular_buffer; //specular and SSS (subsurface scatter)
#else

layout(location = 0) out mediump vec4 frag_color;
#endif // MODE_MULTIPLE_RENDER_TARGETS

#endif // RENDER DEPTH

#include "scene_forward_aa_inc.glsl"

#if !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

/* Make a default specular mode SPECULAR_SCHLICK_GGX. */
#if !defined(SPECULAR_DISABLED) && !defined(SPECULAR_SCHLICK_GGX) && !defined(SPECULAR_BLINN) && !defined(SPECULAR_PHONG) && !defined(SPECULAR_TOON)
#define SPECULAR_SCHLICK_GGX
#endif

#include "scene_forward_lights_inc.glsl"

#endif //!defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

#ifndef MODE_RENDER_DEPTH

/*
	Only supporting normal fog here.
*/

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

#endif //!MODE_RENDER DEPTH

void main() {
#ifdef MODE_DUAL_PARABOLOID

	if (dp_clip > 0.0)
		discard;
#endif

	//lay out everything, whathever is unused is optimized away anyway
	vec3 vertex = vertex_interp;
	vec3 view = -normalize(vertex_interp);
	vec3 albedo = vec3(1.0);
	vec3 backlight = vec3(0.0);
	vec4 transmittance_color = vec4(0.0);
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

	float alpha = 1.0;

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
#ifdef SSS_MODE_SKIN
	transmittance_color.a = sss_strength;
#else
	transmittance_color.a *= sss_strength;
#endif
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

	if (!sc_disable_fog && scene_data.fog_enabled) {
		fog = fog_process(vertex);
	}

#endif //!CUSTOM_FOG_USED

	uint fog_rg = packHalf2x16(fog.rg);
	uint fog_ba = packHalf2x16(fog.ba);

#endif //!MODE_RENDER_DEPTH

	/////////////////////// DECALS ////////////////////////////////

#ifndef MODE_RENDER_DEPTH

	vec3 vertex_ddx = dFdx(vertex);
	vec3 vertex_ddy = dFdy(vertex);

	if (!sc_disable_decals) { //Decals
		// must implement

		uint decal_indices = draw_call.decals.x;
		for (uint i = 0; i < 8; i++) {
			uint decal_index = decal_indices & 0xFF;
			if (i == 4) {
				decal_indices = draw_call.decals.y;
			} else {
				decal_indices = decal_indices >> 8;
			}

			if (decal_index == 0xFF) {
				break;
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
	} //Decals
#endif //!MODE_RENDER_DEPTH

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
#endif // NORMAL_USED
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

#else // USE_RADIANCE_CUBEMAP_ARRAY
		specular_light = textureLod(samplerCube(radiance_cubemap, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), ref_vec, roughness * MAX_ROUGHNESS_LOD).rgb;

#endif //USE_RADIANCE_CUBEMAP_ARRAY
		specular_light *= horizon * horizon;
		specular_light *= scene_data.ambient_light_color_energy.a;
	}

#if defined(CUSTOM_RADIANCE_USED)
	specular_light = mix(specular_light, custom_radiance.rgb, custom_radiance.a);
#endif // CUSTOM_RADIANCE_USED

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
#endif // !USE_LIGHTMAP

#if defined(CUSTOM_IRRADIANCE_USED)
	ambient_light = mix(specular_light, custom_irradiance.rgb, custom_irradiance.a);
#endif // CUSTOM_IRRADIANCE_USED

#endif //!defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

	//radiance

#if !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

#ifdef USE_LIGHTMAP

	//lightmap
	if (bool(draw_call.flags & INSTANCE_FLAGS_USE_LIGHTMAP_CAPTURE)) { //has lightmap capture
		uint index = draw_call.gi_offset;

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

	} else if (bool(draw_call.flags & INSTANCE_FLAGS_USE_LIGHTMAP)) { // has actual lightmap
		bool uses_sh = bool(draw_call.flags & INSTANCE_FLAGS_USE_SH_LIGHTMAP);
		uint ofs = draw_call.gi_offset & 0xFFFF;
		vec3 uvw;
		uvw.xy = uv2 * draw_call.lightmap_uv_scale.zw + draw_call.lightmap_uv_scale.xy;
		uvw.z = float((draw_call.gi_offset >> 16) & 0xFFFF);

		if (uses_sh) {
			uvw.z *= 4.0; //SH textures use 4 times more data
			vec3 lm_light_l0 = textureLod(sampler2DArray(lightmap_textures[ofs], material_samplers[SAMPLER_LINEAR_CLAMP]), uvw + vec3(0.0, 0.0, 0.0), 0.0).rgb;
			vec3 lm_light_l1n1 = textureLod(sampler2DArray(lightmap_textures[ofs], material_samplers[SAMPLER_LINEAR_CLAMP]), uvw + vec3(0.0, 0.0, 1.0), 0.0).rgb;
			vec3 lm_light_l1_0 = textureLod(sampler2DArray(lightmap_textures[ofs], material_samplers[SAMPLER_LINEAR_CLAMP]), uvw + vec3(0.0, 0.0, 2.0), 0.0).rgb;
			vec3 lm_light_l1p1 = textureLod(sampler2DArray(lightmap_textures[ofs], material_samplers[SAMPLER_LINEAR_CLAMP]), uvw + vec3(0.0, 0.0, 3.0), 0.0).rgb;

			uint idx = draw_call.gi_offset >> 20;
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

	// No GI nor non low end mode...

#endif // USE_LIGHTMAP

	// skipping ssao, do we remove ssao totally?

	if (!sc_disable_reflection_probes) { //Reflection probes
		vec4 reflection_accum = vec4(0.0, 0.0, 0.0, 0.0);
		vec4 ambient_accum = vec4(0.0, 0.0, 0.0, 0.0);

		uint reflection_indices = draw_call.reflection_probes.x;
		for (uint i = 0; i < 8; i++) {
			uint reflection_index = reflection_indices & 0xFF;
			if (i == 4) {
				reflection_indices = draw_call.reflection_probes.y;
			} else {
				reflection_indices = reflection_indices >> 8;
			}

			if (reflection_index == 0xFF) {
				break;
			}

			reflection_process(reflection_index, vertex, normal, roughness, ambient_light, specular_light, ambient_accum, reflection_accum);
		}

		if (reflection_accum.a > 0.0) {
			specular_light = reflection_accum.rgb / reflection_accum.a;
		}
	} //Reflection probes

	// finalize ambient light here
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

#endif // !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

#if !defined(MODE_RENDER_DEPTH)
	//this saves some VGPRs
	uint orms = packUnorm4x8(vec4(ao, roughness, metallic, specular));
#endif

// LIGHTING
#if !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

	if (!sc_disable_directional_lights) { //directional light
#ifndef SHADOWS_DISABLED
		// Do shadow and lighting in two passes to reduce register pressure
		uint shadow0 = 0;
		uint shadow1 = 0;

		for (uint i = 0; i < 8; i++) {
			if (i >= scene_data.directional_light_count) {
				break;
			}

			if (!bool(directional_lights.data[i].mask & draw_call.layer_mask)) {
				continue; //not masked
			}

			float shadow = 1.0;

			// Directional light shadow code is basically the same as forward clustered at this point in time minus `LIGHT_TRANSMITTANCE_USED` support.
			// Not sure if there is a reason to change this seeing directional lights are part of our global data
			// Should think about whether we may want to move this code into an include file or function??

#ifdef USE_SOFT_SHADOWS
			//version with soft shadows, more expensive
			if (directional_lights.data[i].shadow_enabled) {
				float depth_z = -vertex.z;

				vec4 pssm_coord;
				vec3 shadow_color = vec3(0.0);
				vec3 light_dir = directional_lights.data[i].direction;

#define BIAS_FUNC(m_var, m_idx)                                                                                                                                       \
	m_var.xyz += light_dir * directional_lights.data[i].shadow_bias[m_idx];                                                                                           \
	vec3 normal_bias = normalize(normal_interp) * (1.0 - max(0.0, dot(light_dir, -normalize(normal_interp)))) * directional_lights.data[i].shadow_normal_bias[m_idx]; \
	normal_bias -= light_dir * dot(light_dir, normal_bias);                                                                                                           \
	m_var.xyz += normal_bias;

				if (depth_z < directional_lights.data[i].shadow_split_offsets.x) {
					vec4 v = vec4(vertex, 1.0);

					BIAS_FUNC(v, 0)

					pssm_coord = (directional_lights.data[i].shadow_matrix1 * v);
					pssm_coord /= pssm_coord.w;

					if (directional_lights.data[i].softshadow_angle > 0) {
						float range_pos = dot(directional_lights.data[i].direction, v.xyz);
						float range_begin = directional_lights.data[i].shadow_range_begin.x;
						float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
						vec2 tex_scale = directional_lights.data[i].uv_scale1 * test_radius;
						shadow = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale);
					} else {
						shadow = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale, pssm_coord);
					}

					shadow_color = directional_lights.data[i].shadow_color1.rgb;

				} else if (depth_z < directional_lights.data[i].shadow_split_offsets.y) {
					vec4 v = vec4(vertex, 1.0);

					BIAS_FUNC(v, 1)

					pssm_coord = (directional_lights.data[i].shadow_matrix2 * v);
					pssm_coord /= pssm_coord.w;

					if (directional_lights.data[i].softshadow_angle > 0) {
						float range_pos = dot(directional_lights.data[i].direction, v.xyz);
						float range_begin = directional_lights.data[i].shadow_range_begin.y;
						float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
						vec2 tex_scale = directional_lights.data[i].uv_scale2 * test_radius;
						shadow = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale);
					} else {
						shadow = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale, pssm_coord);
					}

					shadow_color = directional_lights.data[i].shadow_color2.rgb;
				} else if (depth_z < directional_lights.data[i].shadow_split_offsets.z) {
					vec4 v = vec4(vertex, 1.0);

					BIAS_FUNC(v, 2)

					pssm_coord = (directional_lights.data[i].shadow_matrix3 * v);
					pssm_coord /= pssm_coord.w;

					if (directional_lights.data[i].softshadow_angle > 0) {
						float range_pos = dot(directional_lights.data[i].direction, v.xyz);
						float range_begin = directional_lights.data[i].shadow_range_begin.z;
						float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
						vec2 tex_scale = directional_lights.data[i].uv_scale3 * test_radius;
						shadow = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale);
					} else {
						shadow = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale, pssm_coord);
					}

					shadow_color = directional_lights.data[i].shadow_color3.rgb;

				} else {
					vec4 v = vec4(vertex, 1.0);

					BIAS_FUNC(v, 3)

					pssm_coord = (directional_lights.data[i].shadow_matrix4 * v);
					pssm_coord /= pssm_coord.w;

					if (directional_lights.data[i].softshadow_angle > 0) {
						float range_pos = dot(directional_lights.data[i].direction, v.xyz);
						float range_begin = directional_lights.data[i].shadow_range_begin.w;
						float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
						vec2 tex_scale = directional_lights.data[i].uv_scale4 * test_radius;
						shadow = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale);
					} else {
						shadow = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale, pssm_coord);
					}

					shadow_color = directional_lights.data[i].shadow_color4.rgb;
				}

				if (directional_lights.data[i].blend_splits) {
					vec3 shadow_color_blend = vec3(0.0);
					float pssm_blend;
					float shadow2;

					if (depth_z < directional_lights.data[i].shadow_split_offsets.x) {
						vec4 v = vec4(vertex, 1.0);
						BIAS_FUNC(v, 1)
						pssm_coord = (directional_lights.data[i].shadow_matrix2 * v);
						pssm_coord /= pssm_coord.w;

						if (directional_lights.data[i].softshadow_angle > 0) {
							float range_pos = dot(directional_lights.data[i].direction, v.xyz);
							float range_begin = directional_lights.data[i].shadow_range_begin.y;
							float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
							vec2 tex_scale = directional_lights.data[i].uv_scale2 * test_radius;
							shadow2 = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale);
						} else {
							shadow2 = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale, pssm_coord);
						}

						pssm_blend = smoothstep(0.0, directional_lights.data[i].shadow_split_offsets.x, depth_z);
						shadow_color_blend = directional_lights.data[i].shadow_color2.rgb;
					} else if (depth_z < directional_lights.data[i].shadow_split_offsets.y) {
						vec4 v = vec4(vertex, 1.0);
						BIAS_FUNC(v, 2)
						pssm_coord = (directional_lights.data[i].shadow_matrix3 * v);
						pssm_coord /= pssm_coord.w;

						if (directional_lights.data[i].softshadow_angle > 0) {
							float range_pos = dot(directional_lights.data[i].direction, v.xyz);
							float range_begin = directional_lights.data[i].shadow_range_begin.z;
							float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
							vec2 tex_scale = directional_lights.data[i].uv_scale3 * test_radius;
							shadow2 = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale);
						} else {
							shadow2 = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale, pssm_coord);
						}

						pssm_blend = smoothstep(directional_lights.data[i].shadow_split_offsets.x, directional_lights.data[i].shadow_split_offsets.y, depth_z);

						shadow_color_blend = directional_lights.data[i].shadow_color3.rgb;
					} else if (depth_z < directional_lights.data[i].shadow_split_offsets.z) {
						vec4 v = vec4(vertex, 1.0);
						BIAS_FUNC(v, 3)
						pssm_coord = (directional_lights.data[i].shadow_matrix4 * v);
						pssm_coord /= pssm_coord.w;
						if (directional_lights.data[i].softshadow_angle > 0) {
							float range_pos = dot(directional_lights.data[i].direction, v.xyz);
							float range_begin = directional_lights.data[i].shadow_range_begin.w;
							float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
							vec2 tex_scale = directional_lights.data[i].uv_scale4 * test_radius;
							shadow2 = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale);
						} else {
							shadow2 = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale, pssm_coord);
						}

						pssm_blend = smoothstep(directional_lights.data[i].shadow_split_offsets.y, directional_lights.data[i].shadow_split_offsets.z, depth_z);
						shadow_color_blend = directional_lights.data[i].shadow_color4.rgb;
					} else {
						pssm_blend = 0.0; //if no blend, same coord will be used (divide by z will result in same value, and already cached)
					}

					pssm_blend = sqrt(pssm_blend);

					shadow = mix(shadow, shadow2, pssm_blend);
					shadow_color = mix(shadow_color, shadow_color_blend, pssm_blend);
				}

				shadow = mix(shadow, 1.0, smoothstep(directional_lights.data[i].fade_from, directional_lights.data[i].fade_to, vertex.z)); //done with negative values for performance

#undef BIAS_FUNC
			}
#else
			// Soft shadow disabled version

			if (directional_lights.data[i].shadow_enabled) {
				float depth_z = -vertex.z;

				vec4 pssm_coord;
				vec3 light_dir = directional_lights.data[i].direction;
				vec3 base_normal_bias = normalize(normal_interp) * (1.0 - max(0.0, dot(light_dir, -normalize(normal_interp))));

#define BIAS_FUNC(m_var, m_idx)                                                                 \
	m_var.xyz += light_dir * directional_lights.data[i].shadow_bias[m_idx];                     \
	vec3 normal_bias = base_normal_bias * directional_lights.data[i].shadow_normal_bias[m_idx]; \
	normal_bias -= light_dir * dot(light_dir, normal_bias);                                     \
	m_var.xyz += normal_bias;

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

				shadow = mix(shadow, 1.0, smoothstep(directional_lights.data[i].fade_from, directional_lights.data[i].fade_to, vertex.z)); //done with negative values for performance

#undef BIAS_FUNC
			}
#endif

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

			if (!bool(directional_lights.data[i].mask & draw_call.layer_mask)) {
				continue; //not masked
			}

			// We're not doing light transmittence

			float shadow = 1.0;
#ifndef SHADOWS_DISABLED
			if (i < 4) {
				shadow = float(shadow0 >> (i * 8) & 0xFF) / 255.0;
			} else {
				shadow = float(shadow1 >> ((i - 4) * 8) & 0xFF) / 255.0;
			}
#endif
			blur_shadow(shadow);

			light_compute(normal, directional_lights.data[i].direction, normalize(view), 0.0, directional_lights.data[i].color * directional_lights.data[i].energy, shadow, f0, orms, 1.0, albedo, alpha,
#ifdef LIGHT_BACKLIGHT_USED
					backlight,
#endif
/* not supported here
#ifdef LIGHT_TRANSMITTANCE_USED
					transmittance_color,
					transmittance_depth,
					transmittance_boost,
					transmittance_z,
#endif
*/
#ifdef LIGHT_RIM_USED
					rim, rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
					clearcoat, clearcoat_gloss,
#endif
#ifdef LIGHT_ANISOTROPY_USED
					binormal, tangent, anisotropy,
#endif
#ifdef USE_SOFT_SHADOW
					directional_lights.data[i].size,
#endif
					diffuse_light,
					specular_light);
		}
	} //directional light

	if (!sc_disable_omni_lights) { //omni lights
		uint light_indices = draw_call.omni_lights.x;
		for (uint i = 0; i < 8; i++) {
			uint light_index = light_indices & 0xFF;
			if (i == 4) {
				light_indices = draw_call.omni_lights.y;
			} else {
				light_indices = light_indices >> 8;
			}

			if (light_index == 0xFF) {
				break;
			}

			float shadow = light_process_omni_shadow(light_index, vertex, normal);

			shadow = blur_shadow(shadow);

			light_process_omni(light_index, vertex, view, normal, vertex_ddx, vertex_ddy, f0, orms, shadow, albedo, alpha,
#ifdef LIGHT_BACKLIGHT_USED
					backlight,
#endif
/*
#ifdef LIGHT_TRANSMITTANCE_USED
					transmittance_color,
					transmittance_depth,
					transmittance_boost,
#endif
*/
#ifdef LIGHT_RIM_USED
					rim,
					rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
					clearcoat, clearcoat_gloss,
#endif
#ifdef LIGHT_ANISOTROPY_USED
					tangent, binormal, anisotropy,
#endif
					diffuse_light, specular_light);
		}
	} //omni lights

	if (!sc_disable_spot_lights) { //spot lights

		uint light_indices = draw_call.spot_lights.x;
		for (uint i = 0; i < 8; i++) {
			uint light_index = light_indices & 0xFF;
			if (i == 4) {
				light_indices = draw_call.spot_lights.y;
			} else {
				light_indices = light_indices >> 8;
			}

			if (light_index == 0xFF) {
				break;
			}

			float shadow = light_process_spot_shadow(light_index, vertex, normal);

			shadow = blur_shadow(shadow);

			light_process_spot(light_index, vertex, view, normal, vertex_ddx, vertex_ddy, f0, orms, shadow, albedo, alpha,
#ifdef LIGHT_BACKLIGHT_USED
					backlight,
#endif
/*
#ifdef LIGHT_TRANSMITTANCE_USED
					transmittance_color,
					transmittance_depth,
					transmittance_boost,
#endif
*/
#ifdef LIGHT_RIM_USED
					rim,
					rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
					clearcoat, clearcoat_gloss,
#endif
#ifdef LIGHT_ANISOTROPY_USED
					tangent, binormal, anisotropy,
#endif
					diffuse_light, specular_light);
		}
	} //spot lights

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
#endif // MODE_RENDER_MATERIAL

#else // MODE_RENDER_DEPTH

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

#else // MODE_UNSHADED

#ifdef SSS_MODE_SKIN
	sss_strength = -sss_strength;
#endif // SSS_MODE_SKIN
	diffuse_buffer = vec4(emission + diffuse_light + ambient_light, sss_strength);
	specular_buffer = vec4(specular_light, metallic);
#endif // MODE_UNSHADED

	diffuse_buffer.rgb = mix(diffuse_buffer.rgb, fog.rgb, fog.a);
	specular_buffer.rgb = mix(specular_buffer.rgb, vec3(0.0), fog.a);

#else //MODE_MULTIPLE_RENDER_TARGETS

#ifdef MODE_UNSHADED
	frag_color = vec4(albedo, alpha);
#else // MODE_UNSHADED
	frag_color = vec4(emission + ambient_light + diffuse_light + specular_light, alpha);
#endif // MODE_UNSHADED

	// Draw "fixed" fog before volumetric fog to ensure volumetric fog can appear in front of the sky.
	frag_color.rgb = mix(frag_color.rgb, fog.rgb, fog.a);

	// On mobile we use a UNORM buffer with 10bpp which results in a range from 0.0 - 1.0 resulting in HDR breaking
	// We divide by sc_luminance_multiplier to support a range from 0.0 - 2.0 both increasing precision on bright and darker images
	frag_color.rgb = frag_color.rgb / sc_luminance_multiplier;

#endif //MODE_MULTIPLE_RENDER_TARGETS

#endif //MODE_RENDER_DEPTH
}
