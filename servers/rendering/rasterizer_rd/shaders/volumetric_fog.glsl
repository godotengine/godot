#[compute]

#version 450

VERSION_DEFINES

#if defined(MODE_FOG) || defined(MODE_FILTER)

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#endif

#if defined(MODE_DENSITY)

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

#endif

#include "cluster_data_inc.glsl"

#define M_PI 3.14159265359

layout(set = 0, binding = 1) uniform texture2D shadow_atlas;
layout(set = 0, binding = 2) uniform texture2D directional_shadow_atlas;

layout(set = 0, binding = 3, std430) restrict readonly buffer Lights {
	LightData data[];
}
lights;

layout(set = 0, binding = 4, std140) uniform DirectionalLights {
	DirectionalLightData data[MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS];
}
directional_lights;

layout(set = 0, binding = 5) uniform utexture3D cluster_texture;

layout(set = 0, binding = 6, std430) restrict readonly buffer ClusterData {
	uint indices[];
}
cluster_data;

layout(set = 0, binding = 7) uniform sampler linear_sampler;

#ifdef MODE_DENSITY
layout(rgba16f, set = 0, binding = 8) uniform restrict writeonly image3D density_map;
layout(rgba16f, set = 0, binding = 9) uniform restrict readonly image3D fog_map; //unused
#endif

#ifdef MODE_FOG
layout(rgba16f, set = 0, binding = 8) uniform restrict readonly image3D density_map;
layout(rgba16f, set = 0, binding = 9) uniform restrict writeonly image3D fog_map;
#endif

#ifdef MODE_FILTER
layout(rgba16f, set = 0, binding = 8) uniform restrict readonly image3D source_map;
layout(rgba16f, set = 0, binding = 9) uniform restrict writeonly image3D dest_map;
#endif

layout(set = 0, binding = 10) uniform sampler shadow_sampler;

#define MAX_GI_PROBES 8

struct GIProbeData {
	mat4 xform;
	vec3 bounds;
	float dynamic_range;

	float bias;
	float normal_bias;
	bool blend_ambient;
	uint texture_slot;

	float anisotropy_strength;
	float ambient_occlusion;
	float ambient_occlusion_size;
	uint mipmaps;
};

layout(set = 0, binding = 11, std140) uniform GIProbes {
	GIProbeData data[MAX_GI_PROBES];
}
gi_probes;

layout(set = 0, binding = 12) uniform texture3D gi_probe_textures[MAX_GI_PROBES];

layout(set = 0, binding = 13) uniform sampler linear_sampler_with_mipmaps;

#ifdef ENABLE_SDFGI

// SDFGI Integration on set 1
#define SDFGI_MAX_CASCADES 8

struct SDFGIProbeCascadeData {
	vec3 position;
	float to_probe;
	ivec3 probe_world_offset;
	float to_cell; // 1/bounds * grid_size
};

layout(set = 1, binding = 0, std140) uniform SDFGI {
	vec3 grid_size;
	uint max_cascades;

	bool use_occlusion;
	int probe_axis_size;
	float probe_to_uvw;
	float normal_bias;

	vec3 lightprobe_tex_pixel_size;
	float energy;

	vec3 lightprobe_uv_offset;
	float y_mult;

	vec3 occlusion_clamp;
	uint pad3;

	vec3 occlusion_renormalize;
	uint pad4;

	vec3 cascade_probe_size;
	uint pad5;

	SDFGIProbeCascadeData cascades[SDFGI_MAX_CASCADES];
}
sdfgi;

layout(set = 1, binding = 1) uniform texture2DArray sdfgi_ambient_texture;

layout(set = 1, binding = 2) uniform texture3D sdfgi_occlusion_texture;

#endif //SDFGI

layout(push_constant, binding = 0, std430) uniform Params {
	vec2 fog_frustum_size_begin;
	vec2 fog_frustum_size_end;

	float fog_frustum_end;
	float z_near;
	float z_far;
	int filter_axis;

	ivec3 fog_volume_size;
	uint directional_light_count;

	vec3 light_color;
	float base_density;

	float detail_spread;
	float gi_inject;
	uint max_gi_probes;
	uint pad;

	mat3x4 cam_rotation;
}
params;

float get_depth_at_pos(float cell_depth_size, int z) {
	float d = float(z) * cell_depth_size + cell_depth_size * 0.5; //center of voxels
	d = pow(d, params.detail_spread);
	return params.fog_frustum_end * d;
}

vec3 hash3f(uvec3 x) {
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = (x >> 16) ^ x;
	return vec3(x & 0xFFFFF) / vec3(float(0xFFFFF));
}

void main() {
	vec3 fog_cell_size = 1.0 / vec3(params.fog_volume_size);

#ifdef MODE_DENSITY

	ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);
	if (any(greaterThanEqual(pos, params.fog_volume_size))) {
		return; //do not compute
	}

	vec3 posf = vec3(pos);

	//posf += mix(vec3(0.0),vec3(1.0),0.3) * hash3f(uvec3(pos)) * 2.0 - 1.0;

	vec3 fog_unit_pos = posf * fog_cell_size + fog_cell_size * 0.5; //center of voxels
	fog_unit_pos.z = pow(fog_unit_pos.z, params.detail_spread);

	vec3 view_pos;
	view_pos.xy = (fog_unit_pos.xy * 2.0 - 1.0) * mix(params.fog_frustum_size_begin, params.fog_frustum_size_end, vec2(fog_unit_pos.z));
	view_pos.z = -params.fog_frustum_end * fog_unit_pos.z;
	view_pos.y = -view_pos.y;

	vec3 total_light = params.light_color;

	float total_density = params.base_density;
	float cell_depth_size = abs(view_pos.z - get_depth_at_pos(fog_cell_size.z, pos.z + 1));
	//compute directional lights

	for (uint i = 0; i < params.directional_light_count; i++) {
		vec3 shadow_attenuation = vec3(1.0);

		if (directional_lights.data[i].shadow_enabled) {
			float depth_z = -view_pos.z;

			vec4 pssm_coord;
			vec3 shadow_color = directional_lights.data[i].shadow_color1.rgb;
			vec3 light_dir = directional_lights.data[i].direction;
			vec4 v = vec4(view_pos, 1.0);
			float z_range;

			if (depth_z < directional_lights.data[i].shadow_split_offsets.x) {
				pssm_coord = (directional_lights.data[i].shadow_matrix1 * v);
				pssm_coord /= pssm_coord.w;
				z_range = directional_lights.data[i].shadow_z_range.x;

			} else if (depth_z < directional_lights.data[i].shadow_split_offsets.y) {
				pssm_coord = (directional_lights.data[i].shadow_matrix2 * v);
				pssm_coord /= pssm_coord.w;
				z_range = directional_lights.data[i].shadow_z_range.y;

			} else if (depth_z < directional_lights.data[i].shadow_split_offsets.z) {
				pssm_coord = (directional_lights.data[i].shadow_matrix3 * v);
				pssm_coord /= pssm_coord.w;
				z_range = directional_lights.data[i].shadow_z_range.z;

			} else {
				pssm_coord = (directional_lights.data[i].shadow_matrix4 * v);
				pssm_coord /= pssm_coord.w;
				z_range = directional_lights.data[i].shadow_z_range.w;
			}

			float depth = texture(sampler2D(directional_shadow_atlas, linear_sampler), pssm_coord.xy).r;
			float shadow = exp(min(0.0, (depth - pssm_coord.z)) * z_range * directional_lights.data[i].shadow_volumetric_fog_fade);

			/*
			//float shadow = textureProj(sampler2DShadow(directional_shadow_atlas,shadow_sampler),pssm_coord);
			float shadow = 0.0;
			for(float xi=-1;xi<=1;xi++) {
				for(float yi=-1;yi<=1;yi++) {
					vec2 ofs = vec2(xi,yi) * 1.5 * params.directional_shadow_pixel_size;
					shadow += textureProj(sampler2DShadow(directional_shadow_atlas,shadow_sampler),pssm_coord + vec4(ofs,0.0,0.0));
				}

			}

			shadow /= 3.0 * 3.0;

*/
			shadow = mix(shadow, 1.0, smoothstep(directional_lights.data[i].fade_from, directional_lights.data[i].fade_to, view_pos.z)); //done with negative values for performance

			shadow_attenuation = mix(shadow_color, vec3(1.0), shadow);
		}

		total_light += shadow_attenuation * directional_lights.data[i].color * directional_lights.data[i].energy / M_PI;
	}

	//compute lights from cluster

	vec3 cluster_pos;
	cluster_pos.xy = fog_unit_pos.xy;
	cluster_pos.z = clamp((abs(view_pos.z) - params.z_near) / (params.z_far - params.z_near), 0.0, 1.0);

	uvec4 cluster_cell = texture(usampler3D(cluster_texture, linear_sampler), cluster_pos);

	uint omni_light_count = cluster_cell.x >> CLUSTER_COUNTER_SHIFT;
	uint omni_light_pointer = cluster_cell.x & CLUSTER_POINTER_MASK;

	for (uint i = 0; i < omni_light_count; i++) {
		uint light_index = cluster_data.indices[omni_light_pointer + i];

		vec3 light_pos = lights.data[i].position;
		float d = distance(lights.data[i].position, view_pos) * lights.data[i].inv_radius;
		vec3 shadow_attenuation = vec3(1.0);

		if (d < 1.0) {
			vec2 attenuation_energy = unpackHalf2x16(lights.data[i].attenuation_energy);
			vec4 color_specular = unpackUnorm4x8(lights.data[i].color_specular);

			float attenuation = pow(max(1.0 - d, 0.0), attenuation_energy.x);

			vec3 light = attenuation_energy.y * color_specular.rgb / M_PI;

			vec4 shadow_color_enabled = unpackUnorm4x8(lights.data[i].shadow_color_enabled);

			if (shadow_color_enabled.a > 0.5) {
				//has shadow
				vec4 v = vec4(view_pos, 1.0);

				vec4 splane = (lights.data[i].shadow_matrix * v);
				float shadow_len = length(splane.xyz); //need to remember shadow len from here

				splane.xyz = normalize(splane.xyz);
				vec4 clamp_rect = lights.data[i].atlas_rect;

				if (splane.z >= 0.0) {
					splane.z += 1.0;

					clamp_rect.y += clamp_rect.w;

				} else {
					splane.z = 1.0 - splane.z;
				}

				splane.xy /= splane.z;

				splane.xy = splane.xy * 0.5 + 0.5;
				splane.z = shadow_len * lights.data[i].inv_radius;
				splane.xy = clamp_rect.xy + splane.xy * clamp_rect.zw;
				splane.w = 1.0; //needed? i think it should be 1 already

				float depth = texture(sampler2D(shadow_atlas, linear_sampler), splane.xy).r;
				float shadow = exp(min(0.0, (depth - splane.z)) / lights.data[i].inv_radius * lights.data[i].shadow_volumetric_fog_fade);

				shadow_attenuation = mix(shadow_color_enabled.rgb, vec3(1.0), shadow);
			}
			total_light += light * attenuation * shadow_attenuation;
		}
	}

	uint spot_light_count = cluster_cell.y >> CLUSTER_COUNTER_SHIFT;
	uint spot_light_pointer = cluster_cell.y & CLUSTER_POINTER_MASK;

	for (uint i = 0; i < spot_light_count; i++) {
		uint light_index = cluster_data.indices[spot_light_pointer + i];

		vec3 light_pos = lights.data[i].position;
		vec3 light_rel_vec = lights.data[i].position - view_pos;
		float d = length(light_rel_vec) * lights.data[i].inv_radius;
		vec3 shadow_attenuation = vec3(1.0);

		if (d < 1.0) {
			vec2 attenuation_energy = unpackHalf2x16(lights.data[i].attenuation_energy);
			vec4 color_specular = unpackUnorm4x8(lights.data[i].color_specular);

			float attenuation = pow(max(1.0 - d, 0.0), attenuation_energy.x);

			vec3 spot_dir = lights.data[i].direction;
			vec2 spot_att_angle = unpackHalf2x16(lights.data[i].cone_attenuation_angle);
			float scos = max(dot(-normalize(light_rel_vec), spot_dir), spot_att_angle.y);
			float spot_rim = max(0.0001, (1.0 - scos) / (1.0 - spot_att_angle.y));
			attenuation *= 1.0 - pow(spot_rim, spot_att_angle.x);

			vec3 light = attenuation_energy.y * color_specular.rgb / M_PI;

			vec4 shadow_color_enabled = unpackUnorm4x8(lights.data[i].shadow_color_enabled);

			if (shadow_color_enabled.a > 0.5) {
				//has shadow
				vec4 v = vec4(view_pos, 1.0);

				vec4 splane = (lights.data[i].shadow_matrix * v);
				splane /= splane.w;

				float depth = texture(sampler2D(shadow_atlas, linear_sampler), splane.xy).r;
				float shadow = exp(min(0.0, (depth - splane.z)) / lights.data[i].inv_radius * lights.data[i].shadow_volumetric_fog_fade);

				shadow_attenuation = mix(shadow_color_enabled.rgb, vec3(1.0), shadow);
			}

			total_light += light * attenuation * shadow_attenuation;
		}
	}

	vec3 world_pos = mat3(params.cam_rotation) * view_pos;

	for (uint i = 0; i < params.max_gi_probes; i++) {
		vec3 position = (gi_probes.data[i].xform * vec4(world_pos, 1.0)).xyz;

		//this causes corrupted pixels, i have no idea why..
		if (all(bvec2(all(greaterThanEqual(position, vec3(0.0))), all(lessThan(position, gi_probes.data[i].bounds))))) {
			position /= gi_probes.data[i].bounds;

			vec4 light = vec4(0.0);
			for (uint j = 0; j < gi_probes.data[i].mipmaps; j++) {
				vec4 slight = textureLod(sampler3D(gi_probe_textures[i], linear_sampler_with_mipmaps), position, float(j));
				float a = (1.0 - light.a);
				light += a * slight;
			}

			light.rgb *= gi_probes.data[i].dynamic_range * params.gi_inject;

			total_light += light.rgb;
		}
	}

	//sdfgi
#ifdef ENABLE_SDFGI

	{
		float blend = -1.0;
		vec3 ambient_total = vec3(0.0);

		for (uint i = 0; i < sdfgi.max_cascades; i++) {
			vec3 cascade_pos = (world_pos - sdfgi.cascades[i].position) * sdfgi.cascades[i].to_probe;

			if (any(lessThan(cascade_pos, vec3(0.0))) || any(greaterThanEqual(cascade_pos, sdfgi.cascade_probe_size))) {
				continue; //skip cascade
			}

			vec3 base_pos = floor(cascade_pos);
			ivec3 probe_base_pos = ivec3(base_pos);

			vec4 ambient_accum = vec4(0.0);

			ivec3 tex_pos = ivec3(probe_base_pos.xy, int(i));
			tex_pos.x += probe_base_pos.z * sdfgi.probe_axis_size;

			for (uint j = 0; j < 8; j++) {
				ivec3 offset = (ivec3(j) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1);
				ivec3 probe_posi = probe_base_pos;
				probe_posi += offset;

				// Compute weight

				vec3 probe_pos = vec3(probe_posi);
				vec3 probe_to_pos = cascade_pos - probe_pos;

				vec3 trilinear = vec3(1.0) - abs(probe_to_pos);
				float weight = trilinear.x * trilinear.y * trilinear.z;

				// Compute lightprobe occlusion

				if (sdfgi.use_occlusion) {
					ivec3 occ_indexv = abs((sdfgi.cascades[i].probe_world_offset + probe_posi) & ivec3(1, 1, 1)) * ivec3(1, 2, 4);
					vec4 occ_mask = mix(vec4(0.0), vec4(1.0), equal(ivec4(occ_indexv.x | occ_indexv.y), ivec4(0, 1, 2, 3)));

					vec3 occ_pos = clamp(cascade_pos, probe_pos - sdfgi.occlusion_clamp, probe_pos + sdfgi.occlusion_clamp) * sdfgi.probe_to_uvw;
					occ_pos.z += float(i);
					if (occ_indexv.z != 0) { //z bit is on, means index is >=4, so make it switch to the other half of textures
						occ_pos.x += 1.0;
					}

					occ_pos *= sdfgi.occlusion_renormalize;
					float occlusion = dot(textureLod(sampler3D(sdfgi_occlusion_texture, linear_sampler), occ_pos, 0.0), occ_mask);

					weight *= max(occlusion, 0.01);
				}

				// Compute ambient texture position

				ivec3 uvw = tex_pos;
				uvw.xy += offset.xy;
				uvw.x += offset.z * sdfgi.probe_axis_size;

				vec3 ambient = texelFetch(sampler2DArray(sdfgi_ambient_texture, linear_sampler), uvw, 0).rgb;

				ambient_accum.rgb += ambient * weight;
				ambient_accum.a += weight;
			}

			if (ambient_accum.a > 0) {
				ambient_accum.rgb /= ambient_accum.a;
			}
			ambient_total = ambient_accum.rgb;
			break;
		}

		total_light += ambient_total * params.gi_inject;
	}

#endif

	imageStore(density_map, pos, vec4(total_light, total_density));
#endif

#ifdef MODE_FOG

	ivec3 pos = ivec3(gl_GlobalInvocationID.xy, 0);

	if (any(greaterThanEqual(pos, params.fog_volume_size))) {
		return; //do not compute
	}

	vec4 fog_accum = vec4(0.0);
	float prev_z = 0.0;

	float t = 1.0;

	for (int i = 0; i < params.fog_volume_size.z; i++) {
		//compute fog position
		ivec3 fog_pos = pos + ivec3(0, 0, i);
		//get fog value
		vec4 fog = imageLoad(density_map, fog_pos);

		//get depth at cell pos
		float z = get_depth_at_pos(fog_cell_size.z, i);
		//get distance from previous pos
		float d = abs(prev_z - z);
		//compute exinction based on beer's
		float extinction = t * exp(-d * fog.a);
		//compute alpha based on different of extinctions
		float alpha = t - extinction;
		//update extinction
		t = extinction;

		fog_accum += vec4(fog.rgb * alpha, alpha);
		prev_z = z;

		vec4 fog_value;

		if (fog_accum.a > 0.0) {
			fog_value = vec4(fog_accum.rgb / fog_accum.a, 1.0 - t);
		} else {
			fog_value = vec4(0.0);
		}

		imageStore(fog_map, fog_pos, fog_value);
	}

#endif

#ifdef MODE_FILTER

	ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);

	const float gauss[7] = float[](0.071303, 0.131514, 0.189879, 0.214607, 0.189879, 0.131514, 0.071303);

	const ivec3 filter_dir[3] = ivec3[](ivec3(1, 0, 0), ivec3(0, 1, 0), ivec3(0, 0, 1));
	ivec3 offset = filter_dir[params.filter_axis];

	vec4 accum = vec4(0.0);
	for (int i = -3; i <= 3; i++) {
		accum += imageLoad(source_map, clamp(pos + offset * i, ivec3(0), params.fog_volume_size - ivec3(1))) * gauss[i + 3];
	}

	imageStore(dest_map, pos, accum);

#endif
}
