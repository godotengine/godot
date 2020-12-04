#[compute]

#version 450

VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#define M_PI 3.141592

#define SDFGI_MAX_CASCADES 8

//set 0 for SDFGI and render buffers

layout(set = 0, binding = 1) uniform texture3D sdf_cascades[SDFGI_MAX_CASCADES];
layout(set = 0, binding = 2) uniform texture3D light_cascades[SDFGI_MAX_CASCADES];
layout(set = 0, binding = 3) uniform texture3D aniso0_cascades[SDFGI_MAX_CASCADES];
layout(set = 0, binding = 4) uniform texture3D aniso1_cascades[SDFGI_MAX_CASCADES];
layout(set = 0, binding = 5) uniform texture3D occlusion_texture;

layout(set = 0, binding = 6) uniform sampler linear_sampler;
layout(set = 0, binding = 7) uniform sampler linear_sampler_with_mipmaps;

struct ProbeCascadeData {
	vec3 position;
	float to_probe;
	ivec3 probe_world_offset;
	float to_cell; // 1/bounds * grid_size
};

layout(rgba16f, set = 0, binding = 9) uniform restrict writeonly image2D ambient_buffer;
layout(rgba16f, set = 0, binding = 10) uniform restrict writeonly image2D reflection_buffer;

layout(set = 0, binding = 11) uniform texture2DArray lightprobe_texture;

layout(set = 0, binding = 12) uniform texture2D depth_buffer;
layout(set = 0, binding = 13) uniform texture2D normal_roughness_buffer;
layout(set = 0, binding = 14) uniform utexture2D giprobe_buffer;

layout(set = 0, binding = 15, std140) uniform SDFGI {
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

	ProbeCascadeData cascades[SDFGI_MAX_CASCADES];
}
sdfgi;

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

layout(set = 0, binding = 16, std140) uniform GIProbes {
	GIProbeData data[MAX_GI_PROBES];
}
gi_probes;

layout(set = 0, binding = 17) uniform texture3D gi_probe_textures[MAX_GI_PROBES];

layout(push_constant, binding = 0, std430) uniform Params {
	ivec2 screen_size;
	float z_near;
	float z_far;

	vec4 proj_info;

	uint max_giprobes;
	bool high_quality_vct;
	bool use_sdfgi;
	bool orthogonal;

	vec3 ao_color;
	uint pad;

	mat3x4 cam_rotation;
}
params;

vec2 octahedron_wrap(vec2 v) {
	vec2 signVal;
	signVal.x = v.x >= 0.0 ? 1.0 : -1.0;
	signVal.y = v.y >= 0.0 ? 1.0 : -1.0;
	return (1.0 - abs(v.yx)) * signVal;
}

vec2 octahedron_encode(vec3 n) {
	// https://twitter.com/Stubbesaurus/status/937994790553227264
	n /= (abs(n.x) + abs(n.y) + abs(n.z));
	n.xy = n.z >= 0.0 ? n.xy : octahedron_wrap(n.xy);
	n.xy = n.xy * 0.5 + 0.5;
	return n.xy;
}

vec4 blend_color(vec4 src, vec4 dst) {
	vec4 res;
	float sa = 1.0 - src.a;
	res.a = dst.a * sa + src.a;
	if (res.a == 0.0) {
		res.rgb = vec3(0);
	} else {
		res.rgb = (dst.rgb * dst.a * sa + src.rgb * src.a) / res.a;
	}
	return res;
}

vec3 reconstruct_position(ivec2 screen_pos) {
	vec3 pos;
	pos.z = texelFetch(sampler2D(depth_buffer, linear_sampler), screen_pos, 0).r;

	pos.z = pos.z * 2.0 - 1.0;
	if (params.orthogonal) {
		pos.z = ((pos.z + (params.z_far + params.z_near) / (params.z_far - params.z_near)) * (params.z_far - params.z_near)) / 2.0;
	} else {
		pos.z = 2.0 * params.z_near * params.z_far / (params.z_far + params.z_near - pos.z * (params.z_far - params.z_near));
	}
	pos.z = -pos.z;

	pos.xy = vec2(screen_pos) * params.proj_info.xy + params.proj_info.zw;
	if (!params.orthogonal) {
		pos.xy *= pos.z;
	}

	return pos;
}

void sdfgi_probe_process(uint cascade, vec3 cascade_pos, vec3 cam_pos, vec3 cam_normal, vec3 cam_specular_normal, float roughness, out vec3 diffuse_light, out vec3 specular_light) {
	cascade_pos += cam_normal * sdfgi.normal_bias;

	vec3 base_pos = floor(cascade_pos);
	//cascade_pos += mix(vec3(0.0),vec3(0.01),lessThan(abs(cascade_pos-base_pos),vec3(0.01))) * cam_normal;
	ivec3 probe_base_pos = ivec3(base_pos);

	vec4 diffuse_accum = vec4(0.0);
	vec3 specular_accum;

	ivec3 tex_pos = ivec3(probe_base_pos.xy, int(cascade));
	tex_pos.x += probe_base_pos.z * sdfgi.probe_axis_size;
	tex_pos.xy = tex_pos.xy * (SDFGI_OCT_SIZE + 2) + ivec2(1);

	vec3 diffuse_posf = (vec3(tex_pos) + vec3(octahedron_encode(cam_normal) * float(SDFGI_OCT_SIZE), 0.0)) * sdfgi.lightprobe_tex_pixel_size;

	vec3 specular_posf = (vec3(tex_pos) + vec3(octahedron_encode(cam_specular_normal) * float(SDFGI_OCT_SIZE), 0.0)) * sdfgi.lightprobe_tex_pixel_size;

	specular_accum = vec3(0.0);

	vec4 light_accum = vec4(0.0);
	float weight_accum = 0.0;

	for (uint j = 0; j < 8; j++) {
		ivec3 offset = (ivec3(j) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1);
		ivec3 probe_posi = probe_base_pos;
		probe_posi += offset;

		// Compute weight

		vec3 probe_pos = vec3(probe_posi);
		vec3 probe_to_pos = cascade_pos - probe_pos;
		vec3 probe_dir = normalize(-probe_to_pos);

		vec3 trilinear = vec3(1.0) - abs(probe_to_pos);
		float weight = trilinear.x * trilinear.y * trilinear.z * max(0.005, dot(cam_normal, probe_dir));

		// Compute lightprobe occlusion

		if (sdfgi.use_occlusion) {
			ivec3 occ_indexv = abs((sdfgi.cascades[cascade].probe_world_offset + probe_posi) & ivec3(1, 1, 1)) * ivec3(1, 2, 4);
			vec4 occ_mask = mix(vec4(0.0), vec4(1.0), equal(ivec4(occ_indexv.x | occ_indexv.y), ivec4(0, 1, 2, 3)));

			vec3 occ_pos = clamp(cascade_pos, probe_pos - sdfgi.occlusion_clamp, probe_pos + sdfgi.occlusion_clamp) * sdfgi.probe_to_uvw;
			occ_pos.z += float(cascade);
			if (occ_indexv.z != 0) { //z bit is on, means index is >=4, so make it switch to the other half of textures
				occ_pos.x += 1.0;
			}

			occ_pos *= sdfgi.occlusion_renormalize;
			float occlusion = dot(textureLod(sampler3D(occlusion_texture, linear_sampler), occ_pos, 0.0), occ_mask);

			weight *= max(occlusion, 0.01);
		}

		// Compute lightprobe texture position

		vec3 diffuse;
		vec3 pos_uvw = diffuse_posf;
		pos_uvw.xy += vec2(offset.xy) * sdfgi.lightprobe_uv_offset.xy;
		pos_uvw.x += float(offset.z) * sdfgi.lightprobe_uv_offset.z;
		diffuse = textureLod(sampler2DArray(lightprobe_texture, linear_sampler), pos_uvw, 0.0).rgb;

		diffuse_accum += vec4(diffuse * weight, weight);

		{
			vec3 specular = vec3(0.0);
			vec3 pos_uvw = specular_posf;
			pos_uvw.xy += vec2(offset.xy) * sdfgi.lightprobe_uv_offset.xy;
			pos_uvw.x += float(offset.z) * sdfgi.lightprobe_uv_offset.z;
			if (roughness < 0.99) {
				specular = textureLod(sampler2DArray(lightprobe_texture, linear_sampler), pos_uvw + vec3(0, 0, float(sdfgi.max_cascades)), 0.0).rgb;
			}
			if (roughness > 0.2) {
				specular = mix(specular, textureLod(sampler2DArray(lightprobe_texture, linear_sampler), pos_uvw, 0.0).rgb, (roughness - 0.2) * 1.25);
			}

			specular_accum += specular * weight;
		}
	}

	if (diffuse_accum.a > 0.0) {
		diffuse_accum.rgb /= diffuse_accum.a;
	}

	diffuse_light = diffuse_accum.rgb;

	if (diffuse_accum.a > 0.0) {
		specular_accum /= diffuse_accum.a;
	}

	specular_light = specular_accum;
}

void sdfgi_process(vec3 vertex, vec3 normal, vec3 reflection, float roughness, out vec4 ambient_light, out vec4 reflection_light) {
	//make vertex orientation the world one, but still align to camera
	vertex.y *= sdfgi.y_mult;
	normal.y *= sdfgi.y_mult;
	reflection.y *= sdfgi.y_mult;

	//renormalize
	normal = normalize(normal);
	reflection = normalize(reflection);

	vec3 cam_pos = vertex;
	vec3 cam_normal = normal;

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
		ambient_light = vec4(0, 0, 0, 1);
		reflection_light = vec4(0, 0, 0, 1);

		float blend;
		vec3 diffuse, specular;
		sdfgi_probe_process(cascade, cascade_pos, cam_pos, cam_normal, reflection, roughness, diffuse, specular);

		{
			//process blend
			float blend_from = (float(sdfgi.probe_axis_size - 1) / 2.0) - 2.5;
			float blend_to = blend_from + 2.0;

			vec3 inner_pos = cam_pos * sdfgi.cascades[cascade].to_probe;

			float len = length(inner_pos);

			inner_pos = abs(normalize(inner_pos));
			len *= max(inner_pos.x, max(inner_pos.y, inner_pos.z));

			if (len >= blend_from) {
				blend = smoothstep(blend_from, blend_to, len);
			} else {
				blend = 0.0;
			}
		}

		if (blend > 0.0) {
			//blend
			if (cascade == sdfgi.max_cascades - 1) {
				ambient_light.a = 1.0 - blend;
				reflection_light.a = 1.0 - blend;

			} else {
				vec3 diffuse2, specular2;
				cascade_pos = (cam_pos - sdfgi.cascades[cascade + 1].position) * sdfgi.cascades[cascade + 1].to_probe;
				sdfgi_probe_process(cascade + 1, cascade_pos, cam_pos, cam_normal, reflection, roughness, diffuse2, specular2);
				diffuse = mix(diffuse, diffuse2, blend);
				specular = mix(specular, specular2, blend);
			}
		}

		ambient_light.rgb = diffuse;
#if 1
		if (roughness < 0.2) {
			vec3 pos_to_uvw = 1.0 / sdfgi.grid_size;
			vec4 light_accum = vec4(0.0);

			float blend_size = (sdfgi.grid_size.x / float(sdfgi.probe_axis_size - 1)) * 0.5;

			float radius_sizes[SDFGI_MAX_CASCADES];
			cascade = 0xFFFF;

			float base_distance = length(cam_pos);
			for (uint i = 0; i < sdfgi.max_cascades; i++) {
				radius_sizes[i] = (1.0 / sdfgi.cascades[i].to_cell) * (sdfgi.grid_size.x * 0.5 - blend_size);
				if (cascade == 0xFFFF && base_distance < radius_sizes[i]) {
					cascade = i;
				}
			}

			cascade = min(cascade, sdfgi.max_cascades - 1);

			float max_distance = radius_sizes[sdfgi.max_cascades - 1];
			vec3 ray_pos = cam_pos;
			vec3 ray_dir = reflection;

			{
				float prev_radius = cascade > 0 ? radius_sizes[cascade - 1] : 0.0;
				float base_blend = (base_distance - prev_radius) / (radius_sizes[cascade] - prev_radius);
				float bias = (1.0 + base_blend) * 1.1;
				vec3 abs_ray_dir = abs(ray_dir);
				//ray_pos += ray_dir * (bias / sdfgi.cascades[cascade].to_cell); //bias to avoid self occlusion
				ray_pos += (ray_dir * 1.0 / max(abs_ray_dir.x, max(abs_ray_dir.y, abs_ray_dir.z)) + cam_normal * 1.4) * bias / sdfgi.cascades[cascade].to_cell;
			}

			float softness = 0.2 + min(1.0, roughness * 5.0) * 4.0; //approximation to roughness so it does not seem like a hard fade
			while (length(ray_pos) < max_distance) {
				for (uint i = 0; i < sdfgi.max_cascades; i++) {
					if (i >= cascade && length(ray_pos) < radius_sizes[i]) {
						cascade = max(i, cascade); //never go down

						vec3 pos = ray_pos - sdfgi.cascades[i].position;
						pos *= sdfgi.cascades[i].to_cell * pos_to_uvw;

						float distance = texture(sampler3D(sdf_cascades[i], linear_sampler), pos).r * 255.0 - 1.1;

						vec4 hit_light = vec4(0.0);
						if (distance < softness) {
							hit_light.rgb = texture(sampler3D(light_cascades[i], linear_sampler), pos).rgb;
							hit_light.rgb *= 0.5; //approximation given value read is actually meant for anisotropy
							hit_light.a = clamp(1.0 - (distance / softness), 0.0, 1.0);
							hit_light.rgb *= hit_light.a;
						}

						distance /= sdfgi.cascades[i].to_cell;

						if (i < (sdfgi.max_cascades - 1)) {
							pos = ray_pos - sdfgi.cascades[i + 1].position;
							pos *= sdfgi.cascades[i + 1].to_cell * pos_to_uvw;

							float distance2 = texture(sampler3D(sdf_cascades[i + 1], linear_sampler), pos).r * 255.0 - 1.1;

							vec4 hit_light2 = vec4(0.0);
							if (distance2 < softness) {
								hit_light2.rgb = texture(sampler3D(light_cascades[i + 1], linear_sampler), pos).rgb;
								hit_light2.rgb *= 0.5; //approximation given value read is actually meant for anisotropy
								hit_light2.a = clamp(1.0 - (distance2 / softness), 0.0, 1.0);
								hit_light2.rgb *= hit_light2.a;
							}

							float prev_radius = i == 0 ? 0.0 : radius_sizes[i - 1];
							float blend = clamp((length(ray_pos) - prev_radius) / (radius_sizes[i] - prev_radius), 0.0, 1.0);

							distance2 /= sdfgi.cascades[i + 1].to_cell;

							hit_light = mix(hit_light, hit_light2, blend);
							distance = mix(distance, distance2, blend);
						}

						light_accum += hit_light;
						ray_pos += ray_dir * distance;
						break;
					}
				}

				if (light_accum.a > 0.99) {
					break;
				}
			}

			vec3 light = light_accum.rgb / max(light_accum.a, 0.00001);
			float alpha = min(1.0, light_accum.a);

			float b = min(1.0, roughness * 5.0);

			float sa = 1.0 - b;

			reflection_light.a = alpha * sa + b;
			if (reflection_light.a == 0) {
				specular = vec3(0.0);
			} else {
				specular = (light * alpha * sa + specular * b) / reflection_light.a;
			}
		}

#endif

		reflection_light.rgb = specular;

		ambient_light.rgb *= sdfgi.energy;
		reflection_light.rgb *= sdfgi.energy;
	} else {
		ambient_light = vec4(0);
		reflection_light = vec4(0);
	}
}

//standard voxel cone trace
vec4 voxel_cone_trace(texture3D probe, vec3 cell_size, vec3 pos, vec3 direction, float tan_half_angle, float max_distance, float p_bias) {
	float dist = p_bias;
	vec4 color = vec4(0.0);

	while (dist < max_distance && color.a < 0.95) {
		float diameter = max(1.0, 2.0 * tan_half_angle * dist);
		vec3 uvw_pos = (pos + dist * direction) * cell_size;
		float half_diameter = diameter * 0.5;
		//check if outside, then break
		if (any(greaterThan(abs(uvw_pos - 0.5), vec3(0.5f + half_diameter * cell_size)))) {
			break;
		}
		vec4 scolor = textureLod(sampler3D(probe, linear_sampler_with_mipmaps), uvw_pos, log2(diameter));
		float a = (1.0 - color.a);
		color += a * scolor;
		dist += half_diameter;
	}

	return color;
}

vec4 voxel_cone_trace_45_degrees(texture3D probe, vec3 cell_size, vec3 pos, vec3 direction, float max_distance, float p_bias) {
	float dist = p_bias;
	vec4 color = vec4(0.0);
	float radius = max(0.5, dist);
	float lod_level = log2(radius * 2.0);

	while (dist < max_distance && color.a < 0.95) {
		vec3 uvw_pos = (pos + dist * direction) * cell_size;

		//check if outside, then break
		if (any(greaterThan(abs(uvw_pos - 0.5), vec3(0.5f + radius * cell_size)))) {
			break;
		}
		vec4 scolor = textureLod(sampler3D(probe, linear_sampler_with_mipmaps), uvw_pos, lod_level);
		lod_level += 1.0;

		float a = (1.0 - color.a);
		scolor *= a;
		color += scolor;
		dist += radius;
		radius = max(0.5, dist);
	}
	return color;
}

void gi_probe_compute(uint index, vec3 position, vec3 normal, vec3 ref_vec, mat3 normal_xform, float roughness, inout vec4 out_spec, inout vec4 out_diff, inout float out_blend) {
	position = (gi_probes.data[index].xform * vec4(position, 1.0)).xyz;
	ref_vec = normalize((gi_probes.data[index].xform * vec4(ref_vec, 0.0)).xyz);
	normal = normalize((gi_probes.data[index].xform * vec4(normal, 0.0)).xyz);

	position += normal * gi_probes.data[index].normal_bias;

	//this causes corrupted pixels, i have no idea why..
	if (any(bvec2(any(lessThan(position, vec3(0.0))), any(greaterThan(position, gi_probes.data[index].bounds))))) {
		return;
	}

	mat3 dir_xform = mat3(gi_probes.data[index].xform) * normal_xform;

	vec3 blendv = abs(position / gi_probes.data[index].bounds * 2.0 - 1.0);
	float blend = clamp(1.0 - max(blendv.x, max(blendv.y, blendv.z)), 0.0, 1.0);
	//float blend=1.0;

	float max_distance = length(gi_probes.data[index].bounds);
	vec3 cell_size = 1.0 / gi_probes.data[index].bounds;

	//irradiance

	vec4 light = vec4(0.0);

	if (params.high_quality_vct) {
		const uint cone_dir_count = 6;
		vec3 cone_dirs[cone_dir_count] = vec3[](
				vec3(0.0, 0.0, 1.0),
				vec3(0.866025, 0.0, 0.5),
				vec3(0.267617, 0.823639, 0.5),
				vec3(-0.700629, 0.509037, 0.5),
				vec3(-0.700629, -0.509037, 0.5),
				vec3(0.267617, -0.823639, 0.5));

		float cone_weights[cone_dir_count] = float[](0.25, 0.15, 0.15, 0.15, 0.15, 0.15);
		float cone_angle_tan = 0.577;

		for (uint i = 0; i < cone_dir_count; i++) {
			vec3 dir = normalize(dir_xform * cone_dirs[i]);
			light += cone_weights[i] * voxel_cone_trace(gi_probe_textures[index], cell_size, position, dir, cone_angle_tan, max_distance, gi_probes.data[index].bias);
		}
	} else {
		const uint cone_dir_count = 4;
		vec3 cone_dirs[cone_dir_count] = vec3[](
				vec3(0.707107, 0.0, 0.707107),
				vec3(0.0, 0.707107, 0.707107),
				vec3(-0.707107, 0.0, 0.707107),
				vec3(0.0, -0.707107, 0.707107));

		float cone_weights[cone_dir_count] = float[](0.25, 0.25, 0.25, 0.25);
		for (int i = 0; i < cone_dir_count; i++) {
			vec3 dir = normalize(dir_xform * cone_dirs[i]);
			light += cone_weights[i] * voxel_cone_trace_45_degrees(gi_probe_textures[index], cell_size, position, dir, max_distance, gi_probes.data[index].bias);
		}
	}

	if (gi_probes.data[index].ambient_occlusion > 0.001) {
		float size = 1.0 + gi_probes.data[index].ambient_occlusion_size * 7.0;

		float taps, blend;
		blend = modf(size, taps);
		float ao = 0.0;
		for (float i = 1.0; i <= taps; i++) {
			vec3 ofs = (position + normal * (i * 0.5 + 1.0)) * cell_size;
			ao += textureLod(sampler3D(gi_probe_textures[index], linear_sampler_with_mipmaps), ofs, i - 1.0).a * i;
		}

		if (blend > 0.001) {
			vec3 ofs = (position + normal * ((taps + 1.0) * 0.5 + 1.0)) * cell_size;
			ao += textureLod(sampler3D(gi_probe_textures[index], linear_sampler_with_mipmaps), ofs, taps).a * (taps + 1.0) * blend;
		}

		ao = 1.0 - min(1.0, ao);

		light.rgb = mix(params.ao_color, light.rgb, mix(1.0, ao, gi_probes.data[index].ambient_occlusion));
	}

	light.rgb *= gi_probes.data[index].dynamic_range;
	if (!gi_probes.data[index].blend_ambient) {
		light.a = 1.0;
	}

	out_diff += light * blend;

	//radiance
	vec4 irr_light = voxel_cone_trace(gi_probe_textures[index], cell_size, position, ref_vec, tan(roughness * 0.5 * M_PI * 0.99), max_distance, gi_probes.data[index].bias);
	irr_light.rgb *= gi_probes.data[index].dynamic_range;
	if (!gi_probes.data[index].blend_ambient) {
		irr_light.a = 1.0;
	}

	out_spec += irr_light * blend;

	out_blend += blend;
}

vec4 fetch_normal_and_roughness(ivec2 pos) {
	vec4 normal_roughness = texelFetch(sampler2D(normal_roughness_buffer, linear_sampler), pos, 0);

	normal_roughness.xyz = normalize(normal_roughness.xyz * 2.0 - 1.0);
	return normal_roughness;
}

void main() {
	// Pixel being shaded
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pos, params.screen_size))) { //too large, do nothing
		return;
	}

	vec3 vertex = reconstruct_position(pos);
	vertex.y = -vertex.y;

	vec4 normal_roughness = fetch_normal_and_roughness(pos);
	vec3 normal = normal_roughness.xyz;

	vec4 ambient_light = vec4(0.0), reflection_light = vec4(0.0);

	if (normal.length() > 0.5) {
		//valid normal, can do GI
		float roughness = normal_roughness.w;

		vertex = mat3(params.cam_rotation) * vertex;
		normal = normalize(mat3(params.cam_rotation) * normal);

		vec3 reflection = normalize(reflect(normalize(vertex), normal));

		if (params.use_sdfgi) {
			sdfgi_process(vertex, normal, reflection, roughness, ambient_light, reflection_light);
		}

		if (params.max_giprobes > 0) {
			uvec2 giprobe_tex = texelFetch(usampler2D(giprobe_buffer, linear_sampler), pos, 0).rg;
			roughness *= roughness;
			//find arbitrary tangent and bitangent, then build a matrix
			vec3 v0 = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
			vec3 tangent = normalize(cross(v0, normal));
			vec3 bitangent = normalize(cross(tangent, normal));
			mat3 normal_mat = mat3(tangent, bitangent, normal);

			vec4 amb_accum = vec4(0.0);
			vec4 spec_accum = vec4(0.0);
			float blend_accum = 0.0;

			for (uint i = 0; i < params.max_giprobes; i++) {
				if (any(equal(uvec2(i), giprobe_tex))) {
					gi_probe_compute(i, vertex, normal, reflection, normal_mat, roughness, spec_accum, amb_accum, blend_accum);
				}
			}
			if (blend_accum > 0.0) {
				amb_accum /= blend_accum;
				spec_accum /= blend_accum;
			}

			if (params.use_sdfgi) {
				reflection_light = blend_color(spec_accum, reflection_light);
				ambient_light = blend_color(amb_accum, ambient_light);
			} else {
				reflection_light = spec_accum;
				ambient_light = amb_accum;
			}
		}
	}

	imageStore(ambient_buffer, pos, ambient_light);
	imageStore(reflection_buffer, pos, reflection_light);
}
