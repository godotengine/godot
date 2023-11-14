#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#define MAX_CASCADES 8

layout(set = 0, binding = 1) uniform texture3D sdf_cascades;
layout(set = 0, binding = 2) uniform sampler linear_sampler;

layout(set = 0, binding = 3, std430) restrict readonly buffer DispatchData {
	uint x;
	uint y;
	uint z;
	uint total_count;
}
dispatch_data;

struct ProcessVoxel {
	uint position; // xyz 10 bit packed
	uint albedo_emission_r; // 0 - 15, albedo 16-31 emission R
	uint emission_gb; // 0-15 emission G, 16-31 emission B
	uint normal; // 0-20 normal RG octahedron 21-32 cached occlusion?
};

#define PROCESS_STATIC_PENDING_BIT 0x80000000
#define PROCESS_DYNAMIC_PENDING_BIT 0x40000000

// Can always write, because it needs to set off dirty bits
layout(set = 0, binding = 4, std430) restrict buffer ProcessVoxels {
	ProcessVoxel data[];
}
process_voxels;

layout(r32ui, set = 0, binding = 5) uniform restrict writeonly uimage3D dst_light;

struct CascadeData {
	vec3 offset; //offset of (0,0,0) in world coordinates
	float to_cell; // 1/bounds * grid_size
	ivec3 probe_world_offset;
	uint pad;
	vec4 pad2;
};

layout(set = 0, binding = 8, std140) uniform Cascades {
	CascadeData data[MAX_CASCADES];
}
cascades;

#define LIGHT_TYPE_DIRECTIONAL 0
#define LIGHT_TYPE_OMNI 1
#define LIGHT_TYPE_SPOT 2

struct Light {
	vec3 color;
	float energy;

	vec3 direction;
	bool has_shadow;

	vec3 position;
	float attenuation;

	uint type;
	float cos_spot_angle;
	float inv_spot_attenuation;
	float radius;

};

layout(set = 0, binding = 9, std140) buffer restrict readonly Lights {
	Light data[];
}
lights;

#if 0
layout(set = 0, binding = 10) uniform texture2DArray lightprobe_texture;
layout(set = 0, binding = 11) uniform texture3D occlusion_texture;
#endif

layout(push_constant, std430) uniform Params {
	vec3 grid_size;
	uint max_cascades;

	uint cascade;
	uint light_count;
	uint process_offset;
	uint process_increment;

	int probe_axis_size;
	float bounce_feedback;
	float y_mult;
	bool use_occlusion;

	bool dirty_dynamic_update;
	uint pad0;
	uint pad1;
	uint pad2;

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

vec3 octahedron_decode(vec2 f) {
	// https://twitter.com/Stubbesaurus/status/937994790553227264
	f = f * 2.0 - 1.0;
	vec3 n = vec3(f.x, f.y, 1.0f - abs(f.x) - abs(f.y));
	float t = clamp(-n.z, 0.0, 1.0);
	n.x += n.x >= 0 ? -t : t;
	n.y += n.y >= 0 ? -t : t;
	return normalize(n);
}

float get_omni_attenuation(float distance, float inv_range, float decay) {
	float nd = distance * inv_range;
	nd *= nd;
	nd *= nd; // nd^4
	nd = max(1.0 - nd, 0.0);
	nd *= nd; // nd^2
	return nd * pow(max(distance, 0.0001), -decay);
}

#define REGION_SIZE 8

bool trace_ray(vec3 ray_pos, vec3 ray_dir) {

	// No interpolation
	vec3 inv_dir = 1.0 / ray_dir;

	bool hit = false;
	ivec3 hit_pos;

	int prev_cascade = -1;
	int cascade = int(params.cascade);

	vec3 pos;
	vec3 side;
	vec3 delta;
	ivec3 step;
	ivec3 icell;
	vec3 pos_to_uvw = 1.0 / (params.grid_size * vec3(1,1,float(params.max_cascades)));
	ivec3 iendcell;
	float advance_remainder;

//	uint iters = 0;
	while(true) {

		if (cascade != prev_cascade) {
			pos = ray_pos - cascades.data[cascade].offset;
			pos *= cascades.data[cascade].to_cell;

			if (any(lessThan(pos,vec3(0.0))) || any(greaterThanEqual(pos,params.grid_size))) {
				cascade++;
				if (cascade == params.max_cascades) {
					break;
				}
				continue;
			}

			//find maximum advance distance (until reaching bounds)
			vec3 t0 = -pos * inv_dir;
			vec3 t1 = (params.grid_size - pos) * inv_dir;
			vec3 tmax = max(t0, t1);
			advance_remainder = max(0,min(tmax.x, min(tmax.y, tmax.z)) - 0.1);

			vec3 from_cell = pos / float(REGION_SIZE);

			icell = ivec3(from_cell);

			delta = min(abs(1.0 / ray_dir), params.grid_size / float(REGION_SIZE)); // Use bake_params.grid_size as max to prevent infinity values.
			step = ivec3(sign(ray_dir));
			side = (sign(ray_dir) * (vec3(icell) - from_cell) + (sign(ray_dir) * 0.5) + 0.5) * delta;

			prev_cascade = cascade;

		}


		vec3 lpos = pos - vec3(icell * REGION_SIZE);
		vec3 tmax = (mix(vec3(REGION_SIZE),vec3(0.0),lessThan(ray_dir,vec3(0.0))) - lpos) * inv_dir;
		float max_advance = max(0.0,min(tmax.x, min(tmax.y, tmax.z)));

		vec3 clamp_min = vec3(icell * REGION_SIZE) + 0.5;
		vec3 clamp_max = vec3((icell+ivec3(1)) * REGION_SIZE) - 0.5;

		float advance = 0;
		vec3 uvw;

		while (advance < max_advance) {
			vec3 posf = clamp(pos + ray_dir * advance,clamp_min,clamp_max);
			posf.z+=float(cascade * params.grid_size.x);
			uvw = posf * pos_to_uvw;
			float d = texture(sampler3D(sdf_cascades, linear_sampler), uvw).r * 15.0 - 1.0;
			if (d < -0.001) {

				hit=true;
				break;
			}
			advance += max(d, 0.01);
		}


		if (hit) {
			break;
		}

		pos += ray_dir * max_advance;
		advance_remainder -= max_advance;

		if (advance_remainder <= 0.0) {
			pos /= cascades.data[cascade].to_cell;
			pos += cascades.data[cascade].offset;
			ray_pos = pos;
			cascade++;
			if (cascade == params.max_cascades) {
				break;
			}
			continue;
		}


		bvec3 mask = lessThanEqual(side.xyz, min(side.yzx, side.zxy));
		side += vec3(mask) * delta;
		icell += ivec3(vec3(mask)) * step;

//		iters++;
//		if (iters==1000) {
//			break;
//		}
	}


	return hit;
}


void main() {
	uint voxel_index = uint(gl_GlobalInvocationID.x);

	if (voxel_index >= dispatch_data.total_count) {
		return;
	}

#ifdef MODE_PROCESS_STATIC
	// Discard if not marked for static update
	if (!bool(process_voxels.data[voxel_index].position & PROCESS_STATIC_PENDING_BIT)) {
		return;
	}
#else

	//used for skipping voxels every N frames
	if (params.process_increment > 1) {
		if ( ( (voxel_index + params.process_offset) % params.process_increment) != 0 ) {

			bool still_render = false;
			if (params.dirty_dynamic_update && bool(process_voxels.data[voxel_index].position & PROCESS_DYNAMIC_PENDING_BIT)) {
				//saved	because it still needs dynamic update
			} else {
				return;
			}
		}
	}

	if (params.dirty_dynamic_update) {
		process_voxels.data[voxel_index].position&=~uint(PROCESS_DYNAMIC_PENDING_BIT);
	}
#endif


	// Decode ProcessVoxel

	ivec3 positioni = ivec3((uvec3(process_voxels.data[voxel_index].position) >> uvec3(0, 10, 20)) & uvec3(0x3FF));

	vec3 position = vec3(positioni) + vec3(0.5);
	position /= cascades.data[params.cascade].to_cell;
	position += cascades.data[params.cascade].offset;

	uint voxel_albedo = process_voxels.data[voxel_index].albedo_emission_r;

	vec3 albedo = vec3( (uvec3(process_voxels.data[voxel_index].albedo_emission_r) >> uvec3(0, 5, 11)) & uvec3(0x1F,0x3F,0x1F)) / vec3(0x1F,0x3F,0x1F);
	vec3 emission = vec3(unpackHalf2x16(process_voxels.data[voxel_index].albedo_emission_r).g,unpackHalf2x16(process_voxels.data[voxel_index].emission_gb));
	vec3 normal = octahedron_decode( vec2((uvec2(process_voxels.data[voxel_index].normal) >> uvec2(0, 10)) & uvec2(0x3FF,0x3FF)) / vec2(0x3FF,0x3FF));

	vec3 light_accum = vec3(0.0);

	// Add indirect light first, in order to save computation resources
#if 0
#ifdef MODE_PROCESS_DYNAMIC
	if (params.bounce_feedback > 0.001) {
		vec3 feedback = (params.bounce_feedback < 1.0) ? (albedo * params.bounce_feedback) : mix(albedo, vec3(1.0), params.bounce_feedback - 1.0);
		vec3 pos = (vec3(positioni) + vec3(0.5)) * float(params.probe_axis_size - 1) / params.grid_size;
		ivec3 probe_base_pos = ivec3(pos);

		float weight_accum[6] = float[](0, 0, 0, 0, 0, 0);

		ivec3 tex_pos = ivec3(probe_base_pos.xy, int(params.cascade));
		tex_pos.x += probe_base_pos.z * int(params.probe_axis_size);

		tex_pos.xy = tex_pos.xy * (OCT_SIZE + 2) + ivec2(1);

		vec3 base_tex_posf = vec3(tex_pos);
		vec2 tex_pixel_size = 1.0 / vec2(ivec2((OCT_SIZE + 2) * params.probe_axis_size * params.probe_axis_size, (OCT_SIZE + 2) * params.probe_axis_size));
		vec3 probe_uv_offset = vec3(ivec3(OCT_SIZE + 2, OCT_SIZE + 2, (OCT_SIZE + 2) * params.probe_axis_size)) * tex_pixel_size.xyx;

		for (uint j = 0; j < 8; j++) {
			ivec3 offset = (ivec3(j) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1);
			ivec3 probe_posi = probe_base_pos;
			probe_posi += offset;

			// Compute weight

			vec3 probe_pos = vec3(probe_posi);
			vec3 probe_to_pos = pos - probe_pos;
			vec3 probe_dir = normalize(-probe_to_pos);

			// Compute lightprobe texture position

			vec3 trilinear = vec3(1.0) - abs(probe_to_pos);

			for (uint k = 0; k < 6; k++) {
				if (bool(valid_aniso & (1 << k))) {
					vec3 n = aniso_dir[k];
					float weight = trilinear.x * trilinear.y * trilinear.z * max(0, dot(n, probe_dir));

					if (weight > 0.0 && params.use_occlusion) {
						ivec3 occ_indexv = abs((cascades.data[params.cascade].probe_world_offset + probe_posi) & ivec3(1, 1, 1)) * ivec3(1, 2, 4);
						vec4 occ_mask = mix(vec4(0.0), vec4(1.0), equal(ivec4(occ_indexv.x | occ_indexv.y), ivec4(0, 1, 2, 3)));

						vec3 occ_pos = (vec3(positioni) + aniso_dir[k] + vec3(0.5)) / params.grid_size;
						occ_pos.z += float(params.cascade);
						if (occ_indexv.z != 0) { //z bit is on, means index is >=4, so make it switch to the other half of textures
							occ_pos.x += 1.0;
						}
						occ_pos *= vec3(0.5, 1.0, 1.0 / float(params.max_cascades)); //renormalize
						float occlusion = dot(textureLod(sampler3D(occlusion_texture, linear_sampler), occ_pos, 0.0), occ_mask);

						weight *= occlusion;
					}

					if (weight > 0.0) {
						vec3 tex_posf = base_tex_posf + vec3(octahedron_encode(n) * float(OCT_SIZE), 0.0);
						tex_posf.xy *= tex_pixel_size;

						vec3 pos_uvw = tex_posf;
						pos_uvw.xy += vec2(offset.xy) * probe_uv_offset.xy;
						pos_uvw.x += float(offset.z) * probe_uv_offset.z;
						vec3 indirect_light = textureLod(sampler2DArray(lightprobe_texture, linear_sampler), pos_uvw, 0.0).rgb;

						light_accum[k] += indirect_light * weight;
						weight_accum[k] += weight;
					}
				}
			}
		}

		for (uint k = 0; k < 6; k++) {
			if (weight_accum[k] > 0.0) {
				light_accum[k] /= weight_accum[k];
				light_accum[k] *= feedback;
			}
		}
	}

#endif
#endif

	// Raytrace light

	vec3 pos_to_uvw = 1.0 / params.grid_size;
	vec3 uvw_ofs = pos_to_uvw * 0.5;

	for (uint i = 0; i < params.light_count; i++) {
		float attenuation = 1.0;
		vec3 direction;
		float light_distance = 1e20;

		switch (lights.data[i].type) {
			case LIGHT_TYPE_DIRECTIONAL: {
				direction = -lights.data[i].direction;
				attenuation *= max(0.0,dot(normal,direction));
			} break;
			case LIGHT_TYPE_OMNI: {
				vec3 rel_vec = lights.data[i].position - position;
				direction = normalize(rel_vec);
				light_distance = length(rel_vec);
				rel_vec.y /= params.y_mult;
				attenuation = get_omni_attenuation(light_distance, 1.0 / lights.data[i].radius, lights.data[i].attenuation);
				attenuation *= max(0.0,dot(normal,direction));

			} break;
			case LIGHT_TYPE_SPOT: {
				vec3 rel_vec = lights.data[i].position - position;
				direction = normalize(rel_vec);

				light_distance = length(rel_vec);
				rel_vec.y /= params.y_mult;
				attenuation = get_omni_attenuation(light_distance, 1.0 / lights.data[i].radius, lights.data[i].attenuation);
				attenuation *= max(0.0,dot(normal,direction));

				float cos_spot_angle = lights.data[i].cos_spot_angle;
				float cos_angle = dot(-direction, lights.data[i].direction);

				if (cos_angle < cos_spot_angle) {
					continue;
				}

				float scos = max(cos_angle, cos_spot_angle);
				float spot_rim = max(0.0001, (1.0 - scos) / (1.0 - cos_spot_angle));
				attenuation *= 1.0 - pow(spot_rim, lights.data[i].inv_spot_attenuation);
			} break;
		}

		if (attenuation < 0.001) {
			continue;
		}

		bool hit = false;

		vec3 ray_pos = position;
		vec3 ray_dir = direction;
		vec3 inv_dir = 1.0 / ray_dir;

		//this is how to properly bias outgoing rays
		float cell_size = 1.0 / cascades.data[params.cascade].to_cell;
		ray_pos += sign(direction) * cell_size * 0.48; // go almost to the box edge but remain inside
		ray_pos += ray_dir * 0.4 * cell_size; //apply a small bias from there


		hit = trace_ray(ray_pos,ray_dir);

		if (!hit) {
			light_accum += albedo * lights.data[i].color.rgb * lights.data[i].energy * attenuation;
		}
	}

	light_accum += emission;

#ifdef MODE_PROCESS_STATIC
	// Add to self, since its static.
	process_voxels.data[voxel_index].albedo_emission_r&=0xFFFF; // Keep albedo (lower 16 bits).

	process_voxels.data[voxel_index].albedo_emission_r |= packHalf2x16(vec2(light_accum.r,0))<<16;
	process_voxels.data[voxel_index].emission_gb = packHalf2x16(light_accum.gb);
	process_voxels.data[voxel_index].position&=~uint(PROCESS_STATIC_PENDING_BIT); // Clear process static bit.
#else

	// Store to light texture

	uint light_total_rgbe;

	{
		//compress to RGBE9995 to save space

		const float pow2to9 = 512.0f;
		const float B = 15.0f;
		const float N = 9.0f;
		const float LN2 = 0.6931471805599453094172321215;

		float cRed = clamp(light_accum.r, 0.0, 65408.0);
		float cGreen = clamp(light_accum.g, 0.0, 65408.0);
		float cBlue = clamp(light_accum.b, 0.0, 65408.0);

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

		light_total_rgbe = (uint(sRed) & 0x1FF) | ((uint(sGreen) & 0x1FF) << 9) | ((uint(sBlue) & 0x1FF) << 18) | ((uint(exps) & 0x1F) << 27);
	}

	positioni.z+=int(params.cascade) * int(params.grid_size.x);
	imageStore(dst_light, positioni, uvec4(light_total_rgbe));

#endif

}
