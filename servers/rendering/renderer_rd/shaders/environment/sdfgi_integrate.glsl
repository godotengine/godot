#[compute]

#version 450

#VERSION_DEFINES

#define OCT_SIZE 5
#define REGION_SIZE 8

#define CACHE_IS_VALID 0x80000000
#define CACHE_IS_HIT 0x40000000

layout(local_size_x = OCT_SIZE, local_size_y = OCT_SIZE, local_size_z = 1) in;

#define MAX_CASCADES 8

layout(set = 0, binding = 1) uniform texture3D sdf_cascades;
layout(set = 0, binding = 2) uniform texture3D light_cascades;
layout(set = 0, binding = 3) uniform sampler linear_sampler;
layout(r32ui, set = 0, binding = 4) uniform restrict uimage2DArray lightprobe_texture_data;
layout(r32ui, set = 0, binding = 5) uniform restrict writeonly uimage2DArray lightprobe_filtered_data;
layout(r32ui, set = 0, binding = 6) uniform restrict writeonly uimage2DArray lightprobe_ambient_texture;


struct CascadeData {
	vec3 offset; //offset of (0,0,0) in world coordinates
	float to_cell; // 1/bounds * grid_size
	ivec3 probe_world_offset;
	uint pad;
	vec4 pad2;
};

layout(set = 0, binding = 10, std140) uniform Cascades {
	CascadeData data[MAX_CASCADES];
}
cascades;




layout(set = 0, binding = 11, std140) buffer restrict RayHitCache {
	uint data[MAX_CASCADES];
}
ray_hit_cache;


#ifdef USE_CUBEMAP_ARRAY
layout(set = 1, binding = 0) uniform textureCubeArray sky_irradiance;
#else
layout(set = 1, binding = 0) uniform textureCube sky_irradiance;
#endif
layout(set = 1, binding = 1) uniform sampler linear_sampler_mipmaps;

#define HISTORY_BITS 10

#define SKY_MODE_DISABLED 0
#define SKY_MODE_COLOR 1
#define SKY_MODE_SKY 2

layout(push_constant, std430) uniform Params {
	vec3 grid_size;
	int max_cascades;

	int probe_axis_size;
	int cascade;
	int history_index;
	int history_size;

	int ray_count;
	float ray_bias;
	ivec2 image_size;

	ivec3 world_offset;
	uint sky_mode;

	ivec3 scroll;
	float sky_energy;

	vec3 sky_color;
	float y_mult;

	bool store_ambient_texture;
	uint pad[3];
}
params;


uvec3 hash3(uvec3 x) {
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = (x >> 16) ^ x;
	return x;
}

uint hash(uint x) {
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = (x >> 16) ^ x;
	return x;
}

float hashf3(vec3 co) {
	return fract(sin(dot(co, vec3(12.9898, 78.233, 137.13451))) * 43758.5453);
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


uint rgbe_encode(vec3 color) {
	const float pow2to9 = 512.0f;
	const float B = 15.0f;
	const float N = 9.0f;
	const float LN2 = 0.6931471805599453094172321215;

	float cRed = clamp(color.r, 0.0, 65408.0);
	float cGreen = clamp(color.g, 0.0, 65408.0);
	float cBlue = clamp(color.b, 0.0, 65408.0);

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
	return (uint(sRed) & 0x1FF) | ((uint(sGreen) & 0x1FF) << 9) | ((uint(sBlue) & 0x1FF) << 18) | ((uint(exps) & 0x1F) << 27);
}


vec3 rgbe_decode(uint p_rgbe) {
	vec4 rgbef = vec4((uvec4(p_rgbe) >> uvec4(0,9,18,27)) & uvec4(0x1FF,0x1FF,0x1FF,0x1F));
	return rgbef.rgb * pow( 2.0, rgbef.a - 15.0 - 9.0 );
}

bool trace_ray(vec3 ray_pos, vec3 ray_dir, out ivec3 r_cell,out int r_cascade) {

	// No interpolation
	vec3 inv_dir = 1.0 / ray_dir;

	bool hit = false;
	ivec3 hit_pos;

	int prev_cascade = -1;
	int cascade = params.cascade;

	vec3 pos;
	vec3 side;
	vec3 delta;
	ivec3 step;
	ivec3 icell;
	vec3 pos_to_uvw = 1.0 / (params.grid_size * vec3(1,1,float(params.max_cascades)));
	ivec3 iendcell;
	float advance_remainder;
	vec3 uvw;

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

		while (advance < max_advance) {
			vec3 posf = clamp(pos + ray_dir * advance,clamp_min,clamp_max);
			posf.z+=float(cascade) * params.grid_size.x;
			uvw = posf * pos_to_uvw;
			float d = texture(sampler3D(sdf_cascades, linear_sampler), uvw).r * 15.0 - 1.0;
			if (d < -0.001) {

				// Are we really inside of a voxel?
				ivec3 posi = ivec3(posf);
				float d2 = texelFetch(sampler3D(sdf_cascades, linear_sampler), posi,0).r * 15.0 - 1.0;
				if (d2 < -0.01) {
					// Yes, consider hit.
					icell = posi;
					icell.z -= cascade * int(params.grid_size.x);
					hit = true;
					break;
				} else {
					// No, false positive, we are not, go past to the next voxel.
					vec3 local_pos = posf - vec3(posi);

					vec3 plane = mix(vec3(0.0),vec3(1.0),greaterThan(ray_dir,vec3(0.0)));
					vec3 tv = mix( (plane - local_pos) / ray_dir, vec3(1e20), equal(ray_dir,vec3(0.0)));
					float t = min(tv.x,min(tv.y,tv.z));

					advance += t + 0.1;
					continue;
				}
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

	if (hit) {

		const float EPSILON = 0.001;
		vec3 hit_normal = normalize(vec3(
				texture(sampler3D(sdf_cascades, linear_sampler), uvw + vec3(EPSILON, 0.0, 0.0)).r - texture(sampler3D(sdf_cascades, linear_sampler), uvw - vec3(EPSILON, 0.0, 0.0)).r,
				texture(sampler3D(sdf_cascades, linear_sampler), uvw + vec3(0.0, EPSILON, 0.0)).r - texture(sampler3D(sdf_cascades, linear_sampler), uvw - vec3(0.0, EPSILON, 0.0)).r,
				texture(sampler3D(sdf_cascades, linear_sampler), uvw + vec3(0.0, 0.0, EPSILON / float(params.max_cascades))).r - texture(sampler3D(sdf_cascades, linear_sampler), uvw - vec3(0.0, 0.0, EPSILON / float(params.max_cascades))).r));

		const vec3 axes[3] = vec3[](
			vec3(1,0,0),
			vec3(0,1,0),
			vec3(0,0,1)
		);


		ivec3 normal_ofs;
		float longest_dist = 0.0;
		for(uint i=0;i<3;i++) {
			vec3 axis = axes[i]*hit_normal;
			float d = length(axis);
			if (d > longest_dist) {
				normal_ofs=ivec3(axes[i]*sign(hit_normal));
				longest_dist=d;
			}
		}

		r_cell = icell + normal_ofs;
		r_cascade = cascade;
	}

	return hit;
}


void main() {
	ivec2 pos = ivec2(gl_WorkGroupID.xy);
	ivec2 local_pos = ivec2(gl_LocalInvocationID.xy);
	uint probe_index = gl_LocalInvocationID.x + gl_LocalInvocationID.y * OCT_SIZE;

	float probe_cell_size = params.grid_size.x / float(params.probe_axis_size - 1) / cascades.data[params.cascade].to_cell;

	ivec3 probe_cell;
	probe_cell.x = pos.x % params.probe_axis_size;
	probe_cell.y = pos.y;
	probe_cell.z = pos.x / params.probe_axis_size;

	vec3 ray_pos = cascades.data[params.cascade].offset + vec3(probe_cell) * probe_cell_size;

	// Ensure a unique hash that includes the probe world position, the local octahedron pixel, and the history frame index
	uvec3 h3 = hash3(uvec3(((params.world_offset + probe_cell) * OCT_SIZE * OCT_SIZE + uvec3(probe_index)) * uvec3(params.history_size) + uvec3(params.history_index)));
	uint h = (h3.x ^ h3.y) ^ h3.z;
	vec2 sample_ofs = vec2(ivec2(h>>16,h&0xFFFF)) / vec2(0xFFFF);

	vec3 ray_dir = octahedron_decode(vec2(local_pos) + sample_ofs);

	// Apply bias (by a cell)
	float bias = params.ray_bias;
	vec3 abs_ray_dir = abs(ray_dir);
	ray_pos += ray_dir * 1.0 / max(abs_ray_dir.x, max(abs_ray_dir.y, abs_ray_dir.z)) * bias / cascades.data[params.cascade].to_cell;

	ivec3 hit_cell;
	int hit_cascade;

	bool hit = trace_ray(ray_pos, ray_dir,hit_cell,hit_cascade);


	vec3 light;

	if (hit) {
		hit_cell.z += params.cascade * hit_cascade;
		light = texelFetch(sampler3D(light_cascades, linear_sampler), hit_cell,0).rgb;
	} else if (params.sky_mode == SKY_MODE_SKY) {
#ifdef USE_CUBEMAP_ARRAY
		light = textureLod(samplerCubeArray(sky_irradiance, linear_sampler_mipmaps), vec4(ray_dir, 0.0), 2.0).rgb; // Use second mipmap because we don't usually throw a lot of rays, so this compensates.
#else
		light = textureLod(samplerCube(sky_irradiance, linear_sampler_mipmaps), ray_dir, 2.0).rgb; // Use second mipmap because we don't usually throw a lot of rays, so this compensates.
#endif
		light *= params.sky_energy;
	} else if (params.sky_mode == SKY_MODE_COLOR) {
		light = params.sky_color;
		light *= params.sky_energy;
	} else {
		light = vec3(0);
	}

	light = abs(ray_dir);
	// Blend with existing light

	ivec3 probe_tex_pos = ivec3( pos.xy * OCT_SIZE * 2 + ivec2(1), int(params.cascade) );
	ivec3 probe_read_pos = probe_tex_pos + ivec3(local_pos,0);
	vec3 prev_light = rgbe_decode(imageLoad(lightprobe_texture_data,probe_read_pos).r);

//	light = mix(prev_light,light,0.1);

	// Store in octahedral map

	ivec3 copy_to[4] = ivec3[](ivec3(-2, -2, -2), ivec3(-2, -2, -2), ivec3(-2, -2, -2), ivec3(-2, -2, -2));
	copy_to[0] = probe_read_pos;

	if (local_pos == ivec2(0, 0)) {
		copy_to[1] = probe_tex_pos + ivec3(OCT_SIZE - 1, -1, 0);
		copy_to[2] = probe_tex_pos + ivec3(-1, OCT_SIZE - 1, 0);
		copy_to[3] = probe_tex_pos + ivec3(OCT_SIZE, OCT_SIZE, 0);
	} else if (local_pos == ivec2(OCT_SIZE - 1, 0)) {
		copy_to[1] = probe_tex_pos + ivec3(0, -1, 0);
		copy_to[2] = probe_tex_pos + ivec3(OCT_SIZE, OCT_SIZE - 1, 0);
		copy_to[3] = probe_tex_pos + ivec3(-1, OCT_SIZE, 0);
	} else if (local_pos == ivec2(0, OCT_SIZE - 1)) {
		copy_to[1] = probe_tex_pos + ivec3(-1, 0, 0);
		copy_to[2] = probe_tex_pos + ivec3(OCT_SIZE - 1, OCT_SIZE, 0);
		copy_to[3] = probe_tex_pos + ivec3(OCT_SIZE, -1, 0);
	} else if (local_pos == ivec2(OCT_SIZE - 1, OCT_SIZE - 1)) {
		copy_to[1] = probe_tex_pos + ivec3(0, OCT_SIZE, 0);
		copy_to[2] = probe_tex_pos + ivec3(OCT_SIZE, 0, 0);
		copy_to[3] = probe_tex_pos + ivec3(-1, -1, 0);
	} else if (local_pos.y == 0) {
		copy_to[1] = probe_tex_pos + ivec3(OCT_SIZE - local_pos.x - 1, local_pos.y - 1, 0);
	} else if (local_pos.x == 0) {
		copy_to[1] = probe_tex_pos + ivec3(local_pos.x - 1, OCT_SIZE - local_pos.y - 1, 0);
	} else if (local_pos.y == OCT_SIZE - 1) {
		copy_to[1] = probe_tex_pos + ivec3(OCT_SIZE - local_pos.x - 1, local_pos.y + 1, 0);
	} else if (local_pos.x == OCT_SIZE - 1) {
		copy_to[1] = probe_tex_pos + ivec3(local_pos.x + 1, OCT_SIZE - local_pos.y - 1, 0);
	}

	for (int i = 0; i < 4; i++) {
		if (copy_to[i] == ivec3(-2, -2, -2)) {
			continue;
		}
		imageStore(lightprobe_texture_data, copy_to[i], uvec4(rgbe_encode(light)));
		// also to filtered
	}
}
