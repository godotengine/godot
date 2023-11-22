#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#define MAX_CASCADES 8
#define REGION_SIZE 8

layout(set = 0, binding = 1) uniform texture3D sdf_cascades;
layout(set = 0, binding = 2) uniform texture3D light_cascades;

layout(set = 0, binding = 3) uniform sampler linear_sampler;
layout(set = 0, binding = 8) uniform sampler nearest_sampler;

layout(set = 0, binding = 6) uniform texture2DArray light_probes;
layout(set = 0, binding = 7) uniform texture2DArray occlusion_probes;

layout(rg32ui, set = 0, binding = 9) uniform restrict readonly uimage3D voxel_cascades;
layout(r8ui, set = 0, binding = 10) uniform restrict readonly uimage3D voxel_region_cascades;

struct CascadeData {
	vec3 offset; //offset of (0,0,0) in world coordinates
	float to_cell; // 1/bounds * grid_size
	ivec3 probe_world_offset;
	uint pad;
	vec4 pad2;
};

layout(set = 0, binding = 4, std140) restrict readonly uniform Cascades {
	CascadeData data[MAX_CASCADES];
}
cascades;

layout(rgba16f, set = 0, binding = 5) uniform restrict writeonly image2D screen_buffer;


layout(push_constant, std430) uniform Params {
	vec3 grid_size;
	uint max_cascades;

	ivec2 screen_size;
	float y_mult;

	float z_near;

	mat3x4 inv_projection;
	// We pack these more tightly than mat3 and vec3, which will require some reconstruction trickery.
	float cam_basis[3][3];
	float cam_origin[3];
}
params;

vec3 linear_to_srgb(vec3 color) {
	//if going to srgb, clamp from 0 to 1.
	color = clamp(color, vec3(0.0), vec3(1.0));
	const vec3 a = vec3(0.055f);
	return mix((vec3(1.0f) + a) * pow(color.rgb, vec3(1.0f / 2.4f)) - a, 12.92f * color.rgb, lessThan(color.rgb, vec3(0.0031308f)));
}


bool trace_ray(vec3 ray_pos, vec3 ray_dir, out ivec3 r_cell,out vec3 r_uvw, out int r_cascade) {

	// No interpolation
	vec3 inv_dir = 1.0 / ray_dir;

	bool hit = false;
	ivec3 hit_pos;

	int prev_cascade = -1;
	int cascade = 0;

	vec3 pos;
	vec3 side;
	vec3 delta;
	ivec3 step;
	ivec3 icell;
	vec3 pos_to_uvw = 1.0 / (params.grid_size * vec3(1,float(params.max_cascades),1));
	ivec3 iendcell;
	float advance_remainder;

	uint iters = 0;
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
			vec3 posf = pos + ray_dir * advance;
			vec3 pos_tap = clamp(posf,clamp_min,clamp_max);
			pos_tap.y+=float(cascade * params.grid_size.y);
			uvw = pos_tap * pos_to_uvw;
			float d = texture(sampler3D(sdf_cascades, linear_sampler), uvw).r * 15.0 - 1.0;
			if (d < 1.0) { // Needs to be conservative.
				// Are we really inside of a voxel?
				ivec3 posi = ivec3(posf);
				float d2 = texelFetch(sampler3D(sdf_cascades, nearest_sampler), posi + ivec3(0,cascade * int(params.grid_size.y),0),0).r * 15.0 - 1.0;
				if (d2 < 0.0) {
					// Yes, consider hit.
					r_cell = posi;
					r_cell.y += cascade * int(params.grid_size.y);
					r_uvw = uvw;
					r_cascade = cascade;
					hit = true;
					break;
				} else {
					// No, false positive, we are not, go past to the next voxel.
					vec3 local_pos = posf - vec3(posi);
/*
					vec3 plane = mix(vec3(0.0),vec3(1.0),greaterThan(ray_dir,vec3(0.0)));
					vec3 tv = mix( (plane - local_pos) / ray_dir, vec3(1e20), equal(ray_dir,vec3(0.0)));
					float t = min(tv.x,min(tv.y,tv.z));
*/

					vec3 t0 = -local_pos * inv_dir;
					vec3 t1 = (vec3(1.0) - local_pos) * inv_dir;
					vec3 tmax = max(t0, t1);
					float t = min(tmax.x, min(tmax.y, tmax.z));

					advance += t + 0.001;
					continue;
				}
			}

			advance += max(d, 0.001);
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


		ivec3 mask = ivec3(1,0,0);
		float m = side.x;
		if (side.y < m) {
			mask = ivec3(0,1,0);
			m = side.y;
		}
		if (side.z < m) {
			mask = ivec3(0,0,1);
		}

		//bvec3 mask = lessThanEqual(side.xyz, min(side.yzx, side.zxy));
		side += vec3(mask) * delta;
		icell += mask * step;

		iters++;
		if (iters==1000) {
			break;
		}
	}


	return hit;
}


bool trace_ray_hdda(vec3 ray_pos, vec3 ray_dir,int p_cascade, out ivec3 r_cell,out ivec3 r_side, out int r_cascade) {

	const int LEVEL_CASCADE = -1;
	const int LEVEL_REGION = 0;
	const int LEVEL_BLOCK = 1;
	const int LEVEL_VOXEL = 2;
	const int MAX_LEVEL = 3;


	const int fp_bits = 8;
	const int fp_block_bits = fp_bits + 2;
	const int fp_region_bits = fp_block_bits + 1;
	const int fp_cascade_bits = fp_region_bits + 4;

	bvec3 limit_dir = greaterThan(ray_dir,vec3(0.0));
	ivec3 step = mix(ivec3(0),ivec3(1),limit_dir);
	ivec3 ray_sign = ivec3(sign(ray_dir));

	ivec3 ray_dir_fp = ivec3(ray_dir * float(1<<fp_bits));

	bvec3 ray_zero = lessThan(abs(ray_dir),vec3(1.0/127.0));
	ivec3 inv_ray_dir_fp = ivec3( float(1<<fp_bits) / ray_dir );

	const ivec3 level_masks[MAX_LEVEL]=ivec3[](
		ivec3(1<<fp_region_bits) - ivec3(1),
		ivec3(1<<fp_block_bits) - ivec3(1),
		ivec3(1<<fp_bits) - ivec3(1)
	);

	ivec3 region_offset_mask = (ivec3(params.grid_size) / REGION_SIZE) - ivec3(1);

	ivec3 limits[MAX_LEVEL];

	limits[LEVEL_REGION] = ((ivec3(params.grid_size) << fp_bits) - ivec3(1)) * step; // Region limit does not change, so initialize now.

	// Initialize to cascade
	int level = LEVEL_CASCADE;
	int cascade = p_cascade - 1;

	ivec3 cascade_base;
	ivec3 region_base;
	uvec2 block;
	bool hit = false;

	ivec3 pos;

	while(true) {
		// This loop is written so there is only one single main interation.
		// This ensures that different compute threads working on different
		// levels can still run together without blocking each other.

		if (level == LEVEL_VOXEL) {
			// The first level should be (in a worst case scenario) the most used
			// so it needs to appear first. The rest of the levels go from more to least used order.

			ivec3 block_local = (pos & level_masks[LEVEL_BLOCK]) >> fp_bits;
			uint block_index = uint(block_local.z * 16 + block_local.y * 4 + block_local.x);
			if (block_index < 32) {
				// Low 32 bits.
				if (bool(block.x & uint(1<<block_index))) {
					hit=true;
					break;
				}
			} else {
				// High 32 bits.
				block_index-=32;
				if (bool(block.y & uint(1<<block_index))) {
					hit=true;
					break;
				}
			}
		} else if (level == LEVEL_BLOCK) {
			ivec3 block_local = (pos & level_masks[LEVEL_REGION]) >> fp_block_bits;
			block = imageLoad(voxel_cascades,region_base + block_local).rg;
			if (block != uvec2(0)) {
				// Have voxels inside
				level = LEVEL_VOXEL;
				limits[LEVEL_VOXEL]= pos - (pos & level_masks[LEVEL_BLOCK]) + step * (level_masks[LEVEL_BLOCK] + ivec3(1));
				continue;
			}
		} else if (level == LEVEL_REGION) {
			ivec3 region = pos >> fp_region_bits;
			region = (cascades.data[cascade].probe_world_offset + region) & region_offset_mask; // Scroll to world
			region += cascade_base;
			bool region_used = imageLoad(voxel_region_cascades,region).r > 0;

			if (region_used) {
				// The region has contents.
				region_base = (region<<1);
				level = LEVEL_BLOCK;
				limits[LEVEL_BLOCK]= pos - (pos & level_masks[LEVEL_REGION]) + step * (level_masks[LEVEL_REGION] + ivec3(1));
				continue;
			}
		} else if (level == LEVEL_CASCADE) {
			// Return to global
			if (cascade >= p_cascade) {
				ray_pos = vec3(pos) / float(1<<fp_bits);
				ray_pos /= cascades.data[cascade].to_cell;
				ray_pos += cascades.data[cascade].offset;
			}

			cascade++;
			if (cascade == params.max_cascades) {
				break;
			}

			ray_pos -= cascades.data[cascade].offset;
			ray_pos *= cascades.data[cascade].to_cell;
			pos = ivec3(ray_pos * float(1<<fp_bits));
			if (any(lessThan(pos,ivec3(0))) || any(greaterThanEqual(pos,ivec3(params.grid_size)<<fp_bits))) {
				// Outside this cascade, go to next.
				continue;
			}

			cascade_base = ivec3(0,int(params.grid_size.y/REGION_SIZE) * cascade , 0);
			level = LEVEL_REGION;
			continue;
		}

		// Fixed point, multi-level DDA.

		ivec3 mask = level_masks[level];
		ivec3 box = mask * step;
		ivec3 pos_diff = box - (pos & mask);
		ivec3 tv = mix((pos_diff * inv_ray_dir_fp),ivec3(0x7FFFFFFF),ray_zero) >> fp_bits;
		int t = min(tv.x,min(tv.y,tv.z));

		// The general idea here is that we _always_ need to increment to the closest next cell
		// (this is a DDA after all), so adv_box forces this increment for the minimum axis.

		ivec3 adv_box = pos_diff + ray_sign;
		ivec3 adv_t = (ray_dir_fp * t) >> fp_bits;

		pos += mix(adv_t,adv_box,equal(ivec3(t),tv));

		while(true) {
			bvec3 limit = lessThan(pos,limits[level]);
			bool inside = all(equal(limit,limit_dir));
			if (inside) {
				break;
			}
			level-=1;
			if (level == LEVEL_CASCADE) {
				break;
			}
		}
	}

	if (hit) {

		ivec3 mask = level_masks[LEVEL_VOXEL];
		ivec3 box = mask * (step ^ ivec3(1));
		ivec3 pos_diff = box - (pos & mask);
		ivec3 tv = mix((pos_diff * -inv_ray_dir_fp),ivec3(0x7FFFFFFF),ray_zero);

		int m;
		if (tv.x < tv.y) {
			r_side = ivec3(1,0,0);
			m = tv.x;
		} else {
			r_side = ivec3(0,1,0);
			m = tv.y;
		}
		if (tv.z < m) {
			r_side = ivec3(0,0,1);
		}

		r_side *= -ray_sign;

		r_cell = pos >> fp_bits;

		r_cascade = cascade;
	}

	return hit;
}


#define PROBE_CELLS 8


ivec3 modi(ivec3 value, ivec3 p_y) {
	return ((value % p_y) + p_y) % p_y;
}

ivec2 probe_to_tex(ivec3 local_probe,int p_cascade) {

	ivec3 probe_axis_size = ivec3(params.grid_size) / PROBE_CELLS + ivec3(1);
	ivec3 cell = modi( cascades.data[p_cascade].probe_world_offset + local_probe,probe_axis_size);
	return cell.xy + ivec2(0,cell.z * int(probe_axis_size.y));

}

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

#define OCCLUSION_OCT_SIZE 14
#define OCT_SIZE 4

#define OCC8_DISTANCE_MAX 15.0
#define OCC16_DISTANCE_MAX 256.0

void main() {
	// Pixel being shaded
	ivec2 screen_pos = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(screen_pos, params.screen_size))) { //too large, do nothing
		return;
	}

	vec3 ray_pos;
	vec3 ray_dir;
	vec3 cam_dir;
	{
		ray_pos = vec3(params.cam_origin[0], params.cam_origin[1], params.cam_origin[2]);

		ray_dir.xy = ((vec2(screen_pos) / vec2(params.screen_size)) * 2.0 - 1.0);
		ray_dir.z = params.z_near;

		ray_dir = (vec4(ray_dir, 1.0) * mat4(params.inv_projection)).xyz;

		mat3 cam_basis;
		{
			vec3 c0 = vec3(params.cam_basis[0][0], params.cam_basis[0][1], params.cam_basis[0][2]);
			vec3 c1 = vec3(params.cam_basis[1][0], params.cam_basis[1][1], params.cam_basis[1][2]);
			vec3 c2 = vec3(params.cam_basis[2][0], params.cam_basis[2][1], params.cam_basis[2][2]);
			cam_basis = mat3(c0, c1, c2);
		}
		ray_dir = normalize(cam_basis * ray_dir);
		cam_dir = vec3(params.cam_basis[2][0],params.cam_basis[2][1],params.cam_basis[2][2]);
	}


	ray_pos.y *= params.y_mult;
	ray_dir.y *= params.y_mult;
	ray_dir = normalize(ray_dir);


	vec3 light = vec3(0.0);
	ivec3 hit_cell;
	vec3 hit_uvw;
	int hit_cascade = 0;

#if 1

	ivec3 hit_face;

	if (trace_ray_hdda(ray_pos, ray_dir,0,hit_cell,hit_face,hit_cascade)) {
		hit_cell += hit_face + ivec3(0,(params.grid_size.y * hit_cascade), 0);
		light = texelFetch(sampler3D(light_cascades, linear_sampler), hit_cell,0).rgb;
		//light = vec3(abs(hit_face));

#else
	if (trace_ray(ray_pos,ray_dir,hit_cell,hit_uvw, hit_cascade)) {



		const float EPSILON = 0.001;
		vec3 hit_normal = normalize(vec3(
				texture(sampler3D(sdf_cascades, linear_sampler), hit_uvw + vec3(EPSILON, 0.0, 0.0)).r - texture(sampler3D(sdf_cascades, linear_sampler), hit_uvw - vec3(EPSILON, 0.0, 0.0)).r,
				texture(sampler3D(sdf_cascades, linear_sampler), hit_uvw + vec3(0.0, EPSILON / float(params.max_cascades), 0.0)).r - texture(sampler3D(sdf_cascades, linear_sampler), hit_uvw - vec3(0.0, EPSILON / float(params.max_cascades), 0.0)).r,
				texture(sampler3D(sdf_cascades, linear_sampler), hit_uvw + vec3(0.0, 0.0, EPSILON )).r - texture(sampler3D(sdf_cascades, linear_sampler), hit_uvw - vec3(0.0, 0.0, EPSILON)).r));

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

		light = texelFetch(sampler3D(light_cascades, linear_sampler), hit_cell + normal_ofs,0).rgb;
		light = vec3((ivec3(hit_cascade+1) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1)) * 0.5;
		light += vec3(((hit_cell / 8) + cascades.data[hit_cascade].probe_world_offset) & 0x1) * 0.2;

#endif

#if 0

		{
			// compute occlusion

			int cascade = hit_cell.y / int(params.grid_size.y);
			hit_cell.y%=int(params.grid_size.y);

			ivec3 pos = hit_cell + normal_ofs;

			ivec3 base_probe = pos / PROBE_CELLS;
			vec3 posf = vec3(pos) + 0.5; // Actual point in the center of the box.

			ivec3 probe_axis_size = ivec3(params.grid_size) / PROBE_CELLS + ivec3(1);

			vec2 occ_probe_tex_to_uv = 1.0 / vec2( (OCCLUSION_OCT_SIZE+2) * probe_axis_size.x, (OCCLUSION_OCT_SIZE+2) * probe_axis_size.y * probe_axis_size.z );

			vec4 accum_light = vec4(0.0);

			vec2 light_probe_tex_to_uv = 1.0 / vec2( (OCT_SIZE+2) * probe_axis_size.x, (OCT_SIZE+2) * probe_axis_size.y * probe_axis_size.z );
			vec2 light_uv = octahedron_encode(hit_normal) * float(OCT_SIZE);

			for(int i=0;i<8;i++) {
				ivec3 probe = base_probe + ((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1));

				vec3 probe_pos = vec3(probe * PROBE_CELLS);



				vec3 probe_to_pos = posf - probe_pos;
				vec3 n = normalize(probe_to_pos);
				float d = length(probe_to_pos);

				float weight = 1.0;
				weight *= pow(max(0.0001, (dot(-n, hit_normal) + 1.0) * 0.5),2.0) + 0.2;


				ivec2 tex_pos = probe_to_tex(probe,cascade);
				vec2 tex_uv = vec2(ivec2(tex_pos * (OCCLUSION_OCT_SIZE+2) + ivec2(1))) + octahedron_encode(n) * float(OCCLUSION_OCT_SIZE);
				tex_uv *= occ_probe_tex_to_uv;
				vec2 o_o2 = texture(sampler2DArray(occlusion_probes,linear_sampler),vec3(tex_uv,float(cascade))).rg * OCC16_DISTANCE_MAX;

				float mean = o_o2.x;
				float variance = abs((mean*mean) - o_o2.y);

				 // http://www.punkuser.net/vsm/vsm_paper.pdf; equation 5
				 // Need the max in the denominator because biasing can cause a negative displacement
				float dmean = max(d - mean, 0.0);
				float chebyshev_weight = variance / (variance + dmean*dmean);

				chebyshev_weight = max(pow(chebyshev_weight,3.0), 0.0);

				weight *= (d <= mean) ? 1.0 : chebyshev_weight;

				weight = max(0.000001, weight); // make sure not zero (only trilinear can be zero)

				const float crushThreshold = 0.2;
				if (weight < crushThreshold) {
				      weight *= weight * weight * (1.0 / pow(crushThreshold,2.0));
				}

				vec3 trilinear = vec3(1.0) - abs(probe_to_pos / float(PROBE_CELLS));

				weight *= trilinear.x * trilinear.y * trilinear.z;

				tex_uv = vec2(ivec2(tex_pos * (OCT_SIZE+2) + ivec2(1))) + light_uv;
				tex_uv *= light_probe_tex_to_uv;

				vec3 probe_light = texture(sampler2DArray(light_probes,linear_sampler),vec3(tex_uv,float(cascade))).rgb;

				accum_light+=vec4(probe_light,1.0) * weight;
			}

			light += accum_light.rgb / accum_light.a;

		}

#endif
		//light = abs(hit_normal);//texelFetch(sampler3D(light_cascades, linear_sampler), hit_cell + normal_ofs,0).rgb;
	}


	imageStore(screen_buffer, screen_pos, vec4(linear_to_srgb(light), 1.0));
}
