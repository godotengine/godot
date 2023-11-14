#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#define MAX_CASCADES 8
#define REGION_SIZE 8

layout(set = 0, binding = 1) uniform texture3D sdf_cascades;
layout(set = 0, binding = 2) uniform texture3D light_cascades;

layout(set = 0, binding = 3) uniform sampler linear_sampler;

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

				// Are we really inside of a voxel?
				ivec3 posi = ivec3(posf);
				float d2 = texelFetch(sampler3D(sdf_cascades, linear_sampler), posi,0).r * 15.0 - 1.0;
				if (d2 < -0.01) {
					// Yes, consider hit.
					r_cell = posi;
					r_uvw = uvw;
					r_cascade = cascade;
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


	return hit;
}

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
	int hit_cascade;

	if (trace_ray(ray_pos,ray_dir,hit_cell,hit_uvw, hit_cascade)) {



		const float EPSILON = 0.001;
		vec3 hit_normal = normalize(vec3(
				texture(sampler3D(sdf_cascades, linear_sampler), hit_uvw + vec3(EPSILON, 0.0, 0.0)).r - texture(sampler3D(sdf_cascades, linear_sampler), hit_uvw - vec3(EPSILON, 0.0, 0.0)).r,
				texture(sampler3D(sdf_cascades, linear_sampler), hit_uvw + vec3(0.0, EPSILON, 0.0)).r - texture(sampler3D(sdf_cascades, linear_sampler), hit_uvw - vec3(0.0, EPSILON, 0.0)).r,
				texture(sampler3D(sdf_cascades, linear_sampler), hit_uvw + vec3(0.0, 0.0, EPSILON / float(params.max_cascades))).r - texture(sampler3D(sdf_cascades, linear_sampler), hit_uvw - vec3(0.0, 0.0, EPSILON / float(params.max_cascades))).r));

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
		//light = abs(hit_normal);//texelFetch(sampler3D(light_cascades, linear_sampler), hit_cell + normal_ofs,0).rgb;
	}


	imageStore(screen_buffer, screen_pos, vec4(linear_to_srgb(light), 1.0));
}
