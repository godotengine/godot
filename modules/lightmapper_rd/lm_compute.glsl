#[versions]

primary = "#define MODE_DIRECT_LIGHT";
secondary = "#define MODE_BOUNCE_LIGHT";
dilate = "#define MODE_DILATE";
unocclude = "#define MODE_UNOCCLUDE";
light_probes = "#define MODE_LIGHT_PROBES";

#[compute]

#version 450

#VERSION_DEFINES

// One 2D local group focusing in one layer at a time, though all
// in parallel (no barriers) makes more sense than a 3D local group
// as this can take more advantage of the cache for each group.

#ifdef MODE_LIGHT_PROBES

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#else

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#endif

#include "lm_common_inc.glsl"

#ifdef MODE_LIGHT_PROBES

layout(set = 1, binding = 0, std430) restrict buffer LightProbeData {
	vec4 data[];
}
light_probes;

layout(set = 1, binding = 1) uniform texture2DArray source_light;
layout(set = 1, binding = 2) uniform texture2DArray source_direct_light; //also need the direct light, which was omitted
layout(set = 1, binding = 3) uniform texture2D environment;
#endif

#ifdef MODE_UNOCCLUDE

layout(rgba32f, set = 1, binding = 0) uniform restrict image2DArray position;
layout(rgba32f, set = 1, binding = 1) uniform restrict readonly image2DArray unocclude;

#endif

#if defined(MODE_DIRECT_LIGHT) || defined(MODE_BOUNCE_LIGHT)

layout(rgba16f, set = 1, binding = 0) uniform restrict writeonly image2DArray dest_light;
layout(set = 1, binding = 1) uniform texture2DArray source_light;
layout(set = 1, binding = 2) uniform texture2DArray source_position;
layout(set = 1, binding = 3) uniform texture2DArray source_normal;
layout(rgba16f, set = 1, binding = 4) uniform restrict image2DArray accum_light;

#endif

#ifdef MODE_BOUNCE_LIGHT
layout(rgba32f, set = 1, binding = 5) uniform restrict image2DArray bounce_accum;
layout(set = 1, binding = 6) uniform texture2D environment;
#endif
#ifdef MODE_DIRECT_LIGHT
layout(rgba32f, set = 1, binding = 5) uniform restrict writeonly image2DArray primary_dynamic;
#endif

#ifdef MODE_DILATE
layout(rgba16f, set = 1, binding = 0) uniform restrict writeonly image2DArray dest_light;
layout(set = 1, binding = 1) uniform texture2DArray source_light;
#endif

layout(push_constant, binding = 0, std430) uniform Params {
	ivec2 atlas_size; // x used for light probe mode total probes
	uint ray_count;
	uint ray_to;

	vec3 world_size;
	float bias;

	vec3 to_cell_offset;
	uint ray_from;

	vec3 to_cell_size;
	uint light_count;

	int grid_size;
	int atlas_slice;
	ivec2 region_ofs;

	mat3x4 env_transform;
}
params;

//check it, but also return distance and barycentric coords (for uv lookup)
bool ray_hits_triangle(vec3 from, vec3 dir, float max_dist, vec3 p0, vec3 p1, vec3 p2, out float r_distance, out vec3 r_barycentric) {
	const float EPSILON = 0.00001;
	const vec3 e0 = p1 - p0;
	const vec3 e1 = p0 - p2;
	vec3 triangle_normal = cross(e1, e0);

	float n_dot_dir = dot(triangle_normal, dir);

	if (abs(n_dot_dir) < EPSILON) {
		return false;
	}

	const vec3 e2 = (p0 - from) / n_dot_dir;
	const vec3 i = cross(dir, e2);

	r_barycentric.y = dot(i, e1);
	r_barycentric.z = dot(i, e0);
	r_barycentric.x = 1.0 - (r_barycentric.z + r_barycentric.y);
	r_distance = dot(triangle_normal, e2);

	return (r_distance > params.bias) && (r_distance < max_dist) && all(greaterThanEqual(r_barycentric, vec3(0.0)));
}

const uint RAY_MISS = 0;
const uint RAY_FRONT = 1;
const uint RAY_BACK = 2;
const uint RAY_ANY = 3;

uint trace_ray(vec3 p_from, vec3 p_to
#if defined(MODE_BOUNCE_LIGHT) || defined(MODE_LIGHT_PROBES)
		,
		out uint r_triangle, out vec3 r_barycentric
#endif
#if defined(MODE_UNOCCLUDE)
		,
		out float r_distance, out vec3 r_normal
#endif
) {

	/* world coords */

	vec3 rel = p_to - p_from;
	float rel_len = length(rel);
	vec3 dir = normalize(rel);
	vec3 inv_dir = 1.0 / dir;

	/* cell coords */

	vec3 from_cell = (p_from - params.to_cell_offset) * params.to_cell_size;
	vec3 to_cell = (p_to - params.to_cell_offset) * params.to_cell_size;

	//prepare DDA
	vec3 rel_cell = to_cell - from_cell;
	ivec3 icell = ivec3(from_cell);
	ivec3 iendcell = ivec3(to_cell);
	vec3 dir_cell = normalize(rel_cell);
	vec3 delta = min(abs(1.0 / dir_cell), params.grid_size); // use params.grid_size as max to prevent infinity values
	ivec3 step = ivec3(sign(rel_cell));
	vec3 side = (sign(rel_cell) * (vec3(icell) - from_cell) + (sign(rel_cell) * 0.5) + 0.5) * delta;

	uint iters = 0;
	while (all(greaterThanEqual(icell, ivec3(0))) && all(lessThan(icell, ivec3(params.grid_size))) && iters < 1000) {
		uvec2 cell_data = texelFetch(usampler3D(grid, linear_sampler), icell, 0).xy;
		if (cell_data.x > 0) { //triangles here
			uint hit = RAY_MISS;
			float best_distance = 1e20;

			for (uint i = 0; i < cell_data.x; i++) {
				uint tidx = grid_indices.data[cell_data.y + i];

				//Ray-Box test
				Triangle triangle = triangles.data[tidx];
				vec3 t0 = (triangle.min_bounds - p_from) * inv_dir;
				vec3 t1 = (triangle.max_bounds - p_from) * inv_dir;
				vec3 tmin = min(t0, t1), tmax = max(t0, t1);

				if (max(tmin.x, max(tmin.y, tmin.z)) > min(tmax.x, min(tmax.y, tmax.z))) {
					continue; //ray box failed
				}

				//prepare triangle vertices
				vec3 vtx0 = vertices.data[triangle.indices.x].position;
				vec3 vtx1 = vertices.data[triangle.indices.y].position;
				vec3 vtx2 = vertices.data[triangle.indices.z].position;
#if defined(MODE_UNOCCLUDE) || defined(MODE_BOUNCE_LIGHT) || defined(MODE_LIGHT_PROBES)
				vec3 normal = -normalize(cross((vtx0 - vtx1), (vtx0 - vtx2)));

				bool backface = dot(normal, dir) >= 0.0;
#endif

				float distance;
				vec3 barycentric;

				if (ray_hits_triangle(p_from, dir, rel_len, vtx0, vtx1, vtx2, distance, barycentric)) {
#ifdef MODE_DIRECT_LIGHT
					return RAY_ANY; //any hit good
#endif

#if defined(MODE_UNOCCLUDE) || defined(MODE_BOUNCE_LIGHT) || defined(MODE_LIGHT_PROBES)
					if (!backface) {
						// the case of meshes having both a front and back face in the same plane is more common than
						// expected, so if this is a front-face, bias it closer to the ray origin, so it always wins over the back-face
						distance = max(params.bias, distance - params.bias);
					}

					if (distance < best_distance) {
						hit = backface ? RAY_BACK : RAY_FRONT;
						best_distance = distance;
#if defined(MODE_UNOCCLUDE)
						r_distance = distance;
						r_normal = normal;
#endif
#if defined(MODE_BOUNCE_LIGHT) || defined(MODE_LIGHT_PROBES)
						r_triangle = tidx;
						r_barycentric = barycentric;
#endif
					}
#endif
				}
			}
#if defined(MODE_UNOCCLUDE) || defined(MODE_BOUNCE_LIGHT) || defined(MODE_LIGHT_PROBES)

			if (hit != RAY_MISS) {
				return hit;
			}
#endif
		}

		if (icell == iendcell) {
			break;
		}

		bvec3 mask = lessThanEqual(side.xyz, min(side.yzx, side.zxy));
		side += vec3(mask) * delta;
		icell += ivec3(vec3(mask)) * step;

		iters++;
	}

	return RAY_MISS;
}

// https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
uint hash(uint value) {
	uint state = value * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

uint random_seed(ivec3 seed) {
	return hash(seed.x ^ hash(seed.y ^ hash(seed.z)));
}

// generates a random value in range [0.0, 1.0)
float randomize(inout uint value) {
	value = hash(value);
	return float(value / 4294967296.0);
}

const float PI = 3.14159265f;

// http://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.4.pdf (chapter 15)
vec3 generate_hemisphere_uniform_direction(inout uint noise) {
	float noise1 = randomize(noise);
	float noise2 = randomize(noise) * 2.0 * PI;

	float factor = sqrt(1 - (noise1 * noise1));
	return vec3(factor * cos(noise2), factor * sin(noise2), noise1);
}

vec3 generate_hemisphere_cosine_weighted_direction(inout uint noise) {
	float noise1 = randomize(noise);
	float noise2 = randomize(noise) * 2.0 * PI;

	return vec3(sqrt(noise1) * cos(noise2), sqrt(noise1) * sin(noise2), sqrt(1.0 - noise1));
}

float get_omni_attenuation(float distance, float inv_range, float decay) {
	float nd = distance * inv_range;
	nd *= nd;
	nd *= nd; // nd^4
	nd = max(1.0 - nd, 0.0);
	nd *= nd; // nd^2
	return nd * pow(max(distance, 0.0001), -decay);
}

void main() {
#ifdef MODE_LIGHT_PROBES
	int probe_index = int(gl_GlobalInvocationID.x);
	if (probe_index >= params.atlas_size.x) { //too large, do nothing
		return;
	}

#else
	ivec2 atlas_pos = ivec2(gl_GlobalInvocationID.xy) + params.region_ofs;
	if (any(greaterThanEqual(atlas_pos, params.atlas_size))) { //too large, do nothing
		return;
	}
#endif

#ifdef MODE_DIRECT_LIGHT

	vec3 normal = texelFetch(sampler2DArray(source_normal, linear_sampler), ivec3(atlas_pos, params.atlas_slice), 0).xyz;
	if (length(normal) < 0.5) {
		return; //empty texel, no process
	}
	vec3 position = texelFetch(sampler2DArray(source_position, linear_sampler), ivec3(atlas_pos, params.atlas_slice), 0).xyz;

	//go through all lights
	//start by own light (emissive)
	vec3 static_light = vec3(0.0);
	vec3 dynamic_light = vec3(0.0);

#ifdef USE_SH_LIGHTMAPS
	vec4 sh_accum[4] = vec4[](
			vec4(0.0, 0.0, 0.0, 1.0),
			vec4(0.0, 0.0, 0.0, 1.0),
			vec4(0.0, 0.0, 0.0, 1.0),
			vec4(0.0, 0.0, 0.0, 1.0));
#endif

	for (uint i = 0; i < params.light_count; i++) {
		vec3 light_pos;
		float attenuation;
		if (lights.data[i].type == LIGHT_TYPE_DIRECTIONAL) {
			vec3 light_vec = lights.data[i].direction;
			light_pos = position - light_vec * length(params.world_size);
			attenuation = 1.0;
		} else {
			light_pos = lights.data[i].position;
			float d = distance(position, light_pos);
			if (d > lights.data[i].range) {
				continue;
			}

			attenuation = get_omni_attenuation(d, 1.0 / lights.data[i].range, lights.data[i].attenuation);

			if (lights.data[i].type == LIGHT_TYPE_SPOT) {
				vec3 rel = normalize(position - light_pos);
				float cos_spot_angle = lights.data[i].cos_spot_angle;
				float cos_angle = dot(rel, lights.data[i].direction);

				if (cos_angle < cos_spot_angle) {
					continue; //invisible, dont try
				}

				float scos = max(cos_angle, cos_spot_angle);
				float spot_rim = max(0.0001, (1.0 - scos) / (1.0 - cos_spot_angle));
				attenuation *= 1.0 - pow(spot_rim, lights.data[i].inv_spot_attenuation);
			}
		}

		vec3 light_dir = normalize(light_pos - position);
		attenuation *= max(0.0, dot(normal, light_dir));

		if (attenuation <= 0.0001) {
			continue; //no need to do anything
		}

		if (trace_ray(position + light_dir * params.bias, light_pos) == RAY_MISS) {
			vec3 light = lights.data[i].color * lights.data[i].energy * attenuation;
			if (lights.data[i].static_bake) {
				static_light += light;
#ifdef USE_SH_LIGHTMAPS

				float c[4] = float[](
						0.282095, //l0
						0.488603 * light_dir.y, //l1n1
						0.488603 * light_dir.z, //l1n0
						0.488603 * light_dir.x //l1p1
				);

				for (uint j = 0; j < 4; j++) {
					sh_accum[j].rgb += light * c[j] * (1.0 / 3.0);
				}
#endif

			} else {
				dynamic_light += light;
			}
		}
	}

	vec3 albedo = texelFetch(sampler2DArray(albedo_tex, linear_sampler), ivec3(atlas_pos, params.atlas_slice), 0).rgb;
	vec3 emissive = texelFetch(sampler2DArray(emission_tex, linear_sampler), ivec3(atlas_pos, params.atlas_slice), 0).rgb;

	dynamic_light *= albedo; //if it will bounce, must multiply by albedo
	dynamic_light += emissive;

	//keep for lightprobes
	imageStore(primary_dynamic, ivec3(atlas_pos, params.atlas_slice), vec4(dynamic_light, 1.0));

	dynamic_light += static_light * albedo; //send for bounces
	imageStore(dest_light, ivec3(atlas_pos, params.atlas_slice), vec4(dynamic_light, 1.0));

#ifdef USE_SH_LIGHTMAPS
	//keep for adding at the end
	imageStore(accum_light, ivec3(atlas_pos, params.atlas_slice * 4 + 0), sh_accum[0]);
	imageStore(accum_light, ivec3(atlas_pos, params.atlas_slice * 4 + 1), sh_accum[1]);
	imageStore(accum_light, ivec3(atlas_pos, params.atlas_slice * 4 + 2), sh_accum[2]);
	imageStore(accum_light, ivec3(atlas_pos, params.atlas_slice * 4 + 3), sh_accum[3]);

#else
	imageStore(accum_light, ivec3(atlas_pos, params.atlas_slice), vec4(static_light, 1.0));
#endif

#endif

#ifdef MODE_BOUNCE_LIGHT

	vec3 normal = texelFetch(sampler2DArray(source_normal, linear_sampler), ivec3(atlas_pos, params.atlas_slice), 0).xyz;
	if (length(normal) < 0.5) {
		return; //empty texel, no process
	}

	vec3 position = texelFetch(sampler2DArray(source_position, linear_sampler), ivec3(atlas_pos, params.atlas_slice), 0).xyz;

	vec3 v0 = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
	vec3 tangent = normalize(cross(v0, normal));
	vec3 bitangent = normalize(cross(tangent, normal));
	mat3 normal_mat = mat3(tangent, bitangent, normal);

#ifdef USE_SH_LIGHTMAPS
	vec4 sh_accum[4] = vec4[](
			vec4(0.0, 0.0, 0.0, 1.0),
			vec4(0.0, 0.0, 0.0, 1.0),
			vec4(0.0, 0.0, 0.0, 1.0),
			vec4(0.0, 0.0, 0.0, 1.0));
#endif
	vec3 light_average = vec3(0.0);
	float active_rays = 0.0;
	uint noise = random_seed(ivec3(params.ray_from, atlas_pos));
	for (uint i = params.ray_from; i < params.ray_to; i++) {
		vec3 ray_dir = normal_mat * generate_hemisphere_cosine_weighted_direction(noise);

		uint tidx;
		vec3 barycentric;

		vec3 light = vec3(0.0);
		uint trace_result = trace_ray(position + ray_dir * params.bias, position + ray_dir * length(params.world_size), tidx, barycentric);
		if (trace_result == RAY_FRONT) {
			//hit a triangle
			vec2 uv0 = vertices.data[triangles.data[tidx].indices.x].uv;
			vec2 uv1 = vertices.data[triangles.data[tidx].indices.y].uv;
			vec2 uv2 = vertices.data[triangles.data[tidx].indices.z].uv;
			vec3 uvw = vec3(barycentric.x * uv0 + barycentric.y * uv1 + barycentric.z * uv2, float(triangles.data[tidx].slice));

			light = textureLod(sampler2DArray(source_light, linear_sampler), uvw, 0.0).rgb;
			active_rays += 1.0;
		} else if (trace_result == RAY_MISS) {
			if (params.env_transform[0][3] == 0.0) { // Use env_transform[0][3] to indicate when we are computing the first bounce
				// Did not hit a triangle, reach out for the sky
				vec3 sky_dir = normalize(mat3(params.env_transform) * ray_dir);

				vec2 st = vec2(
						atan(sky_dir.x, sky_dir.z),
						acos(sky_dir.y));

				if (st.x < 0.0)
					st.x += PI * 2.0;

				st /= vec2(PI * 2.0, PI);

				light = textureLod(sampler2D(environment, linear_sampler), st, 0.0).rgb;
			}
			active_rays += 1.0;
		}

		light_average += light;

#ifdef USE_SH_LIGHTMAPS

		float c[4] = float[](
				0.282095, //l0
				0.488603 * ray_dir.y, //l1n1
				0.488603 * ray_dir.z, //l1n0
				0.488603 * ray_dir.x //l1p1
		);

		for (uint j = 0; j < 4; j++) {
			sh_accum[j].rgb += light * c[j] * (8.0 / float(params.ray_count));
		}
#endif
	}

	vec3 light_total;
	if (params.ray_from == 0) {
		light_total = vec3(0.0);
	} else {
		vec4 accum = imageLoad(bounce_accum, ivec3(atlas_pos, params.atlas_slice));
		light_total = accum.rgb;
		active_rays += accum.a;
	}

	light_total += light_average;

#ifdef USE_SH_LIGHTMAPS

	for (int i = 0; i < 4; i++) {
		vec4 accum = imageLoad(accum_light, ivec3(atlas_pos, params.atlas_slice * 4 + i));
		accum.rgb += sh_accum[i].rgb;
		imageStore(accum_light, ivec3(atlas_pos, params.atlas_slice * 4 + i), accum);
	}

#endif
	if (params.ray_to == params.ray_count) {
		if (active_rays > 0) {
			light_total /= active_rays;
		}
		imageStore(dest_light, ivec3(atlas_pos, params.atlas_slice), vec4(light_total, 1.0));
#ifndef USE_SH_LIGHTMAPS
		vec4 accum = imageLoad(accum_light, ivec3(atlas_pos, params.atlas_slice));
		accum.rgb += light_total;
		imageStore(accum_light, ivec3(atlas_pos, params.atlas_slice), accum);
#endif
	} else {
		imageStore(bounce_accum, ivec3(atlas_pos, params.atlas_slice), vec4(light_total, active_rays));
	}

#endif

#ifdef MODE_UNOCCLUDE

	//texel_size = 0.5;
	//compute tangents

	vec4 position_alpha = imageLoad(position, ivec3(atlas_pos, params.atlas_slice));
	if (position_alpha.a < 0.5) {
		return;
	}

	vec3 vertex_pos = position_alpha.xyz;
	vec4 normal_tsize = imageLoad(unocclude, ivec3(atlas_pos, params.atlas_slice));

	vec3 face_normal = normal_tsize.xyz;
	float texel_size = normal_tsize.w;

	vec3 v0 = abs(face_normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
	vec3 tangent = normalize(cross(v0, face_normal));
	vec3 bitangent = normalize(cross(tangent, face_normal));
	vec3 base_pos = vertex_pos + face_normal * params.bias; //raise a bit

	vec3 rays[4] = vec3[](tangent, bitangent, -tangent, -bitangent);
	float min_d = 1e20;
	for (int i = 0; i < 4; i++) {
		vec3 ray_to = base_pos + rays[i] * texel_size;
		float d;
		vec3 norm;

		if (trace_ray(base_pos, ray_to, d, norm) == RAY_BACK) {
			if (d < min_d) {
				vertex_pos = base_pos + rays[i] * d + norm * params.bias * 10.0; //this bias needs to be greater than the regular bias, because otherwise later, rays will go the other side when pointing back.
				min_d = d;
			}
		}
	}

	position_alpha.xyz = vertex_pos;

	imageStore(position, ivec3(atlas_pos, params.atlas_slice), position_alpha);

#endif

#ifdef MODE_LIGHT_PROBES

	vec3 position = probe_positions.data[probe_index].xyz;

	vec4 probe_sh_accum[9] = vec4[](
			vec4(0.0),
			vec4(0.0),
			vec4(0.0),
			vec4(0.0),
			vec4(0.0),
			vec4(0.0),
			vec4(0.0),
			vec4(0.0),
			vec4(0.0));

	uint noise = random_seed(ivec3(params.ray_from, probe_index, 49502741 /* some prime */));
	for (uint i = params.ray_from; i < params.ray_to; i++) {
		vec3 ray_dir = generate_hemisphere_uniform_direction(noise);
		if (bool(i & 1)) {
			//throw to both sides, so alternate them
			ray_dir.z *= -1.0;
		}

		uint tidx;
		vec3 barycentric;
		vec3 light;

		uint trace_result = trace_ray(position + ray_dir * params.bias, position + ray_dir * length(params.world_size), tidx, barycentric);
		if (trace_result == RAY_FRONT) {
			vec2 uv0 = vertices.data[triangles.data[tidx].indices.x].uv;
			vec2 uv1 = vertices.data[triangles.data[tidx].indices.y].uv;
			vec2 uv2 = vertices.data[triangles.data[tidx].indices.z].uv;
			vec3 uvw = vec3(barycentric.x * uv0 + barycentric.y * uv1 + barycentric.z * uv2, float(triangles.data[tidx].slice));

			light = textureLod(sampler2DArray(source_light, linear_sampler), uvw, 0.0).rgb;
			light += textureLod(sampler2DArray(source_direct_light, linear_sampler), uvw, 0.0).rgb;
		} else if (trace_result == RAY_MISS) {
			//did not hit a triangle, reach out for the sky
			vec3 sky_dir = normalize(mat3(params.env_transform) * ray_dir);

			vec2 st = vec2(
					atan(sky_dir.x, sky_dir.z),
					acos(sky_dir.y));

			if (st.x < 0.0)
				st.x += PI * 2.0;

			st /= vec2(PI * 2.0, PI);

			light = textureLod(sampler2D(environment, linear_sampler), st, 0.0).rgb;
		}

		{
			float c[9] = float[](
					0.282095, //l0
					0.488603 * ray_dir.y, //l1n1
					0.488603 * ray_dir.z, //l1n0
					0.488603 * ray_dir.x, //l1p1
					1.092548 * ray_dir.x * ray_dir.y, //l2n2
					1.092548 * ray_dir.y * ray_dir.z, //l2n1
					//0.315392 * (ray_dir.x * ray_dir.x + ray_dir.y * ray_dir.y + 2.0 * ray_dir.z * ray_dir.z), //l20
					0.315392 * (3.0 * ray_dir.z * ray_dir.z - 1.0), //l20
					1.092548 * ray_dir.x * ray_dir.z, //l2p1
					0.546274 * (ray_dir.x * ray_dir.x - ray_dir.y * ray_dir.y) //l2p2
			);

			for (uint j = 0; j < 9; j++) {
				probe_sh_accum[j].rgb += light * c[j];
			}
		}
	}

	if (params.ray_from > 0) {
		for (uint j = 0; j < 9; j++) { //accum from existing
			probe_sh_accum[j] += light_probes.data[probe_index * 9 + j];
		}
	}

	if (params.ray_to == params.ray_count) {
		for (uint j = 0; j < 9; j++) { //accum from existing
			probe_sh_accum[j] *= 4.0 / float(params.ray_count);
		}
	}

	for (uint j = 0; j < 9; j++) { //accum from existing
		light_probes.data[probe_index * 9 + j] = probe_sh_accum[j];
	}

#endif

#ifdef MODE_DILATE

	vec4 c = texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos, params.atlas_slice), 0);
	//sides first, as they are closer
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(-1, 0), params.atlas_slice), 0);
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(0, 1), params.atlas_slice), 0);
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(1, 0), params.atlas_slice), 0);
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(0, -1), params.atlas_slice), 0);
	//endpoints second
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(-1, -1), params.atlas_slice), 0);
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(-1, 1), params.atlas_slice), 0);
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(1, -1), params.atlas_slice), 0);
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(1, 1), params.atlas_slice), 0);

	//far sides third
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(-2, 0), params.atlas_slice), 0);
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(0, 2), params.atlas_slice), 0);
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(2, 0), params.atlas_slice), 0);
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(0, -2), params.atlas_slice), 0);

	//far-mid endpoints
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(-2, -1), params.atlas_slice), 0);
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(-2, 1), params.atlas_slice), 0);
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(2, -1), params.atlas_slice), 0);
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(2, 1), params.atlas_slice), 0);

	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(-1, -2), params.atlas_slice), 0);
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(-1, 2), params.atlas_slice), 0);
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(1, -2), params.atlas_slice), 0);
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(1, 2), params.atlas_slice), 0);
	//far endpoints
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(-2, -2), params.atlas_slice), 0);
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(-2, 2), params.atlas_slice), 0);
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(2, -2), params.atlas_slice), 0);
	c = c.a > 0.5 ? c : texelFetch(sampler2DArray(source_light, linear_sampler), ivec3(atlas_pos + ivec2(2, 2), params.atlas_slice), 0);

	imageStore(dest_light, ivec3(atlas_pos, params.atlas_slice), c);

#endif
}
