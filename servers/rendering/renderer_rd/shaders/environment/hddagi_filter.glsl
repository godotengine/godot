#[compute]

#version 450

#VERSION_DEFINES

#define WG_SIZE 8

layout(local_size_x = WG_SIZE, local_size_y = WG_SIZE, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform texture2D depth_buffer;
layout(set = 0, binding = 1) uniform texture2D normal_roughness_buffer;
layout(set = 0, binding = 2) uniform texture2D specular_buffer;
layout(set = 0, binding = 3) uniform texture2D blend_buffer;
layout(r32ui, set = 0, binding = 4) uniform restrict writeonly uimage2D dst_specular_buffer;
layout(rg8, set = 0, binding = 5) uniform restrict writeonly image2D dst_blend_buffer;
layout(set = 0, binding = 6) uniform sampler linear_sampler;

layout(constant_id = 1) const bool sc_use_full_projection_matrix = false;
layout(constant_id = 0) const bool sc_half_res = false;

#ifdef HALF_SIZE
#define RADIUS 6
#else
#define RADIUS 12
#endif

#define ROUGHNESS_TO_REFLECTION_TRESHOOLD 0.3

layout(set = 0, binding = 7, std140) uniform SceneData {
	mat4x4 inv_projection[2];
	mat4x4 cam_transform;
	vec4 eye_offset[2];

	ivec2 screen_size;
	float pad1;
	float pad2;
}
scene_data;

layout(push_constant, std430) uniform Params {
	bool orthogonal;
	float z_near;
	float z_far;
	uint view_index;

	vec4 proj_info;

	ivec2 filter_dir;
	uvec2 pad;
}
params;

vec3 reconstruct_position(ivec2 screen_pos) {
	if (sc_use_full_projection_matrix) {
		vec4 pos;
		pos.xy = (2.0 * vec2(screen_pos) / vec2(scene_data.screen_size)) - 1.0;
		pos.z = texelFetch(sampler2D(depth_buffer, linear_sampler), screen_pos, 0).r * 2.0 - 1.0;
		pos.w = 1.0;

		pos = scene_data.inv_projection[params.view_index] * pos;

		return pos.xyz / pos.w;
	} else {
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
}

vec4 fetch_normal_and_roughness(ivec2 pos) {
	vec4 normal_roughness = texelFetch(sampler2D(normal_roughness_buffer, linear_sampler), pos, 0);
	if (normal_roughness.xyz != vec3(0)) {
		normal_roughness.xyz = normalize(normal_roughness.xyz * 2.0 - 1.0);
		bool dynamic_object = normal_roughness.a > 0.5;
		if (dynamic_object) {
			normal_roughness.a = 1.0 - normal_roughness.a;
		}
		normal_roughness.a /= (127.0 / 255.0);
	}
	return normal_roughness;
}

uint rgbe_encode(vec3 rgb) {
	const float rgbe_max = uintBitsToFloat(0x477F8000);
	const float rgbe_min = uintBitsToFloat(0x37800000);

	rgb = clamp(rgb, 0, rgbe_max);

	float max_channel = max(max(rgbe_min, rgb.r), max(rgb.g, rgb.b));

	float bias = uintBitsToFloat((floatBitsToUint(max_channel) + 0x07804000) & 0x7F800000);

	uvec3 urgb = floatBitsToUint(rgb + bias);
	uint e = (floatBitsToUint(bias) << 4) + 0x10000000;
	return e | (urgb.b << 18) | (urgb.g << 9) | (urgb.r & 0x1FF);
}

void main() {
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

	ivec2 depth_pos = pos;
	if (sc_half_res) {
		depth_pos <<= 1;
	}

	vec4 normal_roughness = fetch_normal_and_roughness(depth_pos);

	if (normal_roughness.rgb == vec3(0.0) || normal_roughness.a >= ROUGHNESS_TO_REFLECTION_TRESHOOLD) {
		return; // No need to do anything.
	}

	float depth = reconstruct_position(depth_pos).z;

	vec4 specular_accum = vec4(0);
	float total_weight = 0;
	float diffuse_blend = texelFetch(sampler2D(blend_buffer, linear_sampler), pos, 0).r;

	for (int i = -RADIUS; i <= RADIUS; i++) {
		ivec2 read_pos = pos + params.filter_dir * i;
		vec4 specular;
		specular.rgb = texelFetch(sampler2D(specular_buffer, linear_sampler), read_pos, 0).rgb;
		specular.a = texelFetch(sampler2D(blend_buffer, linear_sampler), read_pos, 0).g;
		if (sc_half_res) {
			read_pos <<= 1;
		}
		float d = reconstruct_position(read_pos).z;
		vec4 nr = fetch_normal_and_roughness(read_pos);

		float weight = exp(-abs(depth - d));
		float dp = max(0.0, dot(nr.rgb, normal_roughness.rgb));
		dp = pow(dp, 4.0); // The more curvature, the less filter.
		weight *= dp;
		weight *= max(0.0, 1.0 - abs(nr.a - normal_roughness.a) * 4.0);

		if (weight > 0.0) {
			specular_accum += specular * weight;
			total_weight += weight;
		}
	}

	if (total_weight > 0.0) {
		specular_accum /= total_weight;
	}

	//imageStore(dst_specular_buffer,pos,uvec4(rgbe_encode(specular_accum.rgb)));
	imageStore(dst_specular_buffer, pos, uvec4(rgbe_encode(specular_accum.rgb)));
	uint blend = uint(clamp(specular_accum.a * 0xF, 0, 0xF)) | (uint(clamp(diffuse_blend * 0xF, 0, 0xF)) << 4);
	imageStore(dst_blend_buffer, pos, vec4(diffuse_blend, specular_accum.a, 0, 0));
}
