#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source;
layout(set = 0, binding = 1) uniform restrict writeonly image2D dest;

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	uint mip_level;
}
params;

shared vec4 cache[16][16];

const float WEIGHTS[7] = float[7](
		0.07130343198685299,
		0.1315141208431224,
		0.18987923288883812,
		0.21460642856237303,
		0.18987923288883812,
		0.1315141208431224,
		0.07130343198685299);

float get_weight(vec4 c) {
	return mix(clamp(params.mip_level * 0.2, 0.0, 1.0), 1.0, c.a);
}

vec4 apply_gaus_horz(ivec2 local) {
	vec4 c0 = cache[local.x - 3][local.y];
	float w0 = WEIGHTS[0] * get_weight(c0);

	vec4 c1 = cache[local.x - 2][local.y];
	float w1 = WEIGHTS[1] * get_weight(c1);

	vec4 c2 = cache[local.x - 1][local.y];
	float w2 = WEIGHTS[2] * get_weight(c2);

	vec4 c3 = cache[local.x][local.y];
	float w3 = WEIGHTS[3] * get_weight(c3);

	vec4 c4 = cache[local.x + 1][local.y];
	float w4 = WEIGHTS[4] * get_weight(c4);

	vec4 c5 = cache[local.x + 2][local.y];
	float w5 = WEIGHTS[5] * get_weight(c5);

	vec4 c6 = cache[local.x + 3][local.y];
	float w6 = WEIGHTS[6] * get_weight(c6);

	vec4 c = c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3 + c4 * w4 + c5 * w5 + c6 * w6;
	float w = w0 + w1 + w2 + w3 + w4 + w5 + w6;

	if (w > 0.0) {
		c /= w;
	} else {
		c = vec4(0.0);
	}

	return c;
}

shared vec4 temp_cache[8][16];

vec4 apply_gaus_vert(ivec2 local) {
	vec4 c0 = temp_cache[local.x][local.y - 3];
	float w0 = WEIGHTS[0] * get_weight(c0);

	vec4 c1 = temp_cache[local.x][local.y - 2];
	float w1 = WEIGHTS[1] * get_weight(c1);

	vec4 c2 = temp_cache[local.x][local.y - 1];
	float w2 = WEIGHTS[2] * get_weight(c2);

	vec4 c3 = temp_cache[local.x][local.y];
	float w3 = WEIGHTS[3] * get_weight(c3);

	vec4 c4 = temp_cache[local.x][local.y + 1];
	float w4 = WEIGHTS[4] * get_weight(c4);

	vec4 c5 = temp_cache[local.x][local.y + 2];
	float w5 = WEIGHTS[5] * get_weight(c5);

	vec4 c6 = temp_cache[local.x][local.y + 3];
	float w6 = WEIGHTS[6] * get_weight(c6);

	vec4 c = c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3 + c4 * w4 + c5 * w5 + c6 * w6;
	float w = w0 + w1 + w2 + w3 + w4 + w5 + w6;

	if (w > 0.0) {
		c /= w;
	} else {
		c = vec4(0.0);
	}

	return c;
}

vec4 get_sample(ivec2 pixel_pos) {
	return textureLod(source, (vec2(pixel_pos) + 0.5) / params.screen_size, 0);
}

void main() {
	ivec2 global = ivec2(gl_GlobalInvocationID.xy);
	ivec2 local = ivec2(gl_LocalInvocationID.xy);

	cache[local.x * 2 + 0][local.y * 2 + 0] = get_sample(global + local - 4 + ivec2(0, 0));
	cache[local.x * 2 + 1][local.y * 2 + 0] = get_sample(global + local - 4 + ivec2(1, 0));
	cache[local.x * 2 + 0][local.y * 2 + 1] = get_sample(global + local - 4 + ivec2(0, 1));
	cache[local.x * 2 + 1][local.y * 2 + 1] = get_sample(global + local - 4 + ivec2(1, 1));

	memoryBarrierShared();
	barrier();

	temp_cache[local.x][local.y * 2 + 0] = apply_gaus_horz(ivec2(local.x + 4, local.y * 2 + 0));
	temp_cache[local.x][local.y * 2 + 1] = apply_gaus_horz(ivec2(local.x + 4, local.y * 2 + 1));

	memoryBarrierShared();
	barrier();

	if (any(greaterThanEqual(global, params.screen_size))) {
		return;
	}

	imageStore(dest, global, apply_gaus_vert(local + ivec2(0, 4)));
}
