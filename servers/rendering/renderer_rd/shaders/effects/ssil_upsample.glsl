#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba16, set = 0, binding = 0) uniform restrict writeonly image2D dest_image;
layout(set = 0, binding = 1) uniform sampler2D source_ssil;

layout(r8, set = 1, binding = 0) uniform restrict readonly image2D source_edges;

layout(push_constant, std430) uniform Params {
	float edge_sharpness;
	float pad;
	vec2 half_screen_pixel_size;
}
params;

void add_sample(vec4 p_ssil_value, float p_edge_value, inout vec4 r_sum, inout float r_sum_weight) {
	float weight = p_edge_value;

	r_sum += (weight * p_ssil_value);
	r_sum_weight += weight;
}

vec4 unpack_edges(float p_packed_val) {
	uint packed_val = uint(p_packed_val * 255.5);
	vec4 edgesLRTB;
	edgesLRTB.x = float((packed_val >> 6) & 0x03) / 3.0;
	edgesLRTB.y = float((packed_val >> 4) & 0x03) / 3.0;
	edgesLRTB.z = float((packed_val >> 2) & 0x03) / 3.0;
	edgesLRTB.w = float((packed_val >> 0) & 0x03) / 3.0;

	return clamp(edgesLRTB + params.edge_sharpness, 0.0, 1.0);
}

vec4 bilateral_upsample(ivec2 p_pos, vec2 p_coord) {
	vec4 vC = textureLodOffset(source_ssil, vec2(p_coord), 0.0, ivec2(0, 0));
	vec4 vL = textureLodOffset(source_ssil, vec2(p_coord), 0.0, ivec2(-1, 0));
	vec4 vT = textureLodOffset(source_ssil, vec2(p_coord), 0.0, ivec2(0, -1));
	vec4 vR = textureLodOffset(source_ssil, vec2(p_coord), 0.0, ivec2(1, 0));
	vec4 vB = textureLodOffset(source_ssil, vec2(p_coord), 0.0, ivec2(0, 1));

	float packed_edges = imageLoad(source_edges, p_pos).r;
	vec4 edgesLRTB = unpack_edges(packed_edges);

	float sum_weight = 0.5;
	vec4 sum = vC * sum_weight;

	add_sample(vL, edgesLRTB.x, sum, sum_weight);
	add_sample(vR, edgesLRTB.y, sum, sum_weight);
	add_sample(vT, edgesLRTB.z, sum, sum_weight);
	add_sample(vB, edgesLRTB.w, sum, sum_weight);

	vec4 ssil_avg = sum / sum_weight;

	return ssil_avg;
}

void main() {
	ivec2 ssC = ivec2(gl_GlobalInvocationID.xy);

	vec2 p_coord = (vec2(ssC) + 0.5) * vec2(params.half_screen_pixel_size);

	vec4 upsampled_ssil = bilateral_upsample(ssC, p_coord);

	imageStore(dest_image, ssC, upsampled_ssil);
}
