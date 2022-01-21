#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_ssil;

layout(rgba16, set = 1, binding = 0) uniform restrict writeonly image2D dest_image;

layout(r8, set = 2, binding = 0) uniform restrict readonly image2D source_edges;

layout(push_constant, binding = 1, std430) uniform Params {
	float edge_sharpness;
	float pad;
	vec2 half_screen_pixel_size;
}
params;

vec4 unpack_edges(float p_packed_val) {
	uint packed_val = uint(p_packed_val * 255.5);
	vec4 edgesLRTB;
	edgesLRTB.x = float((packed_val >> 6) & 0x03) / 3.0;
	edgesLRTB.y = float((packed_val >> 4) & 0x03) / 3.0;
	edgesLRTB.z = float((packed_val >> 2) & 0x03) / 3.0;
	edgesLRTB.w = float((packed_val >> 0) & 0x03) / 3.0;

	return clamp(edgesLRTB + params.edge_sharpness, 0.0, 1.0);
}

void add_sample(vec4 p_ssil_value, float p_edge_value, inout vec4 r_sum, inout float r_sum_weight) {
	float weight = p_edge_value;

	r_sum += (weight * p_ssil_value);
	r_sum_weight += weight;
}

#ifdef MODE_WIDE
vec4 sample_blurred_wide(ivec2 p_pos, vec2 p_coord) {
	vec4 ssil_value = textureLodOffset(source_ssil, vec2(p_coord), 0.0, ivec2(0, 0));
	vec4 ssil_valueL = textureLodOffset(source_ssil, vec2(p_coord), 0.0, ivec2(-2, 0));
	vec4 ssil_valueT = textureLodOffset(source_ssil, vec2(p_coord), 0.0, ivec2(0, -2));
	vec4 ssil_valueR = textureLodOffset(source_ssil, vec2(p_coord), 0.0, ivec2(2, 0));
	vec4 ssil_valueB = textureLodOffset(source_ssil, vec2(p_coord), 0.0, ivec2(0, 2));

	vec4 edgesLRTB = unpack_edges(imageLoad(source_edges, p_pos).r);
	edgesLRTB.x *= unpack_edges(imageLoad(source_edges, p_pos + ivec2(-2, 0)).r).y;
	edgesLRTB.z *= unpack_edges(imageLoad(source_edges, p_pos + ivec2(0, -2)).r).w;
	edgesLRTB.y *= unpack_edges(imageLoad(source_edges, p_pos + ivec2(2, 0)).r).x;
	edgesLRTB.w *= unpack_edges(imageLoad(source_edges, p_pos + ivec2(0, 2)).r).z;

	float sum_weight = 0.8;
	vec4 sum = ssil_value * sum_weight;

	add_sample(ssil_valueL, edgesLRTB.x, sum, sum_weight);
	add_sample(ssil_valueR, edgesLRTB.y, sum, sum_weight);
	add_sample(ssil_valueT, edgesLRTB.z, sum, sum_weight);
	add_sample(ssil_valueB, edgesLRTB.w, sum, sum_weight);

	vec4 ssil_avg = sum / sum_weight;

	ssil_value = ssil_avg;

	return ssil_value;
}
#endif

#ifdef MODE_SMART
vec4 sample_blurred(ivec2 p_pos, vec2 p_coord) {
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

	vec4 ssil_value = ssil_avg;

	return ssil_value;
}
#endif

void main() {
	// Pixel being shaded
	ivec2 ssC = ivec2(gl_GlobalInvocationID.xy);

#ifdef MODE_NON_SMART

	vec2 half_pixel = params.half_screen_pixel_size * 0.5;

	vec2 uv = (vec2(gl_GlobalInvocationID.xy) + vec2(0.5, 0.5)) * params.half_screen_pixel_size;

	vec4 centre = textureLod(source_ssil, uv, 0.0);

	vec4 value = textureLod(source_ssil, vec2(uv + vec2(-half_pixel.x * 3, -half_pixel.y)), 0.0) * 0.2;
	value += textureLod(source_ssil, vec2(uv + vec2(+half_pixel.x, -half_pixel.y * 3)), 0.0) * 0.2;
	value += textureLod(source_ssil, vec2(uv + vec2(-half_pixel.x, +half_pixel.y * 3)), 0.0) * 0.2;
	value += textureLod(source_ssil, vec2(uv + vec2(+half_pixel.x * 3, +half_pixel.y)), 0.0) * 0.2;

	vec4 sampled = value + centre * 0.2;

#else
#ifdef MODE_SMART
	vec4 sampled = sample_blurred(ssC, (vec2(gl_GlobalInvocationID.xy) + vec2(0.5, 0.5)) * params.half_screen_pixel_size);
#else // MODE_WIDE
	vec4 sampled = sample_blurred_wide(ssC, (vec2(gl_GlobalInvocationID.xy) + vec2(0.5, 0.5)) * params.half_screen_pixel_size);
#endif
#endif // MODE_NON_SMART
	imageStore(dest_image, ssC, sampled);
}
