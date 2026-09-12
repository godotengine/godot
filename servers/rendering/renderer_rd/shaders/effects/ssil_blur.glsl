///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016, Intel Corporation
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of
// the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// File changes (yyyy-mm-dd)
// 2016-09-07: filip.strugar@intel.com: first commit
// 2020-12-05: clayjohn: convert to Vulkan and Godot
// 2021-05-27: clayjohn: convert SSAO to SSIL
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[compute]

#version 450

#VERSION_DEFINES

// pre-calculated weights and offsets to speed up rendering

const int SAMPLE_COUNT = 8;

const float OFFSETS[8] = float[8](
    -6.4550869992703435,
    -4.468862236297167,
    -2.4826862657413393,
    -0.49653490850373416,
    1.4896094314876247,
    3.475769408144678,
    5.461967313484028,
    7
);

const float WEIGHTS[8] = float[8](
    0.09383640732003992,
    0.12701373852860326,
    0.153999313783728,
    0.16725375352955418,
    0.1627132872924349,
    0.14179514673861507,
    0.1106846237774379,
    0.042703729029586635
);

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_ssil;

layout(rgba16, set = 1, binding = 0) uniform restrict writeonly image2D dest_image;

layout(r8, set = 2, binding = 0) uniform restrict readonly image2D source_edges;

layout(push_constant, std430) uniform Params {
    float edge_sharpness;
    int blur_dir;
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

// vec4 bilateral_blur(ivec2 p_pos, vec2 p_uv) {
//     vec2 blur_offset = params.blur_dir == 0 ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
//     float blur_scale = 1.0;

// 	float weight_total = 0.0;
// 	vec4 result = vec4(0.0);

// 	for (int i = 0; i < SAMPLE_COUNT; ++i) {
// 		vec2 sample_uv = p_uv + ((blur_offset * OFFSETS[i]) * params.half_screen_pixel_size);
//         ivec2 sample_uvi = p_pos + ivec2(blur_offset * OFFSETS[i]);

// 		float packed_edges = imageLoad(source_edges, sample_uvi).r;
//         vec2 edge_weights = unpack_edges(packed_edges);
//         float edge_weight = (OFFSETS[i] > 0.0) ? edge_weights.y : edge_weights.x;

// 		vec4 sample_color = textureLod(source_ssil, sample_uv, 0.0);
// 		float weight = ((WEIGHTS[i]) * edge_weight);
// 		weight_total += weight;
// 		result += weight * sample_color;
// 	}

// 	return weight_total > 0.0 ? result / weight_total : textureLod(source_ssil, p_uv, 0.0);
// }

#ifdef SSIL_BLUR_ACCURATE
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

#ifdef SSIL_BLUR_FAST
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
	ivec2 ssC = ivec2(gl_GlobalInvocationID.xy);

	//vec2 uv = (vec2(ssC) + 0.5) * params.half_screen_pixel_size;

	//vec4 blurred_ssilvb = bilateral_blur(ssC, uv);
#ifdef SSIL_BLUR_ACCURATE
    vec4 sampled = sample_blurred_wide(ssC, (vec2(gl_GlobalInvocationID.xy) + vec2(0.5, 0.5)) * params.half_screen_pixel_size);
#else
	vec4 sampled = sample_blurred(ssC, (vec2(gl_GlobalInvocationID.xy) + vec2(0.5, 0.5)) * params.half_screen_pixel_size);
#endif

	imageStore(dest_image, ssC, sampled);
}