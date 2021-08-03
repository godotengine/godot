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
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_ssao;

layout(rg8, set = 1, binding = 0) uniform restrict writeonly image2D dest_image;

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

void add_sample(float p_ssao_value, float p_edge_value, inout float r_sum, inout float r_sum_weight) {
	float weight = p_edge_value;

	r_sum += (weight * p_ssao_value);
	r_sum_weight += weight;
}

#ifdef MODE_WIDE
vec2 sample_blurred_wide(vec2 p_coord) {
	vec2 vC = textureLodOffset(source_ssao, vec2(p_coord), 0.0, ivec2(0, 0)).xy;
	vec2 vL = textureLodOffset(source_ssao, vec2(p_coord), 0.0, ivec2(-2, 0)).xy;
	vec2 vT = textureLodOffset(source_ssao, vec2(p_coord), 0.0, ivec2(0, -2)).xy;
	vec2 vR = textureLodOffset(source_ssao, vec2(p_coord), 0.0, ivec2(2, 0)).xy;
	vec2 vB = textureLodOffset(source_ssao, vec2(p_coord), 0.0, ivec2(0, 2)).xy;

	float packed_edges = vC.y;
	vec4 edgesLRTB = unpack_edges(packed_edges);
	edgesLRTB.x *= unpack_edges(vL.y).y;
	edgesLRTB.z *= unpack_edges(vT.y).w;
	edgesLRTB.y *= unpack_edges(vR.y).x;
	edgesLRTB.w *= unpack_edges(vB.y).z;

	float ssao_value = vC.x;
	float ssao_valueL = vL.x;
	float ssao_valueT = vT.x;
	float ssao_valueR = vR.x;
	float ssao_valueB = vB.x;

	float sum_weight = 0.8f;
	float sum = ssao_value * sum_weight;

	add_sample(ssao_valueL, edgesLRTB.x, sum, sum_weight);
	add_sample(ssao_valueR, edgesLRTB.y, sum, sum_weight);
	add_sample(ssao_valueT, edgesLRTB.z, sum, sum_weight);
	add_sample(ssao_valueB, edgesLRTB.w, sum, sum_weight);

	float ssao_avg = sum / sum_weight;

	ssao_value = ssao_avg;

	return vec2(ssao_value, packed_edges);
}
#endif

#ifdef MODE_SMART
vec2 sample_blurred(vec3 p_pos, vec2 p_coord) {
	float packed_edges = texelFetch(source_ssao, ivec2(p_pos.xy), 0).y;
	vec4 edgesLRTB = unpack_edges(packed_edges);

	vec4 valuesUL = textureGather(source_ssao, vec2(p_coord - params.half_screen_pixel_size * 0.5));
	vec4 valuesBR = textureGather(source_ssao, vec2(p_coord + params.half_screen_pixel_size * 0.5));

	float ssao_value = valuesUL.y;
	float ssao_valueL = valuesUL.x;
	float ssao_valueT = valuesUL.z;
	float ssao_valueR = valuesBR.z;
	float ssao_valueB = valuesBR.x;

	float sum_weight = 0.5;
	float sum = ssao_value * sum_weight;

	add_sample(ssao_valueL, edgesLRTB.x, sum, sum_weight);
	add_sample(ssao_valueR, edgesLRTB.y, sum, sum_weight);

	add_sample(ssao_valueT, edgesLRTB.z, sum, sum_weight);
	add_sample(ssao_valueB, edgesLRTB.w, sum, sum_weight);

	float ssao_avg = sum / sum_weight;

	ssao_value = ssao_avg;

	return vec2(ssao_value, packed_edges);
}
#endif

void main() {
	// Pixel being shaded
	ivec2 ssC = ivec2(gl_GlobalInvocationID.xy);

#ifdef MODE_NON_SMART

	vec2 half_pixel = params.half_screen_pixel_size * 0.5;

	vec2 uv = (vec2(gl_GlobalInvocationID.xy) + vec2(0.5, 0.5)) * params.half_screen_pixel_size;

	vec2 center = textureLod(source_ssao, vec2(uv), 0.0).xy;

	vec4 vals;
	vals.x = textureLod(source_ssao, vec2(uv + vec2(-half_pixel.x * 3, -half_pixel.y)), 0.0).x;
	vals.y = textureLod(source_ssao, vec2(uv + vec2(+half_pixel.x, -half_pixel.y * 3)), 0.0).x;
	vals.z = textureLod(source_ssao, vec2(uv + vec2(-half_pixel.x, +half_pixel.y * 3)), 0.0).x;
	vals.w = textureLod(source_ssao, vec2(uv + vec2(+half_pixel.x * 3, +half_pixel.y)), 0.0).x;

	vec2 sampled = vec2(dot(vals, vec4(0.2)) + center.x * 0.2, center.y);

#else
#ifdef MODE_SMART
	vec2 sampled = sample_blurred(vec3(gl_GlobalInvocationID), (vec2(gl_GlobalInvocationID.xy) + vec2(0.5, 0.5)) * params.half_screen_pixel_size);
#else // MODE_WIDE
	vec2 sampled = sample_blurred_wide((vec2(gl_GlobalInvocationID.xy) + vec2(0.5, 0.5)) * params.half_screen_pixel_size);
#endif

#endif
	imageStore(dest_image, ivec2(ssC), vec4(sampled, 0.0, 0.0));
}
