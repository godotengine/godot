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
// 2025-11-28: hydrogenc: adapt for GTAO
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba8, set = 0, binding = 0) uniform restrict writeonly image2D dest_image;

layout(set = 1, binding = 0) uniform sampler2DArray source_working_term;

layout(set = 2, binding = 0) uniform sampler2DArray source_edges;

layout(push_constant, std430) uniform Params {
	vec2 pixel_size;
	uint size_modifier;
	int pad;
}
params;

vec4 unpack_edges(float p_packed_val) {
	uint packed_val = uint(p_packed_val * 255.5);
	vec4 edgesLRTB;
	edgesLRTB.x = float((packed_val >> 6) & 0x03) / 3.0;
	edgesLRTB.y = float((packed_val >> 4) & 0x03) / 3.0;
	edgesLRTB.z = float((packed_val >> 2) & 0x03) / 3.0;
	edgesLRTB.w = float((packed_val >> 0) & 0x03) / 3.0;

	return clamp(edgesLRTB, 0.0, 1.0);
}

void main() {
	ivec2 ssC = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(ssC, ivec2(1.0 / params.pixel_size)))) { //too large, do nothing
		return;
	}

	float ao;
	uvec2 pix_pos = uvec2(gl_GlobalInvocationID.xy);
	vec2 uv = (gl_GlobalInvocationID.xy + vec2(0.5)) * params.pixel_size;

	// calculate index in the four deinterleaved source array texture
	int mx = int(pix_pos.x % 2);
	int my = int(pix_pos.y % 2);
	int index_center = mx + my * 2; // center index
	int index_horizontal = (1 - mx) + my * 2; // neighboring, horizontal
	int index_vertical = mx + (1 - my) * 2; // neighboring, vertical
	int index_diagonal = (1 - mx) + (1 - my) * 2; // diagonal

	ao = texelFetch(source_working_term, ivec3(pix_pos / uvec2(params.size_modifier), index_center), 0).x;

	vec4 edgesLRTB = unpack_edges(texelFetch(source_edges, ivec3(pix_pos / uvec2(params.size_modifier), index_center), 0).x);

	// convert index shifts to sampling offsets
	float fmx = float(mx);
	float fmy = float(my);

	// in case of an edge, push sampling offsets away from the edge (towards pixel center)
	float fmxe = (edgesLRTB.y - edgesLRTB.x);
	float fmye = (edgesLRTB.w - edgesLRTB.z);

	// calculate final sampling offsets and sample using bilinear filter
	vec2 uv_horizontal = (gl_GlobalInvocationID.xy + vec2(0.5) + vec2(fmx + fmxe - 0.5, 0.5 - fmy)) * params.pixel_size;
	float ao_horizontal = textureLod(source_working_term, vec3(uv_horizontal, index_horizontal), 0.0).x;
	vec2 uv_vertical = (gl_GlobalInvocationID.xy + vec2(0.5) + vec2(0.5 - fmx, fmy - 0.5 + fmye)) * params.pixel_size;
	float ao_vertical = textureLod(source_working_term, vec3(uv_vertical, index_vertical), 0.0).x;
	vec2 uv_diagonal = (gl_GlobalInvocationID.xy + vec2(0.5) + vec2(fmx - 0.5 + fmxe, fmy - 0.5 + fmye)) * params.pixel_size;
	float ao_diagonal = textureLod(source_working_term, vec3(uv_diagonal, index_diagonal), 0.0).x;

	// reduce weight for samples near edge - if the edge is on both sides, weight goes to 0
	vec4 blendWeights;
	blendWeights.x = 1.0;
	blendWeights.y = (edgesLRTB.x + edgesLRTB.y) * 0.5;
	blendWeights.z = (edgesLRTB.z + edgesLRTB.w) * 0.5;
	blendWeights.w = (blendWeights.y + blendWeights.z) * 0.5;

	// calculate weighted average
	float blendWeightsSum = dot(blendWeights, vec4(1.0, 1.0, 1.0, 1.0));
	ao = dot(vec4(ao, ao_horizontal, ao_vertical, ao_diagonal), blendWeights) / blendWeightsSum;

	imageStore(dest_image, ivec2(gl_GlobalInvocationID.xy), vec4(ao));
}
