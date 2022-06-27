#[versions]

lines = "#define MODE_LINES";
triangles = "#define MODE_TRIANGLES";

#[vertex]

#version 450

#VERSION_DEFINES

#include "lm_common_inc.glsl"

layout(push_constant, std430) uniform Params {
	uint base_index;
	uint slice;
	vec2 uv_offset;
	bool debug;
	float blend;
	uint pad[2];
}
params;

layout(location = 0) out vec3 uv_interp;

void main() {
#ifdef MODE_TRIANGLES
	uint triangle_idx = params.base_index + gl_VertexIndex / 3;
	uint triangle_subidx = gl_VertexIndex % 3;

	vec2 uv;
	if (triangle_subidx == 0) {
		uv = vertices.data[triangles.data[triangle_idx].indices.x].uv;
	} else if (triangle_subidx == 1) {
		uv = vertices.data[triangles.data[triangle_idx].indices.y].uv;
	} else {
		uv = vertices.data[triangles.data[triangle_idx].indices.z].uv;
	}

	uv_interp = vec3(uv, float(params.slice));
	gl_Position = vec4((uv + params.uv_offset) * 2.0 - 1.0, 0.0001, 1.0);
#endif

#ifdef MODE_LINES
	uint seam_idx = params.base_index + gl_VertexIndex / 4;
	uint seam_subidx = gl_VertexIndex % 4;

	uint src_idx;
	uint dst_idx;

	if (seam_subidx == 0) {
		src_idx = seams.data[seam_idx].b.x;
		dst_idx = seams.data[seam_idx].a.x;
	} else if (seam_subidx == 1) {
		src_idx = seams.data[seam_idx].b.y;
		dst_idx = seams.data[seam_idx].a.y;
	} else if (seam_subidx == 2) {
		src_idx = seams.data[seam_idx].a.x;
		dst_idx = seams.data[seam_idx].b.x;
	} else if (seam_subidx == 3) {
		src_idx = seams.data[seam_idx].a.y;
		dst_idx = seams.data[seam_idx].b.y;
	}

	vec2 src_uv = vertices.data[src_idx].uv;
	vec2 dst_uv = vertices.data[dst_idx].uv + params.uv_offset;

	uv_interp = vec3(src_uv, float(params.slice));
	gl_Position = vec4(dst_uv * 2.0 - 1.0, 0.0001, 1.0);
#endif
}

#[fragment]

#version 450

#VERSION_DEFINES

#include "lm_common_inc.glsl"

layout(push_constant, std430) uniform Params {
	uint base_index;
	uint slice;
	vec2 uv_offset;
	bool debug;
	float blend;
	uint pad[2];
}
params;

layout(location = 0) in vec3 uv_interp;

layout(location = 0) out vec4 dst_color;

layout(set = 1, binding = 0) uniform texture2DArray src_color_tex;

void main() {
	if (params.debug) {
#ifdef MODE_TRIANGLES
		dst_color = vec4(1, 0, 1, 1);
#else
		dst_color = vec4(1, 1, 0, 1);
#endif
	} else {
		vec4 src_color = textureLod(sampler2DArray(src_color_tex, linear_sampler), uv_interp, 0.0);
		dst_color = vec4(src_color.rgb, params.blend); //mix
	}
}
