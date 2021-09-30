#[vertex]

#version 450

#VERSION_DEFINES

#include "lm_common_inc.glsl"

layout(location = 0) out vec3 vertex_interp;
layout(location = 1) out vec3 normal_interp;
layout(location = 2) out vec2 uv_interp;
layout(location = 3) out vec3 barycentric;
layout(location = 4) flat out uvec3 vertex_indices;
layout(location = 5) flat out vec3 face_normal;

layout(push_constant, binding = 0, std430) uniform Params {
	vec2 atlas_size;
	vec2 uv_offset;
	vec3 to_cell_size;
	uint base_triangle;
	vec3 to_cell_offset;
	float bias;
	ivec3 grid_size;
	uint pad2;
}
params;

void main() {
	uint triangle_idx = params.base_triangle + gl_VertexIndex / 3;
	uint triangle_subidx = gl_VertexIndex % 3;

	vertex_indices = triangles.data[triangle_idx].indices;

	uint vertex_idx;
	if (triangle_subidx == 0) {
		vertex_idx = vertex_indices.x;
		barycentric = vec3(1, 0, 0);
	} else if (triangle_subidx == 1) {
		vertex_idx = vertex_indices.y;
		barycentric = vec3(0, 1, 0);
	} else {
		vertex_idx = vertex_indices.z;
		barycentric = vec3(0, 0, 1);
	}

	vertex_interp = vertices.data[vertex_idx].position;
	uv_interp = vertices.data[vertex_idx].uv;
	normal_interp = vec3(vertices.data[vertex_idx].normal_xy, vertices.data[vertex_idx].normal_z);

	face_normal = -normalize(cross((vertices.data[vertex_indices.x].position - vertices.data[vertex_indices.y].position), (vertices.data[vertex_indices.x].position - vertices.data[vertex_indices.z].position)));

	gl_Position = vec4((uv_interp + params.uv_offset) * 2.0 - 1.0, 0.0001, 1.0);
}

#[fragment]

#version 450

#VERSION_DEFINES

#include "lm_common_inc.glsl"

layout(push_constant, binding = 0, std430) uniform Params {
	vec2 atlas_size;
	vec2 uv_offset;
	vec3 to_cell_size;
	uint base_triangle;
	vec3 to_cell_offset;
	float bias;
	ivec3 grid_size;
	uint pad2;
}
params;

layout(location = 0) in vec3 vertex_interp;
layout(location = 1) in vec3 normal_interp;
layout(location = 2) in vec2 uv_interp;
layout(location = 3) in vec3 barycentric;
layout(location = 4) in flat uvec3 vertex_indices;
layout(location = 5) in flat vec3 face_normal;

layout(location = 0) out vec4 position;
layout(location = 1) out vec4 normal;
layout(location = 2) out vec4 unocclude;

void main() {
	{
		// unocclusion technique based on:
		// https://ndotl.wordpress.com/2018/08/29/baking-artifact-free-lightmaps/

		/* compute texel size */
		vec3 delta_uv = max(abs(dFdx(vertex_interp)), abs(dFdy(vertex_interp)));
		float texel_size = max(delta_uv.x, max(delta_uv.y, delta_uv.z));
		texel_size *= sqrt(2.0); //expand to unit box edge length (again, worst case)

		unocclude.xyz = face_normal;
		unocclude.w = texel_size;

		//continued on lm_compute.glsl
	}

	position = vec4(vertex_interp, 1.0);
	normal = vec4(normalize(normal_interp), 1.0);
}
