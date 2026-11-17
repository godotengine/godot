#[vertex]

#version 450

#VERSION_DEFINES

#if defined(USE_MULTIVIEW) && defined(has_VK_KHR_multiview)
#extension GL_EXT_multiview : enable
#define ViewIndex gl_ViewIndex
#else
#define ViewIndex 0
#endif

#define MAX_CASCADES 8
#define MAX_VIEWS 2

layout(push_constant, std430) uniform Params {
	uint band_power;
	uint sections_in_band;
	uint band_mask;
	float section_arc;

	vec3 grid_size;
	uint cascade;

	int oct_size;
	float y_mult;
	uvec2 pad;

	ivec3 probe_axis_size;
	uint pad2;
}
params;

vec3 get_sphere_vertex(uint vertex_id) {
	float x_angle = float(vertex_id & 1u) + float(vertex_id >> params.band_power);
	float y_angle = float((vertex_id & params.band_mask) >> 1u) + float((vertex_id >> params.band_power) * params.sections_in_band);
	x_angle *= params.section_arc * 0.5;
	y_angle *= -params.section_arc;
	return vec3(sin(x_angle) * sin(y_angle), cos(x_angle), sin(x_angle) * cos(y_angle));
}

layout(location = 0) out vec3 normal_interp;
layout(location = 1) out flat uint probe_index;

struct CascadeData {
	vec3 offset;
	float to_cell;
	ivec3 region_world_offset;
	uint pad;
	vec4 pad2;
};

layout(set = 0, binding = 1, std140) uniform Cascades {
	CascadeData data[MAX_CASCADES];
}
cascades;

layout(set = 0, binding = 4, std140) uniform SceneData {
	mat4 projection[MAX_VIEWS];
}
scene_data;

void main() {
	probe_index = gl_InstanceIndex;
	ivec3 probe_cell;
	probe_cell.x = int(probe_index) % params.probe_axis_size.x;
	probe_cell.y = (int(probe_index) / params.probe_axis_size.x) % params.probe_axis_size.y;
	probe_cell.z = int(probe_index) / (params.probe_axis_size.x * params.probe_axis_size.y);

	vec3 probe_cell_size = params.grid_size / vec3(params.probe_axis_size - 1) / cascades.data[params.cascade].to_cell;
	normal_interp = get_sphere_vertex(gl_VertexIndex);
	vec3 vertex = normal_interp * 0.2;
	vertex += (cascades.data[params.cascade].offset + vec3(probe_cell) * probe_cell_size) / vec3(1.0, params.y_mult, 1.0);
	gl_Position = scene_data.projection[ViewIndex] * vec4(vertex, 1.0);
}

#[fragment]

#version 450

#VERSION_DEFINES

#include "../oct_inc.glsl"

#define MAX_CASCADES 8

layout(location = 0) out vec4 frag_color;
layout(location = 0) in vec3 normal_interp;
layout(location = 1) in flat uint probe_index;

layout(set = 0, binding = 3) uniform sampler linear_sampler;
layout(set = 0, binding = 5) uniform texture2DArray lightprobe_texture;

layout(push_constant, std430) uniform Params {
	uint band_power;
	uint sections_in_band;
	uint band_mask;
	float section_arc;

	vec3 grid_size;
	uint cascade;

	int oct_size;
	float y_mult;
	uvec2 pad;

	ivec3 probe_axis_size;
	uint pad2;
}
params;

struct CascadeData {
	vec3 offset;
	float to_cell;
	ivec3 region_world_offset;
	uint pad;
	vec4 pad2;
};

layout(set = 0, binding = 1, std140) uniform Cascades {
	CascadeData data[MAX_CASCADES];
}
cascades;

void main() {
	ivec3 probe_cell;
	probe_cell.x = int(probe_index) % params.probe_axis_size.x;
	probe_cell.y = (int(probe_index) / params.probe_axis_size.x) % params.probe_axis_size.y;
	probe_cell.z = int(probe_index) / (params.probe_axis_size.x * params.probe_axis_size.y);
	probe_cell += cascades.data[params.cascade].region_world_offset;
	// GLSL's remainder is undefined for negative operands.
	ivec3 wrapped = abs(probe_cell) % params.probe_axis_size;
	probe_cell = mix(wrapped, (params.probe_axis_size - wrapped) % params.probe_axis_size, lessThan(probe_cell, ivec3(0)));

	ivec2 atlas_size = ivec2(params.probe_axis_size.x, params.probe_axis_size.y * params.probe_axis_size.z) * (params.oct_size + 2);
	vec2 texel = vec2((probe_cell.xy + ivec2(0, probe_cell.z * params.probe_axis_size.y)) * (params.oct_size + 2) + 1);
	texel += vec3_to_oct(normalize(normal_interp)) * float(params.oct_size);
	frag_color = vec4(textureLod(sampler2DArray(lightprobe_texture, linear_sampler), vec3(texel / vec2(atlas_size), float(params.cascade)), 0.0).rgb, 1.0);
}
