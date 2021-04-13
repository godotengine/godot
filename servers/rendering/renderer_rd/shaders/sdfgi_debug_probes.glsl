#[vertex]

#version 450

#VERSION_DEFINES

#define MAX_CASCADES 8

layout(push_constant, binding = 0, std430) uniform Params {
	mat4 projection;

	uint band_power;
	uint sections_in_band;
	uint band_mask;
	float section_arc;

	vec3 grid_size;
	uint cascade;

	uint pad;
	float y_mult;
	uint probe_debug_index;
	int probe_axis_size;
}
params;

// http://in4k.untergrund.net/html_articles/hugi_27_-_coding_corner_polaris_sphere_tessellation_101.htm

vec3 get_sphere_vertex(uint p_vertex_id) {
	float x_angle = float(p_vertex_id & 1u) + (p_vertex_id >> params.band_power);

	float y_angle =
			float((p_vertex_id & params.band_mask) >> 1) + ((p_vertex_id >> params.band_power) * params.sections_in_band);

	x_angle *= params.section_arc * 0.5f; // remember - 180AA x rot not 360
	y_angle *= -params.section_arc;

	vec3 point = vec3(sin(x_angle) * sin(y_angle), cos(x_angle), sin(x_angle) * cos(y_angle));

	return point;
}

#ifdef MODE_PROBES

layout(location = 0) out vec3 normal_interp;
layout(location = 1) out flat uint probe_index;

#endif

#ifdef MODE_VISIBILITY

layout(location = 0) out float visibility;

#endif

struct CascadeData {
	vec3 offset; //offset of (0,0,0) in world coordinates
	float to_cell; // 1/bounds * grid_size
	ivec3 probe_world_offset;
	uint pad;
};

layout(set = 0, binding = 1, std140) uniform Cascades {
	CascadeData data[MAX_CASCADES];
}
cascades;

layout(set = 0, binding = 4) uniform texture3D occlusion_texture;
layout(set = 0, binding = 3) uniform sampler linear_sampler;

void main() {
#ifdef MODE_PROBES
	probe_index = gl_InstanceIndex;

	normal_interp = get_sphere_vertex(gl_VertexIndex);

	vec3 vertex = normal_interp * 0.2;

	float probe_cell_size = float(params.grid_size / float(params.probe_axis_size - 1)) / cascades.data[params.cascade].to_cell;

	ivec3 probe_cell;
	probe_cell.x = int(probe_index % params.probe_axis_size);
	probe_cell.y = int(probe_index / (params.probe_axis_size * params.probe_axis_size));
	probe_cell.z = int((probe_index / params.probe_axis_size) % params.probe_axis_size);

	vertex += (cascades.data[params.cascade].offset + vec3(probe_cell) * probe_cell_size) / vec3(1.0, params.y_mult, 1.0);

	gl_Position = params.projection * vec4(vertex, 1.0);
#endif

#ifdef MODE_VISIBILITY

	int probe_index = int(params.probe_debug_index);

	vec3 vertex = get_sphere_vertex(gl_VertexIndex) * 0.01;

	float probe_cell_size = float(params.grid_size / float(params.probe_axis_size - 1)) / cascades.data[params.cascade].to_cell;

	ivec3 probe_cell;
	probe_cell.x = int(probe_index % params.probe_axis_size);
	probe_cell.y = int((probe_index % (params.probe_axis_size * params.probe_axis_size)) / params.probe_axis_size);
	probe_cell.z = int(probe_index / (params.probe_axis_size * params.probe_axis_size));

	vertex += (cascades.data[params.cascade].offset + vec3(probe_cell) * probe_cell_size) / vec3(1.0, params.y_mult, 1.0);

	int probe_voxels = int(params.grid_size.x) / int(params.probe_axis_size - 1);
	int occluder_index = int(gl_InstanceIndex);

	int diameter = probe_voxels * 2;
	ivec3 occluder_pos;
	occluder_pos.x = int(occluder_index % diameter);
	occluder_pos.y = int(occluder_index / (diameter * diameter));
	occluder_pos.z = int((occluder_index / diameter) % diameter);

	float cell_size = 1.0 / cascades.data[params.cascade].to_cell;

	ivec3 occluder_offset = occluder_pos - ivec3(diameter / 2);
	vertex += ((vec3(occluder_offset) + vec3(0.5)) * cell_size) / vec3(1.0, params.y_mult, 1.0);

	ivec3 global_cell = probe_cell + cascades.data[params.cascade].probe_world_offset;
	uint occlusion_layer = 0;
	if ((global_cell.x & 1) != 0) {
		occlusion_layer |= 1;
	}
	if ((global_cell.y & 1) != 0) {
		occlusion_layer |= 2;
	}
	if ((global_cell.z & 1) != 0) {
		occlusion_layer |= 4;
	}
	ivec3 tex_pos = probe_cell * probe_voxels + occluder_offset;

	const vec4 layer_axis[4] = vec4[](
			vec4(1, 0, 0, 0),
			vec4(0, 1, 0, 0),
			vec4(0, 0, 1, 0),
			vec4(0, 0, 0, 1));

	tex_pos.z += int(params.cascade) * int(params.grid_size);
	if (occlusion_layer >= 4) {
		tex_pos.x += int(params.grid_size.x);
		occlusion_layer &= 3;
	}

	visibility = dot(texelFetch(sampler3D(occlusion_texture, linear_sampler), tex_pos, 0), layer_axis[occlusion_layer]);

	gl_Position = params.projection * vec4(vertex, 1.0);

#endif
}

#[fragment]

#version 450

#VERSION_DEFINES

layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 2) uniform texture2DArray lightprobe_texture;
layout(set = 0, binding = 3) uniform sampler linear_sampler;

layout(push_constant, binding = 0, std430) uniform Params {
	mat4 projection;

	uint band_power;
	uint sections_in_band;
	uint band_mask;
	float section_arc;

	vec3 grid_size;
	uint cascade;

	uint pad;
	float y_mult;
	uint probe_debug_index;
	int probe_axis_size;
}
params;

#ifdef MODE_PROBES

layout(location = 0) in vec3 normal_interp;
layout(location = 1) in flat uint probe_index;

#endif

#ifdef MODE_VISIBILITY
layout(location = 0) in float visibility;
#endif

vec2 octahedron_wrap(vec2 v) {
	vec2 signVal;
	signVal.x = v.x >= 0.0 ? 1.0 : -1.0;
	signVal.y = v.y >= 0.0 ? 1.0 : -1.0;
	return (1.0 - abs(v.yx)) * signVal;
}

vec2 octahedron_encode(vec3 n) {
	// https://twitter.com/Stubbesaurus/status/937994790553227264
	n /= (abs(n.x) + abs(n.y) + abs(n.z));
	n.xy = n.z >= 0.0 ? n.xy : octahedron_wrap(n.xy);
	n.xy = n.xy * 0.5 + 0.5;
	return n.xy;
}

void main() {
#ifdef MODE_PROBES

	ivec3 tex_pos;
	tex_pos.x = int(probe_index) % params.probe_axis_size; //x
	tex_pos.y = int(probe_index) / (params.probe_axis_size * params.probe_axis_size);
	tex_pos.x += params.probe_axis_size * ((int(probe_index) / params.probe_axis_size) % params.probe_axis_size); //z
	tex_pos.z = int(params.cascade);

	vec3 tex_pos_ofs = vec3(octahedron_encode(normal_interp) * float(OCT_SIZE), 0.0);
	vec3 tex_posf = vec3(vec2(tex_pos.xy * (OCT_SIZE + 2) + ivec2(1)), float(tex_pos.z)) + tex_pos_ofs;

	tex_posf.xy /= vec2(ivec2(params.probe_axis_size * params.probe_axis_size * (OCT_SIZE + 2), params.probe_axis_size * (OCT_SIZE + 2)));

	vec4 indirect_light = textureLod(sampler2DArray(lightprobe_texture, linear_sampler), tex_posf, 0.0);

	frag_color = indirect_light;

#endif

#ifdef MODE_VISIBILITY

	frag_color = vec4(vec3(1, visibility, visibility), 1.0);
#endif
}
