#[vertex]

#version 450

#if defined(USE_MULTIVIEW) && defined(has_VK_KHR_multiview)
#extension GL_EXT_multiview : enable
#endif

#ifdef USE_MULTIVIEW
#ifdef has_VK_KHR_multiview
#define ViewIndex gl_ViewIndex
#else // has_VK_KHR_multiview
// !BAS! This needs to become an input once we implement our fallback!
#define ViewIndex 0
#endif // has_VK_KHR_multiview
#else // USE_MULTIVIEW
// Set to zero, not supported in non stereo
#define ViewIndex 0
#endif //USE_MULTIVIEW

#VERSION_DEFINES

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
	uint probe_debug_index;
	uint pad;

	ivec3 probe_axis_size;
	uint pad2;
}
params;


// https://in4k.untergrund.net/html_articles/hugi_27_-_coding_corner_polaris_sphere_tessellation_101.htm

vec3 get_sphere_vertex(uint p_vertex_id) {
	float x_angle = float(p_vertex_id & 1u) + (p_vertex_id >> params.band_power);

	float y_angle =
			float((p_vertex_id & params.band_mask) >> 1) + ((p_vertex_id >> params.band_power) * params.sections_in_band);

	x_angle *= params.section_arc * 0.5f; // remember - 180AA x rot not 360
	y_angle *= -params.section_arc;

	vec3 point = vec3(sin(x_angle) * sin(y_angle), cos(x_angle), sin(x_angle) * cos(y_angle));

	return point;
}


layout(location = 0) out vec3 normal_interp;
layout(location = 1) out flat uint probe_index;
layout(location = 2) out vec3 color_interp;


struct CascadeData {
	vec3 offset; //offset of (0,0,0) in world coordinates
	float to_cell; // 1/bounds * grid_size
	ivec3 probe_world_offset;
	uint pad;
	vec4 pad2;
};

layout(set = 0, binding = 1, std140) uniform Cascades {
	CascadeData data[MAX_CASCADES];
}
cascades;

layout(set = 0, binding = 2) uniform texture2DArray lightprobe_texture;

layout(set = 0, binding = 3) uniform sampler linear_sampler;

layout(set = 0, binding = 4, std140) uniform SceneData {
	mat4 projection[MAX_VIEWS];
}
scene_data;


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

ivec3 modi(ivec3 value, ivec3 p_y) {
	return mix( value % p_y, p_y - ((abs(value)-ivec3(1)) % p_y), lessThan(sign(value), ivec3(0)) );
}

#define OCC_DISTANCE_MAX 15.0

void main() {
#ifdef MODE_OCCLUSION
	probe_index = params.probe_debug_index;
#else
	probe_index = gl_InstanceIndex;
#endif
	normal_interp = get_sphere_vertex(gl_VertexIndex);

	vec3 vertex = normal_interp;

	vec3 probe_cell_size = (params.grid_size / vec3(params.probe_axis_size - 1)) / cascades.data[params.cascade].to_cell;

	ivec3 probe_cell;
	probe_cell.x = int(probe_index) % params.probe_axis_size.x;
	probe_cell.y = (int(probe_index) / params.probe_axis_size.x) % params.probe_axis_size.y;
	probe_cell.z = int(probe_index) / (params.probe_axis_size.x * params.probe_axis_size.y);

#ifdef MODE_OCCLUSION

	{

		int oct_size = params.oct_size;
		int oct_margin = 1;

		ivec3 world_cell=cascades.data[params.cascade].probe_world_offset;

		world_cell = modi(world_cell + probe_cell,ivec3(params.probe_axis_size));

		vec3 tex_posf = vec3(ivec3( (world_cell.xy + ivec2(0,world_cell.z * int(params.probe_axis_size.y))) * (oct_size+oct_margin*2) + ivec2(oct_margin) , params.cascade));

		//vec3 tex_pos_ofs = vec3(octahedron_encode(normal_interp) * float(oct_size), 0.0);
		tex_posf += vec3(octahedron_encode(normalize(normal_interp)) * float(oct_size), 0.0);
		tex_posf.xy /= vec2(ivec2(params.probe_axis_size.x * (oct_size + 2*oct_margin), params.probe_axis_size.z * params.probe_axis_size.y * (oct_size + 2*oct_margin)));

		float occ = textureLod(sampler2DArray(lightprobe_texture, linear_sampler), tex_posf, 0.0).r;
		if (occ == 1.0) {
			color_interp = vec3(0.5,1.0,0.5); // hit nothing
		} else {
			color_interp = vec3(1,0.5,0.5); // hit something
		}
		occ *= OCC_DISTANCE_MAX;
		vertex = vertex * occ / 8.0;
		vertex = clamp(vertex,vec3(-1.0),vec3(1.0));
		vertex *= probe_cell_size;


	}
#else
	vertex *= 0.2;
#endif



	vertex += (cascades.data[params.cascade].offset + vec3(probe_cell) * probe_cell_size) / vec3(1.0, params.y_mult, 1.0);

	gl_Position = scene_data.projection[ViewIndex] * vec4(vertex, 1.0);

}

#[fragment]

#version 450

#if defined(USE_MULTIVIEW) && defined(has_VK_KHR_multiview)
#extension GL_EXT_multiview : enable
#endif

#ifdef USE_MULTIVIEW
#ifdef has_VK_KHR_multiview
#define ViewIndex gl_ViewIndex
#else // has_VK_KHR_multiview
// !BAS! This needs to become an input once we implement our fallback!
#define ViewIndex 0
#endif // has_VK_KHR_multiview
#else // USE_MULTIVIEW
// Set to zero, not supported in non stereo
#define ViewIndex 0
#endif //USE_MULTIVIEW

#VERSION_DEFINES

#define MAX_VIEWS 2
#define MAX_CASCADES 8

layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 2) uniform texture2DArray lightprobe_texture;
layout(set = 0, binding = 3) uniform sampler linear_sampler;

layout(push_constant, std430) uniform Params {
	uint band_power;
	uint sections_in_band;
	uint band_mask;
	float section_arc;

	vec3 grid_size;
	uint cascade;

	int oct_size;
	float y_mult;
	uint probe_debug_index;
	uint pad;

	ivec3 probe_axis_size;
	uint pad2;
}
params;

layout(location = 0) in vec3 normal_interp;
layout(location = 1) in flat uint probe_index;
layout(location = 2) in vec3 color_interp;

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

ivec3 modi(ivec3 value, ivec3 p_y) {
	return mix( value % p_y, p_y - ((abs(value)-ivec3(1)) % p_y), lessThan(sign(value), ivec3(0)) );
}

struct CascadeData {
	vec3 offset; //offset of (0,0,0) in world coordinates
	float to_cell; // 1/bounds * grid_size
	ivec3 probe_world_offset;
	uint pad;
	vec4 pad2;
};

layout(set = 0, binding = 1, std140) uniform Cascades {
	CascadeData data[MAX_CASCADES];
}
cascades;

void main() {

#ifdef MODE_OCCLUSION
	frag_color = vec4(color_interp,0.5);
#else
	int oct_size = params.oct_size;
	int oct_margin = 1;

	ivec3 probe_cell;
	probe_cell.x = int(probe_index) % params.probe_axis_size.x;
	probe_cell.y = (int(probe_index) / params.probe_axis_size.x) % params.probe_axis_size.y;
	probe_cell.z = int(probe_index) / (params.probe_axis_size.x * params.probe_axis_size.y);
	probe_cell+=cascades.data[params.cascade].probe_world_offset;

	probe_cell = modi(probe_cell ,ivec3(params.probe_axis_size));

	vec3 tex_posf = vec3(ivec3( (probe_cell.xy + ivec2(0,probe_cell.z * int(params.probe_axis_size.y))) * (oct_size+oct_margin*2) + ivec2(oct_margin) , params.cascade));

	//vec3 tex_pos_ofs = vec3(octahedron_encode(normal_interp) * float(oct_size), 0.0);
	tex_posf += vec3(octahedron_encode(normalize(normal_interp)) * float(oct_size), 0.0);
	tex_posf.xy /= vec2(ivec2(params.probe_axis_size.x * (oct_size + 2*oct_margin), params.probe_axis_size.z * params.probe_axis_size.y * (oct_size + 2*oct_margin)));

	vec4 indirect_light = textureLod(sampler2DArray(lightprobe_texture, linear_sampler), tex_posf, 0.0);

	frag_color = indirect_light;
#endif
}
