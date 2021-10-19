
/* SET 0, static data that does not change between any call */

struct Vertex {
	vec3 position;
	float normal_z;
	vec2 uv;
	vec2 normal_xy;
};

layout(set = 0, binding = 1, std430) restrict readonly buffer Vertices {
	Vertex data[];
}
vertices;

struct Triangle {
	uvec3 indices;
	uint slice;
	vec3 min_bounds;
	uint pad0;
	vec3 max_bounds;
	uint pad1;
};

layout(set = 0, binding = 2, std430) restrict readonly buffer Triangles {
	Triangle data[];
}
triangles;

layout(set = 0, binding = 3, std430) restrict readonly buffer GridIndices {
	uint data[];
}
grid_indices;

#define LIGHT_TYPE_DIRECTIONAL 0
#define LIGHT_TYPE_OMNI 1
#define LIGHT_TYPE_SPOT 2

struct Light {
	vec3 position;
	uint type;

	vec3 direction;
	float energy;

	vec3 color;
	float size;

	float range;
	float attenuation;
	float cos_spot_angle;
	float inv_spot_attenuation;

	bool static_bake;
	uint pad[3];
};

layout(set = 0, binding = 4, std430) restrict readonly buffer Lights {
	Light data[];
}
lights;

struct Seam {
	uvec2 a;
	uvec2 b;
};

layout(set = 0, binding = 5, std430) restrict readonly buffer Seams {
	Seam data[];
}
seams;

layout(set = 0, binding = 6, std430) restrict readonly buffer Probes {
	vec4 data[];
}
probe_positions;

layout(set = 0, binding = 7) uniform utexture3D grid;

layout(set = 0, binding = 8) uniform texture2DArray albedo_tex;
layout(set = 0, binding = 9) uniform texture2DArray emission_tex;

layout(set = 0, binding = 10) uniform sampler linear_sampler;

// Fragment action constants
const uint FA_NONE = 0;
const uint FA_SMOOTHEN_POSITION = 1;
