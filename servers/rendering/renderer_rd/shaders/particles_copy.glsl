#[compute]

#version 450

VERSION_DEFINES

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct ParticleData {
	mat4 xform;
	vec3 velocity;
	bool is_active;
	vec4 color;
	vec4 custom;
};

layout(set = 0, binding = 1, std430) restrict readonly buffer Particles {
	ParticleData data[];
}
particles;

layout(set = 0, binding = 2, std430) restrict writeonly buffer Transforms {
	vec4 data[];
}
instances;

#ifdef USE_SORT_BUFFER

layout(set = 1, binding = 0, std430) restrict buffer SortBuffer {
	vec2 data[];
}
sort_buffer;

#endif // USE_SORT_BUFFER

layout(push_constant, binding = 0, std430) uniform Params {
	vec3 sort_direction;
	uint total_particles;
}
params;

void main() {
#ifdef MODE_FILL_SORT_BUFFER

	uint particle = gl_GlobalInvocationID.x;
	if (particle >= params.total_particles) {
		return; //discard
	}

	sort_buffer.data[particle].x = dot(params.sort_direction, particles.data[particle].xform[3].xyz);
	sort_buffer.data[particle].y = float(particle);
#endif

#ifdef MODE_FILL_INSTANCES

	uint particle = gl_GlobalInvocationID.x;
	uint write_offset = gl_GlobalInvocationID.x * (3 + 1 + 1); //xform + color + custom

	if (particle >= params.total_particles) {
		return; //discard
	}

#ifdef USE_SORT_BUFFER
	particle = uint(sort_buffer.data[particle].y); //use index from sort buffer
#endif

	mat4 txform;

	if (particles.data[particle].is_active) {
		txform = transpose(particles.data[particle].xform);
	} else {
		txform = mat4(vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0)); //zero scale, becomes invisible
	}

	instances.data[write_offset + 0] = txform[0];
	instances.data[write_offset + 1] = txform[1];
	instances.data[write_offset + 2] = txform[2];
	instances.data[write_offset + 3] = particles.data[particle].color;
	instances.data[write_offset + 4] = particles.data[particle].custom;

#endif
}
