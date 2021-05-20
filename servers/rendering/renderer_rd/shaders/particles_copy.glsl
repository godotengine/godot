#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#define PARTICLE_FLAG_ACTIVE uint(1)
#define PARTICLE_FLAG_STARTED uint(2)
#define PARTICLE_FLAG_TRAILED uint(4)

struct ParticleData {
	mat4 xform;
	vec3 velocity;
	uint flags;
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

layout(set = 2, binding = 0, std430) restrict readonly buffer TrailBindPoses {
	mat4 data[];
}
trail_bind_poses;

layout(push_constant, binding = 0, std430) uniform Params {
	vec3 sort_direction;
	uint total_particles;

	uint trail_size;
	uint trail_total;
	float frame_delta;
	float frame_remainder;

	vec3 align_up;
	uint align_mode;

	bool order_by_lifetime;
	uint lifetime_split;
	bool lifetime_reverse;
	uint pad;
}
params;

#define TRANSFORM_ALIGN_DISABLED 0
#define TRANSFORM_ALIGN_Z_BILLBOARD 1
#define TRANSFORM_ALIGN_Y_TO_VELOCITY 2
#define TRANSFORM_ALIGN_Z_BILLBOARD_Y_TO_VELOCITY 3

void main() {
#ifdef MODE_FILL_SORT_BUFFER

	uint particle = gl_GlobalInvocationID.x;
	if (particle >= params.total_particles) {
		return; //discard
	}

	uint src_particle = particle;
	if (params.trail_size > 1) {
		src_particle = src_particle * params.trail_size + params.trail_size / 2; //use trail center for sorting
	}
	sort_buffer.data[particle].x = dot(params.sort_direction, particles.data[src_particle].xform[3].xyz);
	sort_buffer.data[particle].y = float(particle);
#endif

#ifdef MODE_FILL_INSTANCES

	uint particle = gl_GlobalInvocationID.x;

	if (particle >= params.total_particles) {
		return; //discard
	}

#ifdef USE_SORT_BUFFER

	if (params.trail_size > 1) {
		particle = uint(sort_buffer.data[particle / params.trail_size].y) + (particle % params.trail_size);
	} else {
		particle = uint(sort_buffer.data[particle].y); //use index from sort buffer
	}
#else
	if (params.order_by_lifetime) {
		if (params.trail_size > 1) {
			uint limit = (params.total_particles / params.trail_size) - params.lifetime_split;

			uint base_index = particle / params.trail_size;
			uint base_offset = particle % params.trail_size;

			if (params.lifetime_reverse) {
				base_index = (params.total_particles / params.trail_size) - base_index - 1;
			}

			if (base_index < limit) {
				base_index = params.lifetime_split + base_index;
			} else {
				base_index -= limit;
			}

			particle = base_index * params.trail_size + base_offset;

		} else {
			uint limit = params.total_particles - params.lifetime_split;

			if (params.lifetime_reverse) {
				particle = params.total_particles - particle - 1;
			}

			if (particle < limit) {
				particle = params.lifetime_split + particle;
			} else {
				particle -= limit;
			}
		}
	}
#endif // USE_SORT_BUFFER

	mat4 txform;

	if (bool(particles.data[particle].flags & PARTICLE_FLAG_ACTIVE) || bool(particles.data[particle].flags & PARTICLE_FLAG_TRAILED)) {
		txform = particles.data[particle].xform;
		if (params.trail_size > 1) {
			// since the steps dont fit precisely in the history frames, must do a tiny bit of
			// interpolation to get them close to their intended location.
			uint part_ofs = particle % params.trail_size;
			float natural_ofs = fract((float(part_ofs) / float(params.trail_size)) * float(params.trail_total)) * params.frame_delta;

			txform[3].xyz -= particles.data[particle].velocity * natural_ofs;
		}

		switch (params.align_mode) {
			case TRANSFORM_ALIGN_DISABLED: {
			} break; //nothing
			case TRANSFORM_ALIGN_Z_BILLBOARD: {
				mat3 local = mat3(normalize(cross(params.align_up, params.sort_direction)), params.align_up, params.sort_direction);
				local = local * mat3(txform);
				txform[0].xyz = local[0];
				txform[1].xyz = local[1];
				txform[2].xyz = local[2];

			} break;
			case TRANSFORM_ALIGN_Y_TO_VELOCITY: {
				vec3 v = particles.data[particle].velocity;
				float s = (length(txform[0]) + length(txform[1]) + length(txform[2])) / 3.0;
				if (length(v) > 0.0) {
					txform[1].xyz = normalize(v);
				} else {
					txform[1].xyz = normalize(txform[1].xyz);
				}

				txform[0].xyz = normalize(cross(txform[1].xyz, txform[2].xyz));
				txform[2].xyz = vec3(0.0, 0.0, 1.0) * s;
				txform[0].xyz *= s;
				txform[1].xyz *= s;
			} break;
			case TRANSFORM_ALIGN_Z_BILLBOARD_Y_TO_VELOCITY: {
				vec3 v = particles.data[particle].velocity;
				vec3 sv = v - params.sort_direction * dot(params.sort_direction, v); //screen velocity
				float s = (length(txform[0]) + length(txform[1]) + length(txform[2])) / 3.0;

				if (length(sv) == 0) {
					sv = params.align_up;
				}

				sv = normalize(sv);

				txform[0].xyz = normalize(cross(sv, params.sort_direction)) * s;
				txform[1].xyz = sv * s;
				txform[2].xyz = params.sort_direction * s;

			} break;
		}

		txform[3].xyz += particles.data[particle].velocity * params.frame_remainder;

		if (params.trail_size > 1) {
			uint part_ofs = particle % params.trail_size;
			txform = txform * trail_bind_poses.data[part_ofs];
		}

		txform = transpose(txform);
	} else {
		txform = mat4(vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0)); //zero scale, becomes invisible
	}

#ifdef MODE_2D

	uint write_offset = gl_GlobalInvocationID.x * (2 + 1 + 1); //xform + color + custom

	instances.data[write_offset + 0] = txform[0];
	instances.data[write_offset + 1] = txform[1];
	instances.data[write_offset + 2] = particles.data[particle].color;
	instances.data[write_offset + 3] = particles.data[particle].custom;

#else

	uint write_offset = gl_GlobalInvocationID.x * (3 + 1 + 1); //xform + color + custom

	instances.data[write_offset + 0] = txform[0];
	instances.data[write_offset + 1] = txform[1];
	instances.data[write_offset + 2] = txform[2];
	instances.data[write_offset + 3] = particles.data[particle].color;
	instances.data[write_offset + 4] = particles.data[particle].custom;
#endif //MODE_2D

#endif
}
