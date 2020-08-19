#[compute]

#version 450

VERSION_DEFINES

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct ParticleData {
	mat4 xform;
	vec3 velocity;
	bool active;
	vec4 color;
	vec4 custom;
	uint frame_of_activation; //frame of activation
	uint pad[3];
};

layout(set = 0, binding = 1, std430) restrict buffer Particles {
	ParticleData data[];
}
particles;

// a frame history is kept for trail deterministic behavior
struct FrameParams {
	bool emitting;
	float system_phase;
	float prev_system_phase;
	uint cycle;

	float explosiveness;
	float randomness;
	float time;
	float delta;

	uint random_seed;
	uint pad[3];

	mat4 emission_transform;
};

layout(set = 0, binding = 1, std430) restrict buffer FrameHistory {
	FrameParams data[];
}
frame_history;

layout(push_constant, binding = 0, std430) uniform Params {
	float lifetime;
	bool clear;
	uint total_particles;
	uint trail_size;
}
params;

void main() {
	uint particle = gl_GlobalInvocationID.x;
	uint index = particle / params.trail_size;
	uint frame = (particle % params.trail_size);

#define FRAME frame_history.data[frame]
#define PARTICLE particles.data[particle]

	bool apply_forces = true;
	bool apply_velocity = true;
	float local_delta = FRAME.delta;

	float mass = 1.0;

	float restart_phase = float(index) / float(params.total_particles);

	if (FRAME.randomness > 0.0) {
		uint seed = FRAME.cycle;
		if (restart_phase >= FRAME.system_phase) {
			seed -= uint(1);
		}
		seed *= uint(total_particles);
		seed += uint(index);
		float random = float(hash(seed) % uint(65536)) / 65536.0;
		restart_phase += FRAME.randomness * random * 1.0 / float(params.total_particles);
	}

	restart_phase *= (1.0 - FRAME.explosiveness);

	bool restart = false;
	bool shader_active = PARTICLE.active;

	if (FRAME.system_phase > FRAME.prev_system_phase) {
		// restart_phase >= prev_system_phase is used so particles emit in the first frame they are processed

		if (restart_phase >= FRAME.prev_system_phase && restart_phase < FRAME.system_phase) {
			restart = true;
#ifdef USE_FRACTIONAL_DELTA
			local_delta = (FRAME.system_phase - restart_phase) * params.lifetime;
#endif
		}

	} else if (FRAME.delta > 0.0) {
		if (restart_phase >= FRAME.prev_system_phase) {
			restart = true;
#ifdef USE_FRACTIONAL_DELTA
			local_delta = (1.0 - restart_phase + FRAME.system_phase) * params.lifetime;
#endif
		} else if (restart_phase < FRAME.system_phase) {
			restart = true;
#ifdef USE_FRACTIONAL_DELTA
			local_delta = (FRAME.system_phase - restart_phase) * params.lifetime;
#endif
		}
	}

	uint current_cycle = PARAMS.cycle;

	if (FRAME.system_phase < restart_phase) {
		current_cycle -= uint(1);
	}

	uint particle_number = current_cycle * uint(params.total_particles) + particle;

	if (restart) {
		shader_active = FRAME.emitting;
	}

	mat4 xform;

	if (frame.clear) {
		PARTICLE.color = vec4(1.0);
		PARTICLE.custom = vec4(0.0);
		PARTICLE.velocity = vec3(0.0);
		PARTICLE.active = false;
		PARTICLE.xform = mat4(
				vec4(1.0, 0.0, 0.0, 0.0),
				vec4(0.0, 1.0, 0.0, 0.0),
				vec4(0.0, 0.0, 1.0, 0.0),
				vec4(0.0, 0.0, 0.0, 1.0));
		PARTICLE.frame_of_activation = 0xFFFFFFFF;
	}

	if (shader_active) {
		//execute shader

		{
			/* clang-format off */

VERTEX_SHADER_CODE

			/* clang-format on */
		}

#if !defined(DISABLE_FORCE)

		if (false) {
			vec3 force = vec3(0.0);
			for (int i = 0; i < attractor_count; i++) {
				vec3 rel_vec = xform[3].xyz - attractors[i].pos;
				float dist = length(rel_vec);
				if (attractors[i].radius < dist)
					continue;
				if (attractors[i].eat_radius > 0.0 && attractors[i].eat_radius > dist) {
					out_velocity_active.a = 0.0;
				}

				rel_vec = normalize(rel_vec);

				float attenuation = pow(dist / attractors[i].radius, attractors[i].attenuation);

				if (attractors[i].dir == vec3(0.0)) {
					//towards center
					force += attractors[i].strength * rel_vec * attenuation * mass;
				} else {
					force += attractors[i].strength * attractors[i].dir * attenuation * mass;
				}
			}

			out_velocity_active.xyz += force * local_delta;
		}
#endif

#if !defined(DISABLE_VELOCITY)

		if (true) {
			xform[3].xyz += out_velocity_active.xyz * local_delta;
		}
#endif
	} else {
		xform = mat4(0.0);
	}

	xform = transpose(xform);

	out_velocity_active.a = mix(0.0, 1.0, shader_active);

	out_xform_1 = xform[0];
	out_xform_2 = xform[1];
	out_xform_3 = xform[2];

#endif //PARTICLES_COPY
}
