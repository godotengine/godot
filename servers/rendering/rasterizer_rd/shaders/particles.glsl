#[compute]

#version 450

VERSION_DEFINES

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#define SAMPLER_NEAREST_CLAMP 0
#define SAMPLER_LINEAR_CLAMP 1
#define SAMPLER_NEAREST_WITH_MIPMAPS_CLAMP 2
#define SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP 3
#define SAMPLER_NEAREST_WITH_MIPMAPS_ANISOTROPIC_CLAMP 4
#define SAMPLER_LINEAR_WITH_MIPMAPS_ANISOTROPIC_CLAMP 5
#define SAMPLER_NEAREST_REPEAT 6
#define SAMPLER_LINEAR_REPEAT 7
#define SAMPLER_NEAREST_WITH_MIPMAPS_REPEAT 8
#define SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT 9
#define SAMPLER_NEAREST_WITH_MIPMAPS_ANISOTROPIC_REPEAT 10
#define SAMPLER_LINEAR_WITH_MIPMAPS_ANISOTROPIC_REPEAT 11

/* SET 0: GLOBAL DATA */

layout(set = 0, binding = 1) uniform sampler material_samplers[12];

layout(set = 0, binding = 2, std430) restrict readonly buffer GlobalVariableData {
	vec4 data[];
}
global_variables;

/* Set 1: FRAME AND PARTICLE DATA */

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

layout(set = 1, binding = 0, std430) restrict buffer FrameHistory {
	FrameParams data[];
}
frame_history;

struct ParticleData {
	mat4 xform;
	vec3 velocity;
	bool is_active;
	vec4 color;
	vec4 custom;
};

layout(set = 1, binding = 1, std430) restrict buffer Particles {
	ParticleData data[];
}
particles;

#define EMISSION_FLAG_HAS_POSITION 1
#define EMISSION_FLAG_HAS_ROTATION_SCALE 2
#define EMISSION_FLAG_HAS_VELOCITY 4
#define EMISSION_FLAG_HAS_COLOR 8
#define EMISSION_FLAG_HAS_CUSTOM 16

struct ParticleEmission {
	mat4 xform;
	vec3 velocity;
	uint flags;
	vec4 color;
	vec4 custom;
};

layout(set = 1, binding = 2, std430) restrict volatile coherent buffer SourceEmission {
	int particle_count;
	uint pad0;
	uint pad1;
	uint pad2;
	ParticleEmission data[];
}
src_particles;

layout(set = 1, binding = 3, std430) restrict volatile coherent buffer DestEmission {
	int particle_count;
	int particle_max;
	uint pad1;
	uint pad2;
	ParticleEmission data[];
}
dst_particles;

/* SET 2: MATERIAL */

#ifdef USE_MATERIAL_UNIFORMS
layout(set = 2, binding = 0, std140) uniform MaterialUniforms{
	/* clang-format off */
MATERIAL_UNIFORMS
	/* clang-format on */
} material;
#endif

layout(push_constant, binding = 0, std430) uniform Params {
	float lifetime;
	bool clear;
	uint total_particles;
	uint trail_size;
	bool use_fractional_delta;
	bool sub_emitter_mode;
	bool can_emit;
	uint pad;
}
params;

uint hash(uint x) {
	x = ((x >> uint(16)) ^ x) * uint(0x45d9f3b);
	x = ((x >> uint(16)) ^ x) * uint(0x45d9f3b);
	x = (x >> uint(16)) ^ x;
	return x;
}

bool emit_particle(mat4 p_xform, vec3 p_velocity, vec4 p_color, vec4 p_custom, uint p_flags) {
	if (!params.can_emit) {
		return false;
	}

	bool valid = false;

	int dst_index = atomicAdd(dst_particles.particle_count, 1);

	if (dst_index >= dst_particles.particle_max) {
		atomicAdd(dst_particles.particle_count, -1);
		return false;
	}
	/*
	valid = true;

	int attempts = 256; // never trust compute
	while(attempts-- > 0) {
	    dst_index = dst_particles.particle_count;
	    if (dst_index == dst_particles.particle_max) {
		return false; //can't emit anymore
	    }

	    if (atomicCompSwap(dst_particles.particle_count, dst_index, dst_index +1 ) != dst_index) {
		continue;
	    }
	    valid=true;
	    break;
	}

	barrier();

	if (!valid) {
		return false; //gave up (attempts exhausted)
	}
*/
	dst_particles.data[dst_index].xform = p_xform;
	dst_particles.data[dst_index].velocity = p_velocity;
	dst_particles.data[dst_index].color = p_color;
	dst_particles.data[dst_index].custom = p_custom;
	dst_particles.data[dst_index].flags = p_flags;

	return true;
}

/* clang-format off */

COMPUTE_SHADER_GLOBALS

/* clang-format on */

void main() {
	uint particle = gl_GlobalInvocationID.x;

	if (particle >= params.total_particles * params.trail_size) {
		return; //discard
	}

	uint index = particle / params.trail_size;
	uint frame = (particle % params.trail_size);

#define FRAME frame_history.data[frame]
#define PARTICLE particles.data[particle]

	bool apply_forces = true;
	bool apply_velocity = true;
	float local_delta = FRAME.delta;

	float mass = 1.0;

	bool restart = false;

	bool restart_position = false;
	bool restart_rotation_scale = false;
	bool restart_velocity = false;
	bool restart_color = false;
	bool restart_custom = false;

	if (params.clear) {
		PARTICLE.color = vec4(1.0);
		PARTICLE.custom = vec4(0.0);
		PARTICLE.velocity = vec3(0.0);
		PARTICLE.is_active = false;
		PARTICLE.xform = mat4(
				vec4(1.0, 0.0, 0.0, 0.0),
				vec4(0.0, 1.0, 0.0, 0.0),
				vec4(0.0, 0.0, 1.0, 0.0),
				vec4(0.0, 0.0, 0.0, 1.0));
	}

	if (params.sub_emitter_mode) {
		if (!PARTICLE.is_active) {
			int src_index = atomicAdd(src_particles.particle_count, -1) - 1;

			if (src_index >= 0) {
				PARTICLE.is_active = true;
				restart = true;

				if (bool(src_particles.data[src_index].flags & EMISSION_FLAG_HAS_POSITION)) {
					PARTICLE.xform[3] = src_particles.data[src_index].xform[3];
				} else {
					PARTICLE.xform[3] = vec4(0, 0, 0, 1);
					restart_position = true;
				}
				if (bool(src_particles.data[src_index].flags & EMISSION_FLAG_HAS_ROTATION_SCALE)) {
					PARTICLE.xform[0] = src_particles.data[src_index].xform[0];
					PARTICLE.xform[1] = src_particles.data[src_index].xform[1];
					PARTICLE.xform[2] = src_particles.data[src_index].xform[2];
				} else {
					PARTICLE.xform[0] = vec4(1, 0, 0, 0);
					PARTICLE.xform[1] = vec4(0, 1, 0, 0);
					PARTICLE.xform[2] = vec4(0, 0, 1, 0);
					restart_rotation_scale = true;
				}
				if (bool(src_particles.data[src_index].flags & EMISSION_FLAG_HAS_VELOCITY)) {
					PARTICLE.velocity = src_particles.data[src_index].velocity;
				} else {
					PARTICLE.velocity = vec3(0);
					restart_velocity = true;
				}
				if (bool(src_particles.data[src_index].flags & EMISSION_FLAG_HAS_COLOR)) {
					PARTICLE.color = src_particles.data[src_index].color;
				} else {
					PARTICLE.color = vec4(1);
					restart_color = true;
				}

				if (bool(src_particles.data[src_index].flags & EMISSION_FLAG_HAS_CUSTOM)) {
					PARTICLE.custom = src_particles.data[src_index].custom;
				} else {
					PARTICLE.custom = vec4(0);
					restart_custom = true;
				}
			}
		}

	} else if (FRAME.emitting) {
		float restart_phase = float(index) / float(params.total_particles);

		if (FRAME.randomness > 0.0) {
			uint seed = FRAME.cycle;
			if (restart_phase >= FRAME.system_phase) {
				seed -= uint(1);
			}
			seed *= uint(params.total_particles);
			seed += uint(index);
			float random = float(hash(seed) % uint(65536)) / 65536.0;
			restart_phase += FRAME.randomness * random * 1.0 / float(params.total_particles);
		}

		restart_phase *= (1.0 - FRAME.explosiveness);

		if (FRAME.system_phase > FRAME.prev_system_phase) {
			// restart_phase >= prev_system_phase is used so particles emit in the first frame they are processed

			if (restart_phase >= FRAME.prev_system_phase && restart_phase < FRAME.system_phase) {
				restart = true;
				if (params.use_fractional_delta) {
					local_delta = (FRAME.system_phase - restart_phase) * params.lifetime;
				}
			}

		} else if (FRAME.delta > 0.0) {
			if (restart_phase >= FRAME.prev_system_phase) {
				restart = true;
				if (params.use_fractional_delta) {
					local_delta = (1.0 - restart_phase + FRAME.system_phase) * params.lifetime;
				}

			} else if (restart_phase < FRAME.system_phase) {
				restart = true;
				if (params.use_fractional_delta) {
					local_delta = (FRAME.system_phase - restart_phase) * params.lifetime;
				}
			}
		}

		uint current_cycle = FRAME.cycle;

		if (FRAME.system_phase < restart_phase) {
			current_cycle -= uint(1);
		}

		uint particle_number = current_cycle * uint(params.total_particles) + particle;

		if (restart) {
			PARTICLE.is_active = FRAME.emitting;
			restart_position = true;
			restart_rotation_scale = true;
			restart_velocity = true;
			restart_color = true;
			restart_custom = true;
		}
	}

	if (PARTICLE.is_active) {
		/* clang-format off */

COMPUTE_SHADER_CODE

		/* clang-format on */
	}

#if !defined(DISABLE_VELOCITY)

	if (PARTICLE.is_active) {
		PARTICLE.xform[3].xyz += PARTICLE.velocity * local_delta;
	}
#endif

#if 0
	if (PARTICLE.is_active) {
		//execute shader




		//!defined(DISABLE_FORCE)

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
#endif
}
