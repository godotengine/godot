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
	uint pad[3];
}
params;

uint hash(uint x) {
	x = ((x >> uint(16)) ^ x) * uint(0x45d9f3b);
	x = ((x >> uint(16)) ^ x) * uint(0x45d9f3b);
	x = (x >> uint(16)) ^ x;
	return x;
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

	bool restart = false;

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
	}

#ifdef ENABLE_KEEP_DATA
	if (params.clear) {
#else
	if (params.clear || restart) {
#endif
		PARTICLE.color = vec4(1.0);
		PARTICLE.custom = vec4(0.0);
		PARTICLE.velocity = vec3(0.0);
		if (!restart) {
			PARTICLE.is_active = false;
		}
		PARTICLE.xform = mat4(
				vec4(1.0, 0.0, 0.0, 0.0),
				vec4(0.0, 1.0, 0.0, 0.0),
				vec4(0.0, 0.0, 1.0, 0.0),
				vec4(0.0, 0.0, 0.0, 1.0));
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
