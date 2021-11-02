#[compute]

#version 450

#VERSION_DEFINES

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

#define SDF_MAX_LENGTH 16384.0

/* SET 0: GLOBAL DATA */

layout(set = 0, binding = 1) uniform sampler material_samplers[12];

layout(set = 0, binding = 2, std430) restrict readonly buffer GlobalVariableData {
	vec4 data[];
}
global_variables;

/* Set 1: FRAME AND PARTICLE DATA */

// a frame history is kept for trail deterministic behavior

#define MAX_ATTRACTORS 32

#define ATTRACTOR_TYPE_SPHERE 0
#define ATTRACTOR_TYPE_BOX 1
#define ATTRACTOR_TYPE_VECTOR_FIELD 2

struct Attractor {
	mat4 transform;
	vec3 extents; //exents or radius
	uint type;
	uint texture_index; //texture index for vector field
	float strength;
	float attenuation;
	float directionality;
};

#define MAX_COLLIDERS 32

#define COLLIDER_TYPE_SPHERE 0
#define COLLIDER_TYPE_BOX 1
#define COLLIDER_TYPE_SDF 2
#define COLLIDER_TYPE_HEIGHT_FIELD 3
#define COLLIDER_TYPE_2D_SDF 4

struct Collider {
	mat4 transform;
	vec3 extents; //exents or radius
	uint type;

	uint texture_index; //texture index for vector field
	float scale;
	uint pad[2];
};

struct FrameParams {
	bool emitting;
	float system_phase;
	float prev_system_phase;
	uint cycle;

	float explosiveness;
	float randomness;
	float time;
	float delta;

	uint frame;
	uint pad0;
	uint pad1;
	uint pad2;

	uint random_seed;
	uint attractor_count;
	uint collider_count;
	float particle_size;

	mat4 emission_transform;

	Attractor attractors[MAX_ATTRACTORS];
	Collider colliders[MAX_COLLIDERS];
};

layout(set = 1, binding = 0, std430) restrict buffer FrameHistory {
	FrameParams data[];
}
frame_history;

#define PARTICLE_FLAG_ACTIVE uint(1)
#define PARTICLE_FLAG_STARTED uint(2)
#define PARTICLE_FLAG_TRAILED uint(4)
#define PARTICLE_FRAME_MASK uint(0xFFFF)
#define PARTICLE_FRAME_SHIFT uint(16)

struct ParticleData {
	mat4 xform;
	vec3 velocity;
	uint flags;
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

layout(set = 1, binding = 2, std430) restrict buffer SourceEmission {
	int particle_count;
	uint pad0;
	uint pad1;
	uint pad2;
	ParticleEmission data[];
}
src_particles;

layout(set = 1, binding = 3, std430) restrict buffer DestEmission {
	int particle_count;
	int particle_max;
	uint pad1;
	uint pad2;
	ParticleEmission data[];
}
dst_particles;

/* SET 2: COLLIDER/ATTRACTOR TEXTURES */

#define MAX_3D_TEXTURES 7

layout(set = 2, binding = 0) uniform texture3D sdf_vec_textures[MAX_3D_TEXTURES];
layout(set = 2, binding = 1) uniform texture2D height_field_texture;

/* SET 3: MATERIAL */

#ifdef MATERIAL_UNIFORMS_USED
layout(set = 3, binding = 0, std140) uniform MaterialUniforms{

#MATERIAL_UNIFORMS

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
	bool trail_pass;
}
params;

uint hash(uint x) {
	x = ((x >> uint(16)) ^ x) * uint(0x45d9f3b);
	x = ((x >> uint(16)) ^ x) * uint(0x45d9f3b);
	x = (x >> uint(16)) ^ x;
	return x;
}

bool emit_subparticle(mat4 p_xform, vec3 p_velocity, vec4 p_color, vec4 p_custom, uint p_flags) {
	if (!params.can_emit) {
		return false;
	}

	bool valid = false;

	int dst_index = atomicAdd(dst_particles.particle_count, 1);

	if (dst_index >= dst_particles.particle_max) {
		atomicAdd(dst_particles.particle_count, -1);
		return false;
	}

	dst_particles.data[dst_index].xform = p_xform;
	dst_particles.data[dst_index].velocity = p_velocity;
	dst_particles.data[dst_index].color = p_color;
	dst_particles.data[dst_index].custom = p_custom;
	dst_particles.data[dst_index].flags = p_flags;

	return true;
}

#GLOBALS

void main() {
	uint particle = gl_GlobalInvocationID.x;

	if (params.trail_size > 1) {
		if (params.trail_pass) {
			particle += (particle / (params.trail_size - 1)) + 1;
		} else {
			particle *= params.trail_size;
		}
	}

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
		PARTICLE.flags = 0;
		PARTICLE.xform = mat4(
				vec4(1.0, 0.0, 0.0, 0.0),
				vec4(0.0, 1.0, 0.0, 0.0),
				vec4(0.0, 0.0, 1.0, 0.0),
				vec4(0.0, 0.0, 0.0, 1.0));
	}

	//clear started flag if set

	if (params.trail_pass) {
		//trail started
		uint src_idx = index * params.trail_size;
		if (bool(particles.data[src_idx].flags & PARTICLE_FLAG_STARTED)) {
			//save start conditions for trails
			PARTICLE.color = particles.data[src_idx].color;
			PARTICLE.custom = particles.data[src_idx].custom;
			PARTICLE.velocity = particles.data[src_idx].velocity;
			PARTICLE.flags = PARTICLE_FLAG_TRAILED | ((frame_history.data[0].frame & PARTICLE_FRAME_MASK) << PARTICLE_FRAME_SHIFT); //mark it as trailed, save in which frame it will start
			PARTICLE.xform = particles.data[src_idx].xform;
		}

		if (bool(PARTICLE.flags & PARTICLE_FLAG_TRAILED) && ((PARTICLE.flags >> PARTICLE_FRAME_SHIFT) == (FRAME.frame & PARTICLE_FRAME_MASK))) { //check this is trailed and see if it should start now
			// we just assume that this is the first frame of the particle, the rest is deterministic
			PARTICLE.flags = PARTICLE_FLAG_ACTIVE | (particles.data[src_idx].flags & (PARTICLE_FRAME_MASK << PARTICLE_FRAME_SHIFT));
			return; //- this appears like it should be correct, but it seems not to be.. wonder why.
		}
	} else {
		PARTICLE.flags &= ~PARTICLE_FLAG_STARTED;
	}

	bool collided = false;
	vec3 collision_normal = vec3(0.0);
	float collision_depth = 0.0;

	vec3 attractor_force = vec3(0.0);

#if !defined(DISABLE_VELOCITY)

	if (bool(PARTICLE.flags & PARTICLE_FLAG_ACTIVE)) {
		PARTICLE.xform[3].xyz += PARTICLE.velocity * local_delta;
	}
#endif

	if (!params.trail_pass && params.sub_emitter_mode) {
		if (!bool(PARTICLE.flags & PARTICLE_FLAG_ACTIVE)) {
			int src_index = atomicAdd(src_particles.particle_count, -1) - 1;

			if (src_index >= 0) {
				PARTICLE.flags = (PARTICLE_FLAG_ACTIVE | PARTICLE_FLAG_STARTED | (FRAME.cycle << PARTICLE_FRAME_SHIFT));
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

		if (params.trail_pass) {
			restart = false;
		}

		if (restart) {
			PARTICLE.flags = FRAME.emitting ? (PARTICLE_FLAG_ACTIVE | PARTICLE_FLAG_STARTED | (FRAME.cycle << PARTICLE_FRAME_SHIFT)) : 0;
			restart_position = true;
			restart_rotation_scale = true;
			restart_velocity = true;
			restart_color = true;
			restart_custom = true;
		}
	}

	bool particle_active = bool(PARTICLE.flags & PARTICLE_FLAG_ACTIVE);

	uint particle_number = (PARTICLE.flags >> PARTICLE_FRAME_SHIFT) * uint(params.total_particles) + index;

	if (restart && particle_active) {
#CODE : START
	}

	if (particle_active) {
		for (uint i = 0; i < FRAME.attractor_count; i++) {
			vec3 dir;
			float amount;
			vec3 rel_vec = PARTICLE.xform[3].xyz - FRAME.attractors[i].transform[3].xyz;
			vec3 local_pos = rel_vec * mat3(FRAME.attractors[i].transform);

			switch (FRAME.attractors[i].type) {
				case ATTRACTOR_TYPE_SPHERE: {
					dir = normalize(rel_vec);
					float d = length(local_pos) / FRAME.attractors[i].extents.x;
					if (d > 1.0) {
						continue;
					}
					amount = max(0.0, 1.0 - d);
				} break;
				case ATTRACTOR_TYPE_BOX: {
					dir = normalize(rel_vec);

					vec3 abs_pos = abs(local_pos / FRAME.attractors[i].extents);
					float d = max(abs_pos.x, max(abs_pos.y, abs_pos.z));
					if (d > 1.0) {
						continue;
					}
					amount = max(0.0, 1.0 - d);

				} break;
				case ATTRACTOR_TYPE_VECTOR_FIELD: {
					vec3 uvw_pos = (local_pos / FRAME.attractors[i].extents) * 2.0 - 1.0;
					if (any(lessThan(uvw_pos, vec3(0.0))) || any(greaterThan(uvw_pos, vec3(1.0)))) {
						continue;
					}
					vec3 s = texture(sampler3D(sdf_vec_textures[FRAME.attractors[i].texture_index], material_samplers[SAMPLER_LINEAR_CLAMP]), uvw_pos).xyz;
					dir = mat3(FRAME.attractors[i].transform) * normalize(s); //revert direction
					amount = length(s);

				} break;
			}
			amount = pow(amount, FRAME.attractors[i].attenuation);
			dir = normalize(mix(dir, FRAME.attractors[i].transform[2].xyz, FRAME.attractors[i].directionality));
			attractor_force -= amount * dir * FRAME.attractors[i].strength;
		}

		float particle_size = FRAME.particle_size;

#ifdef USE_COLLISON_SCALE

		particle_size *= dot(vec3(length(PARTICLE.xform[0].xyz), length(PARTICLE.xform[1].xyz), length(PARTICLE.xform[2].xyz)), vec3(0.33333333333));

#endif

		if (FRAME.collider_count == 1 && FRAME.colliders[0].type == COLLIDER_TYPE_2D_SDF) {
			//2D collision

			vec2 pos = PARTICLE.xform[3].xy;
			vec4 to_sdf_x = FRAME.colliders[0].transform[0];
			vec4 to_sdf_y = FRAME.colliders[0].transform[1];
			vec2 sdf_pos = vec2(dot(vec4(pos, 0, 1), to_sdf_x), dot(vec4(pos, 0, 1), to_sdf_y));

			vec4 sdf_to_screen = vec4(FRAME.colliders[0].extents, FRAME.colliders[0].scale);

			vec2 uv_pos = sdf_pos * sdf_to_screen.xy + sdf_to_screen.zw;

			if (all(greaterThan(uv_pos, vec2(0.0))) && all(lessThan(uv_pos, vec2(1.0)))) {
				vec2 pos2 = pos + vec2(0, particle_size);
				vec2 sdf_pos2 = vec2(dot(vec4(pos2, 0, 1), to_sdf_x), dot(vec4(pos2, 0, 1), to_sdf_y));
				float sdf_particle_size = distance(sdf_pos, sdf_pos2);

				float d = texture(sampler2D(height_field_texture, material_samplers[SAMPLER_LINEAR_CLAMP]), uv_pos).r * SDF_MAX_LENGTH;

				d -= sdf_particle_size;

				if (d < 0.0) {
					const float EPSILON = 0.001;
					vec2 n = normalize(vec2(
							texture(sampler2D(height_field_texture, material_samplers[SAMPLER_LINEAR_CLAMP]), uv_pos + vec2(EPSILON, 0.0)).r - texture(sampler2D(height_field_texture, material_samplers[SAMPLER_LINEAR_CLAMP]), uv_pos - vec2(EPSILON, 0.0)).r,
							texture(sampler2D(height_field_texture, material_samplers[SAMPLER_LINEAR_CLAMP]), uv_pos + vec2(0.0, EPSILON)).r - texture(sampler2D(height_field_texture, material_samplers[SAMPLER_LINEAR_CLAMP]), uv_pos - vec2(0.0, EPSILON)).r));

					collided = true;
					sdf_pos2 = sdf_pos + n * d;
					pos2 = vec2(dot(vec4(sdf_pos2, 0, 1), FRAME.colliders[0].transform[2]), dot(vec4(sdf_pos2, 0, 1), FRAME.colliders[0].transform[3]));

					n = pos - pos2;

					collision_normal = normalize(vec3(n, 0.0));
					collision_depth = length(n);
				}
			}

		} else {
			for (uint i = 0; i < FRAME.collider_count; i++) {
				vec3 normal;
				float depth;
				bool col = false;

				vec3 rel_vec = PARTICLE.xform[3].xyz - FRAME.colliders[i].transform[3].xyz;
				vec3 local_pos = rel_vec * mat3(FRAME.colliders[i].transform);

				switch (FRAME.colliders[i].type) {
					case COLLIDER_TYPE_SPHERE: {
						float d = length(rel_vec) - (particle_size + FRAME.colliders[i].extents.x);

						if (d < 0.0) {
							col = true;
							depth = -d;
							normal = normalize(rel_vec);
						}

					} break;
					case COLLIDER_TYPE_BOX: {
						vec3 abs_pos = abs(local_pos);
						vec3 sgn_pos = sign(local_pos);

						if (any(greaterThan(abs_pos, FRAME.colliders[i].extents))) {
							//point outside box

							vec3 closest = min(abs_pos, FRAME.colliders[i].extents);
							vec3 rel = abs_pos - closest;
							depth = length(rel) - particle_size;
							if (depth < 0.0) {
								col = true;
								normal = mat3(FRAME.colliders[i].transform) * (normalize(rel) * sgn_pos);
								depth = -depth;
							}
						} else {
							//point inside box
							vec3 axis_len = FRAME.colliders[i].extents - abs_pos;
							// there has to be a faster way to do this?
							if (all(lessThan(axis_len.xx, axis_len.yz))) {
								normal = vec3(1, 0, 0);
							} else if (all(lessThan(axis_len.yy, axis_len.xz))) {
								normal = vec3(0, 1, 0);
							} else {
								normal = vec3(0, 0, 1);
							}

							col = true;
							depth = dot(normal * axis_len, vec3(1)) + particle_size;
							normal = mat3(FRAME.colliders[i].transform) * (normal * sgn_pos);
						}

					} break;
					case COLLIDER_TYPE_SDF: {
						vec3 apos = abs(local_pos);
						float extra_dist = 0.0;
						if (any(greaterThan(apos, FRAME.colliders[i].extents))) { //outside
							vec3 mpos = min(apos, FRAME.colliders[i].extents);
							extra_dist = distance(mpos, apos);
						}

						if (extra_dist > particle_size) {
							continue;
						}

						vec3 uvw_pos = (local_pos / FRAME.colliders[i].extents) * 0.5 + 0.5;
						float s = texture(sampler3D(sdf_vec_textures[FRAME.colliders[i].texture_index], material_samplers[SAMPLER_LINEAR_CLAMP]), uvw_pos).r;
						s *= FRAME.colliders[i].scale;
						s += extra_dist;
						if (s < particle_size) {
							col = true;
							depth = particle_size - s;
							const float EPSILON = 0.001;
							normal = mat3(FRAME.colliders[i].transform) *
									normalize(
											vec3(
													texture(sampler3D(sdf_vec_textures[FRAME.colliders[i].texture_index], material_samplers[SAMPLER_LINEAR_CLAMP]), uvw_pos + vec3(EPSILON, 0.0, 0.0)).r - texture(sampler3D(sdf_vec_textures[FRAME.colliders[i].texture_index], material_samplers[SAMPLER_LINEAR_CLAMP]), uvw_pos - vec3(EPSILON, 0.0, 0.0)).r,
													texture(sampler3D(sdf_vec_textures[FRAME.colliders[i].texture_index], material_samplers[SAMPLER_LINEAR_CLAMP]), uvw_pos + vec3(0.0, EPSILON, 0.0)).r - texture(sampler3D(sdf_vec_textures[FRAME.colliders[i].texture_index], material_samplers[SAMPLER_LINEAR_CLAMP]), uvw_pos - vec3(0.0, EPSILON, 0.0)).r,
													texture(sampler3D(sdf_vec_textures[FRAME.colliders[i].texture_index], material_samplers[SAMPLER_LINEAR_CLAMP]), uvw_pos + vec3(0.0, 0.0, EPSILON)).r - texture(sampler3D(sdf_vec_textures[FRAME.colliders[i].texture_index], material_samplers[SAMPLER_LINEAR_CLAMP]), uvw_pos - vec3(0.0, 0.0, EPSILON)).r));
						}

					} break;
					case COLLIDER_TYPE_HEIGHT_FIELD: {
						vec3 local_pos_bottom = local_pos;
						local_pos_bottom.y -= particle_size;

						if (any(greaterThan(abs(local_pos_bottom), FRAME.colliders[i].extents))) {
							continue;
						}
						const float DELTA = 1.0 / 8192.0;

						vec3 uvw_pos = vec3(local_pos_bottom / FRAME.colliders[i].extents) * 0.5 + 0.5;

						float y = 1.0 - texture(sampler2D(height_field_texture, material_samplers[SAMPLER_LINEAR_CLAMP]), uvw_pos.xz).r;

						if (y > uvw_pos.y) {
							//inside heightfield

							vec3 pos1 = (vec3(uvw_pos.x, y, uvw_pos.z) * 2.0 - 1.0) * FRAME.colliders[i].extents;
							vec3 pos2 = (vec3(uvw_pos.x + DELTA, 1.0 - texture(sampler2D(height_field_texture, material_samplers[SAMPLER_LINEAR_CLAMP]), uvw_pos.xz + vec2(DELTA, 0)).r, uvw_pos.z) * 2.0 - 1.0) * FRAME.colliders[i].extents;
							vec3 pos3 = (vec3(uvw_pos.x, 1.0 - texture(sampler2D(height_field_texture, material_samplers[SAMPLER_LINEAR_CLAMP]), uvw_pos.xz + vec2(0, DELTA)).r, uvw_pos.z + DELTA) * 2.0 - 1.0) * FRAME.colliders[i].extents;

							normal = normalize(cross(pos1 - pos2, pos1 - pos3));
							float local_y = (vec3(local_pos / FRAME.colliders[i].extents) * 0.5 + 0.5).y;

							col = true;
							depth = dot(normal, pos1) - dot(normal, local_pos_bottom);
						}

					} break;
				}

				if (col) {
					if (!collided) {
						collided = true;
						collision_normal = normal;
						collision_depth = depth;
					} else {
						vec3 c = collision_normal * collision_depth;
						c += normal * max(0.0, depth - dot(normal, c));
						collision_normal = normalize(c);
						collision_depth = length(c);
					}
				}
			}
		}
	}

	if (particle_active) {
#CODE : PROCESS
	}

	PARTICLE.flags &= ~PARTICLE_FLAG_ACTIVE;
	if (particle_active) {
		PARTICLE.flags |= PARTICLE_FLAG_ACTIVE;
	}
}
