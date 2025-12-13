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
#ifdef USERDATA_COUNT
	vec4 userdata[USERDATA_COUNT];
#endif
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

#define PARAMS_FLAG_ORDER_BY_LIFETIME 1
#define PARAMS_FLAG_COPY_MODE_2D 2

layout(push_constant, std430) uniform Params {
	vec3 sort_direction;
	uint total_particles;

	uint trail_size;
	uint trail_total;
	float frame_delta;
	float frame_remainder;

	vec3 align_up;
	uint align_mode;

	uint lifetime_split;
	bool lifetime_reverse;
	uint motion_vectors_current_offset;
	uint flags;

	float inv_emission_transform[12];

	uint align_custom_src;
	uint align_axis;
	uint align_flags;
	uint pad1;
}
params;

#define ALIGN_DISABLED 0
#define ALIGN_BILLBOARD 1
#define ALIGN_Y_TO_VELOCITY 2
#define ALIGN_Z_BILLBOARD_Y_TO_VELOCITY 3
#define ALIGN_ROTATE_AXIS 4
#define ALIGN_LOCAL_BILLBOARD 5

#define CUSTOM_SRC_NONE 0
#define CUSTOM_SRC_X 1
#define CUSTOM_SRC_Y 2
#define CUSTOM_SRC_Z 3
#define CUSTOM_SRC_W 4

#define ALIGN_AXIS_X 0
#define ALIGN_AXIS_Y 1
#define ALIGN_AXIS_Z 2

#define ALIGN_FLAGS_ALIGN_TO_VELOCITY uint(1);


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
	if (bool(params.flags & PARAMS_FLAG_ORDER_BY_LIFETIME)) {
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
			// Since the steps don't fit precisely in the history frames, must do a tiny bit of
			// interpolation to get them close to their intended location.
			uint part_ofs = particle % params.trail_size;
			float natural_ofs = fract((float(part_ofs) / float(params.trail_size)) * float(params.trail_total)) * params.frame_delta;

			txform[3].xyz -= particles.data[particle].velocity * natural_ofs;
		}

		switch (params.align_mode) {
			case ALIGN_DISABLED: {
			} break; //nothing
			case ALIGN_BILLBOARD: {
				float angle = 0.;
				switch (params.align_custom_src) {
					case CUSTOM_SRC_X: {
						angle = particles.data[particle].custom.x;
					} break;
					case CUSTOM_SRC_Y: {
						angle = particles.data[particle].custom.y;
					} break;
					case CUSTOM_SRC_Z: {
						angle = particles.data[particle].custom.z;
					} break;
					case CUSTOM_SRC_W: {
						angle = particles.data[particle].custom.w;
					} break;
				}
				vec3 axis = normalize(params.sort_direction);
				float s = sin(angle);
				float c = cos(angle);
				float oc = 1.0 - c;
				mat3 rotated = mat3(
						oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s,
						oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s,
						oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c
					);
				vec3 new_up = rotated * params.align_up;
				mat3 local = mat3(normalize(cross(new_up, params.sort_direction)), new_up, params.sort_direction);
				local = local * mat3(txform);
				txform[0].xyz = local[0];
				txform[1].xyz = local[1];
				txform[2].xyz = local[2];

			} break;
			case ALIGN_ROTATE_AXIS: {
				if(bool(params.align_flags & uint(1))){
				vec3 v = particles.data[particle].velocity;
				v = normalize(v);

				switch (params.align_axis) {
						case ALIGN_AXIS_X: {
							vec3 len = vec3(
								length(txform[0].xyz),
								length(txform[1].xyz),
								length(txform[2].xyz)
							);

							txform[0].xyz = v;
							txform[1].xyz = normalize(cross(txform[2].xyz/len.z, txform[0].xyz));
							txform[2].xyz = cross(txform[0].xyz, txform[1].xyz);

							txform[0].xyz *= len.x;
							txform[1].xyz *= len.y;
							txform[2].xyz *= len.z;
						} break;
						case ALIGN_AXIS_Y: {
							vec3 len = vec3(
								length(txform[0].xyz),
								length(txform[1].xyz),
								length(txform[2].xyz)
							);

							txform[0].xyz = normalize(cross(v, txform[2].xyz/len.z));
							txform[1].xyz = v;
							txform[2].xyz = cross(txform[0].xyz, txform[1].xyz);

							txform[0].xyz *= len.x;
							txform[1].xyz *= len.y;
							txform[2].xyz *= len.z;
						} break;case ALIGN_AXIS_Z: {
							vec3 len = vec3(
								length(txform[0].xyz),
								length(txform[1].xyz),
								length(txform[2].xyz)
							);

							txform[0].xyz = normalize(cross(txform[1].xyz/len.y, v));
							txform[2].xyz = v;
							txform[1].xyz = normalize(cross(txform[2].xyz, txform[0].xyz));

							txform[0].xyz *= len.x;
							txform[1].xyz *= len.y;
							txform[2].xyz *= len.z;
						} break;
					}
				}
				vec3 axis = vec3(1.0, 0.0, 0.0);
				switch (params.align_axis) {
					case ALIGN_AXIS_X: {
						axis = vec3(1.0, 0.0, 0.0);
					} break;
					case ALIGN_AXIS_Y: {
						axis = vec3(0.0, 1.0, 0.0);
					} break;
					case ALIGN_AXIS_Z: {
						axis = vec3(0.0, 0.0, 1.0);
					} break;
				}
				float angle = 0.;
				switch (params.align_custom_src) {
					case CUSTOM_SRC_X: {
						angle = particles.data[particle].custom.x;
					} break;
					case CUSTOM_SRC_Y: {
						angle = particles.data[particle].custom.y;
					} break;
					case CUSTOM_SRC_Z: {
						angle = particles.data[particle].custom.z;
					} break;
					case CUSTOM_SRC_W: {
						angle = particles.data[particle].custom.w;
					} break;
				}
				axis = normalize(axis);
				float s = sin(angle);
				float c = cos(angle);
				float oc = 1.0 - c;
				vec3 len = vec3(
					length(txform[0].xyz),
					length(txform[1].xyz),
					length(txform[2].xyz)
				);
				mat3 rotated = mat3(
						oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s,
						oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s,
						oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c
					);
				mat3 txform_normalized = mat3(txform);
				txform_normalized[0] /= len.x;
				txform_normalized[1] /= len.y;
				txform_normalized[2] /= len.z;
				rotated = txform_normalized * rotated * mat3(
					len.x, 0.0,0.0,
					0.0,len.y, 0.0,
					0.0, 0.0,len.z
				);
				vec4 origin = txform[3];
				txform = mat4(rotated);
				txform[3] = origin;
			} break;
			case ALIGN_Y_TO_VELOCITY: {
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
			case ALIGN_Z_BILLBOARD_Y_TO_VELOCITY: {
				vec3 v = particles.data[particle].velocity;
				vec3 sv = v - params.sort_direction * dot(params.sort_direction, v); //screen velocity

				if (length(sv) == 0) {
					sv = params.align_up;
				}

				sv = normalize(sv);

				txform[0].xyz = normalize(cross(sv, params.sort_direction)) * length(txform[0]);
				txform[1].xyz = sv * length(txform[1]);
				txform[2].xyz = params.sort_direction * length(txform[2]);

			} break;
			case ALIGN_LOCAL_BILLBOARD: {
				vec3 v = particles.data[particle].velocity;
				v = normalize(v);

				if(bool(params.align_flags & uint(1))){
					switch (params.align_axis) {
						case ALIGN_AXIS_X: {
							vec3 len = vec3(
								length(txform[0].xyz),
								length(txform[1].xyz),
								length(txform[2].xyz)
							);

							txform[0].xyz = v;
							txform[1].xyz = normalize(cross(params.sort_direction, v));
							txform[2].xyz = cross(txform[0].xyz, txform[1].xyz);

							txform[0].xyz *= len.x;
							txform[1].xyz *= len.y;
							txform[2].xyz *= len.z;
						} break;
						case ALIGN_AXIS_Y: {
							vec3 len = vec3(
								length(txform[0].xyz),
								length(txform[1].xyz),
								length(txform[2].xyz)
							);

							txform[0].xyz = normalize(cross(v, params.sort_direction));
							txform[1].xyz = v;
							txform[2].xyz = cross(txform[0].xyz, txform[1].xyz);

							txform[0].xyz *= len.x;
							txform[1].xyz *= len.y;
							txform[2].xyz *= len.z;
						} break;
					}
				} else {
					switch (params.align_axis) {
						case ALIGN_AXIS_X: {
							vec3 len = vec3(
								length(txform[0].xyz),
								length(txform[1].xyz),
								length(txform[2].xyz)
							);

							//txform[0].xyz = v;
							txform[1].xyz = normalize(cross(params.sort_direction, txform[0].xyz));
							txform[2].xyz = cross(txform[0].xyz, txform[1].xyz);

							txform[0].xyz *= len.x;
							txform[1].xyz *= len.y;
							txform[2].xyz *= len.z;
						} break;
						case ALIGN_AXIS_Y: {
							vec3 len = vec3(
								length(txform[0].xyz),
								length(txform[1].xyz),
								length(txform[2].xyz)
							);

							txform[0].xyz = normalize(cross(txform[1].xyz, params.sort_direction));
							txform[2].xyz = cross(txform[0].xyz, txform[1].xyz);

							txform[0].xyz *= len.x;
							txform[1].xyz *= len.y;
							txform[2].xyz *= len.z;
						} break;
					}
				}
				
			}break;
		}

		txform[3].xyz += particles.data[particle].velocity * params.frame_remainder;

		if (params.trail_size > 1) {
			uint part_ofs = particle % params.trail_size;
			txform = txform * trail_bind_poses.data[part_ofs];
		}

		if (bool(params.flags & PARAMS_FLAG_COPY_MODE_2D)) {
			// In global mode, bring 2D particles to local coordinates
			// as they will be drawn with the node position as origin.
			mat4 inv_emission_transform;
			inv_emission_transform[0] = vec4(params.inv_emission_transform[0], params.inv_emission_transform[1], params.inv_emission_transform[2], 0.0);
			inv_emission_transform[1] = vec4(params.inv_emission_transform[3], params.inv_emission_transform[4], params.inv_emission_transform[5], 0.0);
			inv_emission_transform[2] = vec4(params.inv_emission_transform[6], params.inv_emission_transform[7], params.inv_emission_transform[8], 0.0);
			inv_emission_transform[3] = vec4(params.inv_emission_transform[9], params.inv_emission_transform[10], params.inv_emission_transform[11], 1.0);
			inv_emission_transform = transpose(inv_emission_transform);
			txform = inv_emission_transform * txform;
		}
	} else {
		// Set scale to zero and translate to -INF so particle will be invisible
		// even for materials that ignore rotation/scale (i.e. billboards).
		txform = mat4(vec4(0.0), vec4(0.0), vec4(0.0), vec4(-1.0 / 0.0, -1.0 / 0.0, -1.0 / 0.0, 0.0));
	}
	txform = transpose(txform);

	uint instance_index = gl_GlobalInvocationID.x + params.motion_vectors_current_offset;
	if (bool(params.flags & PARAMS_FLAG_COPY_MODE_2D)) {
		uint write_offset = instance_index * (2 + 1 + 1); //xform + color + custom

		instances.data[write_offset + 0] = txform[0];
		instances.data[write_offset + 1] = txform[1];
		instances.data[write_offset + 2] = particles.data[particle].color;
		instances.data[write_offset + 3] = particles.data[particle].custom;
	} else {
		uint write_offset = instance_index * (3 + 1 + 1); //xform + color + custom

		instances.data[write_offset + 0] = txform[0];
		instances.data[write_offset + 1] = txform[1];
		instances.data[write_offset + 2] = txform[2];
		instances.data[write_offset + 3] = particles.data[particle].color;
		instances.data[write_offset + 4] = particles.data[particle].custom;
	}

#endif
}
