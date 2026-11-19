#include "hddagi_voxel_metadata_inc.glsl"
uint hddagi_surface_neighbour_bit(uint index) {
	return (((0x24bc028u >> index) & 1u) << 0u) | (((0x90e2au >> index) & 1u) << 1u) | (((0x2400ab8u >> index) & 1u) << 2u) | (((0x22888a2u >> index) & 1u) << 3u) | (((0x820000u >> index) & 1u) << 4u);
}

bool hddagi_build_surface(ivec3 pos, ivec3 local, out vec4 albedo_accum, out vec4 emission_accum, out vec3 normal_accum, out uint neighbour_voxels) {
	albedo_accum = vec4(0.0);
	emission_accum = vec4(0.0);
	normal_accum = vec3(0.0);
	neighbour_voxels = 0u;
	uint disocclusion = 0u;
	uint solid = get_normal_facing(local);
	bool voxels_found = false;
	const int facing_direction_count = 26;

	bool use_for_filter = false;

	for (int i = 0; i < 6; i++) {
		uint n = get_normal_facing(local + hddagi_axis_direction(uint(i ^ 1)));
		if (n == 0) {
			disocclusion |= (1u << uint(i ^ 1));
		} else if (solid == 0) {
			use_for_filter = true;
		}

		if (solid != 0 || !bool(n & (1u << uint(i)))) {
			continue;
		}

		voxels_found = true;

		uint normals = n >> 6u;
		if ((n & (1u << uint(i ^ 1))) != 0u) {
			uvec3 side = (i & 1) != 0 ? HDDAGI_SURFACE_POSITIVE : HDDAGI_SURFACE_NEGATIVE;
			normals &= i < 2 ? side.x : (i < 4 ? side.y : side.z);
			if (normals == 0u) {
				normals = 1u << uint(i);
			}
		}
		while (normals != 0u) {
			normal_accum += hddagi_surface_normal(uint(findLSB(normals)));
			normals &= normals - 1u;
		}

		ivec3 ofs = pos + hddagi_axis_direction(uint(i ^ 1));

		ivec3 albedo_ofs = ofs >> 1;
		albedo_ofs.z *= 6;
		albedo_ofs.z += i;

		uint a = imageLoad(src_albedo, albedo_ofs).r;
		albedo_accum += vec4(vec3((ivec3(a) >> ivec3(0, 5, 11)) & ivec3(0x1f, 0x3f, 0x1f)) / vec3(31.0, 63.0, 31.0), 1.0);

		uint rgbe = imageLoad(src_emission, ofs >> 1).r;

		vec3 emission = rgbe_decode(rgbe);

		uint rgbe_aniso = imageLoad(src_emission_aniso, ofs >> 1).r;
		float strength = ((rgbe_aniso >> (i * 5)) & 0x1F) / float(0x1F);
		emission_accum += vec4(emission * strength, 1.0);
	}

	if (solid == 0u) {
		for (uint axis = 0u; axis < 3u; axis++) {
			uint pair = 3u << (axis * 2u);
			if ((disocclusion & pair) == pair) {
				neighbour_voxels |= 1u << (HDDAGI_OCCLUSION_PATH_SHIFT + axis);
			}
		}
	}

	if (solid == 0u && use_for_filter) {
		for (int i = 0; i < facing_direction_count; i++) {
			ivec3 neighbor = hddagi_surface_direction(uint(i));
			ivec3 neighbour_pos = local + neighbor;
			uint n = get_normal_facing(neighbour_pos);
			if (n == 0) {
				continue;
			}

			for (int j = 0; j < 6; j++) {
				ivec3 neighbour_neighbour = neighbour_pos + hddagi_axis_direction(uint(j));
				ivec3 nn_rel = neighbour_neighbour - local;

				if (any(lessThan(nn_rel, -ivec3(1))) || any(greaterThan(nn_rel, +ivec3(1)))) {
					continue;
				}

				if (nn_rel == ivec3(0)) {
					continue;
				}

				uint q = get_normal_facing(local + nn_rel);
				if (q != 0) {
					continue;
				}

				ivec3 nn_rel_abs = abs(nn_rel);

				int nn_steps = nn_rel_abs.x + nn_rel_abs.y + nn_rel_abs.z;
				if (nn_steps == 3) {
					continue;
				}
				if (nn_steps > 1) {
					int occlusions = 0;
					for (int k = 0; k < 3; k++) {
						ivec3 test_dir = k == 0 ? ivec3(nn_rel.x, 0, 0) : (k == 1 ? ivec3(0, nn_rel.y, 0) : ivec3(0, 0, nn_rel.z));
						if (test_dir == ivec3(0)) {
							continue;
						}

						q = get_normal_facing(local + test_dir);
						if (q != 0) {
							occlusions++;
						}
					}

					if (occlusions >= 2) {
						continue;
					}
				}

				ivec3 abs_pos = nn_rel + ivec3(1);
				neighbour_voxels |= 1 << hddagi_surface_neighbour_bit(uint(abs_pos.z * 3 * 3 + abs_pos.y * 3 + abs_pos.x));
			}
		}
	}

	neighbour_voxels |= disocclusion << HDDAGI_DISOCCLUSION_SHIFT;
	return voxels_found && solid == 0u;
}

uint hddagi_pack_surface_albedo(vec4 albedo_accum, vec3 normal_accum) {
	albedo_accum.rgb /= albedo_accum.a;
	normal_accum = normalize(normal_accum);
	uint albedo_norm = 0u;
	albedo_norm |= clamp(uint(albedo_accum.r * 31.0), 0u, 31u) << 0;
	albedo_norm |= clamp(uint(albedo_accum.g * 63.0), 0u, 63u) << 5;
	albedo_norm |= clamp(uint(albedo_accum.b * 31.0), 0u, 31u) << 11;
	uvec2 octa_unormal = clamp(uvec2(octahedron_encode(normal_accum) * 255.0), uvec2(0), uvec2(255));
	return albedo_norm | (octa_unormal.x << 16) | (octa_unormal.y << 24);
}
