#include "hddagi_radiance_inc.glsl"

const uvec3 HDDAGI_EMISSION_SHIFTS = uvec3(0u, 10u, 20u);

uint hddagi_pack_emission_lobes(uvec3 negative, uvec3 positive) {
	uvec3 pairs = negative | (positive << 5u);
	return pairs.x | (pairs.y << 10u) | (pairs.z << 20u);
}

void hddagi_store_emission(ivec3 cell, uint radiance, vec3 negative, vec3 positive) {
	uvec3 neg = uvec3(round(clamp(negative, 0.0, 1.0) * 31.0));
	uvec3 pos = uvec3(round(clamp(positive, 0.0, 1.0) * 31.0));
	uint lobes = hddagi_pack_emission_lobes(neg, pos);
#ifdef NO_IMAGE_ATOMICS
	imageStore(emission_grid, cell, uvec4(max(radiance, imageLoad(emission_grid, cell).r)));
	uint previous = imageLoad(emission_aniso_grid, cell).r;
#else
	imageAtomicMax(emission_grid, cell, radiance);
	uint previous = imageAtomicCompSwap(emission_aniso_grid, cell, 0u, lobes);
	if (previous == 0u) {
		return;
	}
#endif
	// Multiple faces can share a material cell. Take the maximum of each 5-bit lobe independently.
	while (previous != lobes) {
		uvec3 pairs = uvec3(previous) >> HDDAGI_EMISSION_SHIFTS;
		uint merged = hddagi_pack_emission_lobes(max(neg, pairs & 31u), max(pos, (pairs >> 5u) & 31u));
		if (merged == previous) {
			break;
		}
#ifdef NO_IMAGE_ATOMICS
		imageStore(emission_aniso_grid, cell, uvec4(merged));
		break;
#else
		uint observed = imageAtomicCompSwap(emission_aniso_grid, cell, previous, merged);
		if (observed == previous) {
			break;
		}
		previous = observed;
#endif
	}
}
