#[compute]

#version 450

#VERSION_DEFINES

#define HDDAGI_IRRADIANCE_CACHE_SET 0
#include "hddagi_screen_probe_irradiance_cache_inc.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

uint hddagi_irradiance_cache_ttl(uint rank) {
	return rank == 0u ? 12u : (rank == 1u ? 8u : 4u);
}

void hddagi_irradiance_cache_clear(uint index) {
	uint capacity = hddagi_irradiance_cache_capacity();
	if (index == 0u) {
		hddagi_irradiance_cache_meta[0] = capacity;
		hddagi_irradiance_cache_meta[1] = 0u;
		hddagi_irradiance_cache_meta[2] = 0u;
		hddagi_irradiance_cache_meta[3] = 0u;
	}

	if (index < uint(HDDAGI_IRRADIANCE_CACHE_GRID_CAPACITY)) {
		hddagi_irradiance_cache_grid[index] = uvec2(HDDAGI_IRRADIANCE_CACHE_EMPTY, 0u);
	}
	if (index < uint(HDDAGI_IRRADIANCE_CACHE_ENTRY_CAPACITY)) {
		hddagi_irradiance_cache_entry_cells[index] = ivec4(0, 0, 0, -1);
		hddagi_irradiance_cache_entry_state[index] = uvec4(0u);
		hddagi_irradiance_cache_clear_entry_payload(index);
		hddagi_irradiance_cache_free_pool[index] = index < capacity ? index : HDDAGI_IRRADIANCE_CACHE_LOCKED;
	}
	if (index < uint(HDDAGI_IRRADIANCE_CACHE_REQUEST_CAPACITY)) {
		uint request_base = index * 3u;
		hddagi_irradiance_cache_requests[request_base + 0u] = vec4(0.0);
		hddagi_irradiance_cache_requests[request_base + 1u] = vec4(0.0);
		hddagi_irradiance_cache_requests[request_base + 2u] = vec4(0.0);
	}
}

void hddagi_irradiance_cache_flush_spatial(uint entry_index, uint generation) {
	uint vote_count = atomicExchange(hddagi_irradiance_cache_vote_count[entry_index], 0u);
	if (vote_count == 0u || generation == 0u) {
		return;
	}
	memoryBarrierBuffer();
	uint spatial_base = entry_index * 2u;
	vec4 proposal_position = hddagi_irradiance_cache_proposals[spatial_base + 0u];
	vec4 proposal_normal = hddagi_irradiance_cache_proposals[spatial_base + 1u];
	if (floatBitsToUint(proposal_position.w) != generation ||
			!hddagi_irradiance_cache_is_finite(proposal_position.xyz) ||
			!hddagi_irradiance_cache_is_finite(proposal_normal.xyz)) {
		return;
	}

	// Publish the generation marker last, mirroring the proposal writer.
	hddagi_irradiance_cache_spatial[spatial_base + 1u] = proposal_normal;
	memoryBarrierBuffer();
	hddagi_irradiance_cache_spatial[spatial_base + 0u] = proposal_position;
	hddagi_irradiance_cache_proposals[spatial_base + 0u] = vec4(0.0);
	hddagi_irradiance_cache_proposals[spatial_base + 1u] = vec4(0.0);
}

void hddagi_irradiance_cache_age(uint entry_index) {
	uint capacity = hddagi_irradiance_cache_capacity();
	if (entry_index >= capacity) {
		return;
	}

	uvec4 state = hddagi_irradiance_cache_entry_state[entry_index];
	hddagi_irradiance_cache_flush_spatial(entry_index, state.y);
	uint accumulation_base = entry_index * uint(HDDAGI_IRRADIANCE_CACHE_ACCUM_STRIDE);
	for (uint coefficient = 0u; coefficient < uint(HDDAGI_IRRADIANCE_CACHE_ACCUM_STRIDE); coefficient++) {
		hddagi_irradiance_cache_accumulation[accumulation_base + coefficient] = 0;
	}

	if (state.y == 0u) {
		return;
	}
	uint age = hddagi_irradiance_cache_params.control.x - state.x;
	if (age <= hddagi_irradiance_cache_ttl(min(state.w, 2u))) {
		return;
	}

	ivec4 entry_cell = hddagi_irradiance_cache_entry_cells[entry_index];
	if (entry_cell.w < 0) {
		return;
	}
	uint slot = uint(entry_cell.w);
	if (slot >= hddagi_irradiance_cache_grid_count()) {
		return;
	}
	uint previous = atomicCompSwap(hddagi_irradiance_cache_grid[slot].x, entry_index, HDDAGI_IRRADIANCE_CACHE_LOCKED);
	if (previous != entry_index) {
		return;
	}
	if (hddagi_irradiance_cache_grid[slot].y != state.y ||
			hddagi_irradiance_cache_entry_state[entry_index].y != state.y ||
			any(notEqual(hddagi_irradiance_cache_entry_cells[entry_index], entry_cell))) {
		atomicExchange(hddagi_irradiance_cache_grid[slot].x, entry_index);
		return;
	}
	// No consumer pops the stack during AGE. Reserve and publish the free slot
	// before destroying the entry so even a defensive overflow failure can
	// restore the grid mapping instead of leaking the entry.
	if (!hddagi_irradiance_cache_push_free(entry_index)) {
		atomicExchange(hddagi_irradiance_cache_grid[slot].x, entry_index);
		return;
	}

	hddagi_irradiance_cache_entry_state[entry_index] = uvec4(0u);
	hddagi_irradiance_cache_entry_cells[entry_index] = ivec4(0, 0, 0, -1);
	hddagi_irradiance_cache_clear_entry_payload(entry_index);
	hddagi_irradiance_cache_grid[slot].y = 0u;
	memoryBarrierBuffer();
	atomicExchange(hddagi_irradiance_cache_grid[slot].x, HDDAGI_IRRADIANCE_CACHE_EMPTY);
}

void hddagi_irradiance_cache_process(uint request_index) {
	uint request_count = min(hddagi_irradiance_cache_meta[1], hddagi_irradiance_cache_request_limit());
	if (request_index >= request_count) {
		return;
	}

	uint request_base = request_index * 3u;
	vec4 request_position = hddagi_irradiance_cache_requests[request_base + 0u];
	vec4 request_normal = hddagi_irradiance_cache_requests[request_base + 1u];
	vec3 radiance = hddagi_irradiance_cache_requests[request_base + 2u].xyz;
	uint entry_index = floatBitsToUint(request_position.w);
	uint generation = floatBitsToUint(request_normal.w);
	if (entry_index >= hddagi_irradiance_cache_capacity() || generation == 0u ||
			!hddagi_irradiance_cache_is_finite(request_position.xyz) ||
			!hddagi_irradiance_cache_is_finite(request_normal.xyz) ||
			!hddagi_irradiance_cache_is_finite(radiance) ||
			hddagi_irradiance_cache_entry_state[entry_index].y != generation) {
		return;
	}

	vec3 normal = hddagi_irradiance_cache_safe_normalize(request_normal.xyz);
	radiance = max(radiance, vec3(0.0));
	// AGE publishes the representative before PROCESS. Revalidate delayed
	// requests against that surface so the first frame cannot mix walls or let a
	// later representative vote reinterpret coefficients trained elsewhere.
	if (!hddagi_irradiance_cache_representative_compatible(entry_index, generation, 0.0, request_position.xyz, normal)) {
		return;
	}
	uint accumulation_base = entry_index * uint(HDDAGI_IRRADIANCE_CACHE_ACCUM_STRIDE);
	int previous_count = atomicAdd(hddagi_irradiance_cache_accumulation[accumulation_base + 12u], 1);
	if (previous_count >= HDDAGI_IRRADIANCE_CACHE_MAX_SAMPLES_PER_ENTRY_PER_FRAME) {
		atomicAdd(hddagi_irradiance_cache_accumulation[accumulation_base + 12u], -1);
		return;
	}
	vec4 basis = vec4(1.0, normal);
	for (uint channel = 0u; channel < 3u; channel++) {
		for (uint coefficient = 0u; coefficient < 4u; coefficient++) {
			// The per-entry request cap keeps 4096 contributions at this bound in
			// signed 32-bit range, retaining HDR values up to roughly 2048.
			float scaled = clamp(radiance[channel] * basis[coefficient] * HDDAGI_IRRADIANCE_CACHE_ACCUM_SCALE, -524287.0, 524287.0);
			atomicAdd(hddagi_irradiance_cache_accumulation[accumulation_base + channel * 4u + coefficient], int(round(scaled)));
		}
	}
	atomicAdd(hddagi_irradiance_cache_accumulation[accumulation_base + 13u], int(round(normal.x * HDDAGI_IRRADIANCE_CACHE_ACCUM_SCALE)));
	atomicAdd(hddagi_irradiance_cache_accumulation[accumulation_base + 14u], int(round(normal.y * HDDAGI_IRRADIANCE_CACHE_ACCUM_SCALE)));
	atomicAdd(hddagi_irradiance_cache_accumulation[accumulation_base + 15u], int(round(normal.z * HDDAGI_IRRADIANCE_CACHE_ACCUM_SCALE)));
}

void hddagi_irradiance_cache_resolve(uint entry_index) {
	uint capacity = hddagi_irradiance_cache_capacity();
	if (entry_index >= capacity) {
		return;
	}

	uvec4 state = hddagi_irradiance_cache_entry_state[entry_index];
	uint accumulation_base = entry_index * uint(HDDAGI_IRRADIANCE_CACHE_ACCUM_STRIDE);
	int signed_count = hddagi_irradiance_cache_accumulation[accumulation_base + 12u];
	if (state.y != 0u && signed_count > 0) {
		float inverse_weight = 1.0 / (HDDAGI_IRRADIANCE_CACHE_ACCUM_SCALE * float(signed_count));
		uint history_sample_cap = hddagi_irradiance_cache_multibounce_enabled() ? HDDAGI_IRRADIANCE_CACHE_MULTIBOUNCE_HISTORY_SAMPLE_CAP : HDDAGI_IRRADIANCE_CACHE_HISTORY_SAMPLE_CAP;
		float history_weight = float(min(state.z, history_sample_cap));
		float update_weight = float(signed_count);
		// A bounded sample-count blend damps sparse refreshes without making
		// well-sampled cells unresponsive.
		float blend = update_weight / (history_weight + update_weight);
		uint sh_base = entry_index * 4u;
		for (uint channel = 0u; channel < 3u; channel++) {
			vec4 estimate = vec4(
									float(hddagi_irradiance_cache_accumulation[accumulation_base + channel * 4u + 0u]),
									float(hddagi_irradiance_cache_accumulation[accumulation_base + channel * 4u + 1u]),
									float(hddagi_irradiance_cache_accumulation[accumulation_base + channel * 4u + 2u]),
									float(hddagi_irradiance_cache_accumulation[accumulation_base + channel * 4u + 3u])) *
					inverse_weight;
			hddagi_irradiance_cache_sh[sh_base + channel] = history_weight == 0.0 ? estimate : mix(hddagi_irradiance_cache_sh[sh_base + channel], estimate, blend);
		}
		vec4 normalization_estimate = vec4(
				1.0,
				float(hddagi_irradiance_cache_accumulation[accumulation_base + 13u]) * inverse_weight,
				float(hddagi_irradiance_cache_accumulation[accumulation_base + 14u]) * inverse_weight,
				float(hddagi_irradiance_cache_accumulation[accumulation_base + 15u]) * inverse_weight);
		hddagi_irradiance_cache_sh[sh_base + 3u] = history_weight == 0.0 ? normalization_estimate : mix(hddagi_irradiance_cache_sh[sh_base + 3u], normalization_estimate, blend);
		uint added_samples = uint(signed_count);
		hddagi_irradiance_cache_entry_state[entry_index].z = state.z > 0xffffffffu - added_samples ? 0xffffffffu : state.z + added_samples;
	}

	// Make RESOLVE idempotent even if AGE is skipped for a frame. AGE also
	// clears this storage before PROCESS as the normal pass ordering requires.
	for (uint coefficient = 0u; coefficient < uint(HDDAGI_IRRADIANCE_CACHE_ACCUM_STRIDE); coefficient++) {
		hddagi_irradiance_cache_accumulation[accumulation_base + coefficient] = 0;
	}
}

void main() {
	uint index = gl_GlobalInvocationID.x;
#if defined(MODE_CLEAR)
	// CLEAR must cover GRID_CAPACITY invocations; it explicitly initializes
	// every backing allocation, including the free stack and request records.
	hddagi_irradiance_cache_clear(index);
#elif defined(MODE_AGE)
	hddagi_irradiance_cache_age(index);
#elif defined(MODE_PROCESS)
	hddagi_irradiance_cache_process(index);
#elif defined(MODE_RESOLVE)
	hddagi_irradiance_cache_resolve(index);
#elif defined(MODE_RESET)
	if (index == 0u) {
		hddagi_irradiance_cache_meta[1] = 0u;
	}
#endif
}
