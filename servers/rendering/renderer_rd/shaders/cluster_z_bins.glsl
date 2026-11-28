#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#define ELEMENT_TYPE_MAX 5

layout(push_constant, std430) uniform Params {
	uint z_bin_count;
	uint z_bin_offset;
	uint z_bin_mask_offset;
	uint max_elements_by_type;

	uint element_counts[ELEMENT_TYPE_MAX];
	uint type; // MODE_MASKS only, dispatched once per element type.
	uint pad1;
	uint pad2;
}
params;

layout(set = 0, binding = 2, std430) buffer restrict ClusterData {
	uint data[];
}
cluster_data;

uint get_mask_index(uint p_type, uint p_word, uint p_z_bin) {
	return params.z_bin_mask_offset + (p_type * (params.max_elements_by_type / 32) + p_word) * params.z_bin_count + p_z_bin;
}

#ifdef MODE_MASKS

// First and last Z bin of every element, packed.
// indexed by type * max_elements_by_type + index.
layout(set = 0, binding = 1, std430) buffer restrict readonly ElementZBins {
	uint data[];
}
element_z_bins;

// One thread per (bin, word). Only the mask words of existing elements are written,
// readers must stay within the bin's range.
void main() {
	uint z_bin = gl_GlobalInvocationID.x;
	uint word = gl_GlobalInvocationID.y;
	uint type = params.type;
	uint element_count = params.element_counts[type];
	if (z_bin >= params.z_bin_count) {
		return;
	}

	uint element_offset = type * params.max_elements_by_type + word * 32;
	uint word_element_count = min(element_count - word * 32, 32);
	uint mask = 0;
	for (uint bit = 0; bit < word_element_count; bit++) {
		uint bins = element_z_bins.data[element_offset + bit];
		if (z_bin >= (bins & 0xFFFF) && z_bin <= (bins >> 16)) {
			mask |= 1u << bit;
		}
	}

	cluster_data.data[get_mask_index(type, word, z_bin)] = mask;
}

#else // MODE_RANGES

// One thread per (bin, type), packs the lowest element index and the highest one plus one (zero is empty).
void main() {
	uint z_bin = gl_GlobalInvocationID.x;
	uint type = gl_GlobalInvocationID.y;
	if (z_bin >= params.z_bin_count) {
		return;
	}

	uint word_count = (params.element_counts[type] + 31) / 32;
	uint item_min = 0;
	uint item_max = 0;
	for (uint word = 0; word < word_count; word++) {
		uint mask = cluster_data.data[get_mask_index(type, word, z_bin)];
		if (mask != 0) {
			if (item_max == 0) {
				item_min = word * 32 + uint(findLSB(mask));
			}
			item_max = word * 32 + uint(findMSB(mask)) + 1;
		}
	}

	cluster_data.data[params.z_bin_offset + type * params.z_bin_count + z_bin] = item_min | (item_max << 16);
}

#endif
