#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant, std430) uniform Params {
	uint cluster_render_data_size; // how much data for a single cluster takes
	uint max_render_element_count_div_32; //divided by 32
	uvec2 cluster_screen_size;
	uint render_element_count_div_32; //divided by 32

	uint max_cluster_element_count_div_32; //divided by 32
	uint pad1;
	uint pad2;
}
params;

layout(set = 0, binding = 1, std430) buffer restrict readonly ClusterRender {
	uint data[];
}
cluster_render;

layout(set = 0, binding = 2, std430) buffer restrict ClusterStore {
	uint data[];
}
cluster_store;

struct RenderElement {
	uint type; //0-4
	bool touches_near;
	bool touches_far;
	uint original_index;
	mat3x4 transform_inv;
	vec3 scale;
	uint pad;
};

layout(set = 0, binding = 3, std430) buffer restrict readonly RenderElements {
	RenderElement data[];
}
render_elements;

void main() {
	uvec2 pos = gl_GlobalInvocationID.xy;
	if (any(greaterThanEqual(pos, params.cluster_screen_size))) {
		return;
	}

	//counter for each type of render_element

	//base offset for this cluster
	uint base_offset = (pos.x + params.cluster_screen_size.x * pos.y);
	uint src_offset = base_offset * params.cluster_render_data_size;

	uint render_element_offset = 0;

	//check all render_elements and see which one was written to
	while (render_element_offset < params.render_element_count_div_32) {
		uint bits = cluster_render.data[src_offset + render_element_offset];
		while (bits != 0) {
			//if bits exist, check the render_element
			uint index_bit = findLSB(bits);
			uint index = render_element_offset * 32 + index_bit;
			uint type = render_elements.data[index].type;

			uint z_range_offset = src_offset + params.max_render_element_count_div_32 + index;
			uint z_range = cluster_render.data[z_range_offset];

			//if object was written, z was written, but check just in case
			if (z_range != 0) { //should always be > 0

				uint from_z = findLSB(z_range);
				uint to_z = findMSB(z_range) + 1;

				if (render_elements.data[index].touches_near) {
					from_z = 0;
				}

				if (render_elements.data[index].touches_far) {
					to_z = 32;
				}

				// find cluster offset in the buffer used for indexing in the renderer
				uint dst_offset = (base_offset + type * (params.cluster_screen_size.x * params.cluster_screen_size.y)) * (params.max_cluster_element_count_div_32 + 32);

				uint orig_index = render_elements.data[index].original_index;
				//store this index in the Z slices by setting the relevant bit
				for (uint i = from_z; i < to_z; i++) {
					uint slice_ofs = dst_offset + params.max_cluster_element_count_div_32 + i;

					uint minmax = cluster_store.data[slice_ofs];

					if (minmax == 0) {
						minmax = 0xFFFF; //min 0, max 0xFFFF
					}

					uint elem_min = min(orig_index, minmax & 0xFFFF);
					uint elem_max = max(orig_index + 1, minmax >> 16); //always store plus one, so zero means range is empty when not written to

					minmax = elem_min | (elem_max << 16);
					cluster_store.data[slice_ofs] = minmax;
				}

				uint store_word = orig_index >> 5;
				uint store_bit = orig_index & 0x1F;

				//store the actual render_element index at the end, so the rendering code can reference it
				cluster_store.data[dst_offset + store_word] |= 1 << store_bit;
			}

			bits &= ~(1 << index_bit); //clear the bit to continue iterating
		}

		render_element_offset++;
	}
}
