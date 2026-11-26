#[vertex]

#version 450

#VERSION_DEFINES

layout(location = 0) in vec3 vertex_attrib;

layout(location = 0) out flat uint element_type;
layout(location = 1) out flat uint element_index;

layout(push_constant, std430) uniform Params {
	uint base_index;
	uint pad0;
	uint pad1;
	uint pad2;
}
params;

layout(set = 0, binding = 1, std140) uniform State {
	mat4 projection;

	uint screen_to_tiles_shift; // shift to obtain coordinates in block indices
	uint tiles_per_row;
	uint bitmasks_per_tile_count; // how many mask words a single tile takes
	uint bitmasks_per_type_count; // how many mask words all tiles of one element type take
}
state;

struct RenderElement {
	uint type; //0-4
	uint original_index;
	uint pad0;
	uint pad1;
	mat3x4 transform_inv;
	vec3 scale;
	uint pad2;
};

layout(set = 0, binding = 2, std430) buffer restrict readonly RenderElements {
	RenderElement data[];
}
render_elements;

void main() {
	uint render_element = params.base_index + gl_InstanceIndex;

	element_type = render_elements.data[render_element].type;
	element_index = render_elements.data[render_element].original_index;

	vec3 vertex = vertex_attrib;
	vertex *= render_elements.data[render_element].scale;

	vertex = vec4(vertex, 1.0) * render_elements.data[render_element].transform_inv;

	gl_Position = state.projection * vec4(vertex, 1.0);
}

#[fragment]

#version 450

#VERSION_DEFINES

#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_vote : enable

// On Apple platforms, gl_HelperInvocation (simd_is_helper_thread()) is unreliable with MSAA, causing rendering artifacts.
// Setting this to false will disable the helper invocation check.
layout(constant_id = 0) const bool sc_use_helper_check = true;

layout(location = 0) in flat uint element_type;
layout(location = 1) in flat uint element_index;

layout(set = 0, binding = 1, std140) uniform State {
	mat4 projection;

	uint screen_to_tiles_shift; // shift to obtain coordinates in block indices
	uint tiles_per_row;
	uint bitmasks_per_tile_count; // how many mask words a single tile takes
	uint bitmasks_per_type_count; // how many mask words all tiles of one element type take
}
state;

//cluster data is layout linearly by element type, each cell contains a bit for every element of that type to mark it as used

layout(set = 0, binding = 3, std430) buffer restrict ClusterData {
	uint data[];
}
cluster_data;

#ifdef USE_ATTACHMENT
layout(location = 0) out vec4 frag_color;
#endif

void main() {
	//convert from screen to tile
	uvec2 tile_pos = uvec2(gl_FragCoord.xy) >> state.screen_to_tiles_shift;

	//get linear tile offset from screen poss
	uint tile_offset = tile_pos.x + state.tiles_per_row * tile_pos.y;
	//multiply by data size to position at the beginning of the element list for this tile
	tile_offset *= state.bitmasks_per_tile_count;

	//find the current element in the list and plot the bit to mark it as used
	uint usage_write_offset = element_type * state.bitmasks_per_type_count + tile_offset + (element_index >> 5);
	uint usage_write_bit = 1 << (element_index & 0x1F);

	uint aux = 0;

	if (sc_use_helper_check) {
		//https://advances.realtimerendering.com/s2017/2017_Sig_Improved_Culling_final.pdf
		if (!gl_HelperInvocation) {
			uvec4 mask;

			while (true) {
				// find the tile offset of the first active thread
				// threads that did break; go inactive and no longer count
				uint first = subgroupBroadcastFirst(tile_offset);
				// update the mask for thread that match this tile
				mask = subgroupBallot(first == tile_offset);
				if (first == tile_offset) {
					// This thread belongs to the group of threads that match this offset,
					// so exit the loop.
					break;
				}
			}

			uint cluster_thread_group_index = subgroupBallotExclusiveBitCount(mask);

			if (cluster_thread_group_index == 0) {
				aux = atomicOr(cluster_data.data[usage_write_offset], usage_write_bit);
			}
		}
	} else {
		aux = atomicOr(cluster_data.data[usage_write_offset], usage_write_bit);
	}

#ifdef USE_ATTACHMENT
	frag_color = vec4(float(aux));
#endif
}
