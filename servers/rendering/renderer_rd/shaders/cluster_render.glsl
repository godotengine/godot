#[vertex]

#version 450

#VERSION_DEFINES

layout(location = 0) in vec3 vertex_attrib;

layout(location = 0) out float depth_interp;
layout(location = 1) out flat uint element_index;

layout(push_constant, binding = 0, std430) uniform Params {
	uint base_index;
	uint pad0;
	uint pad1;
	uint pad2;
}
params;

layout(set = 0, binding = 1, std140) uniform State {
	mat4 projection;

	float inv_z_far;
	uint screen_to_clusters_shift; // shift to obtain coordinates in block indices
	uint cluster_screen_width; //
	uint cluster_data_size; // how much data for a single cluster takes

	uint cluster_depth_offset;
	uint pad0;
	uint pad1;
	uint pad2;
}
state;

struct RenderElement {
	uint type; //0-4
	bool touches_near;
	bool touches_far;
	uint original_index;
	mat3x4 transform_inv;
	vec3 scale;
	uint pad;
};

layout(set = 0, binding = 2, std430) buffer restrict readonly RenderElements {
	RenderElement data[];
}
render_elements;

void main() {
	element_index = params.base_index + gl_InstanceIndex;

	vec3 vertex = vertex_attrib;
	vertex *= render_elements.data[element_index].scale;

	vertex = vec4(vertex, 1.0) * render_elements.data[element_index].transform_inv;
	depth_interp = -vertex.z;

	gl_Position = state.projection * vec4(vertex, 1.0);
}

#[fragment]

#version 450

#VERSION_DEFINES

#if defined(has_GL_KHR_shader_subgroup_ballot) && defined(has_GL_KHR_shader_subgroup_arithmetic) && defined(has_GL_KHR_shader_subgroup_vote)

#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_vote : enable

#define USE_SUBGROUPS
#endif

layout(location = 0) in float depth_interp;
layout(location = 1) in flat uint element_index;

layout(set = 0, binding = 1, std140) uniform State {
	mat4 projection;
	float inv_z_far;
	uint screen_to_clusters_shift; // shift to obtain coordinates in block indices
	uint cluster_screen_width; //
	uint cluster_data_size; // how much data for a single cluster takes
	uint cluster_depth_offset;
	uint pad0;
	uint pad1;
	uint pad2;
}
state;

//cluster data is layout linearly, each cell contains the follow information:
// - list of bits for every element to mark as used, so (max_elem_count/32)*4 uints
// - a uint for each element to mark the depth bits used when rendering (0-31)

layout(set = 0, binding = 3, std430) buffer restrict ClusterRender {
	uint data[];
}
cluster_render;

void main() {
	//convert from screen to cluster
	uvec2 cluster = uvec2(gl_FragCoord.xy) >> state.screen_to_clusters_shift;

	//get linear cluster offset from screen poss
	uint cluster_offset = cluster.x + state.cluster_screen_width * cluster.y;
	//multiply by data size to position at the beginning of the element list for this cluster
	cluster_offset *= state.cluster_data_size;

	//find the current element in the list and plot the bit to mark it as used
	uint usage_write_offset = cluster_offset + (element_index >> 5);
	uint usage_write_bit = 1 << (element_index & 0x1F);

#ifdef USE_SUBGROUPS

	uint cluster_thread_group_index;

	if (!gl_HelperInvocation) {
		//http://advances.realtimerendering.com/s2017/2017_Sig_Improved_Culling_final.pdf

		uvec4 mask;

		while (true) {
			// find the cluster offset of the first active thread
			// threads that did break; go inactive and no longer count
			uint first = subgroupBroadcastFirst(cluster_offset);
			// update the mask for thread that match this cluster
			mask = subgroupBallot(first == cluster_offset);
			if (first == cluster_offset) {
				// This thread belongs to the group of threads that match this offset,
				// so exit the loop.
				break;
			}
		}

		cluster_thread_group_index = subgroupBallotExclusiveBitCount(mask);

		if (cluster_thread_group_index == 0) {
			atomicOr(cluster_render.data[usage_write_offset], usage_write_bit);
		}
	}
#else
	if (!gl_HelperInvocation) {
		atomicOr(cluster_render.data[usage_write_offset], usage_write_bit);
	}
#endif
	//find the current element in the depth usage list and mark the current depth as used
	float unit_depth = depth_interp * state.inv_z_far;

	uint z_bit = clamp(uint(floor(unit_depth * 32.0)), 0, 31);

	uint z_write_offset = cluster_offset + state.cluster_depth_offset + element_index;
	uint z_write_bit = 1 << z_bit;

#ifdef USE_SUBGROUPS
	if (!gl_HelperInvocation) {
		z_write_bit = subgroupOr(z_write_bit); //merge all Zs
		if (cluster_thread_group_index == 0) {
			atomicOr(cluster_render.data[z_write_offset], z_write_bit);
		}
	}
#else
	if (!gl_HelperInvocation) {
		atomicOr(cluster_render.data[z_write_offset], z_write_bit);
	}
#endif
}
