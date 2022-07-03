#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

#define MAX_DISTANCE 100000.0

#define NO_CHILDREN 0xFFFFFFFF

struct CellChildren {
	uint children[8];
};

layout(set = 0, binding = 1, std430) buffer CellChildrenBuffer {
	CellChildren data[];
}
cell_children;

struct CellData {
	uint position; // xyz 10 bits
	uint albedo; //rgb albedo
	uint emission; //rgb normalized with e as multiplier
	uint normal; //RGB normal encoded
};

layout(set = 0, binding = 2, std430) buffer CellDataBuffer {
	CellData data[];
}
cell_data;

layout(r8ui, set = 0, binding = 3) uniform restrict writeonly uimage3D sdf_tex;

layout(push_constant, std430) uniform Params {
	uint offset;
	uint end;
	uint pad0;
	uint pad1;
}
params;

void main() {
	vec3 pos = vec3(gl_GlobalInvocationID);
	float closest_dist = MAX_DISTANCE;

	for (uint i = params.offset; i < params.end; i++) {
		vec3 posu = vec3(uvec3(cell_data.data[i].position & 0x7FF, (cell_data.data[i].position >> 11) & 0x3FF, cell_data.data[i].position >> 21));
		float dist = length(pos - posu);
		if (dist < closest_dist) {
			closest_dist = dist;
		}
	}

	uint dist_8;

	if (closest_dist < 0.0001) { // same cell
		dist_8 = 0; //equals to -1
	} else {
		dist_8 = clamp(uint(closest_dist), 0, 254) + 1; //conservative, 0 is 1, so <1 is considered solid
	}

	imageStore(sdf_tex, ivec3(gl_GlobalInvocationID), uvec4(dist_8));
	//imageStore(sdf_tex,pos,uvec4(pos*2,0));
}

#if 0
layout(push_constant, std430) uniform Params {
	ivec3 limits;
	uint stack_size;
}
params;

float distance_to_aabb(ivec3 pos, ivec3 aabb_pos, ivec3 aabb_size) {
	vec3 delta = vec3(max(ivec3(0), max(aabb_pos - pos, pos - (aabb_pos + aabb_size - ivec3(1)))));
	return length(delta);
}

void main() {
	ivec3 pos = ivec3(gl_GlobalInvocationID);

	uint stack[10] = uint[](0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	uint stack_indices[10] = uint[](0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	ivec3 stack_positions[10] = ivec3[](ivec3(0), ivec3(0), ivec3(0), ivec3(0), ivec3(0), ivec3(0), ivec3(0), ivec3(0), ivec3(0), ivec3(0));

	const uint cell_orders[8] = uint[](
			0x11f58d1,
			0xe2e70a,
			0xd47463,
			0xbb829c,
			0x8d11f5,
			0x70ae2e,
			0x463d47,
			0x29cbb8);

	bool cell_found = false;
	bool cell_found_exact = false;
	ivec3 closest_cell_pos;
	float closest_distance = MAX_DISTANCE;
	int stack_pos = 0;

	while (true) {
		uint index = stack_indices[stack_pos] >> 24;

		if (index == 8) {
			//go up
			if (stack_pos == 0) {
				break; //done going through octree
			}
			stack_pos--;
			continue;
		}

		stack_indices[stack_pos] = (stack_indices[stack_pos] & ((1 << 24) - 1)) | ((index + 1) << 24);

		uint cell_index = (stack_indices[stack_pos] >> (index * 3)) & 0x7;
		uint child_cell = cell_children.data[stack[stack_pos]].children[cell_index];

		if (child_cell == NO_CHILDREN) {
			continue;
		}

		ivec3 child_cell_size = params.limits >> (stack_pos + 1);
		ivec3 child_cell_pos = stack_positions[stack_pos];

		child_cell_pos += mix(ivec3(0), child_cell_size, bvec3(uvec3(index & 1, index & 2, index & 4) != uvec3(0)));

		bool is_leaf = stack_pos == (params.stack_size - 2);

		if (child_cell_pos == pos && is_leaf) {
			//we may actually end up in the exact cell.
			//if this happens, just abort
			cell_found_exact = true;
			break;
		}

		if (cell_found) {
			//discard by distance
			float distance = distance_to_aabb(pos, child_cell_pos, child_cell_size);
			if (distance >= closest_distance) {
				continue; //pointless, just test next child
			} else if (is_leaf) {
				//closer than what we have AND end of stack, save and continue
				closest_cell_pos = child_cell_pos;
				closest_distance = distance;
				continue;
			}
		} else if (is_leaf) {
			//first solid cell we find, save and continue
			closest_distance = distance_to_aabb(pos, child_cell_pos, child_cell_size);
			closest_cell_pos = child_cell_pos;
			cell_found = true;
			continue;
		}

		bvec3 direction = greaterThan((pos - (child_cell_pos + (child_cell_size >> 1))), ivec3(0));
		uint cell_order = 0;
		cell_order |= mix(0, 1, direction.x);
		cell_order |= mix(0, 2, direction.y);
		cell_order |= mix(0, 4, direction.z);

		stack[stack_pos + 1] = child_cell;
		stack_indices[stack_pos + 1] = cell_orders[cell_order]; //start counting
		stack_positions[stack_pos + 1] = child_cell_pos;
		stack_pos++; //go up stack
	}

	uint dist_8;

	if (cell_found_exact) {
		dist_8 = 0; //equals to -1
	} else {
		float closest_distance = length(vec3(pos - closest_cell_pos));
		dist_8 = clamp(uint(closest_distance), 0, 254) + 1; //conservative, 0 is 1, so <1 is considered solid
	}

	imageStore(sdf_tex, pos, uvec4(dist_8));
}
#endif
