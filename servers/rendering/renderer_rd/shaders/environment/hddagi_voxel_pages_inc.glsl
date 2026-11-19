#ifndef HDDAGI_VOXEL_PAGES_INC
#define HDDAGI_VOXEL_PAGES_INC

layout(set = 0, binding = HDDAGI_PAGE_BINDING, std430) restrict readonly buffer VoxelPages {
	uint pages[];
}
hddagi_voxel_pages;

layout(set = 0, binding = HDDAGI_DATA_BINDING, std430) restrict
#ifndef HDDAGI_WRITE_VOXELS
		readonly
#endif
		buffer VoxelData {
	uvec2 cells[];
}
hddagi_voxel_data;

const uint HDDAGI_EMPTY_NEIGHBOURS = 0x0ff80000u;

int hddagi_voxel_index(ivec3 position) {
	ivec3 brick = position >> 3;
	uint page = hddagi_voxel_pages.pages[brick.x + 16 * (brick.z + 16 * brick.y)];
	ivec3 cell = position & 7;
	return (int(page) - 1) * 512 + cell.x + 8 * (cell.y + 8 * cell.z);
}

uint hddagi_read_neighbours(ivec3 position) {
	int index = hddagi_voxel_index(position);
	return (index < 0 ? 0u : hddagi_voxel_data.cells[index].y) ^ HDDAGI_EMPTY_NEIGHBOURS;
}

vec3 hddagi_read_light(ivec3 position) {
	int index = hddagi_voxel_index(position);
	uint packed_light = index < 0 ? 0u : hddagi_voxel_data.cells[index].x;
	return vec3((uvec3(packed_light) >> uvec3(0, 9, 18)) & 511u) * exp2(float(packed_light >> 27u) - 24.0);
}

#endif
