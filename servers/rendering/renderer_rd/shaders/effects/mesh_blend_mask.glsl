#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rg32ui, set = 0, binding = 0) uniform readonly uimage2D vb_vis;
layout(rg16f, set = 0, binding = 1) uniform readonly image2D vb_aux;
layout(rg16f, set = 0, binding = 2) uniform writeonly image2D mesh_mask;
layout(rg32ui, set = 0, binding = 3) uniform writeonly uimage2D mesh_edges;
layout(r32f, set = 0, binding = 4) uniform readonly image2D mesh_depth;

layout(push_constant, std430) uniform Params {
	ivec2 resolution;
	float depth_tolerance;
	int require_pair;
}
params;

const int THREADCOUNT = 8;
const int TILE_BORDER = 1;
const int TILE_SIZE = THREADCOUNT + TILE_BORDER * 2;

shared vec2 cached_mask[TILE_SIZE * TILE_SIZE];
shared float cached_depth[TILE_SIZE * TILE_SIZE];

int coord_to_index(ivec2 coord) {
	coord = clamp(coord, ivec2(0), ivec2(TILE_SIZE - 1));
	return coord.y * TILE_SIZE + coord.x;
}

const ivec2 neighbor_offsets[8] = ivec2[8](
		ivec2(-1, 0),
		ivec2(1, 0),
		ivec2(0, -1),
		ivec2(0, 1),
		ivec2(-1, -1),
		ivec2(-1, 1),
		ivec2(1, -1),
		ivec2(1, 1));

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 resolution = params.resolution;
	if (any(greaterThanEqual(pixel, resolution))) {
		return;
	}

	ivec2 tile_origin = ivec2(gl_WorkGroupID.xy) * THREADCOUNT - ivec2(TILE_BORDER);
	ivec2 local_id = ivec2(gl_LocalInvocationID.xy);

	for (int y = local_id.y; y < TILE_SIZE; y += THREADCOUNT) {
		for (int x = local_id.x; x < TILE_SIZE; x += THREADCOUNT) {
			ivec2 sample_pixel = clamp(tile_origin + ivec2(x, y), ivec2(0), resolution - ivec2(1));
			vec2 value = vec2(0.0);

	uvec4 ids = imageLoad(vb_vis, sample_pixel);
	vec2 aux = imageLoad(vb_aux, sample_pixel).xy;
	float depth_value = imageLoad(mesh_depth, sample_pixel).x;

	if (ids.x != 0u) {
		float raw_weight = aux.x;
		float weight = min(raw_weight, 1.0);
		float id_quantized = floor(aux.y * 255.0 + 0.5) / 255.0;
		value = vec2(id_quantized, weight);
	}

			int cache_idx = coord_to_index(ivec2(x, y));
			cached_mask[cache_idx] = value;
			cached_depth[cache_idx] = depth_value;
		}
	}

	barrier();

	ivec2 local_pixel = local_id + ivec2(TILE_BORDER);
	int current_index = coord_to_index(local_pixel);
	vec2 current = cached_mask[current_index];
	float current_depth = cached_depth[current_index];
	imageStore(mesh_mask, pixel, vec4(current, 0.0, 0.0));

	float current_id = current.x;
	if (current_id <= 0.0) {
		imageStore(mesh_edges, pixel, uvec4(0u));
		return;
	}

	uvec4 edge_store = uvec4(0u);
	for (int i = 0; i < 8; i++) {
		int neighbor_idx = coord_to_index(local_pixel + neighbor_offsets[i]);
		vec2 neighbor = cached_mask[neighbor_idx];
		float neighbor_id = neighbor.x;
		if (neighbor_id <= 0.0 || neighbor_id == current_id) {
			continue;
		}

		float neighbor_depth = cached_depth[neighbor_idx];
		if (abs(neighbor_depth - current_depth) > params.depth_tolerance) {
			continue;
		}

		if (params.require_pair != 0 && neighbor.y <= 0.0) {
			continue;
		}

		ivec2 neighbor_pixel = clamp(pixel + neighbor_offsets[i], ivec2(0), resolution - ivec2(1));
		edge_store.xy = uvec2(neighbor_pixel);
		break;
	}

	imageStore(mesh_edges, pixel, edge_store);
}
