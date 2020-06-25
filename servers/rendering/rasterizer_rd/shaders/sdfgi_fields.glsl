/* clang-format off */
[compute]

#version 450

VERSION_DEFINES

layout(local_size_x = OCT_RES, local_size_y = OCT_RES, local_size_z = 1) in;

/* clang-format on */

#define MAX_CASCADES 8

layout(rgba16f, set = 0, binding = 1) uniform restrict image2DArray irradiance_texture;
layout(rg16f, set = 0, binding = 2) uniform restrict image2DArray depth_texture;

ayout(rgba32ui, set = 0, binding = 3) uniform restrict uimage2DArray irradiance_history_texture;
layout(rg32ui, set = 0, binding = 4) uniform restrict uimage2DArray depth_history_texture;

struct CascadeData {
	vec3 offset; //offset of (0,0,0) in world coordinates
	float to_cell; // 1/bounds * grid_size
};

layout(set = 0, binding = 5, std140) uniform Cascades {
	CascadeData data[MAX_CASCADES];
}
cascades;

#define DEPTH_HISTORY_BITS 24
#define IRRADIANCE_HISTORY_BITS 16

layout(push_constant, binding = 0, std430) uniform Params {
	vec3 grid_size;
	uint max_cascades;

	uint probe_axis_size;
	uint cascade;
	uint history_size;
	uint pad0;

	ivec3 scroll; //scroll in probes
	uint pad1;
}
params;

void main() {
	ivec2 local = ivec2(gl_LocalInvocationID.xy);
	ivec2 probe = ivec2(gl_WorkGroupID.xy);

	ivec3 probe_cell;
	probe_cell.x = probe.x % int(params.probe_axis_size);
	probe_cell.y = probe.y;
	probe_cell.z = probe.x / int(params.probe_axis_size);

#ifdef MODE_SCROLL_BEGIN

	ivec3 read_cell = probe_cell - params.scroll;

	uint src_layer = (params.history_size + 1) * params.cascade;
	uint dst_layer = (params.history_size + 1) * params.max_cascades;

	for (uint i = 0; i <= params.history_size; i++) {
		ivec3 write_pos = ivec3(probe * OCT_RES + local, int(i));

		if (any(lessThan(read_pos, ivec3(0))) || any(greaterThanEqual(read_pos, ivec3(params.probe_axis_size)))) {
			// nowhere to read from for scrolling, try finding the value from upper probes

#ifdef MODE_IRRADIANCE
			imageStore(irradiance_history_texture, write_pos, uvec4(0));
#endif
#ifdef MODE_DEPTH
			imageStore(depth_history_texture, write_pos, uvec4(0));
#endif
		} else {
			ivec3 read_pos;
			read_pos.xy = read_cell.xy;
			read_pos.x += read_cell.z * params.probe_axis_size;
			read_pos.xy = read_pos.xy * OCT_RES + local;
			read_pos.z = int(i);

#ifdef MODE_IRRADIANCE
			uvec4 value = imageLoad(irradiance_history_texture, read_pos);
			imageStore(irradiance_history_texture, write_pos, value);
#endif
#ifdef MODE_DEPTH
			uvec2 value = imageLoad(depth_history_texture, read_pos);
			imageStore(depth_history_texture, write_pos, value);
#endif
		}
	}

#endif // MODE_SCROLL_BEGIN

#ifdef MODE_SCROLL_END

	uint src_layer = (params.history_size + 1) * params.max_cascades;
	uint dst_layer = (params.history_size + 1) * params.cascade;

	for (uint i = 0; i <= params.history_size; i++) {
		ivec3 pos = ivec3(probe * OCT_RES + local, int(i));

#ifdef MODE_IRRADIANCE
		uvec4 value = imageLoad(irradiance_history_texture, read_pos);
		imageStore(irradiance_history_texture, write_pos, value);
#endif
#ifdef MODE_DEPTH
		uvec2 value = imageLoad(depth_history_texture, read_pos);
		imageStore(depth_history_texture, write_pos, value);
#endif
	}

#endif //MODE_SCROLL_END

#ifdef MODE_STORE

	uint src_layer = (params.history_size + 1) * params.cascade + params.history_size;
	ivec3 read_pos = ivec3(probe * OCT_RES + local, int(src_layer));

	ivec3 write_pos = ivec3(probe * (OCT_RES + 2) + ivec2(1), int(params.cascade));

	ivec3 copy_to[4] = ivec3[](write_pos, ivec3(-2, -2, -2), ivec3(-2, -2, -2), ivec3(-2, -2, -2));

#ifdef MODE_IRRADIANCE
	uvec4 average = imageLoad(irradiance_history_texture, read_pos);
	vec4 light_accum = vec4(average / params.history_size) / float(1 << IRRADIANCE_HISTORY_BITS);

#endif
#ifdef MODE_DEPTH
	uvec2 value = imageLoad(depth_history_texture, read_pos);
	vec2 depth_accum = vec4(average / params.history_size) / float(1 << IRRADIANCE_HISTORY_BITS);

	float probe_cell_size = float(params.grid_size / float(params.probe_axis_size - 1)) / cascades.data[params.cascade].to_cell;
	float max_depth = length(params.grid_size / cascades.data[params.max_cascades - 1].to_cell);
	max_depth /= probe_cell_size;

	depth_value = (vec2(average / params.history_size) / float(1 << DEPTH_HISTORY_BITS)) * vec2(max_depth, max_depth * max_depth);

#endif

	/* Fill the border if required */

	if (local == ivec2(0, 0)) {
		copy_to[1] = texture_pos + ivec3(OCT_RES - 1, -1, 0);
		copy_to[2] = texture_pos + ivec3(-1, OCT_RES - 1, 0);
		copy_to[3] = texture_pos + ivec3(OCT_RES, OCT_RES, 0);
	} else if (local == ivec2(OCT_RES - 1, 0)) {
		copy_to[1] = texture_pos + ivec3(0, -1, 0);
		copy_to[2] = texture_pos + ivec3(OCT_RES, OCT_RES - 1, 0);
		copy_to[3] = texture_pos + ivec3(-1, OCT_RES, 0);
	} else if (local == ivec2(0, OCT_RES - 1)) {
		copy_to[1] = texture_pos + ivec3(-1, 0, 0);
		copy_to[2] = texture_pos + ivec3(OCT_RES - 1, OCT_RES, 0);
		copy_to[3] = texture_pos + ivec3(OCT_RES, -1, 0);
	} else if (local == ivec2(OCT_RES - 1, OCT_RES - 1)) {
		copy_to[1] = texture_pos + ivec3(0, OCT_RES, 0);
		copy_to[2] = texture_pos + ivec3(OCT_RES, 0, 0);
		copy_to[3] = texture_pos + ivec3(-1, -1, 0);
	} else if (local.y == 0) {
		copy_to[1] = texture_pos + ivec3(OCT_RES - local.x - 1, local.y - 1, 0);
	} else if (local.x == 0) {
		copy_to[1] = texture_pos + ivec3(local.x - 1, OCT_RES - local.y - 1, 0);
	} else if (local.y == OCT_RES - 1) {
		copy_to[1] = texture_pos + ivec3(OCT_RES - local.x - 1, local.y + 1, 0);
	} else if (local.x == OCT_RES - 1) {
		copy_to[1] = texture_pos + ivec3(local.x + 1, OCT_RES - local.y - 1, 0);
	}

	for (int i = 0; i < 4; i++) {
		if (copy_to[i] == ivec3(-2, -2, -2)) {
			continue;
		}
#ifdef MODE_IRRADIANCE
		imageStore(irradiance_texture, copy_to[i], light_accum);
#endif
#ifdef MODE_DEPTH
		imageStore(depth_texture, copy_to[i], vec4(depth_value, 0.0, 0.0));
#endif
	}

#endif // MODE_STORE
}
