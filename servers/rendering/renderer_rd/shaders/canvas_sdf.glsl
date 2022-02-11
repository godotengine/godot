#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(r8, set = 0, binding = 1) uniform restrict readonly image2D src_pixels;
layout(r16_snorm, set = 0, binding = 2) uniform restrict writeonly image2D dst_sdf;

layout(rg16i, set = 0, binding = 3) uniform restrict readonly iimage2D src_process;
layout(rg16i, set = 0, binding = 4) uniform restrict writeonly iimage2D dst_process;

layout(push_constant, std430) uniform Params {
	ivec2 size;
	int stride;
	int shift;
	ivec2 base_size;
	uvec2 pad;
}
params;

#define SDF_MAX_LENGTH 16384.0

void main() {
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pos, params.size))) { //too large, do nothing
		return;
	}

#ifdef MODE_LOAD

	bool solid = imageLoad(src_pixels, pos).r > 0.5;
	imageStore(dst_process, pos, solid ? ivec4(ivec2(-32767), 0, 0) : ivec4(ivec2(32767), 0, 0));
#endif

#ifdef MODE_LOAD_SHRINK

	int s = 1 << params.shift;
	ivec2 base = pos << params.shift;
	ivec2 center = base + ivec2(params.shift);

	ivec2 rel = ivec2(32767);
	float d = 1e20;
	int found = 0;
	int solid_found = 0;
	for (int i = 0; i < s; i++) {
		for (int j = 0; j < s; j++) {
			ivec2 src_pos = base + ivec2(i, j);
			if (any(greaterThanEqual(src_pos, params.base_size))) {
				continue;
			}
			bool solid = imageLoad(src_pixels, src_pos).r > 0.5;
			if (solid) {
				float dist = length(vec2(src_pos - center));
				if (dist < d) {
					d = dist;
					rel = src_pos;
				}
				solid_found++;
			}
			found++;
		}
	}

	if (solid_found == found) {
		//mark solid only if all are solid
		rel = ivec2(-32767);
	}

	imageStore(dst_process, pos, ivec4(rel, 0, 0));
#endif

#ifdef MODE_PROCESS

	ivec2 base = pos << params.shift;
	ivec2 center = base + ivec2(params.shift);

	ivec2 rel = imageLoad(src_process, pos).xy;

	bool solid = rel.x < 0;

	if (solid) {
		rel = -rel - ivec2(1);
	}

	if (center != rel) {
		//only process if it does not point to itself
		const int ofs_table_size = 8;
		const ivec2 ofs_table[ofs_table_size] = ivec2[](
				ivec2(-1, -1),
				ivec2(0, -1),
				ivec2(+1, -1),

				ivec2(-1, 0),
				ivec2(+1, 0),

				ivec2(-1, +1),
				ivec2(0, +1),
				ivec2(+1, +1));

		float dist = length(vec2(rel - center));
		for (int i = 0; i < ofs_table_size; i++) {
			ivec2 src_pos = pos + ofs_table[i] * params.stride;
			if (any(lessThan(src_pos, ivec2(0))) || any(greaterThanEqual(src_pos, params.size))) {
				continue;
			}
			ivec2 src_rel = imageLoad(src_process, src_pos).xy;
			bool src_solid = src_rel.x < 0;
			if (src_solid) {
				src_rel = -src_rel - ivec2(1);
			}

			if (src_solid != solid) {
				src_rel = ivec2(src_pos << params.shift); //point to itself if of different type
			}

			float src_dist = length(vec2(src_rel - center));
			if (src_dist < dist) {
				dist = src_dist;
				rel = src_rel;
			}
		}
	}

	if (solid) {
		rel = -rel - ivec2(1);
	}

	imageStore(dst_process, pos, ivec4(rel, 0, 0));
#endif

#ifdef MODE_STORE

	ivec2 rel = imageLoad(src_process, pos).xy;

	bool solid = rel.x < 0;

	if (solid) {
		rel = -rel - ivec2(1);
	}

	float d = length(vec2(rel - pos));

	if (solid) {
		d = -d;
	}

	d /= SDF_MAX_LENGTH;
	d = clamp(d, -1.0, 1.0);
	imageStore(dst_sdf, pos, vec4(d));

#endif

#ifdef MODE_STORE_SHRINK

	ivec2 base = pos << params.shift;
	ivec2 center = base + ivec2(params.shift);

	ivec2 rel = imageLoad(src_process, pos).xy;

	bool solid = rel.x < 0;

	if (solid) {
		rel = -rel - ivec2(1);
	}

	float d = length(vec2(rel - center));

	if (solid) {
		d = -d;
	}
	d /= SDF_MAX_LENGTH;
	d = clamp(d, -1.0, 1.0);
	imageStore(dst_sdf, pos, vec4(d));

#endif
}
