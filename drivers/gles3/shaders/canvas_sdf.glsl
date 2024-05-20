/* clang-format off */
#[modes]

mode_load = #define MODE_LOAD
mode_load_shrink = #define MODE_LOAD_SHRINK
mode_process = #define MODE_PROCESS
mode_store = #define MODE_STORE
mode_store_shrink = #define MODE_STORE_SHRINK

#[specializations]

#[vertex]

layout(location = 0) in vec2 vertex_attrib;

/* clang-format on */

uniform ivec2 size;
uniform int stride;
uniform int shift;
uniform ivec2 base_size;

void main() {
	gl_Position = vec4(vertex_attrib, 1.0, 1.0);
}

/* clang-format off */
#[fragment]

#define SDF_MAX_LENGTH 16384.0

#if defined(MODE_LOAD) || defined(MODE_LOAD_SHRINK)
uniform lowp sampler2D src_pixels;//texunit:0
#else
uniform highp isampler2D src_process;//texunit:0
#endif

uniform	ivec2 size;
uniform	int stride;
uniform	int shift;
uniform	ivec2 base_size;

#if defined(MODE_LOAD) || defined(MODE_LOAD_SHRINK) || defined(MODE_PROCESS)
layout(location = 0) out ivec4 distance_field;
#else
layout(location = 0) out vec4 distance_field;
#endif

vec4 float_to_vec4(float p_float) {
    highp vec4 comp = fract(p_float * vec4(255.0 * 255.0 * 255.0, 255.0 * 255.0, 255.0, 1.0));
	comp -= comp.xxyz * vec4(0.0, 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0);
	return comp;
}

void main() {
	ivec2 pos = ivec2(gl_FragCoord.xy);

#ifdef MODE_LOAD

	bool solid = texelFetch(src_pixels, pos, 0).r > 0.5;
	distance_field = solid ? ivec4(ivec2(-32767), 0, 0) : ivec4(ivec2(32767), 0, 0);
#endif

#ifdef MODE_LOAD_SHRINK

	int s = 1 << shift;
	ivec2 base = pos << shift;
	ivec2 center = base + ivec2(shift);

	ivec2 rel = ivec2(32767);
	float d = 1e20;
	int found = 0;
	int solid_found = 0;
	for (int i = 0; i < s; i++) {
		for (int j = 0; j < s; j++) {
			ivec2 src_pos = base + ivec2(i, j);
			if (any(greaterThanEqual(src_pos, base_size))) {
				continue;
			}
			bool solid = texelFetch(src_pixels, src_pos, 0).r > 0.5;
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

	distance_field = ivec4(rel, 0, 0);
#endif

#ifdef MODE_PROCESS

	ivec2 base = pos << shift;
	ivec2 center = base + ivec2(shift);

	ivec2 rel = texelFetch(src_process, pos, 0).xy;

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
			ivec2 src_pos = pos + ofs_table[i] * stride;
			if (any(lessThan(src_pos, ivec2(0))) || any(greaterThanEqual(src_pos, size))) {
				continue;
			}
			ivec2 src_rel = texelFetch(src_process, src_pos, 0).xy;
			bool src_solid = src_rel.x < 0;
			if (src_solid) {
				src_rel = -src_rel - ivec2(1);
			}

			if (src_solid != solid) {
				src_rel = ivec2(src_pos << shift); //point to itself if of different type
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

	distance_field = ivec4(rel, 0, 0);
#endif

#ifdef MODE_STORE

	ivec2 rel = texelFetch(src_process, pos, 0).xy;

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
	distance_field = float_to_vec4(d*0.5+0.5);

#endif

#ifdef MODE_STORE_SHRINK

	ivec2 base = pos << shift;
	ivec2 center = base + ivec2(shift);

	ivec2 rel = texelFetch(src_process, pos, 0).xy;

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
	distance_field = float_to_vec4(d*0.5+0.5);

#endif
}
