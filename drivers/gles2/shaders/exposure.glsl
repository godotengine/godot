/* clang-format off */
[vertex]

layout(location = 0) in highp vec4 vertex_attrib;
/* clang-format on */

void main() {
	gl_Position = vertex_attrib;
}

/* clang-format off */
[fragment]

uniform highp sampler2D source_exposure; //texunit:0
/* clang-format on */

#ifdef EXPOSURE_BEGIN

uniform highp ivec2 source_render_size;
uniform highp ivec2 target_size;

#endif

#ifdef EXPOSURE_END

uniform highp sampler2D prev_exposure; //texunit:1
uniform highp float exposure_adjust;
uniform highp float min_luminance;
uniform highp float max_luminance;

#endif

layout(location = 0) out highp float exposure;

void main() {
#ifdef EXPOSURE_BEGIN

	ivec2 src_pos = ivec2(gl_FragCoord.xy) * source_render_size / target_size;

#if 1
	//more precise and expensive, but less jittery
	ivec2 next_pos = ivec2(gl_FragCoord.xy + ivec2(1)) * source_render_size / target_size;
	next_pos = max(next_pos, src_pos + ivec2(1)); //so it at least reads one pixel
	highp vec3 source_color = vec3(0.0);
	for (int i = src_pos.x; i < next_pos.x; i++) {
		for (int j = src_pos.y; j < next_pos.y; j++) {
			source_color += texelFetch(source_exposure, ivec2(i, j), 0).rgb;
		}
	}

	source_color /= float((next_pos.x - src_pos.x) * (next_pos.y - src_pos.y));
#else
	highp vec3 source_color = texelFetch(source_exposure, src_pos, 0).rgb;

#endif

	exposure = max(source_color.r, max(source_color.g, source_color.b));

#else

	ivec2 coord = ivec2(gl_FragCoord.xy);
	exposure = texelFetch(source_exposure, coord * 3 + ivec2(0, 0), 0).r;
	exposure += texelFetch(source_exposure, coord * 3 + ivec2(1, 0), 0).r;
	exposure += texelFetch(source_exposure, coord * 3 + ivec2(2, 0), 0).r;
	exposure += texelFetch(source_exposure, coord * 3 + ivec2(0, 1), 0).r;
	exposure += texelFetch(source_exposure, coord * 3 + ivec2(1, 1), 0).r;
	exposure += texelFetch(source_exposure, coord * 3 + ivec2(2, 1), 0).r;
	exposure += texelFetch(source_exposure, coord * 3 + ivec2(0, 2), 0).r;
	exposure += texelFetch(source_exposure, coord * 3 + ivec2(1, 2), 0).r;
	exposure += texelFetch(source_exposure, coord * 3 + ivec2(2, 2), 0).r;
	exposure *= (1.0 / 9.0);

#ifdef EXPOSURE_END

#ifdef EXPOSURE_FORCE_SET
	//will stay as is
#else
	highp float prev_lum = texelFetch(prev_exposure, ivec2(0, 0), 0).r; //1 pixel previous exposure
	exposure = clamp(prev_lum + (exposure - prev_lum) * exposure_adjust, min_luminance, max_luminance);

#endif //EXPOSURE_FORCE_SET

#endif //EXPOSURE_END

#endif //EXPOSURE_BEGIN
}
