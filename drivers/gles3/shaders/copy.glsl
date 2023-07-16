/* clang-format off */
#[modes]

mode_default = #define MODE_SIMPLE_COPY
mode_copy_section = #define USE_COPY_SECTION \n#define MODE_SIMPLE_COPY
mode_gaussian_blur = #define MODE_GAUSSIAN_BLUR
mode_mipmap = #define MODE_MIPMAP
mode_simple_color = #define MODE_SIMPLE_COLOR \n#define USE_COPY_SECTION

#[specializations]

#[vertex]

layout(location = 0) in vec2 vertex_attrib;

out vec2 uv_interp;
/* clang-format on */

#if defined(USE_COPY_SECTION) || defined(MODE_GAUSSIAN_BLUR)
// Defined in 0-1 coords.
uniform highp vec4 copy_section;
#endif
#ifdef MODE_GAUSSIAN_BLUR
uniform highp vec4 source_section;
#endif

void main() {
	uv_interp = vertex_attrib * 0.5 + 0.5;
	gl_Position = vec4(vertex_attrib, 1.0, 1.0);

#if defined(USE_COPY_SECTION) || defined(MODE_GAUSSIAN_BLUR)
	gl_Position.xy = (copy_section.xy + uv_interp.xy * copy_section.zw) * 2.0 - 1.0;
#endif
#ifdef MODE_GAUSSIAN_BLUR
	uv_interp = source_section.xy + uv_interp * source_section.zw;
#endif
}

/* clang-format off */
#[fragment]

in vec2 uv_interp;
/* clang-format on */
#ifdef MODE_SIMPLE_COLOR
uniform vec4 color_in;
#endif

#ifdef MODE_GAUSSIAN_BLUR
// Defined in 0-1 coords.
uniform highp vec2 pixel_size;
#endif

uniform sampler2D source; // texunit:0

layout(location = 0) out vec4 frag_color;

void main() {
#ifdef MODE_SIMPLE_COPY
	vec4 color = texture(source, uv_interp);
	frag_color = color;
#endif

#ifdef MODE_SIMPLE_COLOR
	frag_color = color_in;
#endif

// Efficient box filter from Jimenez: http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
// Approximates a Gaussian in a single pass.
#ifdef MODE_GAUSSIAN_BLUR
	vec4 A = textureLod(source, uv_interp + pixel_size * vec2(-1.0, -1.0), 0.0);
	vec4 B = textureLod(source, uv_interp + pixel_size * vec2(0.0, -1.0), 0.0);
	vec4 C = textureLod(source, uv_interp + pixel_size * vec2(1.0, -1.0), 0.0);
	vec4 D = textureLod(source, uv_interp + pixel_size * vec2(-0.5, -0.5), 0.0);
	vec4 E = textureLod(source, uv_interp + pixel_size * vec2(0.5, -0.5), 0.0);
	vec4 F = textureLod(source, uv_interp + pixel_size * vec2(-1.0, 0.0), 0.0);
	vec4 G = textureLod(source, uv_interp, 0.0);
	vec4 H = textureLod(source, uv_interp + pixel_size * vec2(1.0, 0.0), 0.0);
	vec4 I = textureLod(source, uv_interp + pixel_size * vec2(-0.5, 0.5), 0.0);
	vec4 J = textureLod(source, uv_interp + pixel_size * vec2(0.5, 0.5), 0.0);
	vec4 K = textureLod(source, uv_interp + pixel_size * vec2(-1.0, 1.0), 0.0);
	vec4 L = textureLod(source, uv_interp + pixel_size * vec2(0.0, 1.0), 0.0);
	vec4 M = textureLod(source, uv_interp + pixel_size * vec2(1.0, 1.0), 0.0);

	float weight = 0.5 / 4.0;
	float lesser_weight = 0.125 / 4.0;

	frag_color = (D + E + I + J) * weight;
	frag_color += (A + B + G + F) * lesser_weight;
	frag_color += (B + C + H + G) * lesser_weight;
	frag_color += (F + G + L + K) * lesser_weight;
	frag_color += (G + H + M + L) * lesser_weight;
#endif
}
