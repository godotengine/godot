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

#ifdef USE_COPY_SECTION
uniform highp vec4 copy_section;
#endif

void main() {
	uv_interp = vertex_attrib * 0.5 + 0.5;
	gl_Position = vec4(vertex_attrib, 1.0, 1.0);

#ifdef USE_COPY_SECTION
	gl_Position.xy = (copy_section.xy + uv_interp.xy * copy_section.zw) * 2.0 - 1.0;
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
}
