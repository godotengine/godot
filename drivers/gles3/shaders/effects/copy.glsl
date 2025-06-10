/* clang-format off */
#[modes]

mode_default = #define MODE_SIMPLE_COPY
mode_copy_section = #define USE_COPY_SECTION \n#define MODE_SIMPLE_COPY
mode_copy_section_source = #define USE_COPY_SECTION \n#define MODE_SIMPLE_COPY \n#define MODE_COPY_FROM
mode_copy_section_3d = #define USE_COPY_SECTION \n#define MODE_SIMPLE_COPY \n#define USE_TEXTURE_3D
mode_copy_section_2d_array = #define USE_COPY_SECTION \n#define MODE_SIMPLE_COPY \n#define USE_TEXTURE_2D_ARRAY
mode_screen = #define MODE_SIMPLE_COPY \n#define MODE_MULTIPLY
mode_gaussian_blur = #define MODE_GAUSSIAN_BLUR
mode_mipmap = #define MODE_MIPMAP
mode_simple_color = #define MODE_SIMPLE_COLOR \n#define USE_COPY_SECTION
mode_cube_to_octahedral = #define CUBE_TO_OCTAHEDRAL \n#define USE_COPY_SECTION
mode_cube_to_panorama = #define CUBE_TO_PANORAMA

#[specializations]

#[vertex]

layout(location = 0) in vec2 vertex_attrib;

out vec2 uv_interp;
/* clang-format on */

#if defined(USE_COPY_SECTION) || defined(MODE_GAUSSIAN_BLUR)
// Defined in 0-1 coords.
uniform highp vec4 copy_section;
#endif
#if defined(MODE_GAUSSIAN_BLUR) || defined(MODE_COPY_FROM)
uniform highp vec4 source_section;
#endif

void main() {
	uv_interp = vertex_attrib * 0.5 + 0.5;
	gl_Position = vec4(vertex_attrib, 1.0, 1.0);

#if defined(USE_COPY_SECTION) || defined(MODE_GAUSSIAN_BLUR)
	gl_Position.xy = (copy_section.xy + uv_interp.xy * copy_section.zw) * 2.0 - 1.0;
#endif
#if defined(MODE_GAUSSIAN_BLUR) || defined(MODE_COPY_FROM)
	uv_interp = source_section.xy + uv_interp * source_section.zw;
#endif
}

/* clang-format off */
#[fragment]

in vec2 uv_interp;
/* clang-format on */
#if defined(USE_TEXTURE_3D) || defined(USE_TEXTURE_2D_ARRAY)
uniform float layer;
uniform float lod;
#endif

#ifdef MODE_SIMPLE_COLOR
uniform vec4 color_in;
#endif

#ifdef MODE_MULTIPLY
uniform float multiply;
#endif

#ifdef MODE_GAUSSIAN_BLUR
// Defined in 0-1 coords.
uniform highp vec2 pixel_size;
#endif

#ifdef CUBE_TO_OCTAHEDRAL
vec3 oct_to_vec3(vec2 e) {
	vec3 v = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
	float t = max(-v.z, 0.0);
	v.xy += t * -sign(v.xy);
	return normalize(v);
}
#endif

#ifdef CUBE_TO_PANORAMA
uniform lowp float mip_level;
#endif

#if defined(CUBE_TO_OCTAHEDRAL) || defined(CUBE_TO_PANORAMA)
uniform samplerCube source_cube; // texunit:0

#else // ~(defined(CUBE_TO_OCTAHEDRAL) || defined(CUBE_TO_PANORAMA))

#if defined(USE_TEXTURE_3D)
uniform sampler3D source_3d; // texunit:0
#elif defined(USE_TEXTURE_2D_ARRAY)
uniform sampler2DArray source_2d_array; // texunit:0
#else
uniform sampler2D source; // texunit:0
#endif

#endif // !(defined(CUBE_TO_OCTAHEDRAL) || defined(CUBE_TO_PANORAMA))

layout(location = 0) out vec4 frag_color;

// This expects 0-1 range input, outside that range it behaves poorly.
vec3 srgb_to_linear(vec3 color) {
	// Approximation from http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
	return color * (color * (color * 0.305306011 + 0.682171111) + 0.012522878);
}

void main() {
#ifdef MODE_SIMPLE_COPY

#ifdef USE_TEXTURE_3D
	vec4 color = textureLod(source_3d, vec3(uv_interp, layer), lod);
#elif defined(USE_TEXTURE_2D_ARRAY)
	vec4 color = textureLod(source_2d_array, vec3(uv_interp, layer), lod);
#else
	vec4 color = texture(source, uv_interp);
#endif // USE_TEXTURE_3D

#ifdef MODE_MULTIPLY
	color *= multiply;
#endif // MODE_MULTIPLY

	frag_color = color;
#endif // MODE_SIMPLE_COPY

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

#ifdef CUBE_TO_OCTAHEDRAL
	// Treat the UV coordinates as 0-1 encoded octahedral coordinates.
	vec3 dir = oct_to_vec3(uv_interp * 2.0 - 1.0);
	frag_color = texture(source_cube, dir);

#endif

#ifdef CUBE_TO_PANORAMA

	const float PI = 3.14159265359;

	float phi = uv_interp.x * 2.0 * PI;
	float theta = uv_interp.y * PI;

	vec3 normal;
	normal.x = sin(phi) * sin(theta) * -1.0;
	normal.y = cos(theta);
	normal.z = cos(phi) * sin(theta) * -1.0;

	vec3 color = srgb_to_linear(textureLod(source_cube, normal, mip_level).rgb);
	frag_color = vec4(color, 1.0);

#endif
}
