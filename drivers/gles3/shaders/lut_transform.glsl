/* clang-format off */
[vertex]

layout(location = 0) in highp vec4 vertex_attrib;
/* clang-format on */

layout(location = 4) in vec2 uv_in;

uniform highp vec4 screen_rect;

out vec2 uv_interp;

void main() {
	uv_interp = uv_in;
	gl_Position = vec4(screen_rect.xy + vertex_attrib.xy * screen_rect.zw, vertex_attrib.zw);
}

/* clang-format off */
[fragment]

uniform vec3 texel_count;

uniform sampler2D source; //texunit:0
uniform sampler3D lut; //texunit:1
/* clang-format on */

in vec2 uv_interp;

layout(location = 0) out vec4 frag_color;

void main() {
	vec4 color = texture(source, uv_interp);
	vec3 coords = clamp(color.rgb, 0.0, 1.0) * (1.0 - 1.0 / texel_count) + 0.5 / texel_count;
	frag_color = texture(lut, coords);
}
