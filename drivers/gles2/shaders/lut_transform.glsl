/* clang-format off */
[vertex]

attribute highp vec4 vertex_attrib; // attrib:0
/* clang-format on */

attribute vec2 uv_in; // attrib:4

uniform highp vec4 screen_rect;

varying vec2 uv_interp;

void main() {
	uv_interp = uv_in;
	gl_Position = vec4(screen_rect.xy + vertex_attrib.xy * screen_rect.zw, vertex_attrib.zw);
}

/* clang-format off */
[fragment]

uniform vec2 texel_count;
uniform vec2 chunk_count;

uniform sampler2D source; //texunit:0
uniform sampler2D lut; //texunit:1
/* clang-format on */

varying vec2 uv_interp;

vec2 flatten_3d(vec2 chunk_size, vec2 texel_size, vec2 xy, float z_scaled) {
	float row = floor(z_scaled / chunk_count.x);
	float col = mod(z_scaled, chunk_count.x);
	return chunk_size * vec2(col, row) + (chunk_size - texel_size) * xy + texel_size * 0.5;
}

void main() {
	vec4 color = texture2D(source, uv_interp);
	vec3 clamped_rgb = clamp(color.rgb, 0.0, 1.0);

	vec2 texel_size = 1.0 / texel_count;
	vec2 chunk_size = 1.0 / chunk_count;

	float z_scaled = clamped_rgb.b * (chunk_count.x * chunk_count.y - 1.0);
	vec4 low = texture2D(lut, flatten_3d(chunk_size, texel_size, clamped_rgb.rg, floor(z_scaled)));
	vec4 high = texture2D(lut, flatten_3d(chunk_size, texel_size, clamped_rgb.rg, ceil(z_scaled)));

	gl_FragColor = vec4(mix(low.rgb, high.rgb, mod(z_scaled, 1.0)), color.a);
}
