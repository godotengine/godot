/* clang-format off */
[vertex]

layout(location = 0) in highp vec4 vertex_attrib;
/* clang-format on */

uniform vec2 offset;
uniform vec2 scale;

out vec2 uv_interp;

void main() {
	uv_interp = vertex_attrib.xy * 2.0 - 1.0;

	vec2 v = vertex_attrib.xy * scale + offset;
	gl_Position = vec4(v, 0.0, 1.0);
}

/* clang-format off */
[fragment]

uniform sampler2D source; //texunit:0
/* clang-format on */

uniform vec2 eye_center;
uniform float k1;
uniform float k2;
uniform float upscale;
uniform float aspect_ratio;

in vec2 uv_interp;

layout(location = 0) out vec4 frag_color;

void main() {
	vec2 coords = uv_interp;
	vec2 offset = coords - eye_center;

	// take aspect ratio into account
	offset.y /= aspect_ratio;

	// distort
	vec2 offset_sq = offset * offset;
	float radius_sq = offset_sq.x + offset_sq.y;
	float radius_s4 = radius_sq * radius_sq;
	float distortion_scale = 1.0 + (k1 * radius_sq) + (k2 * radius_s4);
	offset *= distortion_scale;

	// reapply aspect ratio
	offset.y *= aspect_ratio;

	// add our eye center back in
	coords = offset + eye_center;
	coords /= upscale;

	// and check our color
	if (coords.x < -1.0 || coords.y < -1.0 || coords.x > 1.0 || coords.y > 1.0) {
		frag_color = vec4(0.0, 0.0, 0.0, 1.0);
	} else {
		coords = (coords + vec2(1.0)) / vec2(2.0);
		frag_color = textureLod(source, coords, 0.0);
	}
}
