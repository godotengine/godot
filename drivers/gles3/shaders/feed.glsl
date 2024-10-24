/* clang-format off */
#[modes]

mode_default =

#[specializations]

USE_EXTERNAL_SAMPLER = false

#[vertex]

layout(location = 0) in vec2 vertex_attrib;

out vec2 uv_interp;


void main() {
	uv_interp = vertex_attrib * 0.5 + 0.5;
	gl_Position = vec4(vertex_attrib, 1.0, 1.0);
}

/* clang-format off */
#[fragment]

layout(location = 0) out vec4 frag_color;
in vec2 uv_interp;

/* clang-format on */
#ifdef USE_EXTERNAL_SAMPLER
uniform samplerExternalOES sourceFeed; // texunit:0
#else
uniform sampler2D sourceFeed; // texunit:0
#endif

void main() {
	vec4 color = texture(sourceFeed, uv_interp);

	frag_color = color;
}
