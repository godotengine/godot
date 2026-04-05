/* clang-format off */
#[vertex]

#version 450

#VERSION_DEFINES

layout(location = 0) out vec2 uv_interp;
/* clang-format on */

void main() {
	vec2 base_arr[3] = vec2[](vec2(-1.0, -1.0), vec2(-1.0, 3.0), vec2(3.0, -1.0));
	gl_Position = vec4(base_arr[gl_VertexIndex], 0.0, 1.0);
	uv_interp = clamp(gl_Position.xy, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0; // saturate(x) * 2.0
}

/* clang-format off */
#[fragment]

#version 450

#VERSION_DEFINES

layout(location = 0) in vec2 uv_interp;

layout(set = 0, binding = 0) uniform sampler2DMS source_depth;

layout(push_constant, std430) uniform Params {
    ivec2 pad;
	int sample_count;
    int pad2;
}
params;

layout (location = 0) out float out_depth;

void main() {
    ivec2 pos = ivec2(gl_FragCoord.xy);

	float depth_avg = 0.0;
	for (int i = 0; i < params.sample_count; i++) {
		depth_avg += texelFetch(source_depth, pos, i).r;
	}
	depth_avg /= float(params.sample_count);
	out_depth = depth_avg;
}
