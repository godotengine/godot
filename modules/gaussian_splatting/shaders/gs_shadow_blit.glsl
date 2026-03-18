#[vertex]

#version 450

layout(location = 0) out vec2 uv_interp;

layout(push_constant, std430) uniform Params {
    vec4 uv_scale_offset; // xy = scale, zw = offset
    float invert_depth;   // 1.0 to invert (reversed depth)
    float _pad0;
    float _pad1;
    float _pad2;
} params;

void main() {
    vec2 base_arr[4] = vec2[](vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0));
    uv_interp = base_arr[gl_VertexIndex];

    vec2 vpos = uv_interp;
    gl_Position = vec4(vpos * 2.0 - 1.0, 0.0, 1.0);

    uv_interp = uv_interp * params.uv_scale_offset.xy + params.uv_scale_offset.zw;
}

#[fragment]

#version 450

layout(location = 0) in vec2 uv_interp;

layout(set = 0, binding = 0) uniform sampler2D source_depth;

layout(push_constant, std430) uniform Params {
    vec4 uv_scale_offset; // xy = scale, zw = offset
    float invert_depth;   // 1.0 to invert (reversed depth)
    float _pad0;
    float _pad1;
    float _pad2;
} params;

void main() {
    float depth = texture(source_depth, uv_interp).r;
    depth = clamp(depth, 0.0, 1.0);
    if (params.invert_depth > 0.5) {
        depth = 1.0 - depth;
    }
    gl_FragDepth = depth;
}
