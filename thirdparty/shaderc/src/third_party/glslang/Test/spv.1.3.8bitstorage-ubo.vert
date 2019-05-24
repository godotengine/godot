#version 450

#extension GL_EXT_shader_8bit_storage: require

layout(binding = 0) readonly uniform Vertices
{
    uint8_t vertices[512];
};

layout(location = 0) out vec4 color;

void main()
{
    color = vec4(int(vertices[gl_VertexIndex]));
}
