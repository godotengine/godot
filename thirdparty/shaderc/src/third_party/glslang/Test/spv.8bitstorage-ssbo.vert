#version 450

#extension GL_EXT_shader_8bit_storage: require

layout(binding = 0) readonly buffer Vertices
{
    uint8_t vertices[];
};

layout(location = 0) out vec4 color;

void main()
{
    color = vec4(int(vertices[gl_VertexIndex]));
}
