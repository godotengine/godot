#version 450 core

in gl_PerVertex {
    float gl_CullDistance[3];
} gl_in[];

out gl_PerVertex {
    float gl_CullDistance[3];
};

layout(triangles) in;

void main()
{
    gl_in[3].gl_Position; // ERROR, out of range
    gl_CullDistance[2] = gl_in[1].gl_CullDistance[2];
}

layout(points) in float f[3];  // ERROR, must be standalone
