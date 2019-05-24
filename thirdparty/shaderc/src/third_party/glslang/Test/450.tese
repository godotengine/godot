#version 450 core

in gl_PerVertex {
    float gl_CullDistance[3];
} gl_in[gl_MaxPatchVertices];

out gl_PerVertex {
    float gl_CullDistance[3];
};

void main()
{
    gl_CullDistance[2] = gl_in[1].gl_CullDistance[2];
}

layout(equal_spacing)           in float f1[];  // ERROR, must be standalone
layout(fractional_even_spacing) in float f2[];  // ERROR, must be standalone
layout(fractional_odd_spacing)  in float f3[];  // ERROR, must be standalone
layout(cw)                      in float f4[];  // ERROR, must be standalone
layout(ccw)                     in float f5[];  // ERROR, must be standalone
layout(point_mode)              in float f6[];  // ERROR, must be standalone
