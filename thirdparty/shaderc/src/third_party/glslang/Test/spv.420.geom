#version 420 core

layout(triangles) in;

in gl_PerVertex {
    float gl_PointSize;
} gl_in[];

out gl_PerVertex {
    float gl_PointSize;
};

layout(line_strip) out;
layout(max_vertices = 127) out;
layout(invocations = 4) in;

uniform sampler2D s2D;
in vec2 coord[];

int i;

void main()
{
    float p = gl_in[1].gl_PointSize;
    gl_PointSize = p;
    gl_ViewportIndex = 7;

    EmitStreamVertex(1);
    EndStreamPrimitive(0);
    EmitVertex();
    EndPrimitive();
    int id = gl_InvocationID;

    const ivec2 offsets[5] =
    {
        ivec2(0,1),
        ivec2(1,-2),
        ivec2(0,3),
        ivec2(-3,0),
        ivec2(2,1)
    };
    vec4 v = textureGatherOffset(s2D, coord[0], offsets[i].xy);
}
