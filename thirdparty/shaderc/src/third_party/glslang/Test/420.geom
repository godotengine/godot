#version 420 core

// testing input arrays without a gl_in[] block redeclaration, see 400.geom for with

int i;

void foo()
{
    gl_in.length();  // ERROR
    gl_in[1].gl_Position;
    gl_in[i].gl_Position;  // ERROR
}

layout(triangles) in;

in vec4 color3[3];

void foo3()
{
    gl_in.length();
    gl_in[i].gl_Position;
    color3.length();
}

uniform sampler2D s2D;
in vec2 coord[];
uniform vec4 v4;

void foo4()
{
    const ivec2 offsets[5] =
    {
        ivec2(0,1),
        ivec2(1,-2),
        ivec2(0,3),
        ivec2(-3,0),
        ivec2(2,1)
    };

    vec4 v = textureGatherOffset(s2D, coord[0], offsets[i].xy);

    offsets[i].xy = ivec2(3);  // ERROR
    v4.x = 3.2;                // ERROR
    v4.xy;   // should have non-uniform type
}

out gl_PerVertex {
    float gl_PointSize[1];  // ERROR, adding array
    float gl_ClipDistance;  // ERROR, removing array
};

float foo5()
{
    return i;  // implicit conversion of return type
}
