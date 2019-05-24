#version 420 core

// testing input arrays without a gl_in[] block redeclaration, see 400.geom for with

int i;

layout(triangles) in;
in vec4 colorun[];
in vec4 color3[3];

void foo()
{
    gl_in.length();
    gl_in[1].gl_Position;
    gl_in.length();
    gl_in[i].gl_Position;   // should be sized to 3 by 'triangles'
}

in gl_PerVertex {  // ERROR, already used
    vec4 gl_Position;
} gl_in[];
