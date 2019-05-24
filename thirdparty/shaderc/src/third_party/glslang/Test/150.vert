#version 150 core

#ifndef GL_core_profile
#	error standard macro GL_core_profile not defined
#endif

in vec4 iv4;

uniform float ps;

invariant gl_Position;

void main()
{
    gl_Position = iv4;
    gl_PointSize = ps;
    gl_ClipDistance[2] = iv4.x;
    gl_ClipVertex = iv4;
}

out float gl_ClipDistance[4];

uniform foob {
    int a[];
};
int a[5]; // ERROR, resizing user-block member

#line 3000
#error line of this error should be 3001
