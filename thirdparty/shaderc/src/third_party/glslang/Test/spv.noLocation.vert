#version 450

layout(location = 1) in vec4 in1;
in vec4 in2;                        // ERROR
layout(location = 3) in vec4 in3;

layout(location = 1) out vec4 out1;
out vec4 out2;                      // ERROR
layout(location = 3) out vec4 out3;

layout(location = 10) out inb1 { 
    vec4 a;
    vec4 b;
} inbi1;
out inb2 { 
    layout(location = 12) vec4 a;
    layout(location = 13) vec4 b;
} inbi2;
out inb3 {                          // ERROR
    vec4 a;
    vec4 b;
} inbi3;

layout(location = 14) out struct S1 { vec4 a; } s1;
out struct S2 { vec4 a; } s2;       // ERROR

struct SS { int a; };
out layout(location = 15) SS ss1;
out SS ss2;                         // ERROR

out gl_PerVertex {
    vec4 gl_Position;
    float gl_ClipDistance[2];
};

void main()
{
    gl_ClipDistance[0] = 1.0;
}
