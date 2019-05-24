#version 410 core

void main()
{
    gl_ViewportIndex = 7;
}

in gl_PerVertex {
    float gl_PointSize;
} myIn[];  // ERROR, can't redeclare a different name

in gl_PerVertex {
    float gl_PointSize;
} gl_myIn[];  // ERROR, can't redeclare a different name

in gl_PerVertex {
    float gl_PointSize;
} gl_in[];

in gl_PerVertex {
    float gl_PointSize;
} gl_in[];     // ERROR, can't do it again

out gl_PerVertex {
    float gl_PointSize;
};

void foo()
{
    float p = gl_in[1].gl_PointSize;  // use of redeclared
    gl_PointSize = p;                 // use of redeclared
    vec4 v = gl_in[1].gl_Position;    // ERROR, not included in the redeclaration
    gl_Position = vec4(1.0);          // ERROR, not included in the redeclaration
}

float foo5()
{
    return 4;  // implicit conversion of return type
}
