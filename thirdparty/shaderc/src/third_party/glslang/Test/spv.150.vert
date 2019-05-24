#version 150 core

in vec4 iv4;

in float ps;
in int ui;
uniform sampler2D s2D;

invariant gl_Position;

struct s1 {
    int a;
    int a2;
    vec4 b[3];
};

struct s2 {
    int c;
    s1 d[4];
};

out s2 s2out;

void main()
{
    gl_Position = iv4;
    gl_PointSize = ps;
    gl_ClipDistance[2] = iv4.x;
    int i;
    s2out.d[i].b[2].w = ps;

    // test non-implicit lod
    texture(s2D, vec2(0.5));
    textureProj(s2D, vec3(0.5));
    textureLod(s2D, vec2(0.5), 3.2);
}

out float gl_ClipDistance[4];
