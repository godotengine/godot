#version 310 es

layout(location = 7) in vec3 c;
layout(LocatioN = 3) in vec4 p;
layout(location = 9) in ivec2 aiv2;
out vec4 pos;
out vec3 color;
flat out int iout;

layout(row_major) uniform; // default is now row_major

layout(std140) uniform Transform { // layout of this block is std140
    mat4 M1; // row_major
    layout(column_major) mat4 M2; // column major
    mat3 N1; // row_major
    int iuin;
} tblock;

uniform T2 { // layout of this block is shared
    bool b;
    mat4 t2m;
};

layout(column_major) uniform T3 { // shared and column_major
    mat4 M3; // column_major
    layout(row_major) mat4 M4; // row major
    mat2x3 N2; // column_major
    layout(align=16, offset=2048) uvec3 uv3a[4];
};

in uint uiuin;

struct S {
    vec3 c;
    float f;
};

out S s;

void main()
{
    pos = p * (tblock.M1 + tblock.M2 + M4 + M3 + t2m);
    color = c * tblock.N1;
    iout = tblock.iuin + int(uiuin) + aiv2.y;
    s.c = c;
    s.f = p.x;
    if (N2[1] != vec3(1.0) || uv3a[2] != uvec3(5))
        ++s.c;
}
