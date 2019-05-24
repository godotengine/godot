#version 300 es

struct s { vec4 v; };

layout(location = 7) in vec3 c;
layout(LocatioN = 3) in vec4 p;
layout(LocatioN = 9) in vec4 q[4]; // ERROR, no array
layout(LocatioN = 10) in s r[4];   // ERROR, no struct, ERROR, location overlap
out vec4 pos;
out vec3 color;

layout(shared, column_major) uniform mat4 badm4; // ERROR
layout(shared, column_major, row_major) uniform; // default is now shared and row_major

layout(std140) uniform Transform { // layout of this block is std140
    mat4 M1; // row_major
    layout(column_major) mat4 M2; // column major
    mat3 N1; // row_major
    centroid float badf;  // ERROR
    in float badg;        // ERROR
    layout(std140) float bad1;
    layout(shared) float bad2;
    layout(packed) float bad3;
} tblock;

uniform T2 { // layout of this block is shared
    bool b;
    mat4 t2m;
};

layout(column_major) uniform T3 { // shared and column_major
    mat4 M3; // column_major
    layout(row_major) mat4 M4; // row major
    mat3 N2; // column_major
    int b;  // ERROR, redefinition (needs to be last member of block for testing, following members are skipped)
};

out badout {  // ERROR
    float f;
};

layout (location = 10) out vec4 badoutA;  // ERROR

void main()
{
    pos = p * (tblock.M1 + tblock.M2 + M4 + M3 + t2m);
    color = c * tblock.N1;
}

shared vec4 compute_only;  // ERROR

layout(packed) uniform;

layout(packed) uniform float aoeuntaoeu;  // ERROR, packed on variable

layout(location = 40) in float cd;
layout(location = 37) in mat4x3 ce; // ERROR, overlap
