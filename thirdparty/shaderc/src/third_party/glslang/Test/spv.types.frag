#version 140

bool  u_b;
bvec2 u_b2;
bvec3 u_b3;
bvec4 u_b4;
flat in int   u_i;
flat in ivec2 u_i2;
flat in ivec3 u_i3;
flat in ivec4 u_i4;
     in float u_f;
     in vec2 u_f2;
     in vec3 u_f3;
     in vec4 u_f4;
bool  i_b;
bvec2 i_b2;
bvec3 i_b3;
bvec4 i_b4;

flat in int   i_i;
flat in ivec2 i_i2;
flat in ivec3 i_i3;
flat in ivec4 i_i4;

in float i_f;
in vec2 i_f2;
in vec3 i_f3;
in vec4 i_f4;

void main()
{
    bool  b = u_b && i_b;
    bvec2 b2 = bvec2(u_b2.x && i_b2.x && u_b2.y && i_b2.y);
    bvec3 b3 = bvec3(u_b3.x && i_b3.x && u_b3.y && i_b3.y && u_b3.z && i_b3.z);
    bvec4 b4 = bvec4(u_b4.x && i_b4.x && u_b4.y && i_b4.y && u_b4.z && i_b4.z && u_b4.w && i_b4.w);

    int   i = u_i + i_i;
    ivec2 i2 = u_i2 + i_i2;
    ivec3 i3 = u_i3 + i_i3;
    ivec4 i4 = u_i4 + i_i4;

    float f = u_f + i_f;
    vec2  f2 = u_f2 + i_f2;
    vec3  f3 = u_f3 + i_f3;
    vec4  f4 = u_f4 + i_f4;

    gl_FragColor = 
        b || 
        b2.x ||
        b2.y ||
        b3.x ||
        b3.y ||
        b3.z ||
        b4.x ||
        b4.y ||
        b4.z ||
        b4.w ? vec4(
        i  +
        i2.x +
        i2.y +
        i3.x +
        i3.y +
        i3.z +
        i4.x +
        i4.y +
        i4.z +
        i4.w +
        f  +
        f2.x +
        f2.y +
        f3.x +
        f3.y +
        f3.z +
        f4.x +
        f4.y +
        f4.z +
        f4.w) : vec4(1.0);        
}
