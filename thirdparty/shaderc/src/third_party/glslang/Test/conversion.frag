#version 130

uniform bool  u_b;
uniform bvec2 u_b2;
uniform bvec3 u_b3;
uniform bvec4 u_b4;

uniform int   u_i;
uniform ivec2 u_i2;
uniform ivec3 u_i3;
uniform ivec4 u_i4;
 
uniform float u_f;
uniform vec2 u_f2;
uniform vec3 u_f3;
uniform vec4 u_f4;

uniform bool  i_b;
uniform bvec2 i_b2;
uniform bvec3 i_b3;
uniform bvec4 i_b4;

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
    bool   b = bool(u_i) ^^ bool(u_f);
    bvec2 b2 = bvec2(u_i, u_f);
    bvec3 b3 = bvec3(u_i, u_f, i_i);
    bvec4 b4 = bvec4(u_i, u_f, i_i, i_f);

    int    i = int(u_f)    + int(b);
    ivec2 i2 = ivec2(u_f2) + ivec2(b2);
    ivec3 i3 = ivec3(u_f3) + ivec3(b3);
    ivec4 i4 = ivec4(u_f4) + ivec4(b4);

    float f = i;
    vec2 f2 = i2;
    vec3 f3 = i3;
    vec4 f4 = i4;

    f  += (float(i) + float(b));
    f2 -= vec2(i2) + vec2(b2);
    f3 /= vec3(i3) + vec3(b3);
    f4 += vec4(i4) + vec4(b4);

    f4 += vec4(bvec4(i_i4));
    f4 += vec4(bvec4(u_f4));
    
    f  += f                 - i;
    f2 += vec2(f, i)       + i2;
    f3 += i3 + vec3(f, i, f);
    f4 += vec4(b, i, f, i) + i4;
    
    f2 += vec2(f, i)       * i;
    f3 += vec3(f, i, f)    + i;
    f4 += i - vec4(b, i, f, i);

    i2 += ivec2(f, i);
    i3 += ivec3(f, i, f);
    i4 += ivec4(b, i, f, i);

    if (f < i || i < f ||
        f2 == i2 ||
        i3 != f3)
        f = (b ? i : f2.x) + (b2.x ? f3.x : i2.y);

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

    // with constants...
    ivec4 cv2 = ivec4(1.0);
    bvec4 cv5 = bvec4(cv2);
    gl_FragColor += float(cv5);
}
