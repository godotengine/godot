#version 430

const int a = 1;
const int b = 2;
const int c = a + b; // 3
const int d = c - a; // 2
const float e = float(d); // 2.0
const float f = e * float(c); // 6.0
const float g = f / float(d); // 3.0

const vec2 pytho = vec2(3.0, 4.0);

in vec4 inv;
out vec4 FragColor;
out vec2 out2;
out vec4 out3;
out vec4 out4;
out ivec4 out5;
out vec3 out6;
out vec4 out7;
out vec4 out8;
out vec4 out9;
out vec4 out10;
out vec4 out11; 
out ivec2 out12;
out uvec3 out13;

void main()
{
    vec4 dx = dFdx(inv);
    const ivec4 v = ivec4(a, b, c, d);
    vec4 array2[v.y];                  // 2
    const ivec4 u = ~v;

    const float h = degrees(g); // 171.88

    FragColor = vec4(e, f, g, h);  // 2, 6, 3, 171.88

    vec4 array3[c];               // 3
    vec4 arrayMax[int(max(float(array2.length()), float(array3.length())))];
    vec4 arrayMin[int(min(float(array2.length()), float(array3.length())))];
    FragColor = vec4(arrayMax.length(), arrayMin.length(), sin(3.14), cos(3.14));  // 3, 2, .00159, -.999
    out2 = length(pytho) + normalize(pytho) + dFdx(pytho) + dFdy(pytho) + fwidth(pytho); // 5+3/5, 5+4/5
    out3 = vec4(exp(3.0), log(10.0), exp2(4.0), log2(256.0));  //  20.08, 2.3, 16, 8
    out4 = vec4(sqrt(100.0), inversesqrt(100.0), abs(-4.7), abs(10.9)); // 10, .1, 4.7, 10.9
    out5 = ivec4(abs(-8) + sign(0), abs(17), sign(-12), sign(9));  // 8, 17, -1, 1
    out6 = vec3(sign(-8.8), sign(18.0), sign(0.0)); // -1.0, 1.0, 0.0
    out7 = vec4(floor(4.2), ceil(-4.1), trunc(5.9), trunc(-5.9)); // 4, -4, 5, -5
    out8 = vec4(round(4.4), round(4.6), roundEven(4.5), roundEven(-5.5)); // 4, 5, 4, -6
    out9 = vec4(roundEven(7.5), roundEven(-4.5), fract(2.345), fract(-2.6)); // 8, -4, .345, 0.4
    out10 = vec4(isinf(4.0/0.0), isinf(-3.0/0.0), isinf(0.0/0.0), isinf(-93048593405938405938405.0));  // true, true, false, false -> 1.0, 1.0, 0.0, 0.0
    out11 = vec4(isnan(4.0/0.0), isnan(-3.0/0.0), isnan(0.0/0.0), isnan(-93048593405938405938405.0));  // false, false, true, false -> 0.0, 1.0, 0.0, 0.0
    out11 = vec4(tan(0.8), atan(1.029), atan(8.0, 10.0), atan(10000.0));                               // 1.029, 0.8, 0.6747, 1.57
    out11 = vec4(asin(0.0), asin(0.5), acos(0.0), acos(0.5));                                          // 0.0, .523599, 1.57, 1.047

    const vec4 v1 = vec4(1.0, 0.0, 0.5, -0.2);
    const vec4 v2 = vec4(0.2, 0.3, 0.4, 0.5);
    out11 = atan(v1, v2);                      // 1.373401, 0.0, 0.896055, -0.380506

    const ivec2 v3 = ivec2(15.0, 17.0);
    const ivec2 v4 = ivec2(17.0, 15.0);
    out12 = min(v3, 16);                      // 15, 16
    out12 = max(v3, v4);                      // 17, 17
    out2 = pow(vec2(v3), vec2(2.5, 3.0));     // 871.4, 4913
    out13 = clamp(uvec3(1, 20, 50), 10u, 30u);  // 10, 20, 30
    out2 = mix(vec2(3.0, 4.0), vec2(5.0, 6.0), bvec2(false, true));  // 3.0, 6.0
    out2 = mix(vec2(3.0, 4.0), vec2(5.0, 6.0), 0.25);  // 3.5, 4.5
    out2 = step(0.5, vec2(0.2, 0.6));                  // 0.0, 1.0
    out11 = smoothstep(50.0, 60.0, vec4(40.0, 51.0, 55.0, 70.0)); // 0.0, 0.028, 0.5, 1.0
}

const struct S {
    vec3 v3;
    ivec2 iv2;
    mat2x4 m;
} s = S(vec3(3.0), ivec2(3, a + b), mat2x4(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0));

void foo()
{
    float a[s.iv2.y];  // 3 element array
    a[0] = s.m[1].z;     // 7.0
    b % 0;  // int
    b / 0;
    e / 0;  -e / 0;  0.0 / 0.0;
    const uint ua = 5;
    const uvec2 ub = uvec2(6, 7);
    const uint uc = 8;
    ub % 4u;
    0u % uc;
    ub % 0u;
}

const mat2 m2 = mat2(2, 3, 4, 5);
const mat3 m3 = mat3(m2);
const int mc = int(m3[2][2]);
float a1[mc];
float a2[int(m3[2][1]) + 2];  // size 2
float a3[int(m3[1][0])];  // size 4
const vec2 v2 = vec2(1, 2);
const vec3 v3 = vec3(3, 4, 5);
float a4[uint(mat3(v2, v3, v2, v2)[2][2])];  // size 2

void foo2()
{
    a1[0]; // array size 1
    a2[0]; // array size 2
    a3[0]; // array size 4
    a4[0]; // array size 2
    v2[-1]; // ERROR
    v3[4];  // ERROR
    m3[0][-2];  // ERROR
    m2[-1][1];  // ERROR
    m3[1][3];   // ERROR
    m3[3][1];   // ERROR
    int p;
    p = -2147483647 / -1;
    p = -2147483648 / -1;
    p =  2147483647 / -1;
    float f = vec4(7.8 < 2.4 ? -1.333 : 1.444).a;
    f = vec4(inv.x < 2.4 ? -1.0 : 1.0).a;  // not folded, ensuring no propagation
}

const mat2 mm2 = mat2(1.0, 2.0, 3.0, 4.0);
const mat3x2 mm32 = mat3x2(10.0, 11.0, 12.0, 13.0, 14.0, 15.0);

void foo3()
{
    mat3x2 r32 = mm2 * mm32;
}

struct cag {
    int   i;
    float f;
    bool  b;
};
const cag a0[3] = cag[3](cag(3, 2.0, true), cag(1, 5.0, true), cag(1, 9.0, false));

void foo4()
{
    int a = int(a0[2].f);
}

const bool cval1 = all(bvec4(true, true, true, true));
const bool cval2 = all(bvec4(false, false, false, false));
const bool cval3 = all(bvec4(true, true, false, true));
const bool cval4 = any(bvec4(true, true, true, true));
const bool cval5 = any(bvec4(false, false, false, false));
const bool cval6 = any(bvec4(false, true, false, false));
