#version 300 es

uniform block {
    mediump float f;
} instanceName;

struct S {
    int i;
} s;

float a[5];

void main()
{
    bool b;
    float f;
    int i;
    uint u;
    bvec3 b3;
    vec3 v3;
    ivec3 iv3;
    uvec3 uv3;
    vec4 v4;
    ivec4 iv4;
    uvec4 uv4;
    mat2 m2;
    mat4 m4;

    // These are all errors:
    instanceName + instanceName;
    s + s;
    i + f;
    u + f;
    u + i;
    iv3 *= iv4;
    iv4 / uv4;
    i - v3;
    iv3 + uv3;
    a * a;
    b / b;

    f % f;
    i % f;
    f % u;
    instanceName++;
    ++s;
    a--;
    ++b3;

    iv3 < uv3;
    m2 > m2;
    m2 != m4;
    i >= u;
    a <= a;
    b > b;

    b && b3;
    b3 ^^ b3;
    b3 || b;
    i && i;
    u || u;
    m2 ^^ m2;

    !u;
    !i;
    !m2;
    !v3;
    !a;

    ~f;
    ~m4;
    ~v3;
    ~a;
    ~instanceName;

    i << iv3;
    u << uv3;
    i >> f;
    f >> i;
    m4 >> i;
    a >> u;
    iv3 >> iv4;

    i & u;    
    u &= uv3;
    i | uv3;
    u & f;
    m2 | m2;
    s ^ s;
    (f = f) = f;

    // These are all okay:
    f * v4;
    u + u;
    uv4 / u;
    iv3 -= iv3;
    
    i %= 3;
    uv3 % 4u;
    --m2;
    iv4++;

    m4 != m4;
    m2 == m2;
    i <= i;
    a == a;
    s != s;

    b && b;
    b || b;
    b ^^ b;

    !b, uv3;

    ~i;
    ~u;
    ~uv3;
    ~iv3;

    uv3 <<= i;
    i >> i;
    u << u;
    iv3 >> iv3;

    i & i;
    u | u;
    iv3 ^ iv3;
    u & uv3;
    uv3 | u;
    uv3 &= u;
    int arr[0x222 & 0xf];
    arr[1]; // size 2
    int arr2[(uvec2(0, 0x2) | 0x1u).y];
    arr2[2]; // size 3
}
