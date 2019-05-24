#version 130

uniform int c;
uniform usampler2D us2D;

in ivec2 x;
in vec2 v2a;
in float c1D;
in vec2  c2D;
in vec3  c3D;
smooth vec4  c4D;  // ??

uniform vec4 v4;

void main()
{
    float f = 3;
    switch (c) {     // full switch testing in switch.frag
    case 1:
        f = sin(f);
        break;
    case 2:
        f = f * f;
    default:
        f = 3.0;
    }

    uint i;
    i = texture(us2D, x).w;          // full uint testing in uint.frag
    i << 3u | 0x8Au >> 1u & 0xFFu;

    vec3 modfOut, modfIn;
    vec3 v11 = modf(modfIn, modfOut);
    float t = trunc(f);
    vec2 v12 = round(v2a);
    vec2 v13 = roundEven(v2a);
    bvec2 b10 = isnan(v2a);
    bvec4 b11 = isinf(v4);

    sinh(c1D) +
    cosh(c1D) * tanh(c2D);
    asinh(c4D) + acosh(c4D);
    atanh(c3D);

    int id = gl_VertexID;
    gl_ClipDistance[1] = 0.3;
}

// version 140 features

//uniform isamplerBuffer sbuf;

//layout(std140) uniform blockName {
//    int anonMem;
//};

void foo88()
{
    int id = gl_InstanceID;    // ERROR
    //id += anonMem;
    id += texelFetch(id, 8);

    gl_ClipVertex;         // these are all present...
    gl_Color;
    gl_LightSource[0];
    gl_DepthRange.far;
    gl_TexCoord;
    gl_FogFragCoord;
    gl_FrontColor;
}

// token pasting

#define mac abc##def
int mac;

#define macr(A,B) A##B
int macr(qrs,tuv);
