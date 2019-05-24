#version 450

uniform layout(binding=0) sampler       g_sSamp1;
uniform layout(binding=1) sampler       g_sSamp2;
uniform layout(binding=2) sampler       g_sSamp3[2];
uniform layout(binding=3) sampler       g_sSamp4[3];
uniform layout(binding=4) sampler       g_sSamp5;

uniform layout(binding=5) sampler       g_sSamp_unused1;
uniform layout(binding=6) sampler       g_sSamp_unused2;

uniform layout(binding=7) texture1D          g_tTex1;
uniform layout(binding=8) texture1D          g_tTex2;
uniform layout(binding=9) texture1D          g_tTex3[2];
uniform layout(binding=10) texture1D          g_tTex4[3];
uniform layout(binding=11) texture1D          g_tTex5;

uniform layout(binding=12) texture1D          g_tTex_unused1;
uniform layout(binding=13) texture1D          g_tTex_unused2;
uniform layout(binding=14) texture1D          g_tTex_unused3;

struct MyStruct_t {
    int a;
    float b;
    vec3 c;
};

uniform layout(binding=4) myblock {
    MyStruct_t mystruct;
    vec4 myvec4_a;
    vec4 myvec4_b;
    ivec4 myint4_a;
};

vec4 Func1()
{
    return
        texture(sampler1D(g_tTex1, g_sSamp1), 0.1) +
        texture(sampler1D(g_tTex2, g_sSamp2), 0.2) +
        texture(sampler1D(g_tTex3[0], g_sSamp3[0]), 0.3) +
        texture(sampler1D(g_tTex3[1], g_sSamp3[1]), 0.3) +
        texture(sampler1D(g_tTex4[1], g_sSamp4[1]), 0.4) +
        texture(sampler1D(g_tTex4[2], g_sSamp4[2]), 0.4) +
        texture(sampler1D(g_tTex5, g_sSamp5), 0.5) +
        mystruct.c[1];
}

vec4 Func2()
{
    return
        texture(sampler1D(g_tTex1, g_sSamp1), 0.1) +
        texture(sampler1D(g_tTex3[1], g_sSamp3[1]), 0.3);
}

// Not called from entry point:
vec4 Func2_unused()
{
    return
        texture(sampler1D(g_tTex_unused1, g_sSamp_unused1), 1.1) +
        texture(sampler1D(g_tTex_unused2, g_sSamp_unused2), 1.2);
}

out vec4 FragColor;

void main()
{
    FragColor = Func1() + Func2();
}
