
// uniform Texture1D g_tex3[3][2];  // TODO: legal in HLSL, but we don't handle it yet.

uniform Texture1D g_tex[3];
uniform Texture1D g_tex_explicit[3] : register(t1);

SamplerState g_samp[3];
SamplerState g_samp_explicit[3] : register(s5);

uniform float3x3 g_mats[4];
uniform float3x3 g_mats_explicit[4] : register(b10);
uniform float g_floats[4];

// uniform float g_floats[4] = { 10, 11, 12, 13 };  // TODO: ... add when initializer lists can be flattened.

float4 TestFn1()
{
    return g_tex[1].Sample(g_samp[1], 0.2);
}

float4 TestFn2(Texture1D l_tex[3], SamplerState l_samp[3])
{
    return l_tex[2].Sample(l_samp[2], 0.2);
}

static int not_flattened_a[5] = { 1, 2, 3, 4, 5 };

struct PS_OUTPUT { float4 color : SV_Target0; };

void main(out PS_OUTPUT ps_output)
{
    // test flattening for local assignment initialization
    SamplerState local_sampler_array[3] = g_samp;
    Texture1D local_texture_array[3]    = g_tex;
    float local_float_array[4]          = g_floats;

    ps_output.color = TestFn1() + TestFn2(g_tex, g_samp);
}
