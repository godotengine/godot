SamplerState g_samp;
Texture1D    g_tex;

struct tex_t {
    SamplerState samp;
    Texture1D tex;
    int nonopaque_thing;
};

struct tex_with_arrays_t {
    SamplerState samp[2];
    Texture1D tex[2];
    int nonopaque_thing;
};

uniform tex_t g_texdata;
uniform tex_t g_texdata_array[3];
uniform tex_with_arrays_t g_texdata_array2[3];

struct PS_OUTPUT { float4 color : SV_Target0; };

void main(out PS_OUTPUT ps_output)
{
    ps_output.color =
        g_texdata.tex.Sample(g_texdata.samp, 0.5) +
        g_texdata_array[1].tex.Sample(g_texdata_array[1].samp, 0.4) +
        g_texdata_array2[1].tex[0].Sample(g_texdata_array2[1].samp[0], 0.3);
}
