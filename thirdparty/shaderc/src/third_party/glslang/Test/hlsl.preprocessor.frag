#define DEFINE_TEXTURE(name) Texture2D name; SamplerState name##_ss;
#define SAMPLE_TEXTURE(name, uv) name.Sample(name##_ss, (uv).xy)

#define test_texture2 test_texture

DEFINE_TEXTURE(test_texture)

float4 main(float4 input : TEXCOORD0) : SV_TARGET
{
    float4 tex = SAMPLE_TEXTURE(test_texture2, input.xy);
    return tex;
}

