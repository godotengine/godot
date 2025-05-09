
Texture2D theTextureY : register(t0);
Texture2D theTextureU : register(t1);
Texture2D theTextureV : register(t2);

SamplerState theSampler = sampler_state
{
    addressU = Clamp;
    addressV = Clamp;
    mipfilter = NONE;
    minfilter = LINEAR;
    magfilter = LINEAR;
};

struct PixelShaderInput
{
    float4 pos : SV_POSITION;
    float2 tex : TEXCOORD0;
    float4 color : COLOR0;
};

cbuffer Constants : register(b0)
{
    float4 Yoffset;
    float4 Rcoeff;
    float4 Gcoeff;
    float4 Bcoeff;
};


float4 main(PixelShaderInput input) : SV_TARGET
{
    float4 Output;

    float3 yuv;
    yuv.x = theTextureY.Sample(theSampler, input.tex).r;
    yuv.y = theTextureU.Sample(theSampler, input.tex).r;
    yuv.z = theTextureV.Sample(theSampler, input.tex).r;

    yuv += Yoffset.xyz;
    Output.r = dot(yuv, Rcoeff.xyz);
    Output.g = dot(yuv, Gcoeff.xyz);
    Output.b = dot(yuv, Bcoeff.xyz);
    Output.a = 1.0f;

    return Output * input.color;
}
