
Texture2D<float>  tf1;
Texture2D<float4> tf4;

RWTexture2D<float>  twf1;
RWTexture2D<float4> twf4;

float Func(Texture2D<float> DummyTex) { return 1.0f; }
float4 Func(Texture2D<float4> DummyTex) { return float4(0,0,0,0); }

float Func(RWTexture2D<float> DummyTex) { return 1.0f; }
float4 Func(RWTexture2D<float4> DummyTex) { return float4(0,0,0,0); }

float4 main() : SV_TARGET
{
    return Func(tf1) + Func(tf4) + Func(twf1) + Func(twf4);
}
