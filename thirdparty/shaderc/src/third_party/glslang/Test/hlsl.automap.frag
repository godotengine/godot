// Test register class offsets for different resource types

SamplerState       s1 : register(s1);
SamplerComparisonState s2 : register(s2);

Texture1D <float4> t1 : register(t1);
Texture2D <float4> t2 : register(t2);
Texture3D <float4> t3 : register(t3);
StructuredBuffer<float4> t4 : register(t4);
ByteAddressBuffer t5 : register(t5);
Buffer<float4> t6 : register(t6);

RWTexture1D <float4> u1 : register(u1);
RWTexture2D <float4> u2 : register(u2);
RWTexture3D <float4> u3 : register(u3);

RWBuffer <float> u4 : register(u4);
RWByteAddressBuffer u5 : register(u5);
RWStructuredBuffer<float> u6 : register(u6);
AppendStructuredBuffer<float> u7 : register(u7);
ConsumeStructuredBuffer<float> u8 : register(u8);

cbuffer cb : register(b1) {
    int cb1;
};

tbuffer tb : register(t7) {
    int tb1;
};

float4 main() : SV_Target0
{
    t1;
    t2;
    t3;
    t4[0];
    t5.Load(0);
    t6;

    s1;
    s2;

    u1;
    u2;
    u3;

    u4[0];
    u5.Load(0);
    u6[0];
    u7;
    u8;

    cb1;
    tb1;

    return 0;
}
