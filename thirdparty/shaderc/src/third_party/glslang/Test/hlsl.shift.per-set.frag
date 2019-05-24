// Test register class offsets for different resource types

SamplerState       s1 : register(s1, space1);
SamplerComparisonState s2 : register(s2, space2);

Texture1D <float4> t1 : register(t1, space1);
Texture2D <float4> t2 : register(t2, space1);
Texture3D <float4> t3 : register(t1, space2);
Texture3D <float4> ts6 : register(t1, space6);
StructuredBuffer<float4> t4 : register(t1, space3);

ByteAddressBuffer t5 : register(t2, space3);
Buffer<float4> t6 : register(t3, space3);

RWTexture1D <float4> u1 : register(u1, space1);
RWTexture2D <float4> u2 : register(u2, space2);
RWTexture3D <float4> u3 : register(u3, space2);

RWBuffer <float> u4 : register(u4, space1);
RWByteAddressBuffer u5 : register(u4, space2);
RWStructuredBuffer<float> u6 : register(u4, space3);
AppendStructuredBuffer<float> u7 : register(u4, space4);
ConsumeStructuredBuffer<float> u8 : register(u4, space5);

cbuffer cb : register(b1, space6) {
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
    ts6;

    return 0;
}
