cbuffer buf1 {
    float4 v1;
}; // extraneous ;

tbuffer buf2 {
    float4 v2;
}; // extraneous ;

cbuffer cbufName {
    float4 v3 : packoffset(c0);
    int i3    : packoffset(c1.y);
}

tbuffer tbufName : register(t8) {
    float4 v4 : packoffset(c1);
    int i4    : packoffset(c3);
    float f1  : packoffset(c3.w);
    float f3  : packoffset(c4.x);
    float f4  : packoffset(c4.y);
    float f5  : packoffset(c4.z);
    float f6  : packoffset(c);
    float f7  : packoffset(c8);
                 float3x4 m1 : packoffset(c7);
       row_major float3x4 m2 : packoffset(c11);
    column_major float3x4 m3 : packoffset(c15);
                 float3x4 m4 : packoffset(c19);
}

float foo() // float looks like identifier, but can't be part of tbuffer
{
    return 1.0;
}

struct id {
    float4 a;
};

cbuffer cbufName2 {
    float4 v24;
}

id PixelShaderFunction(float4 input : SV_POSITION) : SV_TARGET0  // id looks like id for cbuffer name, but can't be
{
    id ret;
    ret.a = v24 + (input + v1 + v2 + v3 + v4) * foo();
    return ret;
}
