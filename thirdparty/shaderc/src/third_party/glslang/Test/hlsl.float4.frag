float4 AmbientColor = float4(1, 0.5, 0, 1);

bool ff1 : SV_IsFrontFace;
float ff2 : packoffset(c1.y);
float4 ff3 : packoffset(c2) : register(ps_5_0, s0) ;
float4 ff4 : VPOS : packoffset(c3) : register(ps_5_0, s1) <int bambam=30;> ;

float4 ShaderFunction(float4 input) : COLOR0
{
    return input * AmbientColor;
}
