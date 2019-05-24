
uniform float u1 : register(b2);

uniform SamplerState s1     : register(s5);
uniform SamplerState s1a[3] : register(s6);

uniform Texture1D t1     : register(t15);
uniform Texture1D t1a[3] : register(t16);

cbuffer cbuff1 : register(b2) {
    float4 c1_a;
    int c1_b;
    float c1_c;
};

cbuffer cbuff2 : register(b3) {
    float4 c2_a;
    int c2_b;
    float c2_c;
};

struct PS_OUTPUT
{
    float4 Color : Sv_Target0;
};

void main(out PS_OUTPUT psout)
{
    psout.Color = 
        t1.Sample(s1, 0.3) +
        t1a[0].Sample(s1a[0], 0.3) +
        c1_a + c1_b + c1_c +
        c2_a + c2_b + c2_c;
}
