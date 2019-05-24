
struct VS_INPUT
{
    int    x0_in  : foo0;
    float4 Pos_in : SV_Position;
    int    x1_in  : foo1;
};

struct VS_OUTPUT
{
    int    x0_out  : foo0;
    float4 Pos_out : SV_Position;
    int    x1_out  : foo1;
};

VS_OUTPUT main(VS_INPUT vsin, float4 Pos_loose : SV_Position)
{
    VS_OUTPUT vsout;

    vsout.x0_out  = vsin.x0_in;
    vsout.Pos_out = vsin.Pos_in + Pos_loose;
    vsout.x1_out  = vsin.x1_in;

    return vsout;
}
