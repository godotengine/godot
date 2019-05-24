// Test passing split structs to functions.

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

void Fn1(VS_INPUT fn1_in, VS_OUTPUT fn1_out) {
    fn1_in.Pos_in + fn1_out.Pos_out;
}

VS_OUTPUT main(VS_INPUT vsin)
{
    VS_OUTPUT vsout;

    vsout.x0_out  = vsin.x0_in;
    vsout.Pos_out = vsin.Pos_in;
    vsout.x1_out  = vsin.x1_in;

    Fn1(vsin, vsout);

    return vsout;
}
