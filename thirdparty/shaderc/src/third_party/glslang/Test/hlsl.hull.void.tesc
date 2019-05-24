// *** 
// void patchconstantfunction input and return
// ***

struct VS_OUT
{
    float3 cpoint : CPOINT;
};

struct HS_CONSTANT_OUT
{
    float edges[2] : SV_TessFactor;
};

struct HS_OUT
{
    float3 cpoint : CPOINT;
};

[domain("tri")]
[partitioning("fractional_even")]
[outputtopology("triangle_ccw")]
[outputcontrolpoints(3)]
[patchconstantfunc("PCF")]
HS_OUT main(InputPatch<VS_OUT, 3> ip)
{
    HS_OUT output;
    output.cpoint = ip[0].cpoint;
    return output;
}

void PCF()
{
}
