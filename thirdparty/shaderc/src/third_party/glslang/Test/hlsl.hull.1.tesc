// *** 
// invocation ID coming from input to entry point
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

[domain("isoline")]
[partitioning("integer")]
[outputtopology("line")]
[outputcontrolpoints(4)]
[patchconstantfunc("PCF")]
HS_OUT main(InputPatch<VS_OUT, 4> ip, uint m_cpid : SV_OutputControlPointID)
{
    HS_OUT output;
    output.cpoint = ip[0].cpoint;
    return output;
}

HS_CONSTANT_OUT PCF(uint pid : SV_PrimitiveId)
{
    HS_CONSTANT_OUT output;
    
    output.edges[0] = 2.0f;
    output.edges[1] = 8.0f;
    return output;
}
