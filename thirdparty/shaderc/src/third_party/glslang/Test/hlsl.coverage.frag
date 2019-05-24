
// Verify that coverage mask is an array, as required by SPIR-V.

struct PS_INPUT
{
};

struct PS_OUTPUT
{
    float4 vColor : SV_Target0;
    uint nCoverageMask : SV_Coverage;
};

PS_OUTPUT main( PS_INPUT i )
{
    PS_OUTPUT o;
    o.vColor = float4(1.0, 0.0, 0.0, 1.0);
    o.nCoverageMask = 0;
    return o;
}
