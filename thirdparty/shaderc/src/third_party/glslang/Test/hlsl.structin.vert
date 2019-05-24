struct VI {
    float4 m[2] : mysemA;
    float4 coord : SV_POSITION;
    linear float4 b : mysemB;
};

VI main(float4 d : mysem, VI vi, float4 e : mysem)
{
    VI local;

    local.b = vi.m[1] + vi.m[0] + (float4)vi.coord.x + d + e;
    local.coord = (float4)1;
    local.m[0] = (float4)2;
    local.m[1] = (float4)3;

    return local;
}
