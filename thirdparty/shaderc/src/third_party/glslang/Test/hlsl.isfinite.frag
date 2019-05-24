
uniform float  f;
uniform float2 f2;
uniform float3 f3;

bool test1(float v)
{
    return !isnan(v) && isfinite(v);
}

float4 main() : SV_Target0
{
    isfinite(f);
    isfinite(f2);
    isfinite(f3);

    return 0;
}
