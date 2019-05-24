
struct PS_OUTPUT { precise float4 color : SV_Target0; };

static precise float precisefloat;

void MyFunction(in precise float myfloat, out precise float3 myfloat3) { }

PS_OUTPUT main()
{
    PS_OUTPUT ps_output;
    ps_output.color = 1.0;
    return ps_output;
}

