
uniform float4 col4;

int4 main() : SV_Target0
{
    return D3DCOLORtoUBYTE4(col4);
}
