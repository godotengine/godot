[[vk::global_cbuffer_binding(5, 2)]]
float4 u1;
float4 u2;

float4 main() : SV_Target0
{
    return u1 + u2;
}