float4 main(in float4 pos : SV_Position,
            in float2 clip[2] : SV_ClipDistance,               // array of vector float
            in float2 cull[2] : SV_CullDistance)  : SV_Target0 // array of vector float
{

    return pos + clip[0][0] + cull[0][0];
}
