float4 main(in float4 pos : SV_Position,
            in float clip[2] : SV_ClipDistance,              // array of scalar float
            in float cull[2] : SV_CullDistance) : SV_Target0 // array of scalar float
{
    return pos + clip[0] + cull[0];
}
