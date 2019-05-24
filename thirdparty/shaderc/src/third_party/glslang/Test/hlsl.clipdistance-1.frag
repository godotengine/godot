float4 main(in float4 pos : SV_Position, 
            in float clip : SV_ClipDistance,
            in float cull : SV_CullDistance) : SV_Target0
{
    return pos + clip + cull;
}
