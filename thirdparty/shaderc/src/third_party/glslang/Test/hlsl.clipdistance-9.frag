
// Test packing 0 and 1 semantics into single array[4], from in fn params.
float4 main(in float4 Position : SV_Position,
            in float3 clip0 : SV_ClipDistance0,
            in float clip1 : SV_ClipDistance1) : SV_Target0
{
    return Position + clip0.x + clip1;
}
