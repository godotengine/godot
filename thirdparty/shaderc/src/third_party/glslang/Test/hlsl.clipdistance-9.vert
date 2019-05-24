struct VS_OUTPUT        {
    float4 Position             : SV_Position;
};

// Test packing 0 and 1 semantics into single array[4] output, from out fn params.
VS_OUTPUT main(out float3 clip0 : SV_ClipDistance0, out float clip1 : SV_ClipDistance1)
{
    VS_OUTPUT           Output;
    Output.Position     = 0;

    clip0.x = 0;
    clip0.y = 1;
    clip0.z = 2;

    // Position 3 is packed from clip1's float
    clip1   = 3;

    return Output;
}
