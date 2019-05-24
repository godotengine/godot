struct VS_OUTPUT        {
    float4 Position             : SV_Position;
    float3 clip0                : SV_ClipDistance0;  // multiple semantic IDs, vec3+float (pack)
    float  clip1                : SV_ClipDistance1;  // ...
};

VS_OUTPUT main()
{
    VS_OUTPUT           Output;
    Output.Position     = 0;

    Output.clip0.x = 0;
    Output.clip0.y = 1;
    Output.clip0.z = 2;

    // Position 3 is packed from clip1's float
    Output.clip1   = 3;

    return Output;
}
