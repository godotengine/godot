struct VS_OUTPUT        {
    float4 Position             : SV_Position;
    float4 clip0                : SV_ClipDistance0;  // multiple semantic IDs, two vec4s (no extra packing)
    float4 clip1                : SV_ClipDistance1;  // ...
};

VS_OUTPUT main()
{
    VS_OUTPUT           Output;
    Output.Position     = 0;

    Output.clip0.x = 0;
    Output.clip0.y = 1;
    Output.clip0.z = 2;
    Output.clip0.w = 3;

    Output.clip1.x = 4;
    Output.clip1.y = 5;
    Output.clip1.z = 6;
    Output.clip1.w = 7;

    return Output;
}
