struct VS_INPUT         {
    float4 Position             : POSITION;
};

struct VS_OUTPUT        {
    float4 Position             : SV_Position;
    float4 ClipRect             : SV_ClipDistance0;  // vector in split struct
};

VS_OUTPUT main(const VS_INPUT v)
{
    VS_OUTPUT           Output;
    Output.Position     = 0;

    Output.ClipRect.x = 1;
    Output.ClipRect.y = 2;
    Output.ClipRect.z = 3;
    Output.ClipRect.w = 4;

    return Output;
}
