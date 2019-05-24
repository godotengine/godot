struct VS_INPUT         {
    float4 Position             : POSITION;
};

struct VS_OUTPUT        {
    float4 Position             : SV_Position;
    float2 ClipRect[2]          : SV_ClipDistance0;  // array of float2 in split struct
};

VS_OUTPUT main(const VS_INPUT v)
{
    VS_OUTPUT           Output;
    Output.Position     = 0;

    Output.ClipRect[0].x = 1;
    Output.ClipRect[0].y = 2;
    Output.ClipRect[1].x = 3;
    Output.ClipRect[1].y = 4;

    return Output;
}
