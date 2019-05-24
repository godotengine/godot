struct VS_OUTPUT        {
    float4 Position             : SV_Position;
    float2 ClipRect[2]          : SV_ClipDistance0;  // array of float2 in split struct
};

float4 main(const VS_OUTPUT v) : SV_Target0
{
    return v.Position + v.ClipRect[0].x + v.ClipRect[1].x;
}
