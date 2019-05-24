struct VS_OUTPUT        {
    float4 Position             : SV_Position;
    float4 ClipRect             : SV_ClipDistance0;  // vector in split struct
};

float4 main(const VS_OUTPUT v) : SV_Target0
{
    return v.Position + v.ClipRect;
}
