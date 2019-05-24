struct VS_OUTPUT        {
    float4 Position             : SV_Position;
    float3 clip0                : SV_ClipDistance0;  // multiple semantic IDs, vec3+vec4 (skip)
    float4 clip1                : SV_ClipDistance1;  // ...
};

float4 main(VS_OUTPUT v) : SV_Target0
{
    return v.Position + v.clip0.x + v.clip1.x;
}
