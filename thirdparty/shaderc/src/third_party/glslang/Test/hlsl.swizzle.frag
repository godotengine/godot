static float4 AmbientColor = float4(1, 0.5, 0, 1);

float4 ShaderFunction(float4 input) : COLOR0
{
    return input.wwyx * float4(AmbientColor.z);
}
