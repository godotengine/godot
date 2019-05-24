float4 AmbientColor = float4(1, 0.5, 0, 1);
float AmbientIntensity = 0.1;

float4 PixelShaderFunction(float4 input) : COLOR0
{
    return input * AmbientIntensity + AmbientColor;
    return input * input + input * input;
    return input + input * input + input;
    return ++input * -+-+--input;
    return input++ + ++input;
    return sin(input);
}
