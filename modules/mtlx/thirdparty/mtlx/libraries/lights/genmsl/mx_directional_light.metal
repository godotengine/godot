void mx_directional_light(LightData light, float3 position, thread lightshader& result)
{
    result.direction = -light.direction;
    result.intensity = light.color * light.intensity;
}
