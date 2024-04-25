void mx_directional_light(LightData light, vec3 position, out lightshader result)
{
    result.direction = -light.direction;
    result.intensity = light.color * light.intensity;
}
