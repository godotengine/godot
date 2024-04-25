void mx_point_light(LightData light, float3 position, thread lightshader& result)
{
    result.direction = light.position - position;
    float distance = length(result.direction) + M_FLOAT_EPS;
    float attenuation = pow(distance + 1.0, light.decay_rate + M_FLOAT_EPS);
    result.intensity = light.color * light.intensity / attenuation;
    result.direction /= distance;
}
