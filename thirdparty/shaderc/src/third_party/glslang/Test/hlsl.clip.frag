
float GetEntitySelectClip()
{
    return 1.0f;
}

float4 main() : SV_TARGET
{
    clip(GetEntitySelectClip());

    return 0;
}
