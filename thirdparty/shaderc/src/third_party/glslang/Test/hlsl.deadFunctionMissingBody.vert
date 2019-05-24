float4 main(): SV_Target0 { return 0; }
struct Surface { float3 albedo; };
Surface surfaceShader(float fade);
Surface surfaceShaderExec()
{
    float fade = 0;
    return surfaceShader(0);
}
