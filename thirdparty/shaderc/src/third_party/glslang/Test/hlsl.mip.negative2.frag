Texture2D          g_tTex2df4;

float4 main() : SV_Target0
{
    g_tTex2df4.r[2][uint2(3, 4)]; // '.r' not valid on texture object

    return 0;
}
    
