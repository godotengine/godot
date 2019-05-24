Texture2D          g_tTex2df4;

float4 main() : SV_Target0
{
    g_tTex2df4.mips.mips[2][uint2(3, 4)]; // error to chain like this

    return 0;
}
    
