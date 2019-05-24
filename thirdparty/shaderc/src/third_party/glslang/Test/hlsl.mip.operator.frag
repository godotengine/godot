Texture2DArray     g_tTex2df4a;
Texture2D          g_tTex2df4;

float4 main() : SV_Target0
{
    return g_tTex2df4.mips[2][uint2(3, 4)] +

        // test float->uint cast on the mip arg
        g_tTex2df4a.mips[5.2][uint3(6, 7, 8)] +

        // Test nesting involving .mips operators:
        //               ....outer operator mip level......     .....outer operator coordinate....
        g_tTex2df4.mips[ g_tTex2df4.mips[9][uint2(10,11)][0] ][ g_tTex2df4.mips[13][uint2(14,15)].xy ];
}
