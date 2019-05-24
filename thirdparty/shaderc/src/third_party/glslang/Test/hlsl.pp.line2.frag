#line 1 "foo.frag"
Texture2D g_tColor[ 128 ] ;

layout (push_constant) cbuffer PerViewConstantBuffer_t
{
    uint g_nDataIdx;
    uint g_nDataIdx2;
    bool g_B;
} ;

SamplerState g_sAniso;

struct PS_INPUT
{
    float2 vTextureCoords : TEXCOORD2 ;
} ;

struct PS_OUTPUT
{
    float4 vColor : SV_Target0 ;
} ;

PS_OUTPUT MainPs ( PS_INPUT i )
{
    PS_OUTPUT ps_output ;

    uint u;
#line 47
    if (g_B)
#line 3 "foo.h"
        u = g_nDataIdx;
    else
#line 67
        u = g_nDataIdx2;
#line 7 "foo2.h"
    ps_output . vColor = g_tColor [ u ] . Sample ( g_sAniso , i . vTextureCoords . xy );
#line 105
    return ps_output ;
}

