#line 1 "C:\\Users\\Greg\\shaders\\line\\foo4.frag"
Texture2D g_tColor [ 128 ] ;

layout ( push_constant ) cbuffer PerViewConstantBuffer_t
{
    uint g_nDataIdx ;
    uint g_nDataIdx2 ;
    bool g_B ;
} ;

#line 12
SamplerState g_sAniso ;

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

    uint u ;
    if ( g_B )


#line 1 "C:\\Users\\Greg\\shaders\\line\\u1.h"
    u = g_nDataIdx ;


#line 31 "C:\\Users\\Greg\\shaders\\line\\foo4.frag"
    else
    u = g_nDataIdx2 ;
    ps_output . vColor = g_tColor [ u ] . Sample ( g_sAniso , i . vTextureCoords . xy ) ;
    return ps_output ;
}

