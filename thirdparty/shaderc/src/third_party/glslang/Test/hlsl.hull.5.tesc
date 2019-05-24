
// Test mixed InputPatch structure: user and builtin members.  Hull shaders involve extra
// logic in this case due to patch constant function call synthesis.

// This example tests the PCF EP having an InputPatch, but the main EP does not.

struct HS_Main_Output
{ 
    float4 m_Position : SV_POSITION ; 
}; 

struct HS_Output 
{ 
    float fTessFactor [ 3 ] : SV_TessFactor ; 
    float fInsideTessFactor : SV_InsideTessFactor ; 
}; 

struct HS_Input 
{ 
    float4 m_Position : SV_POSITION; 
    float4 m_Normal   : TEXCOORD2; 
}; 

HS_Output HS_ConstFunc ( InputPatch < HS_Input , 3 > I ) 
{ 
    HS_Output O = (HS_Output)0;

    O.fInsideTessFactor = I [ 0 ].m_Position.w + I [ 0 ].m_Normal.w;

    return O;
} 

[ domain ( "tri" ) ] 
[ partitioning ( "fractional_odd" ) ] 
[ outputtopology ( "triangle_cw" ) ] 
[ patchconstantfunc ( "HS_ConstFunc" ) ] 
[ outputcontrolpoints ( 3 ) ] 
HS_Main_Output main( uint cpid : SV_OutputControlPointID ) 
{ 
    HS_Main_Output output = ( HS_Main_Output ) 0 ; 
    output.m_Position = 0;
    return output ; 
}
