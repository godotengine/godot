
struct STRUCT_WITH_NO_BUILTIN_INTERSTAGE_IO
{
    float m0_array[2] : mysemA;
    int   m1 : mysemB;
};

struct PS_IN 
{ 
    float4 pos : SV_Position; 
    float2 tc  : TEXCOORD0;
    // float  c           : SV_ClipDistance0;
}; 

struct GS_OUT
{ 
    PS_IN psIn;
    STRUCT_WITH_NO_BUILTIN_INTERSTAGE_IO contains_no_builtin_io;
};


[maxvertexcount(3)] 
void main(triangle PS_IN tin[3], inout TriangleStream <GS_OUT> ts ) 
{
    GS_OUT o;

    o.psIn.pos = float4(1,2,3,4);
    o.psIn.tc  = float2(5,6);
    o.contains_no_builtin_io.m0_array[0] = 2.3;
    o.contains_no_builtin_io.m0_array[1] = 2.3;
    o.contains_no_builtin_io.m1 = 2;

    ts.Append(o);
}
