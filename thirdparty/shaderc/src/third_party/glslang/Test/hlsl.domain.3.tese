// Test vec2 tessellation coordinate: the IO form should be a vec3, copied to a vec2
// at the entry point boundary.

struct ds_in_t 
{ 
    float4 pos  : POSITION; 
    float3 norm : TEXCOORD0; 
}; 

struct pcf_in_t 
{ 
    float flTessFactor [3]   : SV_TessFactor; 
    float flInsideTessFactor : SV_InsideTessFactor; 
}; 

struct gs_in_t 
{ 
    float4 pos  : POSITION; 
    float3 norm : TEXCOORD0; 
}; 

[domain ( "isoline" )]
gs_in_t main (const OutputPatch <ds_in_t, 2> i, float2 tesscoord : SV_DomainLocation, pcf_in_t pcf_data ) 
{ 
    gs_in_t o; 

    o.pos  = i[0].pos + tesscoord.x;
    o.norm = i[0].norm + tesscoord.y;

    return o; 
}
