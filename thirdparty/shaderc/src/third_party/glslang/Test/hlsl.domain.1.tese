
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

[domain ( "tri" )] 
gs_in_t main (const OutputPatch <ds_in_t, 3> i, float f : msem, float3 tesscoord : SV_DomainLocation, pcf_in_t pcf_data ) 
{ 
    gs_in_t o; 

    o.pos  = i[0].pos + tesscoord.x * f;
    o.norm = i[0].norm + tesscoord.y;

    tesscoord.z;
    
    return o; 
}

