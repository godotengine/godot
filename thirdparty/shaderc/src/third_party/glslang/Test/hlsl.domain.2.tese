// This will test having the PCF input to the domain shader not be given at the end of
// the argument list.  We must move it to the end of the linkage in this case.

struct ds_in_t 
{ 
    float4 pos  : POSITION; 
    float3 norm : TEXCOORD0; 
}; 

struct pcf_in_t 
{ 
    float flTessFactor [3]   : SV_TessFactor; 
    float flInsideTessFactor : SV_InsideTessFactor; 
    float foo : PCF_FOO;
}; 

struct gs_in_t 
{ 
    float4 pos  : POSITION; 
    float3 norm : TEXCOORD0; 
}; 

[domain ( "tri" )] 
gs_in_t main (pcf_in_t pcf_data, const OutputPatch <ds_in_t, 3> i, float3 tesscoord : SV_DomainLocation) 
{ 
    gs_in_t o; 

    o.pos  = i[0].pos + tesscoord.x;
    o.norm = i[0].norm + tesscoord.y;

    tesscoord.z;
    
    return o; 
}

