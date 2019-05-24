#define DLAYER 3

#define DMACRO1 TEXCOORD1
#define DMACRO(num) TEXCOORD##num

struct S {
	float4 pos	: POSITION;
	float2 UV0    	: TEXCOORD0;
	float2 UV1	: DMACRO1;
	float2 UV2	: DMACRO(2);
	float2 UV3	: DMACRO(DLAYER);
};


S main(float4 v : POSITION)
{
    S s;
    s.pos = v;
    s.UV0 = float2(v.x,v.x);
    s.UV1 = float2(v.y,v.y);
    s.UV2 = float2(v.z,v.z);
    s.UV3 = float2(v.w,v.w);
    return s;
}
