
struct PS_IN 
{ 
    float4 pos : SV_Position; 
}; 

struct GS_OUT
{ 
    float4 pos : SV_Position; 
};

[maxvertexcount(3)] 
void main(triangle PS_IN i[3], inout TriangleStream <GS_OUT> ts)
{
    GS_OUT o;

    for (int x=0; x<3; ++x) {
        o.pos = i[x].pos;
        ts.Append(o);
    }
}
