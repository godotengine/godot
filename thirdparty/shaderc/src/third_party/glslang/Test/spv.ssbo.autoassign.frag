
cbuffer TestCB
{ 
    uint W;
    uint H;
}; 

struct BufType 
{ 
    float4 va; 
    float4 vb; 
};

StructuredBuffer < BufType > SB0; 
RWStructuredBuffer < BufType > SB1;

float4 main(float4 pos : POS) : SV_Target0
{ 
    float4 vTmp = SB0[pos.y * W + pos.x].va + SB0[pos.y * W + pos.x].vb;

    vTmp += SB1[pos.y * W + pos.x].va + SB1[pos.y * W + pos.x].vb;

    return vTmp;
} 
