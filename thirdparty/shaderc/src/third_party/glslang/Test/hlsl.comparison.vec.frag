uniform float4 uf4;

void Bug1( float4 a )
{
    float4 v04 = float4( 0.0, 0.0, 0.0, 0.0 );
    float  v01 = 0.0; 

    bool4 r00 = a == v04;
    bool4 r01 = a != v04;
    bool4 r02 = a < v04;
    bool4 r03 = a > v04;
    
    bool4 r10 = a == v01;
    bool4 r11 = a != v01;
    bool4 r12 = a < v01;
    bool4 r13 = a > v01;
    
    bool4 r20 = v01 == a;
    bool4 r21 = v01 != a;
    bool4 r22 = v01 < a;
    bool4 r23 = v01 > a;
}

struct PS_OUTPUT
{
    float4 Color : SV_Target0;
};

PS_OUTPUT main()
{
   PS_OUTPUT psout;
   psout.Color = 0;
   return psout;
}
