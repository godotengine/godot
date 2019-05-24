struct T {
    float f : packoffset(c4.y);    // artificial, but validates all different treatments: uniform offset
    centroid float g;              // interpolant input
    float d: SV_DepthGreaterEqual; // fragment output
    float4 normal;                 // non-IO
};

T s;  // loose uniform

cbuffer buff {
    T t : packoffset(c5.z);
};

T main(T t : myInput) : SV_Target0
{
    T local;
    return local;
}
