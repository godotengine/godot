cbuffer b {
    float m0;
    float3 m4;
    //////
    float m16;
    float3 m20 : packoffset(c1.y);
    /////
    float3 m36 : packoffset(c2.y);
    /////
    float2 m56 : packoffset(c3.z);
    /////
    float m64;
    float2 m68;
    float m76;
    //////
    float m80;
    float2 m96[1];
};

void main() {}
