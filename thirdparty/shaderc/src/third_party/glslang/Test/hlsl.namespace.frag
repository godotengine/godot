static float4 v1;
static float4 v2;

namespace N1 {
    float4 getVec() { return v1; }
}

namespace N2 {
    static float gf;
    float4 getVec() { return v2; }
    namespace N3 {
        float4 getVec() { return v2; }
        
        class C1 {
            float4 getVec() { return v2; }
        };
    }
}

float4 main() : SV_Target0
{
    return N1::getVec() + N2::getVec() + N2::N3::getVec() + N2::N3::C1::getVec() * N2::gf;
}
