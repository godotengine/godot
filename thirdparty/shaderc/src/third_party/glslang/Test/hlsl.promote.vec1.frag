
float4 main() : SV_Target
{
    float f1a;
    float1 f1b;

    f1a = f1b;  // convert float1 to float
    f1b = f1a;  // convert float to float1

    float3 f3;
    step(0.0, f3);

    sin(f1b); // test 1-vectors in intrinsics

    return float4(0,0,0,0);
}
