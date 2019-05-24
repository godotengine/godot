
float main(float4 input) : COLOR0
{
    vector r00 = float4(1,2,3,4);  // vector means float4
    float4 r01 = vector(2,3,4,5);  // vector means float4

    vector<bool, 1>   r12 = bool1(false);
    vector<int, 1>    r13 = int1(1);
    vector<float, 1>  r14 = float1(1);
    vector<double, 1> r15 = double1(1);
    vector<uint, 1>   r16 = uint1(1);

    vector<bool, 2>   r20 = bool2(false, true);
    vector<int, 2>    r21 = int2(1,2);
    vector<float, 2>  r22 = float2(1,2);
    vector<double, 2> r23 = double2(1,2);
    vector<uint, 2>   r24 = uint2(1,2);
    
    vector<bool, 3>   r30 = bool3(false, true, true);
    vector<int, 3>    r31 = int3(1,2,3);
    vector<float, 3>  r32 = float3(1,2,3);
    vector<double, 3> r33 = double3(1,2,3);
    vector<uint, 3>   r34 = uint3(1,2,3);

    vector<bool, 4>   r40 = bool4(false, true, true, false);
    vector<int, 4>    r41 = int4(1,2,3,4);
    vector<float, 4>  r42 = float4(1,2,3,4);
    vector<double, 4> r43 = double4(1,2,3,4);
    vector<uint, 4>   r44 = uint4(1,2,3,4);

    matrix   r50 = float4x4(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15); // matrix means float4x4
    float4x4 r51 = matrix(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);   // matrix means float4x4

    // matrix<bool, 2, 3>  r60 = bool2x3(false, true, false, true, false, true);   // TODO: 
    matrix<float, 2, 3> r61 = float2x3(1,2,3,4,5,6);
    matrix<float, 3, 2> r62 = float3x2(1,2,3,4,5,6);
    // matrix<float, 4, 1> r63 = float4x1(1,2,3,4);  // TODO: 
    // matrix<float, 1, 4> r64 = float1x4(1,2,3,4);  // TODO: 
    matrix<float, 4, 2> r65 = float4x2(1,2,3,4,5,6,7,8);
    matrix<float, 4, 3> r66 = float4x3(1,2,3,4,5,6,7,8,9,10,11,12);

    // TODO: bool mats
    // TODO: int mats
    
    return 0.0;
}

