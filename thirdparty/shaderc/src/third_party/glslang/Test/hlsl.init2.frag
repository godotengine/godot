
void Test1()
{
    struct mystruct { float2 a; };
    mystruct test1 = {
        { 1, 2, },          // test trailing commas in list
    };

    mystruct test2 = {
        float2(3, 4),
    };

    // mystruct test3 = {
    //     { { 5, 6, } },   // TODO: test unneeded levels
    // };

    float test4 = { 7, } ;   // test scalar initialization

    struct mystruct2 { float a; float b; float c; };
    mystruct2 test5 = { {8,}, {9,}, {10}, };
    const mystruct2 constTest5 = { {8,}, {9,}, {10}, };
    constTest5.c;

    const float step = 1.f;
    float n = 0;
    const float3 a[8] = {
            normalize(float3(1, 1, 1)) * (n += step),
            normalize(float3(-1, -1, -1)) * (n += step),
            normalize(float3(-1, -1, 1)) * (n += step),
            normalize(float3(-1, 1, -1)) * (n += step),
            normalize(float3(-1, 1, 1)) * (n += step),
            normalize(float3(1, -1, -1)) * (n += step),
            normalize(float3(1, -1, 1)) * (n += step),
            normalize(float3(1, 1, -1)) * (n += step) };

    const struct one { float3 a; } oneNonConst = { normalize(float3(-1, 1, 1)) * (n += step) };
    const struct two { float3 a;
                       float3 b; } twoNonConst = { normalize(float3(-1, 1, 1)) * (n += step),
                                                   normalize(float3(-1, 1, 1)) * (n += step) };
}

struct PS_OUTPUT { float4 color : SV_Target0; };

PS_OUTPUT main()
{
    Test1();

    PS_OUTPUT ps_output;
    ps_output.color = 1.0;
    return ps_output;
}
