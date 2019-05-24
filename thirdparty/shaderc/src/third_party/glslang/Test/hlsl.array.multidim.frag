
float float_array[5][4][3];

struct PS_OUTPUT
{
    float4 Color : SV_Target0;
};

PS_OUTPUT main()
{
    float4 float4_array_1[2][3];
    float4 float4_array_2[5][3];

    float4_array_1[1][2] = float_array[2][3][1];
    float4_array_2[1] = float4_array_1[0];

    PS_OUTPUT psout;
    psout.Color = float4_array_1[1][2];
    return psout;
}
