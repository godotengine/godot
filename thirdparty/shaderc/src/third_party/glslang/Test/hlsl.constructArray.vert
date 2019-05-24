float4 main() :  SV_POSITION 
{
    int4 int4_array[3];
    float4 float4_array_times[2] = (float4[2])int4_array;
    float2 float2_array_times2[4] = (float2[4])int4_array;
    int4 int4_array2[2] = (int4[2])int4_array;
    int int1_array[2] = (int[2])int4_array;

    return (float4)0.0;
}
