
struct MyStruct {
    sample        float a;
    noperspective float b;
    linear        float c;
    centroid      float d;
};

int sample(int x) { return x; } // HLSL allows this as an identifier as well.

float4 main() : SV_Target0
{
    // HLSL allows this as an identifier as well.
    // However, this is not true of other qualifier keywords such as "linear".
    float4 sample = float4(3,4,5,6);

    return sample.rgba; // 'sample' can participate in an expression.
}
