
float4 main() : SV_Target0
{
    half  h0 = 0;
    half1 h1 = 1;
    half2 h2 = 2;
    half3 h3 = 3;
    half4 h4 = 4;

    half1x1 h11;
    half1x2 h12;
    half1x3 h13;
    half1x4 h14;
    half2x1 h21;
    half2x2 h22 = half2x2(1,2,3,4);
    half2x3 h23 = (half2x3)4.9;
    half2x4 h24;
    half3x1 h31;
    half3x2 h32;
    half3x3 h33;
    half3x4 h34;
    half4x1 h41;
    half4x2 h42;
    half4x3 h43;
    half4x4 h44;

    return h23._11 + h4.y + h0;
}
