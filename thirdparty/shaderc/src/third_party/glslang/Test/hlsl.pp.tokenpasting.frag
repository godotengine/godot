
#define foobarblee zzzz

#define ar qqqq

#define MACRO1(x,y) foo##x##y
// #define MACRO2 abc##def

// #define SPACE_IN_MACRO int var1

float4 main() : SV_Target0
{
    // float MACRO2 = 10;
    float MACRO1(b##ar,blee) = 3;

    return float4(foobarblee,0,0,0);
}

