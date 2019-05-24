layout(set=3,binding=5) tbuffer tbufName {
    layout(offset = 16) float4 v1;
};

layout(push_constant) tbuffer tbufName2 {
    float4 v5;
};

layout(constant_id=17) const int specConst = 10;

tbuffer tbufName2 : layout(set=4,binding=7) {
    layout(offset = 16) float4 v1PostLayout;
};

float4 PixelShaderFunction(float4 input) : COLOR0
{
    float4 layout = 2.0;
    return input + v1 + v5 + v1PostLayout * layout;
}
