void foo1() {}
void foo2(void) {}

void PixelShaderFunction(float4 input) : COLOR0
{
    foo1();
    foo2();
    return;
}