// Test Y negation from entry point out parameter

float4 pos;

void main(out float4 position : SV_Position)
{
    position = pos;
}
