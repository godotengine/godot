static bool a, b = true;
float4 main() : SV_Position
{
    int r = 0;

    r += a + b;
    r += a - b;
    r += a * b;
    r += a / b;
    r += a % b;

    r += a & b;
    r += a | b;
    r += a ^ b;

    r += a << b;
    r += a >> b;

    return r;
}
