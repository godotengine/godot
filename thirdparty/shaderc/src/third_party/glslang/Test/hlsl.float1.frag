static float1 f1 = float1(1.0);
static float scalar = 2.0;

float1 ShaderFunction(float1 inFloat1 : COLOR, float inScalar) : COLOR0
{
    return f1 * scalar + inFloat1 * inScalar;
}
