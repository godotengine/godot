float1 f1 = float1(1.0);
float1x1 fmat11;
float4x1 fmat41;
float1x2 fmat12;
double2x3 dmat23;
int4x4 int44;

float1 ShaderFunction(float1 inFloat1, float inScalar) : COLOR0
{
    return inFloat1;
}
