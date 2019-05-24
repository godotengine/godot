#version 400

layout(constant_id = 16) const int arraySize = 5;
in vec4 ucol[arraySize];

layout(constant_id = 17) const bool spBool = true;
layout(constant_id = 18) const float spFloat = 3.14;
layout(constant_id = 19) const double spDouble = 3.1415926535897932384626433832795;
layout(constant_id = 22) const uint scale = 2;

layout(constant_id = 24) gl_MaxImageUnits;

out vec4 color;
out int size;

// parameter should be considered same type as ucol
void foo(vec4 p[arraySize]);

void main()
{
    color = ucol[2];
    size = arraySize;
    if (spBool)
        color *= scale;
    color += float(spDouble / spFloat);

    foo(ucol);
}

layout(constant_id = 116) const int dupArraySize = 12;
in vec4 dupUcol[dupArraySize];

layout(constant_id = 117) const bool spDupBool = true;
layout(constant_id = 118) const float spDupFloat = 3.14;
layout(constant_id = 119) const double spDupDouble = 3.1415926535897932384626433832795;
layout(constant_id = 122) const uint dupScale = 2;

void foo(vec4 p[arraySize])
{
    color += dupUcol[2];
    size += dupArraySize;
    if (spDupBool)
        color *= dupScale;
    color += float(spDupDouble / spDupFloat);
}

int builtin_spec_constant()
{
    int result = gl_MaxImageUnits;
    return result;
}
