#include "bar.h"
#include "./inc1/bar.h"
#include "inc2\bar.h"

float4 main() : SV_Position
{
    return i1 + i2 + i3 + i4 + i5 + i6;
}
