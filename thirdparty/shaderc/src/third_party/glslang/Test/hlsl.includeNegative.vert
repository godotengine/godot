#include "foo.h"
#include "inc2/../foo.h"
#include "inc1/badInc.h"

float4 main() : SV_Position
{
#error in main
}
