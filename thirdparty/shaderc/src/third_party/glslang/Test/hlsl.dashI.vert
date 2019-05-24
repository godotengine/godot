// For -Iinc1/path1 -Iinc1/path2
// bar.h from local directory will define i1, while the ones from inc1/path1 and inc1/path2 will not.
// notHere.h is only in inc1/path1 and inc2/path2, and only inc1/path1 defines p1, p2, and p3
// parent.h is local again, and defines i4

#include "bar.h"
#include "notHere.h"
#include "parent.h"

float4 main() : SV_Position
{
    return i1 + i4 + p1 + p2 + p3;
}
