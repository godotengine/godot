
#pragma once

#include "edge-segments.h"

namespace msdfgen {

/// For curves a, b converging at P = a->point(1) = b->point(0) with the same (opposite) direction, determines the relative ordering in which they exit P (i.e. whether a is to the left or right of b at the smallest positive radius around P)
int convergentCurveOrdering(const EdgeSegment *a, const EdgeSegment *b);

}
