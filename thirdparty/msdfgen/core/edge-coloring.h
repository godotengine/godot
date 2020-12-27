
#pragma once

#include "Shape.h"

#define MSDFGEN_EDGE_LENGTH_PRECISION 4

namespace msdfgen {

/** Assigns colors to edges of the shape in accordance to the multi-channel distance field technique.
 *  May split some edges if necessary.
 *  angleThreshold specifies the maximum angle (in radians) to be considered a corner, for example 3 (~172 degrees).
 *  Values below 1/2 PI will be treated as the external angle.
 */
void edgeColoringSimple(Shape &shape, double angleThreshold, unsigned long long seed = 0);

/** The alternative "ink trap" coloring strategy is designed for better results with typefaces
 *  that use ink traps as a design feature. It guarantees that even if all edges that are shorter than
 *  both their neighboring edges are removed, the coloring remains consistent with the established rules.
 */
void edgeColoringInkTrap(Shape &shape, double angleThreshold, unsigned long long seed = 0);

}
