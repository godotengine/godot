/*
 * Copyright 2015 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

/*
http://stackoverflow.com/questions/2009160/how-do-i-convert-the-2-control-points-of-a-cubic-curve-to-the-single-control-poi
*/

/*
Let's call the control points of the cubic Q0..Q3 and the control points of the quadratic P0..P2.
Then for degree elevation, the equations are:

Q0 = P0
Q1 = 1/3 P0 + 2/3 P1
Q2 = 2/3 P1 + 1/3 P2
Q3 = P2
In your case you have Q0..Q3 and you're solving for P0..P2. There are two ways to compute P1 from
 the equations above:

P1 = 3/2 Q1 - 1/2 Q0
P1 = 3/2 Q2 - 1/2 Q3
If this is a degree-elevated cubic, then both equations will give the same answer for P1. Since
 it's likely not, your best bet is to average them. So,

P1 = -1/4 Q0 + 3/4 Q1 + 3/4 Q2 - 1/4 Q3
*/

#include "src/pathops/SkPathOpsCubic.h"
#include "src/pathops/SkPathOpsQuad.h"

// used for testing only
SkDQuad SkDCubic::toQuad() const {
    SkDQuad quad;
    quad[0] = fPts[0];
    const SkDPoint fromC1 = {(3 * fPts[1].fX - fPts[0].fX) / 2, (3 * fPts[1].fY - fPts[0].fY) / 2};
    const SkDPoint fromC2 = {(3 * fPts[2].fX - fPts[3].fX) / 2, (3 * fPts[2].fY - fPts[3].fY) / 2};
    quad[1].fX = (fromC1.fX + fromC2.fX) / 2;
    quad[1].fY = (fromC1.fY + fromC2.fY) / 2;
    quad[2] = fPts[3];
    return quad;
}
