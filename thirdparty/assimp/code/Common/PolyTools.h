/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2020, assimp team


All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the
following conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----------------------------------------------------------------------
*/

/** @file PolyTools.h, various utilities for our dealings with arbitrary polygons */

#ifndef AI_POLYTOOLS_H_INCLUDED
#define AI_POLYTOOLS_H_INCLUDED

#include <assimp/material.h>
#include <assimp/ai_assert.h>

namespace Assimp {

// -------------------------------------------------------------------------------
/** Compute the signed area of a triangle.
 *  The function accepts an unconstrained template parameter for use with
 *  both aiVector3D and aiVector2D, but generally ignores the third coordinate.*/
template <typename T>
inline double GetArea2D(const T& v1, const T& v2, const T& v3)
{
    return 0.5 * (v1.x * ((double)v3.y - v2.y) + v2.x * ((double)v1.y - v3.y) + v3.x * ((double)v2.y - v1.y));
}

// -------------------------------------------------------------------------------
/** Test if a given point p2 is on the left side of the line formed by p0-p1.
 *  The function accepts an unconstrained template parameter for use with
 *  both aiVector3D and aiVector2D, but generally ignores the third coordinate.*/
template <typename T>
inline bool OnLeftSideOfLine2D(const T& p0, const T& p1,const T& p2)
{
    return GetArea2D(p0,p2,p1) > 0;
}

// -------------------------------------------------------------------------------
/** Test if a given point is inside a given triangle in R2.
 * The function accepts an unconstrained template parameter for use with
 *  both aiVector3D and aiVector2D, but generally ignores the third coordinate.*/
template <typename T>
inline bool PointInTriangle2D(const T& p0, const T& p1,const T& p2, const T& pp)
{
    // Point in triangle test using baryzentric coordinates
    const aiVector2D v0 = p1 - p0;
    const aiVector2D v1 = p2 - p0;
    const aiVector2D v2 = pp - p0;

    double dot00 = v0 * v0;
    double dot01 = v0 * v1;
    double dot02 = v0 * v2;
    double dot11 = v1 * v1;
    double dot12 = v1 * v2;

    const double invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
    dot11 = (dot11 * dot02 - dot01 * dot12) * invDenom;
    dot00 = (dot00 * dot12 - dot01 * dot02) * invDenom;

    return (dot11 > 0) && (dot00 > 0) && (dot11 + dot00 < 1);
}


// -------------------------------------------------------------------------------
/** Check whether the winding order of a given polygon is counter-clockwise.
 *  The function accepts an unconstrained template parameter, but is intended
 *  to be used only with aiVector2D and aiVector3D (z axis is ignored, only
 *  x and y are taken into account).
 * @note Code taken from http://cgm.cs.mcgill.ca/~godfried/teaching/cg-projects/97/Ian/applet1.html and translated to C++
 */
template <typename T>
inline bool IsCCW(T* in, size_t npoints) {
    double aa, bb, cc, b, c, theta;
    double convex_turn;
    double convex_sum = 0;

    ai_assert(npoints >= 3);

    for (size_t i = 0; i < npoints - 2; i++) {
        aa = ((in[i+2].x - in[i].x) * (in[i+2].x - in[i].x)) +
            ((-in[i+2].y + in[i].y) * (-in[i+2].y + in[i].y));

        bb = ((in[i+1].x - in[i].x) * (in[i+1].x - in[i].x)) +
            ((-in[i+1].y + in[i].y) * (-in[i+1].y + in[i].y));

        cc = ((in[i+2].x - in[i+1].x) *
            (in[i+2].x - in[i+1].x)) +
            ((-in[i+2].y + in[i+1].y) *
            (-in[i+2].y + in[i+1].y));

        b = std::sqrt(bb);
        c = std::sqrt(cc);
        theta = std::acos((bb + cc - aa) / (2 * b * c));

        if (OnLeftSideOfLine2D(in[i],in[i+2],in[i+1])) {
            //  if (convex(in[i].x, in[i].y,
            //      in[i+1].x, in[i+1].y,
            //      in[i+2].x, in[i+2].y)) {
            convex_turn = AI_MATH_PI_F - theta;
            convex_sum += convex_turn;
        }
        else {
            convex_sum -= AI_MATH_PI_F - theta;
        }
    }
    aa = ((in[1].x - in[npoints-2].x) *
        (in[1].x - in[npoints-2].x)) +
        ((-in[1].y + in[npoints-2].y) *
        (-in[1].y + in[npoints-2].y));

    bb = ((in[0].x - in[npoints-2].x) *
        (in[0].x - in[npoints-2].x)) +
        ((-in[0].y + in[npoints-2].y) *
        (-in[0].y + in[npoints-2].y));

    cc = ((in[1].x - in[0].x) * (in[1].x - in[0].x)) +
        ((-in[1].y + in[0].y) * (-in[1].y + in[0].y));

    b = std::sqrt(bb);
    c = std::sqrt(cc);
    theta = std::acos((bb + cc - aa) / (2 * b * c));

    //if (convex(in[npoints-2].x, in[npoints-2].y,
    //  in[0].x, in[0].y,
    //  in[1].x, in[1].y)) {
    if (OnLeftSideOfLine2D(in[npoints-2],in[1],in[0])) {
        convex_turn = AI_MATH_PI_F - theta;
        convex_sum += convex_turn;
    }
    else {
        convex_sum -= AI_MATH_PI_F - theta;
    }

    return convex_sum >= (2 * AI_MATH_PI_F);
}


// -------------------------------------------------------------------------------
/** Compute the normal of an arbitrary polygon in R3.
 *
 *  The code is based on Newell's formula, that is a polygons normal is the ratio
 *  of its area when projected onto the three coordinate axes.
 *
 *  @param out Receives the output normal
 *  @param num Number of input vertices
 *  @param x X data source. x[ofs_x*n] is the n'th element.
 *  @param y Y data source. y[ofs_y*n] is the y'th element
 *  @param z Z data source. z[ofs_z*n] is the z'th element
 *
 *  @note The data arrays must have storage for at least num+2 elements. Using
 *  this method is much faster than the 'other' NewellNormal()
 */
template <int ofs_x, int ofs_y, int ofs_z, typename TReal>
inline void NewellNormal (aiVector3t<TReal>& out, int num, TReal* x, TReal* y, TReal* z)
{
    // Duplicate the first two vertices at the end
    x[(num+0)*ofs_x] = x[0];
    x[(num+1)*ofs_x] = x[ofs_x];

    y[(num+0)*ofs_y] = y[0];
    y[(num+1)*ofs_y] = y[ofs_y];

    z[(num+0)*ofs_z] = z[0];
    z[(num+1)*ofs_z] = z[ofs_z];

    TReal sum_xy = 0.0, sum_yz = 0.0, sum_zx = 0.0;

    TReal *xptr = x +ofs_x, *xlow = x, *xhigh = x + ofs_x*2;
    TReal *yptr = y +ofs_y, *ylow = y, *yhigh = y + ofs_y*2;
    TReal *zptr = z +ofs_z, *zlow = z, *zhigh = z + ofs_z*2;

    for (int tmp=0; tmp < num; tmp++) {
        sum_xy += (*xptr) * ( (*yhigh) - (*ylow) );
        sum_yz += (*yptr) * ( (*zhigh) - (*zlow) );
        sum_zx += (*zptr) * ( (*xhigh) - (*xlow) );

        xptr  += ofs_x;
        xlow  += ofs_x;
        xhigh += ofs_x;

        yptr  += ofs_y;
        ylow  += ofs_y;
        yhigh += ofs_y;

        zptr  += ofs_z;
        zlow  += ofs_z;
        zhigh += ofs_z;
    }
    out = aiVector3t<TReal>(sum_yz,sum_zx,sum_xy);
}

} // ! Assimp

#endif
