/* -----------------------------------------------------------------------------

    Copyright (c) 2006 Simon Brown                          si@sjbrown.co.uk
    Copyright (c) 2007 Ignacio Castano                   icastano@nvidia.com

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

   -------------------------------------------------------------------------- */

#ifndef SQUISH_CLUSTERFIT_H
#define SQUISH_CLUSTERFIT_H

#include "squish.h"
#include "maths.h"
#include "simd.h"
#include "colourfit.h"

namespace squish {

class ClusterFit : public ColourFit
{
public:
    ClusterFit( ColourSet const* colours, int flags, float* metric );

private:
    bool ConstructOrdering( Vec3 const& axis, int iteration );

    virtual void Compress3( void* block );
    virtual void Compress4( void* block );

    enum { kMaxIterations = 8 };

    int m_iterationCount;
    Vec3 m_principle;
    u8 m_order[16*kMaxIterations];
    Vec4 m_points_weights[16];
    Vec4 m_xsum_wsum;
    Vec4 m_metric;
    Vec4 m_besterror;
};

} // namespace squish

#endif // ndef SQUISH_CLUSTERFIT_H
