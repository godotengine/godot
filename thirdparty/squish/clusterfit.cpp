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

#include "clusterfit.h"
#include "colourset.h"
#include "colourblock.h"
#include <cfloat>

namespace squish {

ClusterFit::ClusterFit( ColourSet const* colours, int flags, float* metric )
  : ColourFit( colours, flags )
{
    // set the iteration count
    m_iterationCount = ( m_flags & kColourIterativeClusterFit ) ? kMaxIterations : 1;

    // initialise the metric (old perceptual = 0.2126f, 0.7152f, 0.0722f)
    if( metric )
        m_metric = Vec4( metric[0], metric[1], metric[2], 1.0f );
    else
        m_metric = VEC4_CONST( 1.0f );

    // initialise the best error
    m_besterror = VEC4_CONST( FLT_MAX );

    // cache some values
    int const count = m_colours->GetCount();
    Vec3 const* values = m_colours->GetPoints();

    // get the covariance matrix
    Sym3x3 covariance = ComputeWeightedCovariance( count, values, m_colours->GetWeights() );

    // compute the principle component
    m_principle = ComputePrincipleComponent( covariance );
}

bool ClusterFit::ConstructOrdering( Vec3 const& axis, int iteration )
{
    // cache some values
    int const count = m_colours->GetCount();
    Vec3 const* values = m_colours->GetPoints();

    // build the list of dot products
    float dps[16];
    u8* order = ( u8* )m_order + 16*iteration;
    for( int i = 0; i < count; ++i )
    {
        dps[i] = Dot( values[i], axis );
        order[i] = ( u8 )i;
    }

    // stable sort using them
    for( int i = 0; i < count; ++i )
    {
        for( int j = i; j > 0 && dps[j] < dps[j - 1]; --j )
        {
            std::swap( dps[j], dps[j - 1] );
            std::swap( order[j], order[j - 1] );
        }
    }

    // check this ordering is unique
    for( int it = 0; it < iteration; ++it )
    {
        u8 const* prev = ( u8* )m_order + 16*it;
        bool same = true;
        for( int i = 0; i < count; ++i )
        {
            if( order[i] != prev[i] )
            {
                same = false;
                break;
            }
        }
        if( same )
            return false;
    }

    // copy the ordering and weight all the points
    Vec3 const* unweighted = m_colours->GetPoints();
    float const* weights = m_colours->GetWeights();
    m_xsum_wsum = VEC4_CONST( 0.0f );
    for( int i = 0; i < count; ++i )
    {
        int j = order[i];
        Vec4 p( unweighted[j].X(), unweighted[j].Y(), unweighted[j].Z(), 1.0f );
        Vec4 w( weights[j] );
        Vec4 x = p*w;
        m_points_weights[i] = x;
        m_xsum_wsum += x;
    }
    return true;
}

void ClusterFit::Compress3( void* block )
{
    // declare variables
    int const count = m_colours->GetCount();
    Vec4 const two = VEC4_CONST( 2.0 );
    Vec4 const one = VEC4_CONST( 1.0f );
    Vec4 const half_half2( 0.5f, 0.5f, 0.5f, 0.25f );
    Vec4 const zero = VEC4_CONST( 0.0f );
    Vec4 const half = VEC4_CONST( 0.5f );
    Vec4 const grid( 31.0f, 63.0f, 31.0f, 0.0f );
    Vec4 const gridrcp( 1.0f/31.0f, 1.0f/63.0f, 1.0f/31.0f, 0.0f );

    // prepare an ordering using the principle axis
    ConstructOrdering( m_principle, 0 );

    // check all possible clusters and iterate on the total order
    Vec4 beststart = VEC4_CONST( 0.0f );
    Vec4 bestend = VEC4_CONST( 0.0f );
    Vec4 besterror = m_besterror;
    u8 bestindices[16];
    int bestiteration = 0;
    int besti = 0, bestj = 0;

    // loop over iterations (we avoid the case that all points in first or last cluster)
    for( int iterationIndex = 0;; )
    {
        // first cluster [0,i) is at the start
        Vec4 part0 = VEC4_CONST( 0.0f );
        for( int i = 0; i < count; ++i )
        {
            // second cluster [i,j) is half along
            Vec4 part1 = ( i == 0 ) ? m_points_weights[0] : VEC4_CONST( 0.0f );
            int jmin = ( i == 0 ) ? 1 : i;
            for( int j = jmin;; )
            {
                // last cluster [j,count) is at the end
                Vec4 part2 = m_xsum_wsum - part1 - part0;

                // compute least squares terms directly
                Vec4 alphax_sum = MultiplyAdd( part1, half_half2, part0 );
                Vec4 alpha2_sum = alphax_sum.SplatW();

                Vec4 betax_sum = MultiplyAdd( part1, half_half2, part2 );
                Vec4 beta2_sum = betax_sum.SplatW();

                Vec4 alphabeta_sum = ( part1*half_half2 ).SplatW();

                // compute the least-squares optimal points
                Vec4 factor = Reciprocal( NegativeMultiplySubtract( alphabeta_sum, alphabeta_sum, alpha2_sum*beta2_sum ) );
                Vec4 a = NegativeMultiplySubtract( betax_sum, alphabeta_sum, alphax_sum*beta2_sum )*factor;
                Vec4 b = NegativeMultiplySubtract( alphax_sum, alphabeta_sum, betax_sum*alpha2_sum )*factor;

                // clamp to the grid
                a = Min( one, Max( zero, a ) );
                b = Min( one, Max( zero, b ) );
                a = Truncate( MultiplyAdd( grid, a, half ) )*gridrcp;
                b = Truncate( MultiplyAdd( grid, b, half ) )*gridrcp;

                // compute the error (we skip the constant xxsum)
                Vec4 e1 = MultiplyAdd( a*a, alpha2_sum, b*b*beta2_sum );
                Vec4 e2 = NegativeMultiplySubtract( a, alphax_sum, a*b*alphabeta_sum );
                Vec4 e3 = NegativeMultiplySubtract( b, betax_sum, e2 );
                Vec4 e4 = MultiplyAdd( two, e3, e1 );

                // apply the metric to the error term
                Vec4 e5 = e4*m_metric;
                Vec4 error = e5.SplatX() + e5.SplatY() + e5.SplatZ();

                // keep the solution if it wins
                if( CompareAnyLessThan( error, besterror ) )
                {
                    beststart = a;
                    bestend = b;
                    besti = i;
                    bestj = j;
                    besterror = error;
                    bestiteration = iterationIndex;
                }

                // advance
                if( j == count )
                    break;
                part1 += m_points_weights[j];
                ++j;
            }

            // advance
            part0 += m_points_weights[i];
        }

        // stop if we didn't improve in this iteration
        if( bestiteration != iterationIndex )
            break;

        // advance if possible
        ++iterationIndex;
        if( iterationIndex == m_iterationCount )
            break;

        // stop if a new iteration is an ordering that has already been tried
        Vec3 axis = ( bestend - beststart ).GetVec3();
        if( !ConstructOrdering( axis, iterationIndex ) )
            break;
    }

    // save the block if necessary
    if( CompareAnyLessThan( besterror, m_besterror ) )
    {
        // remap the indices
        u8 const* order = ( u8* )m_order + 16*bestiteration;

        u8 unordered[16];
        for( int m = 0; m < besti; ++m )
            unordered[order[m]] = 0;
        for( int m = besti; m < bestj; ++m )
            unordered[order[m]] = 2;
        for( int m = bestj; m < count; ++m )
            unordered[order[m]] = 1;

        m_colours->RemapIndices( unordered, bestindices );

        // save the block
        WriteColourBlock3( beststart.GetVec3(), bestend.GetVec3(), bestindices, block );

        // save the error
        m_besterror = besterror;
    }
}

void ClusterFit::Compress4( void* block )
{
    // declare variables
    int const count = m_colours->GetCount();
    Vec4 const two = VEC4_CONST( 2.0f );
    Vec4 const one = VEC4_CONST( 1.0f );
    Vec4 const onethird_onethird2( 1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 1.0f/9.0f );
    Vec4 const twothirds_twothirds2( 2.0f/3.0f, 2.0f/3.0f, 2.0f/3.0f, 4.0f/9.0f );
    Vec4 const twonineths = VEC4_CONST( 2.0f/9.0f );
    Vec4 const zero = VEC4_CONST( 0.0f );
    Vec4 const half = VEC4_CONST( 0.5f );
    Vec4 const grid( 31.0f, 63.0f, 31.0f, 0.0f );
    Vec4 const gridrcp( 1.0f/31.0f, 1.0f/63.0f, 1.0f/31.0f, 0.0f );

    // prepare an ordering using the principle axis
    ConstructOrdering( m_principle, 0 );

    // check all possible clusters and iterate on the total order
    Vec4 beststart = VEC4_CONST( 0.0f );
    Vec4 bestend = VEC4_CONST( 0.0f );
    Vec4 besterror = m_besterror;
    u8 bestindices[16];
    int bestiteration = 0;
    int besti = 0, bestj = 0, bestk = 0;

    // loop over iterations (we avoid the case that all points in first or last cluster)
    for( int iterationIndex = 0;; )
    {
        // first cluster [0,i) is at the start
        Vec4 part0 = VEC4_CONST( 0.0f );
        for( int i = 0; i < count; ++i )
        {
            // second cluster [i,j) is one third along
            Vec4 part1 = VEC4_CONST( 0.0f );
            for( int j = i;; )
            {
                // third cluster [j,k) is two thirds along
                Vec4 part2 = ( j == 0 ) ? m_points_weights[0] : VEC4_CONST( 0.0f );
                int kmin = ( j == 0 ) ? 1 : j;
                for( int k = kmin;; )
                {
                    // last cluster [k,count) is at the end
                    Vec4 part3 = m_xsum_wsum - part2 - part1 - part0;

                    // compute least squares terms directly
                    Vec4 const alphax_sum = MultiplyAdd( part2, onethird_onethird2, MultiplyAdd( part1, twothirds_twothirds2, part0 ) );
                    Vec4 const alpha2_sum = alphax_sum.SplatW();

                    Vec4 const betax_sum = MultiplyAdd( part1, onethird_onethird2, MultiplyAdd( part2, twothirds_twothirds2, part3 ) );
                    Vec4 const beta2_sum = betax_sum.SplatW();

                    Vec4 const alphabeta_sum = twonineths*( part1 + part2 ).SplatW();

                    // compute the least-squares optimal points
                    Vec4 factor = Reciprocal( NegativeMultiplySubtract( alphabeta_sum, alphabeta_sum, alpha2_sum*beta2_sum ) );
                    Vec4 a = NegativeMultiplySubtract( betax_sum, alphabeta_sum, alphax_sum*beta2_sum )*factor;
                    Vec4 b = NegativeMultiplySubtract( alphax_sum, alphabeta_sum, betax_sum*alpha2_sum )*factor;

                    // clamp to the grid
                    a = Min( one, Max( zero, a ) );
                    b = Min( one, Max( zero, b ) );
                    a = Truncate( MultiplyAdd( grid, a, half ) )*gridrcp;
                    b = Truncate( MultiplyAdd( grid, b, half ) )*gridrcp;

                    // compute the error (we skip the constant xxsum)
                    Vec4 e1 = MultiplyAdd( a*a, alpha2_sum, b*b*beta2_sum );
                    Vec4 e2 = NegativeMultiplySubtract( a, alphax_sum, a*b*alphabeta_sum );
                    Vec4 e3 = NegativeMultiplySubtract( b, betax_sum, e2 );
                    Vec4 e4 = MultiplyAdd( two, e3, e1 );

                    // apply the metric to the error term
                    Vec4 e5 = e4*m_metric;
                    Vec4 error = e5.SplatX() + e5.SplatY() + e5.SplatZ();

                    // keep the solution if it wins
                    if( CompareAnyLessThan( error, besterror ) )
                    {
                        beststart = a;
                        bestend = b;
                        besterror = error;
                        besti = i;
                        bestj = j;
                        bestk = k;
                        bestiteration = iterationIndex;
                    }

                    // advance
                    if( k == count )
                        break;
                    part2 += m_points_weights[k];
                    ++k;
                }

                // advance
                if( j == count )
                    break;
                part1 += m_points_weights[j];
                ++j;
            }

            // advance
            part0 += m_points_weights[i];
        }

        // stop if we didn't improve in this iteration
        if( bestiteration != iterationIndex )
            break;

        // advance if possible
        ++iterationIndex;
        if( iterationIndex == m_iterationCount )
            break;

        // stop if a new iteration is an ordering that has already been tried
        Vec3 axis = ( bestend - beststart ).GetVec3();
        if( !ConstructOrdering( axis, iterationIndex ) )
            break;
    }

    // save the block if necessary
    if( CompareAnyLessThan( besterror, m_besterror ) )
    {
        // remap the indices
        u8 const* order = ( u8* )m_order + 16*bestiteration;

        u8 unordered[16];
        for( int m = 0; m < besti; ++m )
            unordered[order[m]] = 0;
        for( int m = besti; m < bestj; ++m )
            unordered[order[m]] = 2;
        for( int m = bestj; m < bestk; ++m )
            unordered[order[m]] = 3;
        for( int m = bestk; m < count; ++m )
            unordered[order[m]] = 1;

        m_colours->RemapIndices( unordered, bestindices );

        // save the block
        WriteColourBlock4( beststart.GetVec3(), bestend.GetVec3(), bestindices, block );

        // save the error
        m_besterror = besterror;
    }
}

} // namespace squish
