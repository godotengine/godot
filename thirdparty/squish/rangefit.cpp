/* -----------------------------------------------------------------------------

    Copyright (c) 2006 Simon Brown                          si@sjbrown.co.uk

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

#include "rangefit.h"
#include "colourset.h"
#include "colourblock.h"
#include <cfloat>

namespace squish {

RangeFit::RangeFit( ColourSet const* colours, int flags, float* metric )
  : ColourFit( colours, flags )
{
    // initialise the metric (old perceptual = 0.2126f, 0.7152f, 0.0722f)
    if( metric )
        m_metric = Vec3( metric[0], metric[1], metric[2] );
    else
        m_metric = Vec3( 1.0f );

    // initialise the best error
    m_besterror = FLT_MAX;

    // cache some values
    int const count = m_colours->GetCount();
    Vec3 const* values = m_colours->GetPoints();
    float const* weights = m_colours->GetWeights();

    // get the covariance matrix
    Sym3x3 covariance = ComputeWeightedCovariance( count, values, weights );

    // compute the principle component
    Vec3 principle = ComputePrincipleComponent( covariance );

    // get the min and max range as the codebook endpoints
    Vec3 start( 0.0f );
    Vec3 end( 0.0f );
    if( count > 0 )
    {
        float min, max;

        // compute the range
        start = end = values[0];
        min = max = Dot( values[0], principle );
        for( int i = 1; i < count; ++i )
        {
            float val = Dot( values[i], principle );
            if( val < min )
            {
                start = values[i];
                min = val;
            }
            else if( val > max )
            {
                end = values[i];
                max = val;
            }
        }
    }

    // clamp the output to [0, 1]
    Vec3 const one( 1.0f );
    Vec3 const zero( 0.0f );
    start = Min( one, Max( zero, start ) );
    end = Min( one, Max( zero, end ) );

    // clamp to the grid and save
    Vec3 const grid( 31.0f, 63.0f, 31.0f );
    Vec3 const gridrcp( 1.0f/31.0f, 1.0f/63.0f, 1.0f/31.0f );
    Vec3 const half( 0.5f );
    m_start = Truncate( grid*start + half )*gridrcp;
    m_end = Truncate( grid*end + half )*gridrcp;
}

void RangeFit::Compress3( void* block )
{
    // cache some values
    int const count = m_colours->GetCount();
    Vec3 const* values = m_colours->GetPoints();

    // create a codebook
    Vec3 codes[3];
    codes[0] = m_start;
    codes[1] = m_end;
    codes[2] = 0.5f*m_start + 0.5f*m_end;

    // match each point to the closest code
    u8 closest[16];
    float error = 0.0f;
    for( int i = 0; i < count; ++i )
    {
        // find the closest code
        float dist = FLT_MAX;
        int idx = 0;
        for( int j = 0; j < 3; ++j )
        {
            float d = LengthSquared( m_metric*( values[i] - codes[j] ) );
            if( d < dist )
            {
                dist = d;
                idx = j;
            }
        }

        // save the index
        closest[i] = ( u8 )idx;

        // accumulate the error
        error += dist;
    }

    // save this scheme if it wins
    if( error < m_besterror )
    {
        // remap the indices
        u8 indices[16];
        m_colours->RemapIndices( closest, indices );

        // save the block
        WriteColourBlock3( m_start, m_end, indices, block );

        // save the error
        m_besterror = error;
    }
}

void RangeFit::Compress4( void* block )
{
    // cache some values
    int const count = m_colours->GetCount();
    Vec3 const* values = m_colours->GetPoints();

    // create a codebook
    Vec3 codes[4];
    codes[0] = m_start;
    codes[1] = m_end;
    codes[2] = ( 2.0f/3.0f )*m_start + ( 1.0f/3.0f )*m_end;
    codes[3] = ( 1.0f/3.0f )*m_start + ( 2.0f/3.0f )*m_end;

    // match each point to the closest code
    u8 closest[16];
    float error = 0.0f;
    for( int i = 0; i < count; ++i )
    {
        // find the closest code
        float dist = FLT_MAX;
        int idx = 0;
        for( int j = 0; j < 4; ++j )
        {
            float d = LengthSquared( m_metric*( values[i] - codes[j] ) );
            if( d < dist )
            {
                dist = d;
                idx = j;
            }
        }

        // save the index
        closest[i] = ( u8 )idx;

        // accumulate the error
        error += dist;
    }

    // save this scheme if it wins
    if( error < m_besterror )
    {
        // remap the indices
        u8 indices[16];
        m_colours->RemapIndices( closest, indices );

        // save the block
        WriteColourBlock4( m_start, m_end, indices, block );

        // save the error
        m_besterror = error;
    }
}

} // namespace squish
