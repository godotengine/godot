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

#include "singlecolourfit.h"
#include "colourset.h"
#include "colourblock.h"

namespace squish {

struct SourceBlock
{
    u8 start;
    u8 end;
    u8 error;
};

struct SingleColourLookup
{
    SourceBlock sources[2];
};

#include "singlecolourlookup.inl"

static int FloatToInt( float a, int limit )
{
    // use ANSI round-to-zero behaviour to get round-to-nearest
    int i = ( int )( a + 0.5f );

    // clamp to the limit
    if( i < 0 )
        i = 0;
    else if( i > limit )
        i = limit;

    // done
    return i;
}

SingleColourFit::SingleColourFit( ColourSet const* colours, int flags )
  : ColourFit( colours, flags )
{
    // grab the single colour
    Vec3 const* values = m_colours->GetPoints();
    m_colour[0] = ( u8 )FloatToInt( 255.0f*values->X(), 255 );
    m_colour[1] = ( u8 )FloatToInt( 255.0f*values->Y(), 255 );
    m_colour[2] = ( u8 )FloatToInt( 255.0f*values->Z(), 255 );

    // initialise the best error
    m_besterror = INT_MAX;
}

void SingleColourFit::Compress3( void* block )
{
    // build the table of lookups
    SingleColourLookup const* const lookups[] =
    {
        lookup_5_3,
        lookup_6_3,
        lookup_5_3
    };

    // find the best end-points and index
    ComputeEndPoints( lookups );

    // build the block if we win
    if( m_error < m_besterror )
    {
        // remap the indices
        u8 indices[16];
        m_colours->RemapIndices( &m_index, indices );

        // save the block
        WriteColourBlock3( m_start, m_end, indices, block );

        // save the error
        m_besterror = m_error;
    }
}

void SingleColourFit::Compress4( void* block )
{
    // build the table of lookups
    SingleColourLookup const* const lookups[] =
    {
        lookup_5_4,
        lookup_6_4,
        lookup_5_4
    };

    // find the best end-points and index
    ComputeEndPoints( lookups );

    // build the block if we win
    if( m_error < m_besterror )
    {
        // remap the indices
        u8 indices[16];
        m_colours->RemapIndices( &m_index, indices );

        // save the block
        WriteColourBlock4( m_start, m_end, indices, block );

        // save the error
        m_besterror = m_error;
    }
}

void SingleColourFit::ComputeEndPoints( SingleColourLookup const* const* lookups )
{
    // check each index combination (endpoint or intermediate)
    m_error = INT_MAX;
    for( int index = 0; index < 2; ++index )
    {
        // check the error for this codebook index
        SourceBlock const* sources[3];
        int error = 0;
        for( int channel = 0; channel < 3; ++channel )
        {
            // grab the lookup table and index for this channel
            SingleColourLookup const* lookup = lookups[channel];
            int target = m_colour[channel];

            // store a pointer to the source for this channel
            sources[channel] = lookup[target].sources + index;

            // accumulate the error
            int diff = sources[channel]->error;
            error += diff*diff;
        }

        // keep it if the error is lower
        if( error < m_error )
        {
            m_start = Vec3(
                ( float )sources[0]->start/31.0f,
                ( float )sources[1]->start/63.0f,
                ( float )sources[2]->start/31.0f
            );
            m_end = Vec3(
                ( float )sources[0]->end/31.0f,
                ( float )sources[1]->end/63.0f,
                ( float )sources[2]->end/31.0f
            );
            m_index = ( u8 )( 2*index );
            m_error = error;
        }
    }
}

} // namespace squish
