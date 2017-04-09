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

#include <string.h>
#include "squish.h"
#include "colourset.h"
#include "maths.h"
#include "rangefit.h"
#include "clusterfit.h"
#include "colourblock.h"
#include "alpha.h"
#include "singlecolourfit.h"

namespace squish {

static int FixFlags( int flags )
{
    // grab the flag bits
    int method = flags & ( kDxt1 | kDxt3 | kDxt5 | kBc4 | kBc5 );
    int fit = flags & ( kColourIterativeClusterFit | kColourClusterFit | kColourRangeFit );
    int extra = flags & kWeightColourByAlpha;

    // set defaults
    if ( method != kDxt3
    &&   method != kDxt5
    &&   method != kBc4
    &&   method != kBc5 )
    {
        method = kDxt1;
    }
    if( fit != kColourRangeFit && fit != kColourIterativeClusterFit )
        fit = kColourClusterFit;

    // done
    return method | fit | extra;
}

void CompressMasked( u8 const* rgba, int mask, void* block, int flags, float* metric )
{
    // fix any bad flags
    flags = FixFlags( flags );

    if ( ( flags & ( kBc4 | kBc5 ) ) != 0 )
    {
        u8 alpha[16*4];
        for( int i = 0; i < 16; ++i )
        {
            alpha[i*4 + 3] = rgba[i*4 + 0]; // copy R to A
        }

        u8* rBlock = reinterpret_cast< u8* >( block );
        CompressAlphaDxt5( alpha, mask, rBlock );

        if ( ( flags & ( kBc5 ) ) != 0 )
        {
            for( int i = 0; i < 16; ++i )
            {
                alpha[i*4 + 3] = rgba[i*4 + 1]; // copy G to A
            }

            u8* gBlock = reinterpret_cast< u8* >( block ) + 8;
            CompressAlphaDxt5( alpha, mask, gBlock );
        }

        return;
    }

    // get the block locations
    void* colourBlock = block;
    void* alphaBlock = block;
    if( ( flags & ( kDxt3 | kDxt5 ) ) != 0 )
        colourBlock = reinterpret_cast< u8* >( block ) + 8;

    // create the minimal point set
    ColourSet colours( rgba, mask, flags );

    // check the compression type and compress colour
    if( colours.GetCount() == 1 )
    {
        // always do a single colour fit
        SingleColourFit fit( &colours, flags );
        fit.Compress( colourBlock );
    }
    else if( ( flags & kColourRangeFit ) != 0 || colours.GetCount() == 0 )
    {
        // do a range fit
        RangeFit fit( &colours, flags, metric );
        fit.Compress( colourBlock );
    }
    else
    {
        // default to a cluster fit (could be iterative or not)
        ClusterFit fit( &colours, flags, metric );
        fit.Compress( colourBlock );
    }

    // compress alpha separately if necessary
    if( ( flags & kDxt3 ) != 0 )
        CompressAlphaDxt3( rgba, mask, alphaBlock );
    else if( ( flags & kDxt5 ) != 0 )
        CompressAlphaDxt5( rgba, mask, alphaBlock );
}

void Decompress( u8* rgba, void const* block, int flags )
{
    // fix any bad flags
    flags = FixFlags( flags );

    // get the block locations
    void const* colourBlock = block;
    void const* alphaBlock = block;
    if( ( flags & ( kDxt3 | kDxt5 ) ) != 0 )
        colourBlock = reinterpret_cast< u8 const* >( block ) + 8;

    // decompress colour
    DecompressColour( rgba, colourBlock, ( flags & kDxt1 ) != 0 );

    // decompress alpha separately if necessary
    if( ( flags & kDxt3 ) != 0 )
        DecompressAlphaDxt3( rgba, alphaBlock );
    else if( ( flags & kDxt5 ) != 0 )
        DecompressAlphaDxt5( rgba, alphaBlock );
}

int GetStorageRequirements( int width, int height, int flags )
{
    // fix any bad flags
    flags = FixFlags( flags );

    // compute the storage requirements
    int blockcount = ( ( width + 3 )/4 ) * ( ( height + 3 )/4 );
    int blocksize = ( ( flags & ( kDxt1 | kBc4 ) ) != 0 ) ? 8 : 16;
    return blockcount*blocksize;
}

void CopyRGBA( u8 const* source, u8* dest, int flags )
{
    if (flags & kSourceBGRA)
    {
        // convert from bgra to rgba
        dest[0] = source[2];
        dest[1] = source[1];
        dest[2] = source[0];
        dest[3] = source[3];
    }
    else
    {
        for( int i = 0; i < 4; ++i )
            *dest++ = *source++;
    }
}

void CompressImage( u8 const* rgba, int width, int height, int pitch, void* blocks, int flags, float* metric )
{
    // fix any bad flags
    flags = FixFlags( flags );

    // loop over blocks
#ifdef SQUISH_USE_OPENMP
#   pragma omp parallel for
#endif
    for( int y = 0; y < height; y += 4 )
    {
        // initialise the block output
        u8* targetBlock = reinterpret_cast< u8* >( blocks );
        int bytesPerBlock = ( ( flags & ( kDxt1 | kBc4 ) ) != 0 ) ? 8 : 16;
        targetBlock += ( (y / 4) * ( (width + 3) / 4) ) * bytesPerBlock;

        for( int x = 0; x < width; x += 4 )
        {
            // build the 4x4 block of pixels
            u8 sourceRgba[16*4];
            u8* targetPixel = sourceRgba;
            int mask = 0;
            for( int py = 0; py < 4; ++py )
            {
                for( int px = 0; px < 4; ++px )
                {
                    // get the source pixel in the image
                    int sx = x + px;
                    int sy = y + py;

                    // enable if we're in the image
                    if( sx < width && sy < height )
                    {
                        // copy the rgba value
                        u8 const* sourcePixel = rgba + pitch*sy + 4*sx;
                        CopyRGBA(sourcePixel, targetPixel, flags);
                        // enable this pixel
                        mask |= ( 1 << ( 4*py + px ) );
                    }

                    // advance to the next pixel
                    targetPixel += 4;
                }
            }

            // compress it into the output
            CompressMasked( sourceRgba, mask, targetBlock, flags, metric );

            // advance
            targetBlock += bytesPerBlock;
        }
    }
}

void CompressImage( u8 const* rgba, int width, int height, void* blocks, int flags, float* metric )
{
    CompressImage(rgba, width, height, width*4, blocks, flags, metric);
}

void DecompressImage( u8* rgba, int width, int height, int pitch, void const* blocks, int flags )
{
    // fix any bad flags
    flags = FixFlags( flags );

    // loop over blocks
#ifdef SQUISH_USE_OPENMP
#   pragma omp parallel for
#endif
    for( int y = 0; y < height; y += 4 )
    {
        // initialise the block input
        u8 const* sourceBlock = reinterpret_cast< u8 const* >( blocks );
        int bytesPerBlock = ( ( flags & ( kDxt1 | kBc4 ) ) != 0 ) ? 8 : 16;
        sourceBlock += ( (y / 4) * ( (width + 3) / 4) ) * bytesPerBlock;

        for( int x = 0; x < width; x += 4 )
        {
            // decompress the block
            u8 targetRgba[4*16];
            Decompress( targetRgba, sourceBlock, flags );

            // write the decompressed pixels to the correct image locations
            u8 const* sourcePixel = targetRgba;
            for( int py = 0; py < 4; ++py )
            {
                for( int px = 0; px < 4; ++px )
                {
                    // get the target location
                    int sx = x + px;
                    int sy = y + py;

                    // write if we're in the image
                    if( sx < width && sy < height )
                    {
                        // copy the rgba value
                        u8* targetPixel = rgba + pitch*sy + 4*sx;
                        CopyRGBA(sourcePixel, targetPixel, flags);
                    }

                    // advance to the next pixel
                    sourcePixel += 4;
                }
            }

            // advance
            sourceBlock += bytesPerBlock;
        }
    }
}

void DecompressImage( u8* rgba, int width, int height, void const* blocks, int flags )
{
    DecompressImage( rgba, width, height, width*4, blocks, flags );
}

static double ErrorSq(double x, double y)
{
    return (x - y) * (x - y);
}

static void ComputeBlockWMSE(u8 const *original, u8 const *compressed, unsigned int w, unsigned int h, double &cmse, double &amse)
{
    // Computes the MSE for the block and weights it by the variance of the original block.
    // If the variance of the original block is less than 4 (i.e. a standard deviation of 1 per channel)
    // then the block is close to being a single colour. Quantisation errors in single colour blocks
    // are easier to see than similar errors in blocks that contain more colours, particularly when there
    // are many such blocks in a large area (eg a blue sky background) as they cause banding.  Given that
    // banding is easier to see than small errors in "complex" blocks, we weight the errors by a factor
    // of 5. This implies that images with large, single colour areas will have a higher potential WMSE
    // than images with lots of detail.

    cmse = amse = 0;
    unsigned int sum_p[4];  // per channel sum of pixels
    unsigned int sum_p2[4]; // per channel sum of pixels squared
    memset(sum_p, 0, sizeof(sum_p));
    memset(sum_p2, 0, sizeof(sum_p2));
    for( unsigned int py = 0; py < 4; ++py )
    {
        for( unsigned int px = 0; px < 4; ++px )
        {
            if( px < w && py < h )
            {
                double pixelCMSE = 0;
                for( int i = 0; i < 3; ++i )
                {
                    pixelCMSE += ErrorSq(original[i], compressed[i]);
                    sum_p[i] += original[i];
                    sum_p2[i] += (unsigned int)original[i]*original[i];
                }
                if( original[3] == 0 && compressed[3] == 0 )
                    pixelCMSE = 0; // transparent in both, so colour is inconsequential
                amse += ErrorSq(original[3], compressed[3]);
                cmse += pixelCMSE;
                sum_p[3] += original[3];
                sum_p2[3] += (unsigned int)original[3]*original[3];
            }
            original += 4;
            compressed += 4;
        }
    }
    unsigned int variance = 0;
    for( int i = 0; i < 4; ++i )
        variance += w*h*sum_p2[i] - sum_p[i]*sum_p[i];
    if( variance < 4 * w * w * h * h )
    {
        amse *= 5;
        cmse *= 5;
    }
}

void ComputeMSE( u8 const *rgba, int width, int height, int pitch, u8 const *dxt, int flags, double &colourMSE, double &alphaMSE )
{
    // fix any bad flags
    flags = FixFlags( flags );
    colourMSE = alphaMSE = 0;

    // initialise the block input
    squish::u8 const* sourceBlock = dxt;
    int bytesPerBlock = ( ( flags & squish::kDxt1 ) != 0 ) ? 8 : 16;

    // loop over blocks
    for( int y = 0; y < height; y += 4 )
    {
        for( int x = 0; x < width; x += 4 )
        {
            // decompress the block
            u8 targetRgba[4*16];
            Decompress( targetRgba, sourceBlock, flags );
            u8 const* sourcePixel = targetRgba;

            // copy across to a similar pixel block
            u8 originalRgba[4*16];
            u8* originalPixel = originalRgba;

            for( int py = 0; py < 4; ++py )
            {
                for( int px = 0; px < 4; ++px )
                {
                    int sx = x + px;
                    int sy = y + py;
                    if( sx < width && sy < height )
                    {
                        u8 const* targetPixel = rgba + pitch*sy + 4*sx;
                        CopyRGBA(targetPixel, originalPixel, flags);
                    }
                    sourcePixel += 4;
                    originalPixel += 4;
                }
            }

            // compute the weighted MSE of the block
            double blockCMSE, blockAMSE;
            ComputeBlockWMSE(originalRgba, targetRgba, std::min(4, width - x), std::min(4, height - y), blockCMSE, blockAMSE);
            colourMSE += blockCMSE;
            alphaMSE += blockAMSE;
            // advance
            sourceBlock += bytesPerBlock;
        }
    }
    colourMSE /= (width * height * 3);
    alphaMSE /= (width * height);
}

void ComputeMSE( u8 const *rgba, int width, int height, u8 const *dxt, int flags, double &colourMSE, double &alphaMSE )
{
    ComputeMSE(rgba, width, height, width*4, dxt, flags, colourMSE, alphaMSE);
}

} // namespace squish
