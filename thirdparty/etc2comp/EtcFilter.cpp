#include <stdlib.h>
#include <math.h>
#include "EtcFilter.h"


namespace Etc
{

static const double PiConst = 3.14159265358979323846;

inline double sinc(double x) 
{
    if ( x == 0.0 ) 
    {
        return 1.0;
    }

    return sin(PiConst * x) / (PiConst * x);
}

//inline float sincf( float x )
//{
//    x *= F_PI;
//    if (x < 0.01f && x > -0.01f)
//    {
//        return 1.0f + x*x*(-1.0f/6.0f + x*x*1.0f/120.0f);
//    }
//
//    return sinf(x)/x;
//}
//
//double bessel0(double x) 
//{
//    const double EPSILON_RATIO = 1E-16;
//    double xh, sum, pow, ds;
//    int k;
//
//    xh = 0.5 * x;
//    sum = 1.0;
//    pow = 1.0;
//    k = 0;
//    ds = 1.0;
//    while (ds > sum * EPSILON_RATIO) 
//    {
//        ++k;
//        pow = pow * (xh / k);
//        ds = pow * pow;
//        sum = sum + ds;
//    }
//
//    return sum;
//}

//**--------------------------------------------------------------------------
//** Name: kaiser(double alpha, double half_width, double x) 
//** Returns:
//** Description: Alpha controls shape of filter.  We are using 4.
//**--------------------------------------------------------------------------
//inline double kaiser(double alpha, double half_width, double x) 
//{
//    double ratio = (x / half_width);
//    return bessel0(alpha * sqrt(1 - ratio * ratio)) / bessel0(alpha);
//}
//
//float Filter_Lanczos4Sinc(float x)
//{
//    if (x <= -4.0f || x >= 4.0f)    // half-width of 4
//    {
//        return 0.0;
//    }
//
//    return sinc(0.875f * x) * sinc(0.25f * x);
//}
//
//double Filter_Kaiser4( double t )
//{
//    return kaiser( 4.0, 3.0, t);
//}
//
//double Filter_KaiserOptimal( double t )
//{
//    return kaiser( 8.93, 3.0f, t);
//}                  

double FilterLanczos3( double t )
{
	if ( t <= -3.0 || t >= 3.0 ) 
    {
        return 0.0;
    }

    return sinc( t ) * sinc( t / 3.0 );
}

double FilterBox( double t )
{
    return ( t > -0.5 && t < 0.5) ? 1.0 : 0.0;
}

double FilterLinear( double t )
{
	if (t < 0.0) t = -t;

    return (t < 1.0) ? (1.0 - t) : 0.0;
}


//**--------------------------------------------------------------------------
//** Name: CalcContributions( int srcSize, 
//**                          int destSize, 
//**                          double filterSize, 
//**						  bool wrap,
//**                          double (*FilterProc)(double), 
//**                          FilterWeights contrib[] )
//** Returns: void
//** Description:
//**--------------------------------------------------------------------------
void CalcContributions( int srcSize, int destSize, double filterSize, bool wrap, double (*FilterProc)(double), FilterWeights contrib[] )
{
    double scale;
    double filterScale;
    double center;
    double totalWeight;
    double weight;
    int   iRight;
    int   iLeft;
    int   iDest;

    scale = (double)destSize / srcSize;
    if ( scale < 1.0 )
    {
        filterSize = filterSize / scale;
        filterScale = scale;
    }
    else
    {
        filterScale = 1.0;
    }

    if ( filterSize > (double)MaxFilterSize )
    {
        filterSize = (double)MaxFilterSize;
    }

    for ( iDest = 0; iDest < destSize; ++iDest )
    {
        center = (double)iDest / scale;

        iLeft = (int)ceil(center - filterSize);
		iRight = (int)floor(center + filterSize);

		if ( !wrap )
		{
        if ( iLeft < 0 )
        {
            iLeft = 0;
        }

        if ( iRight >= srcSize )
        {
            iRight = srcSize - 1;
        }
		}

		int numWeights = iRight - iLeft + 1;

        contrib[iDest].first = iLeft;
        contrib[iDest].numWeights = numWeights;

        totalWeight = 0;
		double t = ((double)iLeft - center) * filterScale;
		for (int i = 0; i < numWeights; i++)
        {
			weight = (*FilterProc)(t) * filterScale;
            totalWeight += weight;
			contrib[iDest].weight[i] = weight;
			t += filterScale;
        }

        //**--------------------------------------------------------
        //** Normalize weights by dividing by the sum of the weights
        //**--------------------------------------------------------
        if ( totalWeight > 0.0 )
        {   
            for ( int i = 0; i < numWeights; i++)
            {
                contrib[iDest].weight[i] /= totalWeight;
            }
        }
    }
}

//**-------------------------------------------------------------------------
//** Name: Filter_TwoPass( RGBCOLOR *pSrcImage, 
//**                       int srcWidth, int srcHeight, 
//**                       RGBCOLOR *pDestImage, 
//**                       int destWidth, int destHeight, 
//**                       double (*FilterProc)(double) )
//** Returns: 0 on failure and 1 on success
//** Description: Filters a 2d image with a two pass filter by averaging the
//**    weighted contributions of the pixels within the filter region.  The
//**    contributions are determined by a weighting function parameter.
//**-------------------------------------------------------------------------
int FilterTwoPass( RGBCOLOR *pSrcImage, int srcWidth, int srcHeight, 
                    RGBCOLOR *pDestImage, int destWidth, int destHeight, unsigned int wrapFlags, double (*FilterProc)(double) )
{
    FilterWeights *contrib;
    RGBCOLOR *pPixel;
    RGBCOLOR *pSrcPixel;
    RGBCOLOR *pTempImage;
    int iRow;
    int iCol;
    int iSrcCol;
    int iSrcRow;
    int iWeight;
    double dRed;
    double dGreen;
    double dBlue;
    double dAlpha;
    double filterSize = 3.0;

	int maxDim = (srcWidth>srcHeight)?srcWidth:srcHeight;
	contrib = (FilterWeights*)malloc(maxDim * sizeof(FilterWeights));

	//**------------------------------------------------------------------------
    //** Need to create a temporary image to stuff the horizontally scaled image
    //**------------------------------------------------------------------------
    pTempImage = (RGBCOLOR *)malloc( destWidth * srcHeight * sizeof(RGBCOLOR) );
    if ( pTempImage == NULL )
    {
        // -- GODOT start --
        free( contrib );
        // -- GODOT end --
        return 0;
    }

    //**-------------------------------------------------------
    //** Horizontally filter the image into the temporary image
    //**-------------------------------------------------------
	bool bWrapHorizontal = !!(wrapFlags&FILTER_WRAP_X);
	CalcContributions( srcWidth, destWidth, filterSize, bWrapHorizontal, FilterProc, contrib );
    for ( iRow = 0; iRow < srcHeight; iRow++ )
    {
        for ( iCol = 0; iCol < destWidth; iCol++ )
        {
            dRed   = 0;
            dGreen = 0;
            dBlue  = 0;
            dAlpha = 0;

            for ( iWeight = 0; iWeight < contrib[iCol].numWeights; iWeight++ )
            {
                iSrcCol = iWeight + contrib[iCol].first;
				if (bWrapHorizontal)
				{
					iSrcCol = (iSrcCol < 0) ? (srcWidth + iSrcCol) : (iSrcCol >= srcWidth) ? (iSrcCol - srcWidth) : iSrcCol;
				}
                pSrcPixel = pSrcImage + (iRow * srcWidth) + iSrcCol;
                dRed   += contrib[iCol].weight[iWeight] * pSrcPixel->rgba[0];
                dGreen += contrib[iCol].weight[iWeight] * pSrcPixel->rgba[1];
                dBlue  += contrib[iCol].weight[iWeight] * pSrcPixel->rgba[2];
                dAlpha += contrib[iCol].weight[iWeight] * pSrcPixel->rgba[3];
            }

            pPixel = pTempImage + (iRow * destWidth) + iCol;
			pPixel->rgba[0] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, dRed)));
			pPixel->rgba[1] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, dGreen)));
			pPixel->rgba[2] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, dBlue)));
			pPixel->rgba[3] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, dAlpha)));
        }
    }

    //**-------------------------------------------------------
    //** Vertically filter the image into the destination image
    //**-------------------------------------------------------
	bool bWrapVertical = !!(wrapFlags&FILTER_WRAP_Y);
	CalcContributions(srcHeight, destHeight, filterSize, bWrapVertical, FilterProc, contrib);
    for ( iCol = 0; iCol < destWidth; iCol++ )
    {
        for ( iRow = 0; iRow < destHeight; iRow++ )
        {
            dRed   = 0;
            dGreen = 0;
            dBlue  = 0;
            dAlpha = 0;

            for ( iWeight = 0; iWeight < contrib[iRow].numWeights; iWeight++ )
            {
                iSrcRow = iWeight + contrib[iRow].first;
				if (bWrapVertical)
				{
					iSrcRow = (iSrcRow < 0) ? (srcHeight + iSrcRow) : (iSrcRow >= srcHeight) ? (iSrcRow - srcHeight) : iSrcRow;
				}
                pSrcPixel = pTempImage + (iSrcRow * destWidth) + iCol;
                dRed   += contrib[iRow].weight[iWeight] * pSrcPixel->rgba[0];
                dGreen += contrib[iRow].weight[iWeight] * pSrcPixel->rgba[1];
                dBlue  += contrib[iRow].weight[iWeight] * pSrcPixel->rgba[2];
                dAlpha += contrib[iRow].weight[iWeight] * pSrcPixel->rgba[3];
            }

            pPixel = pDestImage + (iRow * destWidth) + iCol;
            pPixel->rgba[0]   = (unsigned char)(std::max( 0.0, std::min( 255.0, dRed)));
            pPixel->rgba[1] = (unsigned char)(std::max( 0.0, std::min( 255.0, dGreen)));
            pPixel->rgba[2]  = (unsigned char)(std::max( 0.0, std::min( 255.0, dBlue)));
            pPixel->rgba[3] = (unsigned char)(std::max( 0.0, std::min( 255.0, dAlpha)));
        }
    }

    free( pTempImage );
	free( contrib );

    return 1;
}

//**-------------------------------------------------------------------------
//** Name: FilterResample(RGBCOLOR *pSrcImage, int srcWidth, int srcHeight, 
//**                       RGBCOLOR *pDstImage, int dstWidth, int dstHeight)
//** Returns: 1
//** Description: This function runs a 2d box filter over the srouce image
//** to produce the destination image.
//**-------------------------------------------------------------------------
void FilterResample( RGBCOLOR *pSrcImage, int srcWidth, int srcHeight, 
                     RGBCOLOR *pDstImage, int dstWidth, int dstHeight )
{
    int iRow;
    int iCol;
    int iSampleRow;
    int iSampleCol;
    int iFirstSampleRow;
    int iFirstSampleCol;
    int iLastSampleRow;
    int iLastSampleCol;
    int red;
    int green;
    int blue;
    int alpha;
    int samples;
    float xScale;
    float yScale;

    RGBCOLOR *pSrcPixel;
    RGBCOLOR *pDstPixel;

    xScale = (float)srcWidth / dstWidth;
    yScale = (float)srcHeight / dstHeight;

    for ( iRow = 0; iRow < dstHeight; iRow++ )
    {
        for ( iCol = 0; iCol < dstWidth; iCol++ )
        {
            iFirstSampleRow = (int)(iRow * yScale);
            iLastSampleRow = (int)ceil(iFirstSampleRow + yScale - 1);
            if ( iLastSampleRow >= srcHeight )
            {
                iLastSampleRow = srcHeight - 1;
            }

            iFirstSampleCol = (int)(iCol * xScale);
            iLastSampleCol = (int)ceil(iFirstSampleCol + xScale - 1);
            if ( iLastSampleCol >= srcWidth )
            {
                iLastSampleCol = srcWidth - 1;
            }

            samples = 0;
            red     = 0;
            green   = 0;
            blue    = 0;
            alpha   = 0;
            for ( iSampleRow = iFirstSampleRow; iSampleRow <= iLastSampleRow; iSampleRow++ )
            {
                for ( iSampleCol = iFirstSampleCol; iSampleCol <= iLastSampleCol; iSampleCol++ )
                {
                    pSrcPixel = pSrcImage + iSampleRow * srcWidth + iSampleCol;
                    red   += pSrcPixel->rgba[0];
                    green += pSrcPixel->rgba[1];
                    blue  += pSrcPixel->rgba[2];
                    alpha += pSrcPixel->rgba[3];

                    samples++;
                }
            }

            pDstPixel = pDstImage + iRow * dstWidth + iCol;
            if ( samples > 0 )
            {
                pDstPixel->rgba[0] = static_cast<uint8_t>(red / samples);
                pDstPixel->rgba[1] = static_cast<uint8_t>(green / samples);
                pDstPixel->rgba[2] = static_cast<uint8_t>(blue / samples);
                pDstPixel->rgba[3] = static_cast<uint8_t>(alpha / samples);
            }
            else
            {
                pDstPixel->rgba[0] = static_cast<uint8_t>(red);
                pDstPixel->rgba[1] = static_cast<uint8_t>(green);
                pDstPixel->rgba[2] = static_cast<uint8_t>(blue);
                pDstPixel->rgba[3] = static_cast<uint8_t>(alpha);
            }
        }
    }
}


}