#pragma once
#include <stdint.h>
#include <algorithm>

namespace Etc
{

enum FilterEnums
{
	MaxFilterSize = 32
};

enum WrapFlags
{
	FILTER_WRAP_NONE = 0,
	FILTER_WRAP_X = 0x1,
	FILTER_WRAP_Y = 0x2
};

typedef struct tagFilterWeights
{
	int   first;
	int   numWeights;
	double weight[MaxFilterSize * 2 + 1];
} FilterWeights;

typedef struct tagRGBCOLOR
{
	union
	{
		uint32_t ulColor;
		uint8_t rgba[4];
	};
} RGBCOLOR;


double FilterBox( double t );
double FilterLinear( double t );
double FilterLanczos3( double t );

int FilterTwoPass( RGBCOLOR *pSrcImage, int srcWidth, int srcHeight, 
                    RGBCOLOR *pDestImage, int destWidth, int destHeight, unsigned int wrapFlags, double (*FilterProc)(double) );
void FilterResample( RGBCOLOR *pSrcImage, int srcWidth, int srcHeight, 
                     RGBCOLOR *pDstImage, int dstWidth, int dstHeight );


void CalcContributions(int srcSize, int destSize, double filterSize, bool wrap, double(*FilterProc)(double), FilterWeights contrib[]);

template <typename T>
void FilterResample(T *pSrcImage, int srcWidth, int srcHeight, T *pDstImage, int dstWidth, int dstHeight)
{
	float xScale;
	float yScale;

	T *pSrcPixel;
	T *pDstPixel;

	xScale = (float)srcWidth / dstWidth;
	yScale = (float)srcHeight / dstHeight;

	for (int iRow = 0; iRow < dstHeight; iRow++)
	{
		for (int iCol = 0; iCol < dstWidth; iCol++)
		{
			int samples;
			int iFirstSampleRow;
			int iFirstSampleCol;
			int iLastSampleRow;
			int iLastSampleCol;
			float red;
			float green;
			float blue;
			float alpha;

			iFirstSampleRow = (int)(iRow * yScale);
			iLastSampleRow = (int)ceil(iFirstSampleRow + yScale - 1);
			if (iLastSampleRow >= srcHeight)
			{
				iLastSampleRow = srcHeight - 1;
			}

			iFirstSampleCol = (int)(iCol * xScale);
			iLastSampleCol = (int)ceil(iFirstSampleCol + xScale - 1);
			if (iLastSampleCol >= srcWidth)
			{
				iLastSampleCol = srcWidth - 1;
			}

			samples = 0;
			red = 0.f;
			green = 0.f;
			blue = 0.f;
			alpha = 0.f;
			for (int iSampleRow = iFirstSampleRow; iSampleRow <= iLastSampleRow; iSampleRow++)
			{
				for (int iSampleCol = iFirstSampleCol; iSampleCol <= iLastSampleCol; iSampleCol++)
				{
					pSrcPixel = pSrcImage + (iSampleRow * srcWidth + iSampleCol) * 4;
					red += static_cast<float>(pSrcPixel[0]);
					green += static_cast<float>(pSrcPixel[1]);
					blue += static_cast<float>(pSrcPixel[2]);
					alpha += static_cast<float>(pSrcPixel[3]);

					samples++;
				}
			}

			pDstPixel = pDstImage + (iRow * dstWidth + iCol) * 4;
			if (samples > 0)
			{
				pDstPixel[0] = static_cast<T>(red / samples);
				pDstPixel[1] = static_cast<T>(green / samples);
				pDstPixel[2] = static_cast<T>(blue / samples);
				pDstPixel[3] = static_cast<T>(alpha / samples);
			}
			else
			{
				pDstPixel[0] = static_cast<T>(red);
				pDstPixel[1] = static_cast<T>(green);
				pDstPixel[2] = static_cast<T>(blue);
				pDstPixel[3] = static_cast<T>(alpha);
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
template <typename T>
int FilterTwoPass(T *pSrcImage, int srcWidth, int srcHeight,
	T *pDestImage, int destWidth, int destHeight, unsigned int wrapFlags, double(*FilterProc)(double))
{
	const int numComponents = 4;
	FilterWeights *contrib;
	T *pPixel;
	T *pTempImage;
	double dRed;
	double dGreen;
	double dBlue;
	double dAlpha;
	double filterSize = 3.0;

	int maxDim = (srcWidth>srcHeight) ? srcWidth : srcHeight;
	contrib = new FilterWeights[maxDim];

	//**------------------------------------------------------------------------
	//** Need to create a temporary image to stuff the horizontally scaled image
	//**------------------------------------------------------------------------
	pTempImage = new T[destWidth * srcHeight * numComponents];
	if (pTempImage == NULL)
	{
		return 0;
	}

	//**-------------------------------------------------------
	//** Horizontally filter the image into the temporary image
	//**-------------------------------------------------------
	bool bWrapHorizontal = !!(wrapFlags&FILTER_WRAP_X);
	CalcContributions(srcWidth, destWidth, filterSize, bWrapHorizontal, FilterProc, contrib);
	for (int iRow = 0; iRow < srcHeight; iRow++)
	{
		for (int iCol = 0; iCol < destWidth; iCol++)
		{
			dRed = 0;
			dGreen = 0;
			dBlue = 0;
			dAlpha = 0;

			for (int iWeight = 0; iWeight < contrib[iCol].numWeights; iWeight++)
			{
				int iSrcCol = iWeight + contrib[iCol].first;
				if(bWrapHorizontal)
				{
					iSrcCol = (iSrcCol < 0)?(srcWidth+iSrcCol):(iSrcCol >= srcWidth)?(iSrcCol-srcWidth):iSrcCol;
				}
				T* pSrcPixel = pSrcImage + ((iRow * srcWidth) + iSrcCol)*numComponents;
				dRed += contrib[iCol].weight[iWeight] * pSrcPixel[0];
				dGreen += contrib[iCol].weight[iWeight] * pSrcPixel[1];
				dBlue += contrib[iCol].weight[iWeight] * pSrcPixel[2];
				dAlpha += contrib[iCol].weight[iWeight] * pSrcPixel[3];
			}

			pPixel = pTempImage + ((iRow * destWidth) + iCol)*numComponents;
			pPixel[0] = static_cast<T>(std::max(0.0, std::min(255.0, dRed)));
			pPixel[1] = static_cast<T>(std::max(0.0, std::min(255.0, dGreen)));
			pPixel[2] = static_cast<T>(std::max(0.0, std::min(255.0, dBlue)));
			pPixel[3] = static_cast<T>(std::max(0.0, std::min(255.0, dAlpha)));
		}
	}

	//**-------------------------------------------------------
	//** Vertically filter the image into the destination image
	//**-------------------------------------------------------
	bool bWrapVertical = !!(wrapFlags&FILTER_WRAP_Y);
	CalcContributions(srcHeight, destHeight, filterSize, bWrapVertical, FilterProc, contrib);
	for (int iCol = 0; iCol < destWidth; iCol++)
	{
		for (int iRow = 0; iRow < destHeight; iRow++)
		{
			dRed = 0;
			dGreen = 0;
			dBlue = 0;
			dAlpha = 0;

			for (int iWeight = 0; iWeight < contrib[iRow].numWeights; iWeight++)
			{
				int iSrcRow = iWeight + contrib[iRow].first;
				if (bWrapVertical)
				{
					iSrcRow = (iSrcRow < 0) ? (srcHeight + iSrcRow) : (iSrcRow >= srcHeight) ? (iSrcRow - srcHeight) : iSrcRow;
				}
				T* pSrcPixel = pTempImage + ((iSrcRow * destWidth) + iCol)*numComponents;
				dRed += contrib[iRow].weight[iWeight] * pSrcPixel[0];
				dGreen += contrib[iRow].weight[iWeight] * pSrcPixel[1];
				dBlue += contrib[iRow].weight[iWeight] * pSrcPixel[2];
				dAlpha += contrib[iRow].weight[iWeight] * pSrcPixel[3];
			}

			pPixel = pDestImage + ((iRow * destWidth) + iCol)*numComponents;
			pPixel[0] = static_cast<T>(std::max(0.0, std::min(255.0, dRed)));
			pPixel[1] = static_cast<T>(std::max(0.0, std::min(255.0, dGreen)));
			pPixel[2] = static_cast<T>(std::max(0.0, std::min(255.0, dBlue)));
			pPixel[3] = static_cast<T>(std::max(0.0, std::min(255.0, dAlpha)));
		}
	}

	delete[] pTempImage;
	delete[] contrib;

	return 1;
}


}