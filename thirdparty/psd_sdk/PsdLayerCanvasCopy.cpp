// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#include "PsdPch.h"
#include "PsdLayerCanvasCopy.h"

#include "PsdAssert.h"
#include <cstring>


PSD_NAMESPACE_BEGIN

namespace
{
	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static inline bool IsOutside(int layerLeft, int layerTop, int layerRight, int layerBottom, unsigned int canvasWidth, unsigned int canvasHeight)
	{
		// layer data can be completely outside the canvas, or overlapping, or completely inside.
		// find the overlapping rectangle first.
		const int w = static_cast<int>(canvasWidth);
		const int h = static_cast<int>(canvasHeight);
		if ((layerLeft >= w) || (layerTop >= h) || (layerRight < 0) || (layerBottom < 0))
		{
			// layer data is completely outside
			return true;
		}

		return false;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static inline bool IsSameRegion(int layerLeft, int layerTop, int layerRight, int layerBottom, unsigned int canvasWidth, unsigned int canvasHeight)
	{
		const int w = static_cast<int>(canvasWidth);
		const int h = static_cast<int>(canvasHeight);
		if ((layerLeft == 0) && (layerTop == 0) && (layerRight == w) && (layerBottom == h))
		{
			// layer region exactly matches the canvas
			return true;
		}

		return false;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	static void CopyLayerDataImpl(const T* PSD_RESTRICT layerData, T* PSD_RESTRICT canvasData, int layerLeft, int layerTop, int layerRight, int layerBottom, unsigned int canvasWidth, unsigned int canvasHeight)
	{
		PSD_ASSERT_NOT_NULL(layerData);
		PSD_ASSERT_NOT_NULL(canvasData);

		const bool isOutside = IsOutside(layerLeft, layerTop, layerRight, layerBottom, canvasWidth, canvasHeight);
		if (isOutside)
			return;

		const bool isSameRegion = IsSameRegion(layerLeft, layerTop, layerRight, layerBottom, canvasWidth, canvasHeight);
		if (isSameRegion)
		{
			// fast path, the layer is exactly the same size as the canvas
			memcpy(canvasData, layerData, canvasWidth*canvasHeight*sizeof(T));
			return;
		}

		// slower path, find the extents of the overlapping region to copy
		const int w = static_cast<int>(canvasWidth);
		const int h = static_cast<int>(canvasHeight);
		const int left = layerLeft > 0 ? layerLeft : 0;
		const int top = layerTop > 0 ? layerTop : 0;
		const int right = layerRight < w ? layerRight : w;
		const int bottom = layerBottom < h ? layerBottom : h;

		// setup source and destination data so we can copy row by row
		const int regionWidth = right-left;
		const int regionHeight = bottom-top;
		const int planarWidth = layerRight-layerLeft;
		const T* PSD_RESTRICT src = layerData + (top - layerTop)*planarWidth + (left - layerLeft);
		T* PSD_RESTRICT dest = canvasData + static_cast<unsigned int>(top)*canvasWidth + static_cast<unsigned int>(left);

		for (int y=0; y < regionHeight; ++y)
		{
			memcpy(dest, src, static_cast<unsigned int>(regionWidth)*sizeof(T));
			dest += canvasWidth;
			src += planarWidth;
		}
	}
}


namespace imageUtil
{
	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void CopyLayerData(const uint8_t* PSD_RESTRICT layerData, uint8_t* PSD_RESTRICT canvasData, int layerLeft, int layerTop, int layerRight, int layerBottom, unsigned int canvasWidth, unsigned int canvasHeight)
	{
		CopyLayerDataImpl(layerData, canvasData, layerLeft, layerTop, layerRight, layerBottom, canvasWidth, canvasHeight);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void CopyLayerData(const uint16_t* PSD_RESTRICT layerData, uint16_t* PSD_RESTRICT canvasData, int layerLeft, int layerTop, int layerRight, int layerBottom, unsigned int canvasWidth, unsigned int canvasHeight)
	{
		CopyLayerDataImpl(layerData, canvasData, layerLeft, layerTop, layerRight, layerBottom, canvasWidth, canvasHeight);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void CopyLayerData(const float32_t* PSD_RESTRICT layerData, float32_t* PSD_RESTRICT canvasData, int layerLeft, int layerTop, int layerRight, int layerBottom, unsigned int canvasWidth, unsigned int canvasHeight)
	{
		CopyLayerDataImpl(layerData, canvasData, layerLeft, layerTop, layerRight, layerBottom, canvasWidth, canvasHeight);
	}
}

PSD_NAMESPACE_END
