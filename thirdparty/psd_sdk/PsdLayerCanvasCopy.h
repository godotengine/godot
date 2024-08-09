// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

namespace imageUtil
{
	/// \ingroup ImageUtil
	/// Copies 8-bit planar layer data to a canvas. Only the parts overlapping the canvas will be copied to it.
	void CopyLayerData(const uint8_t* PSD_RESTRICT layerData, uint8_t* PSD_RESTRICT canvasData, int layerLeft, int layerTop, int layerRight, int layerBottom, unsigned int canvasWidth, unsigned int canvasHeight);

	/// \ingroup ImageUtil
	/// Copies 16-bit planar layer data to a canvas. Only the parts overlapping the canvas will be copied to it.
	void CopyLayerData(const uint16_t* PSD_RESTRICT layerData, uint16_t* PSD_RESTRICT canvasData, int layerLeft, int layerTop, int layerRight, int layerBottom, unsigned int canvasWidth, unsigned int canvasHeight);

	/// \ingroup ImageUtil
	/// Copies 32-bit planar layer data to a canvas. Only the parts overlapping the canvas will be copied to it.
	void CopyLayerData(const float32_t* PSD_RESTRICT layerData, float32_t* PSD_RESTRICT canvasData, int layerLeft, int layerTop, int layerRight, int layerBottom, unsigned int canvasWidth, unsigned int canvasHeight);
}

PSD_NAMESPACE_END
