// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

/// \ingroup Types
/// \class LayerMask
/// \brief A struct representing a layer mask as stored in the layers of the Layer Mask section.
/// \sa Layer VectorMask
struct LayerMask
{
	int32_t top;					///< Top coordinate of the rectangle that encloses the mask.
	int32_t left;					///< Left coordinate of the rectangle that encloses the mask.
	int32_t bottom;					///< Bottom coordinate of the rectangle that encloses the mask.
	int32_t right;					///< Right coordinate of the rectangle that encloses the mask.

	uint64_t fileOffset;			///< The offset from the start of the file where the channel's data is stored.

	void* data;						///< Planar data, having a size of (right-left)*(bottom-top)*bytesPerPixel.

	float64_t feather;				///< The mask's feather value.
	uint8_t density;				///< The mask's density value.
	uint8_t defaultColor;			///< The mask's default color regions outside the enclosing rectangle.
};

PSD_NAMESPACE_END
