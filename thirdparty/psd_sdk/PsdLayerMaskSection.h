// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

struct Layer;


/// \ingroup Sections
/// \class LayerMaskSection
/// \brief A struct representing the information extracted from the Layer Mask section.
/// \sa Layer
struct LayerMaskSection
{
	Layer* layers;						///< An array of layers, having layerCount entries.
	unsigned int layerCount;			///< The number of layers stored in the array.

	uint16_t overlayColorSpace;			///< The color space of the overlay (undocumented, not used yet).
	uint16_t opacity;					///< The global opacity level (0 = transparent, 100 = opaque, not used yet).
	uint8_t kind;						///< The global kind of layer (not used yet).

	bool hasTransparencyMask;			///< Whether the layer data contains a transparency mask or not.
};

PSD_NAMESPACE_END
