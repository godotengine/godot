// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once

#include "PsdFixedSizeString.h"


PSD_NAMESPACE_BEGIN

/// \ingroup Types
/// \class AlphaChannel
/// \brief A struct representing an alpha channel as stored in the image resources section.
/// \remark Note that the image data for alpha channels is stored in the image data section.
/// \sa ImageResourcesSection
struct AlphaChannel
{
	struct Mode
	{
		enum Enum
		{
			ALPHA = 0,									///< The channel stores alpha data.
			INVERTED_ALPHA = 1,							///< The channel stores inverted alpha data.
			SPOT = 2									///< The channel stores spot color data.
		};
	};

	util::FixedSizeString asciiName;				///< The channel's ASCII name.
	uint16_t colorSpace;							///< The color space the colors are stored in.
	uint16_t color[4];								///< 16-bit color data with 0 being black and 65535 being white (assuming RGBA).
	uint16_t opacity;								///< The channel's opacity in the range [0, 100].
	uint8_t mode;									///< The channel's mode, one of AlphaChannel::Mode.
};

PSD_NAMESPACE_END
