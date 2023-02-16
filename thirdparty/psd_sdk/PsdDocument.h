// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once

#include "PsdSection.h"


PSD_NAMESPACE_BEGIN

/// \ingroup Types
/// \class Document
/// \brief A struct storing the document-wide information and sections contained in a .PSD file.
/// \sa Section
struct Document
{
	unsigned int width;							///< The width of the document.
	unsigned int height;						///< The height of the document.
	unsigned int channelCount;					///< The number of channels stored in the document, including any additional alpha channels.
	unsigned int bitsPerChannel;				///< The bits per channel (8, 16 or 32).
	unsigned int colorMode;						///< The color mode the document is stored in, can be any of \ref colorMode::Enum.

	Section colorModeDataSection;				///< Color mode data section.
	Section imageResourcesSection;				///< Image Resources section.
	Section layerMaskInfoSection;				///< Layer Mask Info section.
	Section imageDataSection;					///< Image Data section.
};

PSD_NAMESPACE_END
