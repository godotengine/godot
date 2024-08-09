// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

/// \ingroup Types
/// \class Thumbnail
/// \brief A struct representing a thumbnail as stored in the image resources section.
/// \sa ImageResourcesSection
struct Thumbnail
{
	uint32_t width;
	uint32_t height;
	uint32_t binaryJpegSize;
	uint8_t* binaryJpeg;
};

PSD_NAMESPACE_END
