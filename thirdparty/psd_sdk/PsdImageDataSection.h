// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

struct PlanarImage;


/// \ingroup Sections
/// \class ImageDataSection
/// \brief A struct representing the information extracted from the Image Data section.
/// \sa PlanarImage
struct ImageDataSection
{
	PlanarImage* images;					///< An array of planar images, having imageCount entries.
	unsigned int imageCount;				///< The number of planar images stored in the array.
};

PSD_NAMESPACE_END
