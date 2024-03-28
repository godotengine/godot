// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

/// \ingroup Types
/// \class PlanarImage
/// \brief A struct representing a planar image as stored in the Image Data section.
/// \sa ImageDataSection
struct PlanarImage
{
	void* data;					///< Planar data the size of the document's canvas.
};

PSD_NAMESPACE_END
