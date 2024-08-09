// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

/// \ingroup Types
/// \namespace compressionType
/// \brief A namespace holding compression types known by Photoshop.
namespace compressionType
{
	enum Enum
	{
		RAW = 0,								///< Raw data.
		RLE = 1,								///< RLE-compressed data (using the PackBits algorithm).
		ZIP = 2,								///< ZIP-compressed data.
		ZIP_WITH_PREDICTION = 3					///< ZIP-compressed data with prediction (delta-encoding).
	};
}

PSD_NAMESPACE_END
