// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

/// \ingroup Types
/// \namespace channelType
/// \brief A namespace holding constants to distinguish between the types of data a channel can hold.
namespace channelType
{
	enum Enum
	{
		INVALID = 32767,					///< Internal value. Used to denote that a channel no longer holds valid data.

		R = 0,								///< Type denoting the R channel, not necessarily the first in a RGB Color Mode document.
		G = 1,								///< Type denoting the G channel, not necessarily the second in a RGB Color Mode document.
		B = 2,								///< Type denoting the B channel, not necessarily the third in a RGB Color Mode document.

		TRANSPARENCY_MASK = -1,				///< The layer's channel data is a transparency mask.
		LAYER_OR_VECTOR_MASK = -2,			///< The layer's channel data is either a layer or vector mask.
		LAYER_MASK = -3						///< The layer's channel data is a layer mask.
	};
}

PSD_NAMESPACE_END
