// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

/// \ingroup Types
/// \class Channel
/// \brief A struct representing a channel as stored in the layers of the Layer Mask section.
/// \sa Layer
struct Channel
{
	uint64_t fileOffset;				///< The offset from the start of the file where the channel's data is stored.
	uint32_t size;						///< The size of the channel data to be read from the file.
	void* data;							///< Planar data the size of the layer the channel belongs to. Data is only valid if the type member indicates so.
	int16_t type;						///< One of the \ref channelType constants denoting the type of data.
};

PSD_NAMESPACE_END
