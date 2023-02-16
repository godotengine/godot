// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

/// \ingroup Types
/// \class ExportLayer
/// \brief A struct representing a layer as exported to the Layer Mask section.
struct ExportLayer
{
	// the SDK currently supports R, G, B, A
	static const unsigned int MAX_CHANNEL_COUNT = 4u;

	int32_t top;
	int32_t left;
	int32_t bottom;
	int32_t right;
	char* name;

	void* channelData[MAX_CHANNEL_COUNT];
	uint32_t channelSize[MAX_CHANNEL_COUNT];
	uint16_t channelCompression[MAX_CHANNEL_COUNT];
};

PSD_NAMESPACE_END
