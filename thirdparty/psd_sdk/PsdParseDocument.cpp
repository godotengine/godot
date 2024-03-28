// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#include "PsdPch.h"
#include "PsdParseDocument.h"

#include "PsdDocument.h"
#include "PsdSyncFileReader.h"
#include "PsdSyncFileUtil.h"
#include "PsdKey.h"
#include "PsdMemoryUtil.h"
#include "PsdAllocator.h"
#include "PsdFile.h"
#include "PsdLog.h"
#include <cstring>

#include "core/io/stream_peer.h"

PSD_NAMESPACE_BEGIN

// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
Document* CreateDocument(Ref<StreamPeerBuffer> file, Allocator* allocator)
{
	ERR_FAIL_COND_V(file.is_null(), nullptr);
	file->seek(0u);

	// check signature, must be "8BPS"
	{
		const uint32_t signature = file->get_u32();
		if (signature != util::Key<'8', 'B', 'P', 'S'>::VALUE)
		{
			PSD_ERROR("PsdExtract", "File seems to be corrupt, signature does not match \"8BPS\".");
			return nullptr;
		}
	}

	// check version, must be 1
	{
		const uint16_t version = file->get_u16();
		if (version != 1)
		{
			PSD_ERROR("PsdExtract", "File seems to be corrupt, version does not match 1.");
			return nullptr;
		}
	}

	// check reserved bytes, must be zero
	{
		const uint8_t expected[6] = { 0u, 0u, 0u, 0u, 0u, 0u };
		uint8_t zeroes[6] = {};
		file->get_data(zeroes, 6u);

		if (memcmp(zeroes, expected, sizeof(uint8_t)*6) != 0)
		{
			PSD_ERROR("PsdExtract", "File seems to be corrupt, reserved bytes are not zero.");
			return nullptr;
		}
	}

	Document* document = memoryUtil::Allocate<Document>(allocator);

	// read in the number of channels.
	// this is the number of channels contained in the document for all layers, including any alpha channels.
	// e.g. for an RGB document with 3 alpha channels, this would be 3 (RGB) + 3 (Alpha) = 6 channels.
	// however, note that individual layers can have extra channels for transparency masks, vector masks, and user masks.
	// this is different from layer to layer.
	document->channelCount = file->get_u16();

	// read rest of header information
	document->height = file->get_u32();
	document->width = file->get_u32();
	document->bitsPerChannel = file->get_u16();
	document->colorMode = file->get_u16();

	// grab offsets into different sections
	{
		const uint32_t length = file->get_u32();

		document->colorModeDataSection.offset = file->get_position();
		document->colorModeDataSection.length = length;

		file->seek(file->get_position() + length);
	}
	{
		const uint32_t length = file->get_u32();

		document->imageResourcesSection.offset = file->get_position();
		document->imageResourcesSection.length = length;

		file->seek(file->get_position() + length);
	}
	{
		const uint32_t length = file->get_u32();

		document->layerMaskInfoSection.offset = file->get_position();
		document->layerMaskInfoSection.length = length;

		file->seek(file->get_position() + length);
	}
	{
		// note that the image data section does NOT store its length in the first 4 bytes
		document->imageDataSection.offset = file->get_position();
		document->imageDataSection.length = static_cast<uint32_t>(file->get_size() - file->get_position());
	}

	return document;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void DestroyDocument(Document*& document, Allocator* allocator)
{
	PSD_ASSERT_NOT_NULL(document);
	PSD_ASSERT_NOT_NULL(allocator);

	memoryUtil::Free(allocator, document);
}

PSD_NAMESPACE_END
