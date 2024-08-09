// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#include "PsdPch.h"
#include "PsdParseImageResourcesSection.h"

#include "PsdImageResourcesSection.h"
#include "PsdDocument.h"
#include "PsdImageResourceType.h"
#include "PsdAlphaChannel.h"
#include "PsdThumbnail.h"
#include "PsdKey.h"
#include "PsdBitUtil.h"
#include "PsdSyncFileReader.h"
#include "PsdSyncFileUtil.h"
#include "PsdMemoryUtil.h"
#include "PsdAllocator.h"
#include "PsdLog.h"


PSD_NAMESPACE_BEGIN

// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
ImageResourcesSection* ParseImageResourcesSection(const Document* document, Ref<StreamPeerBuffer> file, Allocator* allocator)
{
	PSD_ASSERT_NOT_NULL(file);
	PSD_ASSERT_NOT_NULL(allocator);

	ImageResourcesSection* imageResources = memoryUtil::Allocate<ImageResourcesSection>(allocator);
	imageResources->alphaChannels = nullptr;
	imageResources->alphaChannelCount = 0u;
	imageResources->iccProfile = nullptr;
	imageResources->sizeOfICCProfile = 0;
	imageResources->exifData = nullptr;
	imageResources->sizeOfExifData = 0;
	imageResources->containsRealMergedData = true;
	imageResources->xmpMetadata = nullptr;
	imageResources->thumbnail = nullptr;

	file->seek(document->imageResourcesSection.offset);

	int64_t leftToRead = document->imageResourcesSection.length;
	while (leftToRead > 0)
	{
		const uint32_t signature = file->get_u32();
		if ((signature != util::Key<'8', 'B', 'I', 'M'>::VALUE) && (signature != util::Key<'p', 's', 'd', 'M'>::VALUE))
		{
			PSD_ERROR("ImageResources", "Image resources section seems to be corrupt, signature does not match \"8BIM\".");
			return imageResources;
		}

		const uint16_t id = file->get_u16();

		// the resource name is stored as a Pascal string. note that the string is padded to make the size even.
		uint8_t name[512] = {};
		const uint8_t nameLength = file->get_u8();
		const uint32_t paddedNameLength = bitUtil::RoundUpToMultiple(nameLength+1u, 2u);
		file->get_data(name, paddedNameLength - 1u);

		// the resource data size is also padded to make the size even
		uint32_t resourceSize = file->get_u32();
		resourceSize = bitUtil::RoundUpToMultiple(resourceSize, 2u);

		// work out the next position we need to read from once, no matter which image resource we're going to read
		const uint64_t nextReaderPosition = file->get_position() + resourceSize;

		switch (id)
		{
			case imageResource::IPTC_NAA:
			case imageResource::CAPTION_DIGEST:
			case imageResource::PRINT_INFORMATION:
			case imageResource::PRINT_STYLE:
			case imageResource::PRINT_SCALE:
			case imageResource::PRINT_FLAGS:
			case imageResource::PRINT_FLAGS_INFO:
			case imageResource::PRINT_INFO:
			case imageResource::RESOLUTION_INFO:
				// we are currently not interested in this resource
				break;

			case imageResource::DISPLAY_INFO:
			{
				// the display info resource stores color information and opacity for extra channels contained
				// in the document. these extra channels could be alpha/transparency, as well as spot color
				// channels used for printing.

				// check whether storage for alpha channels has been allocated yet
				// (imageResource::ALPHA_CHANNEL_ASCII_NAMES stores the channel names)
				if (!imageResources->alphaChannels)
				{
					// note that this assumes RGB mode
					const unsigned int channelCount = document->channelCount - 3;
					imageResources->alphaChannelCount = channelCount;
					imageResources->alphaChannels = memoryUtil::AllocateArray<AlphaChannel>(allocator, channelCount);
				}

				const uint32_t version = file->get_u32();
				PSD_UNUSED(version);

				for (unsigned int i = 0u; i < imageResources->alphaChannelCount; ++i)
				{
					AlphaChannel* channel = imageResources->alphaChannels + i;
					channel->colorSpace = file->get_u16();
					channel->color[0] = file->get_u16();
					channel->color[1] = file->get_u16();
					channel->color[2] = file->get_u16();
					channel->color[3] = file->get_u16();
					channel->opacity = file->get_u16();
					channel->mode = file->get_u8();
				}
			}
			break;

			case imageResource::GLOBAL_ANGLE:
			case imageResource::GLOBAL_ALTITUDE:
			case imageResource::COLOR_HALFTONING_INFO:
			case imageResource::COLOR_TRANSFER_FUNCTIONS:
			case imageResource::MULTICHANNEL_HALFTONING_INFO:
			case imageResource::MULTICHANNEL_TRANSFER_FUNCTIONS:
			case imageResource::LAYER_STATE_INFORMATION:
			case imageResource::LAYER_GROUP_INFORMATION:
			case imageResource::LAYER_GROUP_ENABLED_ID:
			case imageResource::LAYER_SELECTION_ID:
			case imageResource::GRID_GUIDES_INFO:
			case imageResource::URL_LIST:
			case imageResource::SLICES:
			case imageResource::PIXEL_ASPECT_RATIO:
			case imageResource::ICC_UNTAGGED_PROFILE:
			case imageResource::ID_SEED_NUMBER:
			case imageResource::BACKGROUND_COLOR:
			case imageResource::ALPHA_CHANNEL_UNICODE_NAMES:
			case imageResource::ALPHA_IDENTIFIERS:
			case imageResource::COPYRIGHT_FLAG:
			case imageResource::PATH_SELECTION_STATE:
			case imageResource::ONION_SKINS:
			case imageResource::TIMELINE_INFO:
			case imageResource::SHEET_DISCLOSURE:
			case imageResource::WORKING_PATH:
			case imageResource::MAC_PRINT_MANAGER_INFO:
			case imageResource::WINDOWS_DEVMODE:
				// we are currently not interested in this resource
				break;

			case imageResource::VERSION_INFO:
			{
				const uint32_t version = file->get_u32();
				PSD_UNUSED(version);

				const uint8_t hasRealMergedData = file->get_u8();
				imageResources->containsRealMergedData = (hasRealMergedData != 0u);
			}
			break;

			case imageResource::THUMBNAIL_RESOURCE:
			{
				Thumbnail* thumbnail = memoryUtil::Allocate<Thumbnail>(allocator);
				imageResources->thumbnail = thumbnail;

				const uint32_t format = file->get_u32();
				PSD_UNUSED(format);

				const uint32_t width = file->get_u32();
				const uint32_t height = file->get_u32(); 

				const uint32_t widthInBytes = file->get_u32();
				PSD_UNUSED(widthInBytes);

				const uint32_t totalSize = file->get_u32();
				PSD_UNUSED(totalSize);

				const uint32_t binaryJpegSize = file->get_u32();

				const uint16_t bitsPerPixel = file->get_u16();
				PSD_UNUSED(bitsPerPixel);
				const uint16_t numberOfPlanes = file->get_u16();
				PSD_UNUSED(numberOfPlanes);

				thumbnail->width = width;
				thumbnail->height = height;
				thumbnail->binaryJpegSize = binaryJpegSize;
				thumbnail->binaryJpeg = memoryUtil::AllocateArray<uint8_t>(allocator, binaryJpegSize);

				file->get_data(thumbnail->binaryJpeg, binaryJpegSize);
			}
			break;

			case imageResource::XMP_METADATA:
			{
				// load the XMP metadata as raw data
				PSD_ASSERT(!imageResources->xmpMetadata, "File contains more than one XMP metadata resource.");
				imageResources->xmpMetadata = memoryUtil::AllocateArray<char>(allocator, resourceSize);
				file->get_data((uint8_t *)imageResources->xmpMetadata, resourceSize);
			}
			break;

			case imageResource::ICC_PROFILE:
			{
				// load the ICC profile as raw data
				PSD_ASSERT(!imageResources->iccProfile, "File contains more than one ICC profile.");
				imageResources->iccProfile = memoryUtil::AllocateArray<uint8_t>(allocator, resourceSize);
				imageResources->sizeOfICCProfile = resourceSize;
				file->get_data(imageResources->iccProfile, resourceSize);
			}
			break;

			case imageResource::EXIF_DATA:
			{
				// load the EXIF data as raw data
				PSD_ASSERT(!imageResources->exifData, "File contains more than one EXIF data block.");
				imageResources->exifData = memoryUtil::AllocateArray<uint8_t>(allocator, resourceSize);
				imageResources->sizeOfExifData = resourceSize;
				file->get_data(imageResources->exifData, resourceSize);
			}
			break;

			case imageResource::ALPHA_CHANNEL_ASCII_NAMES:
			{
				// check whether storage for alpha channels has been allocated yet
				// (imageResource::DISPLAY_INFO stores the channel color data)
				if (!imageResources->alphaChannels)
				{
					// note that this assumes RGB mode
					const unsigned int channelCount = document->channelCount - 3;
					imageResources->alphaChannelCount = channelCount;
					imageResources->alphaChannels = memoryUtil::AllocateArray<AlphaChannel>(allocator, channelCount);
				}

				// the names of the alpha channels are stored as a series of Pascal strings
				unsigned int channel = 0;
				int64_t remaining = resourceSize;
				while (remaining > 0)
				{
					char channelName[512] = {};
					const uint8_t channelNameLength = file->get_u8();
					if (channelNameLength > 0)
					{
						file->get_data((uint8_t *)channelName, channelNameLength);
					}

					remaining -= 1 + channelNameLength;

					if (channel < imageResources->alphaChannelCount)
					{
						imageResources->alphaChannels[channel].asciiName.Assign(channelName);
						++channel;
					}
				}
			}
			break;

			default:
				// this is a resource we know nothing about
				break;
		};

		file->seek(nextReaderPosition);
		leftToRead = static_cast<int64_t>(document->imageResourcesSection.offset + document->imageResourcesSection.length) - static_cast<int64_t>(nextReaderPosition);
	}

	return imageResources;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void DestroyImageResourcesSection(ImageResourcesSection*& section, Allocator* allocator)
{
	PSD_ASSERT_NOT_NULL(section);
	PSD_ASSERT_NOT_NULL(allocator);

	if (section->thumbnail)
	{
		memoryUtil::FreeArray(allocator, section->thumbnail->binaryJpeg);
	}

	memoryUtil::Free(allocator, section->thumbnail);
	memoryUtil::FreeArray(allocator, section->xmpMetadata);
	memoryUtil::FreeArray(allocator, section->exifData);
	memoryUtil::FreeArray(allocator, section->iccProfile);
	memoryUtil::FreeArray(allocator, section->alphaChannels);
	memoryUtil::Free(allocator, section);
}

PSD_NAMESPACE_END
