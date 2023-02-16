// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#include "PsdPch.h"
#include "PsdParseImageDataSection.h"

#include "PsdImageDataSection.h"
#include "PsdDocument.h"
#include "PsdCompressionType.h"
#include "PsdPlanarImage.h"
#include "PsdFile.h"
#include "PsdAllocator.h"
#include "PsdEndianConversion.h"
#include "PsdSyncFileReader.h"
#include "PsdSyncFileUtil.h"
#include "PsdMemoryUtil.h"
#include "PsdDecompressRle.h"
#include "PsdAssert.h"
#include "PsdLog.h"


PSD_NAMESPACE_BEGIN

namespace
{
	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	void EndianConvert(PlanarImage* images, unsigned int width, unsigned int height, unsigned int channelCount)
	{
		PSD_ASSERT_NOT_NULL(images);

		const unsigned int size = width*height;
		for (unsigned int i=0; i < channelCount; ++i)
		{
			T* planarData = static_cast<T*>(images[i].data);
			for (unsigned int j=0; j < size; ++j)
			{
				planarData[j] = endianUtil::BigEndianToNative(planarData[j]);
			}
		}
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static ImageDataSection* ReadImageDataSectionRaw(SyncFileReader& reader, Allocator* allocator, unsigned int width, unsigned int height, unsigned int channelCount, unsigned int bytesPerPixel)
	{
		const unsigned int size = width*height;
		if (size == 0)
			return nullptr;

		ImageDataSection* imageData = memoryUtil::Allocate<ImageDataSection>(allocator);
		imageData->imageCount = channelCount;
		imageData->images = memoryUtil::AllocateArray<PlanarImage>(allocator, channelCount);

		// read data for all channels at once
		for (unsigned int i=0; i < channelCount; ++i)
		{
			void* planarData = allocator->Allocate(size*bytesPerPixel, 16u);
			imageData->images[i].data = planarData;

			reader.Read(planarData, size*bytesPerPixel);
		}

		return imageData;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static ImageDataSection* ReadImageDataSectionRLE(SyncFileReader& reader, Allocator* allocator, unsigned int width, unsigned int height, unsigned int channelCount, unsigned int bytesPerPixel)
	{
		// the RLE-compressed data is preceded by a 2-byte data count for each scan line, per channel.
		// we store the size of the RLE data per channel, and assume a maximum of 256 channels.
		PSD_ASSERT(channelCount < 256, "Image data section has too many channels (%d).", channelCount);
		unsigned int channelSize[256] = {};
		unsigned int totalSize = 0;
		for (unsigned int i=0; i < channelCount; ++i)
		{
			unsigned int size = 0u;
			for (unsigned int j=0; j < height; ++j)
			{
				const uint16_t dataCount = fileUtil::ReadFromFileBE<uint16_t>(reader);
				size += dataCount;
			}

			channelSize[i] = size;
			totalSize += size;
		}

		if (totalSize == 0)
			return nullptr;

		const unsigned int size = width*height;
		ImageDataSection* imageData = memoryUtil::Allocate<ImageDataSection>(allocator);
		imageData->imageCount = channelCount;
		imageData->images = memoryUtil::AllocateArray<PlanarImage>(allocator, channelCount);

		for (unsigned int i=0; i < channelCount; ++i)
		{
			void* planarData = allocator->Allocate(size*bytesPerPixel, 16u);
			imageData->images[i].data = planarData;

			// read RLE data, and uncompress into planar buffer
			const unsigned int rleSize = channelSize[i];
			uint8_t* rleData = static_cast<uint8_t*>(allocator->Allocate(rleSize, 4u));
			reader.Read(rleData, rleSize);

			imageUtil::DecompressRle(rleData, rleSize, static_cast<uint8_t*>(planarData), width*height*bytesPerPixel);

			allocator->Free(rleData);
		}

		return imageData;
	}
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
ImageDataSection* ParseImageDataSection(const Document* document, File* file, Allocator* allocator)
{
	PSD_ASSERT_NOT_NULL(file);
	PSD_ASSERT_NOT_NULL(allocator);

	// this is the merged image. it is only stored if "maximize compatibility" is turned on when saving a PSD file.
	// image data is stored in planar order: first red data, then green data, and so on.
	// each plane is stored in scan-line order, with no padding bytes.

	// 8-bit values are stored directly.
	// 16-bit values are stored directly, even though they are stored as 15-bit+1 integers in the range 0...32768
	// internally in Photoshop, see https://forums.adobe.com/message/3472269
	// 32-bit values are stored directly as IEEE 32-bit floats.
	const Section& section = document->imageDataSection;
	if (section.length == 0)
	{
		PSD_ERROR("PSD", "Document does not contain an image data section.");
		return nullptr;
	}

	SyncFileReader reader(file);
	reader.SetPosition(section.offset);

	ImageDataSection* imageData = nullptr;
	const unsigned int width = document->width;
	const unsigned int height = document->height;
	const unsigned int bitsPerChannel = document->bitsPerChannel;
	const unsigned int channelCount = document->channelCount;
	const uint16_t compressionType = fileUtil::ReadFromFileBE<uint16_t>(reader);
	if (compressionType == compressionType::RAW)
	{
		imageData = ReadImageDataSectionRaw(reader, allocator, width, height, channelCount, bitsPerChannel / 8u);
	}
	else if (compressionType == compressionType::RLE)
	{
		imageData = ReadImageDataSectionRLE(reader, allocator, width, height, channelCount, bitsPerChannel / 8u);
	}
	else
	{
		PSD_ERROR("ImageData", "Unhandled compression type %u.", compressionType);
	}

	if (!imageData)
		return nullptr;

	if (!imageData->images)
		return imageData;

	// endian-convert the data
	switch (bitsPerChannel)
	{
		case 8:
			EndianConvert<uint8_t>(imageData->images, width, height, channelCount);
			break;

		case 16:
			EndianConvert<uint16_t>(imageData->images, width, height, channelCount);
			break;

		case 32:
			EndianConvert<float32_t>(imageData->images, width, height, channelCount);
			break;

		default:
			PSD_ERROR("ImageData", "Unhandled bits per channel: %u.", bitsPerChannel);
			break;
	}

	return imageData;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void DestroyImageDataSection(ImageDataSection*& section, Allocator* allocator)
{
	PSD_ASSERT_NOT_NULL(section);
	PSD_ASSERT_NOT_NULL(allocator);

	for (unsigned int i=0; i < section->imageCount; ++i)
	{
		allocator->Free(section->images[i].data);
	}

	memoryUtil::FreeArray(allocator, section->images);
	memoryUtil::Free(allocator, section);
}

PSD_NAMESPACE_END
