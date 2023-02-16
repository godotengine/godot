// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#include "PsdPch.h"
#include "PsdExport.h"

#include "PsdMemoryUtil.h"
#include "PsdImageResourceType.h"
#include "PsdExportDocument.h"
#include "PsdDecompressRle.h"
#include "PsdSyncFileWriter.h"
#include "PsdSyncFileUtil.h"
#include "PsdKey.h"
#include "PsdChannelType.h"
#include "PsdBitUtil.h"
#include "PsdThumbnail.h"
#include "Psdminiz.h"
#include <string.h>


PSD_NAMESPACE_BEGIN

namespace
{
	static const char XMP_HEADER[]= "<x:xmpmeta xmlns:x = \"adobe:ns:meta/\">\n"
		"<rdf:RDF xmlns:rdf = \"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n"
		"<rdf:Description rdf:about=\"\"\n"
		"xmlns:xmp = \"http://ns.adobe.com/xap/1.0/\"\n"
		"xmlns:dc = \"http://purl.org/dc/elements/1.1/\"\n"
		"xmlns:photoshop = \"http://ns.adobe.com/photoshop/1.0/\"\n"
		"xmlns:xmpMM = \"http://ns.adobe.com/xap/1.0/mm/\"\n"
		"xmlns:stEvt = \"http://ns.adobe.com/xap/1.0/sType/ResourceEvent#\">\n";

	static const char XMP_FOOTER[] =
		"</rdf:Description>\n"
		"</rdf:RDF>\n"
		"</x:xmpmeta>\n";


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T> struct Mask {};
	template <> struct Mask<uint8_t> { static const uint32_t Value = 0xFFu; };
	template <> struct Mask<uint16_t> { static const uint32_t Value = 0xFFFFu; };

	const uint32_t Mask<uint8_t>::Value;
	const uint32_t Mask<uint16_t>::Value;


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static char* CreateString(Allocator* allocator, const char* str)
	{
		const size_t length = strlen(str);
		const size_t paddedLength = bitUtil::RoundUpToMultiple(length + 1u, static_cast<size_t>(4u));
		char* newString = memoryUtil::AllocateArray<char>(allocator, paddedLength);

		// clear and copy null terminator as well
		memset(newString, 0, paddedLength);
		memcpy(newString, str, length + 1u);

		return newString;
	}

	
	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static void DestroyString(Allocator* allocator, char*& str)
	{
		memoryUtil::FreeArray(allocator, str);
		str = nullptr;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static uint16_t GetChannelCount(ExportLayer* layer)
	{
		uint16_t count = 0u;
		for (unsigned int i = 0u; i < ExportLayer::MAX_CHANNEL_COUNT; ++i)
		{
			if (layer->channelData[i])
			{
				++count;
			}
		}

		return count;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static uint32_t GetChannelDataSize(ExportLayer* layer)
	{
		uint32_t size = 0u;
		for (unsigned int i = 0u; i < ExportLayer::MAX_CHANNEL_COUNT; ++i)
		{
			if (layer->channelData[i])
			{
				size += layer->channelSize[i];
			}
		}

		return size;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static unsigned int GetChannelIndex(exportChannel::Enum channel)
	{
		switch (channel)
		{
			case exportChannel::GRAY:
				return 0u;

			case exportChannel::RED:
				return 0u;

			case exportChannel::GREEN:
				return 1u;

			case exportChannel::BLUE:
				return 2u;

			case exportChannel::ALPHA:
				return 3u;

			default:
				return 0u;
		}
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static int16_t GetChannelId(unsigned int channelIndex)
	{
		switch (channelIndex)
		{
			case 0u:
				return channelType::R;

			case 1u:
				return channelType::G;

			case 2u:
				return channelType::B;

			case 3u:
				return channelType::TRANSPARENCY_MASK;

			default:
				return 0u;
		}
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static uint32_t GetExtraDataLength(ExportLayer* layer)
	{
		const uint8_t nameLength = static_cast<uint8_t>(strlen(layer->name));
		const uint32_t paddedNameLength = bitUtil::RoundUpToMultiple(nameLength + 1u, 4u);

		// includes the lengths of the layer mask data and layer blending ranges data
		return (4u + 4u + paddedNameLength);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static uint32_t GetLayerInfoSectionLength(ExportDocument* document)
	{
		// the layer info section includes the following data:
		// - layer count (2)
		//   per layer:
		//   - top, left, bottom, right (16)
		//   - channel count (2)
		//     per channel
		//     - channel ID and size (6)
		//   - blend mode signature (4)
		//   - blend mode key (4)
		//   - opacity, clipping, flags, filler (4)
		//   - extra data (variable)
		//     - length (4)
		//     - layer mask data length (4)
		//     - layer blending ranges length (4)
		//     - padded name (variable)
		// - all channel data (variable)
		//   - compression (2)
		//   - channel data (variable)

		uint32_t size = 2u + 4u;
		for (unsigned int i = 0u; i < document->layerCount; ++i)
		{
			ExportLayer* layer = document->layers + i;
			size += 16u + 2u + GetChannelCount(layer) * 6u + 4u + 4u + 4u + GetExtraDataLength(layer) + 4u;
			size += GetChannelDataSize(layer) + GetChannelCount(layer) * 2u;
		}

		return size;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static uint32_t GetImageResourceSize(void)
	{
		uint32_t size = 0u;
		size += sizeof(uint32_t);			// signature
		size += sizeof(uint16_t);			// resource ID		
		size += 2u;							// padded name, 2 zero bytes
		size += sizeof(uint32_t);			// resource size

		return size;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static uint32_t GetMetaDataResourceSize(ExportDocument* document)
	{
		size_t metaDataSize = sizeof(XMP_HEADER)-1u;
		for (unsigned int i = 0u; i < document->attributeCount; ++i)
		{
			metaDataSize += strlen("<xmp:>");
			metaDataSize += strlen(document->attributes[i].name)*2u;
			metaDataSize += strlen(document->attributes[i].value);
			metaDataSize += strlen("</xmp:>\n");
		}
		metaDataSize += sizeof(XMP_FOOTER)-1u;

		return static_cast<uint32_t>(metaDataSize);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static uint32_t GetIccProfileResourceSize(ExportDocument* document)
	{
		return document->sizeOfICCProfile;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static uint32_t GetExifDataResourceSize(ExportDocument* document)
	{
		return document->sizeOfExifData;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static uint32_t GetThumbnailResourceSize(ExportDocument* document)
	{
		return document->thumbnail->binaryJpegSize + 28u;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static uint32_t GetDisplayInfoResourceSize(ExportDocument* document)
	{
		// display info consists of 4-byte version, followed by 13 bytes per channel
		return sizeof(uint32_t) + 13u * document->alphaChannelCount;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static uint32_t GetChannelNamesResourceSize(ExportDocument* document)
	{
		size_t size = 0u;
		for (unsigned int i = 0u; i < document->alphaChannelCount; ++i)
		{
			size += document->alphaChannels[i].asciiName.GetLength() + 1u;
		}

		return static_cast<uint32_t>(size);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static uint32_t GetUnicodeChannelNamesResourceSize(ExportDocument* document)
	{
		size_t size = 0u;
		for (unsigned int i = 0u; i < document->alphaChannelCount; ++i)
		{
			// unicode strings are null terminated
			size += (document->alphaChannels[i].asciiName.GetLength() + 1u)*2u + 4u;
		}

		return static_cast<uint32_t>(size);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static void WriteImageResource(SyncFileWriter& writer, uint16_t id, uint32_t resourceSize)
	{
		const uint32_t signature = util::Key<'8', 'B', 'I', 'M'>::VALUE;
		fileUtil::WriteToFileBE(writer, signature);
		fileUtil::WriteToFileBE(writer, id);

		// padded name, unused
		fileUtil::WriteToFileBE(writer, uint8_t(0u));
		fileUtil::WriteToFileBE(writer, uint8_t(0u));

		fileUtil::WriteToFileBE(writer, resourceSize);
	}
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
ExportDocument* CreateExportDocument(Allocator* allocator, unsigned int canvasWidth, unsigned int canvasHeight, unsigned int bitsPerChannel, exportColorMode::Enum colorMode)
{
	ExportDocument* document = memoryUtil::Allocate<ExportDocument>(allocator);
	memset(document, 0, sizeof(ExportDocument));

	document->width = canvasWidth;
	document->height = canvasHeight;
	document->bitsPerChannel = static_cast<uint16_t>(bitsPerChannel);
	document->colorMode = colorMode;

	document->attributeCount = 0u;
	document->layerCount = 0u;

	document->mergedImageData[0] = nullptr;
	document->mergedImageData[1] = nullptr;
	document->mergedImageData[2] = nullptr;

	document->alphaChannelCount = 0u;

	document->iccProfile = nullptr;
	document->sizeOfICCProfile = 0u;

	document->exifData = nullptr;
	document->sizeOfExifData = 0u;

	document->thumbnail = nullptr;

	return document;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void DestroyExportDocument(ExportDocument*& document, Allocator* allocator)
{
	PSD_ASSERT_NOT_NULL(document);
	PSD_ASSERT_NOT_NULL(allocator);

	if (document->thumbnail)
	{
		memoryUtil::FreeArray(allocator, document->thumbnail->binaryJpeg);
	}
	memoryUtil::Free(allocator, document->thumbnail);

	memoryUtil::FreeArray(allocator, document->exifData);
	memoryUtil::FreeArray(allocator, document->iccProfile);

	memoryUtil::FreeArray(allocator, document->mergedImageData[0]);
	memoryUtil::FreeArray(allocator, document->mergedImageData[1]);
	memoryUtil::FreeArray(allocator, document->mergedImageData[2]);

	for (unsigned int i = 0u; i < document->alphaChannelCount; ++i)
	{
		memoryUtil::FreeArray(allocator, document->alphaChannelData[i]);
	}

	for (unsigned int i = 0u; i < document->attributeCount; ++i)
	{
		DestroyString(allocator, document->attributes[i].name);
		DestroyString(allocator, document->attributes[i].value);
	}

	for (unsigned int i = 0u; i < document->layerCount; ++i)
	{
		DestroyString(allocator, document->layers[i].name);

		for (unsigned int j = 0u; j < ExportLayer::MAX_CHANNEL_COUNT; ++j)
		{
			const uint16_t compression = document->layers[i].channelCompression[j];
			void*& data = document->layers[i].channelData[j];
			if ((compression == compressionType::ZIP) ||
				(compression == compressionType::ZIP_WITH_PREDICTION))
			{
				// data was allocated by miniz
				free(data);
			}
			else
			{
				memoryUtil::FreeArray(allocator, data);
			}
			data = nullptr;
		}
	}

	memoryUtil::Free(allocator, document);
	document = nullptr;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
unsigned int AddMetaData(ExportDocument* document, Allocator* allocator, const char* name, const char* value)
{
	const unsigned int index = document->attributeCount;
	++document->attributeCount;

	UpdateMetaData(document, allocator, index, name, value);

	return index;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void UpdateMetaData(ExportDocument* document, Allocator* allocator, unsigned int index, const char* name, const char* value)
{
	ExportMetaDataAttribute* attribute = document->attributes + index;
	DestroyString(allocator, attribute->name);
	DestroyString(allocator, attribute->value);
	attribute->name = CreateString(allocator, name);
	attribute->value = CreateString(allocator, value);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void SetICCProfile(ExportDocument* document, Allocator* allocator, void* rawProfileData, uint32_t size)
{
	memoryUtil::FreeArray(allocator, document->iccProfile);
	document->iccProfile = memoryUtil::AllocateArray<uint8_t>(allocator, size);
	document->sizeOfICCProfile = size;

	memcpy(document->iccProfile, rawProfileData, size);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void SetEXIFData(ExportDocument* document, Allocator* allocator, void* rawExifData, uint32_t size)
{
	memoryUtil::FreeArray(allocator, document->exifData);
	document->exifData = memoryUtil::AllocateArray<uint8_t>(allocator, size);
	document->sizeOfExifData = size;
	memcpy(document->exifData, rawExifData, size);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void SetJpegThumbnail(ExportDocument* document, Allocator* allocator, uint32_t width, uint32_t height, void* rawJpegData, uint32_t size)
{
	if (document->thumbnail)
	{
		memoryUtil::FreeArray(allocator, document->thumbnail->binaryJpeg);
	}
	memoryUtil::Free(allocator, document->thumbnail);

	Thumbnail* thumbnail = memoryUtil::Allocate<Thumbnail>(allocator);
	thumbnail->width = width;
	thumbnail->height = height;
	thumbnail->binaryJpeg = memoryUtil::AllocateArray<uint8_t>(allocator, size);
	thumbnail->binaryJpegSize = size;

	memcpy(thumbnail->binaryJpeg, rawJpegData, size);

	document->thumbnail = thumbnail;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
unsigned int AddLayer(ExportDocument* document, Allocator* allocator, const char* name)
{
	const unsigned int index = document->layerCount;
	++document->layerCount;

	ExportLayer* layer = document->layers + index;
	layer->name = CreateString(allocator, name);

	return index;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
template <typename T>
static void CreateDataRaw(Allocator* allocator, ExportLayer* layer, unsigned int channelIndex, const T* planarData, uint32_t width, uint32_t height)
{
	const uint32_t size = width*height;

	T* bigEndianData = memoryUtil::AllocateArray<T>(allocator, size);
	for (unsigned int i = 0u; i < size; ++i)
	{
		bigEndianData[i] = endianUtil::NativeToBigEndian(planarData[i]);
	}

	layer->channelData[channelIndex] = bigEndianData;
	layer->channelSize[channelIndex] = size*sizeof(T);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
template <typename T>
static void CreateDataRLE(Allocator* allocator, ExportLayer* layer, unsigned int channelIndex, const T* planarData, uint32_t width, uint32_t height)
{
	const uint32_t size = width*height;

	// each row needs two additional bytes for storing the size of the row's data.
	// we pack the data row by row, and copy it into the final buffer.
	uint8_t* rleData = memoryUtil::AllocateArray<uint8_t>(allocator, height*sizeof(uint16_t) + size*sizeof(T) * 2u);

	uint8_t* rleRowData = memoryUtil::AllocateArray<uint8_t>(allocator, width*sizeof(T) * 2u);
	T* bigEndianRowData = memoryUtil::AllocateArray<T>(allocator, width);
	unsigned int offset = 0u;
	for (unsigned int y = 0u; y < height; ++y)
	{
		for (unsigned int x = 0u; x < width; ++x)
		{
			bigEndianRowData[x] = endianUtil::NativeToBigEndian(planarData[y*width + x]);
		}

		const unsigned int compressedSize = imageUtil::CompressRle(reinterpret_cast<const uint8_t*>(bigEndianRowData), rleRowData, width*sizeof(T));
		PSD_ASSERT(compressedSize <= width*sizeof(T) * 2u, "RLE compressed data doesn't fit into provided buffer.");

		const uint16_t rleRowSize = endianUtil::NativeToBigEndian(static_cast<uint16_t>(compressedSize));

		// copy 2 bytes row size, and copy RLE data
		memcpy(rleData + y * sizeof(uint16_t), &rleRowSize, sizeof(uint16_t));
		memcpy(rleData + height*sizeof(uint16_t) + offset, rleRowData, compressedSize);

		offset += compressedSize;
	}
	memoryUtil::FreeArray(allocator, bigEndianRowData);
	memoryUtil::FreeArray(allocator, rleRowData);

	layer->channelData[channelIndex] = rleData;
	layer->channelSize[channelIndex] = offset + height * sizeof(uint16_t);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
template <typename T>
static void CreateDataZipPrediction(Allocator* allocator, ExportLayer* layer, unsigned int channelIndex, const T* planarData, uint32_t width, uint32_t height)
{
	const uint32_t size = width*height;

	T* deltaData = memoryUtil::AllocateArray<T>(allocator, size);
	T* allocation = deltaData;
	for (unsigned int y = 0; y < height; ++y)
	{
		*deltaData++ = *planarData++;
		for (unsigned int x = 1; x < width; ++x)
		{
			const uint32_t previous = planarData[-1];
			const uint32_t current = planarData[0];
			const uint32_t value = current - previous;

			*deltaData++ = static_cast<T>(value & Mask<T>::Value);
			++planarData;
		}
	}

	// convert to big endian
	for (unsigned int i = 0u; i < size; ++i)
	{
		allocation[i] = endianUtil::NativeToBigEndian(allocation[i]);
	}

	size_t zipDataSize = 0u;
	void* zipData = tdefl_compress_mem_to_heap(allocation, size*sizeof(T), &zipDataSize, TDEFL_WRITE_ZLIB_HEADER);

	layer->channelData[channelIndex] = zipData;
	layer->channelSize[channelIndex] = static_cast<uint32_t>(zipDataSize);

	memoryUtil::FreeArray(allocator, allocation);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
template <>
void CreateDataZipPrediction<float32_t>(Allocator* allocator, ExportLayer* layer, unsigned int channelIndex, const float32_t* planarData, uint32_t width, uint32_t height)
{
	const uint32_t size = width*height;

	// float data is first converted into planar data to allow for better compression.
	// this is done row by row, so if the bytes of the floats in a row consist of "1234123412341234" they will be turned into "1111222233334444".
	// the data is also converted to big-endian in the same loop.
	uint8_t* bigEndianPlanarData = memoryUtil::AllocateArray<uint8_t>(allocator, size*sizeof(float32_t));
	for (unsigned int y = 0u; y < height; ++y)
	{
		for (unsigned int x = 0u; x < width; ++x)
		{
			uint8_t asBytes[sizeof(float32_t)] = {};
			memcpy(asBytes, planarData + y*width + x, sizeof(float32_t));
			bigEndianPlanarData[y*width * sizeof(float32_t) + x + width * 0] = asBytes[3];
			bigEndianPlanarData[y*width * sizeof(float32_t) + x + width * 1] = asBytes[2];
			bigEndianPlanarData[y*width * sizeof(float32_t) + x + width * 2] = asBytes[1];
			bigEndianPlanarData[y*width * sizeof(float32_t) + x + width * 3] = asBytes[0];
		}
	}

	// now delta encode the individual bytes row by row
	uint8_t* deltaData = memoryUtil::AllocateArray<uint8_t>(allocator, size*sizeof(float32_t));
	for (unsigned int y = 0; y < height; ++y)
	{
		deltaData[y*width * sizeof(float32_t)] = bigEndianPlanarData[y*width * sizeof(float32_t)];
		for (unsigned int x = 1; x < width*4u; ++x)
		{
			const uint32_t previous = bigEndianPlanarData[y*width * sizeof(float32_t) + x - 1];
			const uint32_t current = bigEndianPlanarData[y*width * sizeof(float32_t) + x];
			const uint32_t value = current - previous;

			deltaData[y*width * sizeof(float32_t) + x] = static_cast<uint8_t>(value & 0xFFu);
		}
	}

	size_t zipDataSize = 0u;
	void* zipData = tdefl_compress_mem_to_heap(deltaData, size*sizeof(float32_t), &zipDataSize, TDEFL_WRITE_ZLIB_HEADER);

	layer->channelData[channelIndex] = zipData;
	layer->channelSize[channelIndex] = static_cast<uint32_t>(zipDataSize);

	memoryUtil::FreeArray(allocator, deltaData);
	memoryUtil::FreeArray(allocator, bigEndianPlanarData);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
template <typename T>
static void CreateDataZip(Allocator* allocator, ExportLayer* layer, unsigned int channelIndex, const T* planarData, uint32_t width, uint32_t height)
{
	const uint32_t size = width*height;

	T* bigEndianData = memoryUtil::AllocateArray<T>(allocator, size);
	for (unsigned int i = 0u; i < size; ++i)
	{
		bigEndianData[i] = endianUtil::NativeToBigEndian(planarData[i]);
	}

	size_t zipDataSize = 0u;
	void* zipData = tdefl_compress_mem_to_heap(bigEndianData, size*sizeof(T), &zipDataSize, TDEFL_WRITE_ZLIB_HEADER);

	layer->channelData[channelIndex] = zipData;
	layer->channelSize[channelIndex] = static_cast<uint32_t>(zipDataSize);

	memoryUtil::FreeArray(allocator, bigEndianData);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
template <>
void CreateDataZip<float32_t>(Allocator* allocator, ExportLayer* layer, unsigned int channelIndex, const float32_t* planarData, uint32_t width, uint32_t height)
{
	// yes, this specialization is *not *a bug.
	// in 32 bit per channel mode, Photoshop treats ZIP and ZIP_WITH_PREDICTION as being the same compression mode.
	// it insists on delta-encoding the data before zipping, presumably to get better compression.
	return CreateDataZipPrediction(allocator, layer, channelIndex, planarData, width, height);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
template <typename T>
void UpdateLayerImpl(ExportDocument* document, Allocator* allocator, unsigned int layerIndex, exportChannel::Enum channel, int left, int top, int right, int bottom, const T* planarData, compressionType::Enum compression)
{
	if (document->colorMode == exportColorMode::GRAYSCALE)
	{
		PSD_ASSERT((channel == exportChannel::GRAY) || (channel == exportChannel::ALPHA), "Wrong channel for this color mode.");
	}
	else if (document->colorMode == exportColorMode::RGB)
	{
		PSD_ASSERT((channel == exportChannel::RED) || (channel == exportChannel::GREEN) || (channel == exportChannel::BLUE) || (channel == exportChannel::ALPHA), "Wrong channel for this color mode.");
	}

	ExportLayer* layer = document->layers + layerIndex;
	const unsigned int channelIndex = GetChannelIndex(channel);

	// free old data
	{
		void* data = layer->channelData[channelIndex];
		if (data)
		{
			const uint16_t type = layer->channelCompression[channelIndex];
			if ((type == compressionType::ZIP) ||
				(type == compressionType::ZIP_WITH_PREDICTION))
			{
				// data was allocated by miniz
				free(data);
			}
			else
			{
				memoryUtil::FreeArray(allocator, data);
			}
		}
	}

	// prepare new data
	layer->top = top;
	layer->left = left;
	layer->bottom = bottom;
	layer->right = right;
	layer->channelCompression[channelIndex] = static_cast<uint16_t>(compression);

	PSD_ASSERT(right >= left, "Invalid layer bounds.");
	PSD_ASSERT(bottom >= top, "Invalid layer bounds.");
	const uint32_t width = static_cast<uint32_t>(right - left);
	const uint32_t height = static_cast<uint32_t>(bottom - top);

	if (compression == compressionType::RAW)
	{
		// raw data, copy directly and convert to big endian
		CreateDataRaw(allocator, layer, channelIndex, planarData, width, height);
	}
	else if (compression == compressionType::RLE)
	{
		// compress with RLE
		CreateDataRLE(allocator, layer, channelIndex, planarData, width, height);
	}
	else if (compression == compressionType::ZIP)
	{
		// compress with ZIP
		// note that this has a template specialization for 32-bit float data that forwards to ZipWithPrediction.
		CreateDataZip(allocator, layer, channelIndex, planarData, width, height);
	}
	else if (compression == compressionType::ZIP_WITH_PREDICTION)
	{
		// delta-encode, then compress with ZIP
		CreateDataZipPrediction(allocator, layer, channelIndex, planarData, width, height);
	}
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void UpdateLayer(ExportDocument* document, Allocator* allocator, unsigned int layerIndex, exportChannel::Enum channel, int left, int top, int right, int bottom, const uint8_t* planarData, compressionType::Enum compression)
{
	UpdateLayerImpl(document, allocator, layerIndex, channel, left, top, right, bottom, planarData, compression);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void UpdateLayer(ExportDocument* document, Allocator* allocator, unsigned int layerIndex, exportChannel::Enum channel, int left, int top, int right, int bottom, const uint16_t* planarData, compressionType::Enum compression)
{
	UpdateLayerImpl(document, allocator, layerIndex, channel, left, top, right, bottom, planarData, compression);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void UpdateLayer(ExportDocument* document, Allocator* allocator, unsigned int layerIndex, exportChannel::Enum channel, int left, int top, int right, int bottom, const float32_t* planarData, compressionType::Enum compression)
{
	UpdateLayerImpl(document, allocator, layerIndex, channel, left, top, right, bottom, planarData, compression);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
unsigned int AddAlphaChannel(ExportDocument* document, Allocator* allocator, const char* name, uint16_t r, uint16_t g, uint16_t b, uint16_t a, uint16_t opacity, AlphaChannel::Mode::Enum mode)
{
	PSD_UNUSED(allocator);

	const unsigned int index = document->alphaChannelCount;
	++document->alphaChannelCount;

	AlphaChannel* channel = document->alphaChannels + index;
	channel->asciiName.Assign(name);
	channel->colorSpace = 0u;
	channel->color[0] = r;
	channel->color[1] = g;
	channel->color[2] = b;
	channel->color[3] = a;
	channel->opacity = opacity;
	channel->mode = static_cast<uint8_t>(mode);

	return index;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
template <typename T>
void UpdateChannelImpl(ExportDocument* document, Allocator* allocator, unsigned int channelIndex, const T* data)
{
	// free old data
	memoryUtil::FreeArray(allocator, document->alphaChannelData[channelIndex]);

	// copy raw data
	const uint32_t size = document->width*document->height;
	T* channelData = memoryUtil::AllocateArray<T>(allocator, size);
	for (unsigned int i = 0u; i < size; ++i)
	{
		channelData[i] = endianUtil::NativeToBigEndian(data[i]);
	}
	document->alphaChannelData[channelIndex] = channelData;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void UpdateChannel(ExportDocument* document, Allocator* allocator, unsigned int channelIndex, const uint8_t* data)
{
	UpdateChannelImpl(document, allocator, channelIndex, data);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void UpdateChannel(ExportDocument* document, Allocator* allocator, unsigned int channelIndex, const uint16_t* data)
{
	UpdateChannelImpl(document, allocator, channelIndex, data);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void UpdateChannel(ExportDocument* document, Allocator* allocator, unsigned int channelIndex, const float32_t* data)
{
	UpdateChannelImpl(document, allocator, channelIndex, data);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
template <typename T>
void UpdateMergedImageImpl(ExportDocument* document, Allocator* allocator, const T* planarDataR, const T* planarDataG, const T* planarDataB)
{
	// free old data
	memoryUtil::FreeArray(allocator, document->mergedImageData[0]);
	memoryUtil::FreeArray(allocator, document->mergedImageData[1]);
	memoryUtil::FreeArray(allocator, document->mergedImageData[2]);

	// copy raw data
	const uint32_t size = document->width*document->height;
	T* memoryR = memoryUtil::AllocateArray<T>(allocator, size);
	T* memoryG = memoryUtil::AllocateArray<T>(allocator, size);
	T* memoryB = memoryUtil::AllocateArray<T>(allocator, size);
	for (unsigned int i = 0u; i < size; ++i)
	{
		memoryR[i] = endianUtil::NativeToBigEndian(planarDataR[i]);
		memoryG[i] = endianUtil::NativeToBigEndian(planarDataG[i]);
		memoryB[i] = endianUtil::NativeToBigEndian(planarDataB[i]);
	}
	document->mergedImageData[0] = memoryR;
	document->mergedImageData[1] = memoryG;
	document->mergedImageData[2] = memoryB;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void UpdateMergedImage(ExportDocument* document, Allocator* allocator, const uint8_t* planarDataR, const uint8_t* planarDataG, const uint8_t* planarDataB)
{
	UpdateMergedImageImpl(document, allocator, planarDataR, planarDataG, planarDataB);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void UpdateMergedImage(ExportDocument* document, Allocator* allocator, const uint16_t* planarDataR, const uint16_t* planarDataG, const uint16_t* planarDataB)
{
	UpdateMergedImageImpl(document, allocator, planarDataR, planarDataG, planarDataB);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void UpdateMergedImage(ExportDocument* document, Allocator* allocator, const float32_t* planarDataR, const float32_t* planarDataG, const float32_t* planarDataB)
{
	UpdateMergedImageImpl(document, allocator, planarDataR, planarDataG, planarDataB);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void WriteDocument(ExportDocument* document, Allocator* allocator, File* file)
{
	SyncFileWriter writer(file);

	// signature
	fileUtil::WriteToFileBE(writer, util::Key<'8', 'B', 'P', 'S'>::VALUE);

	// version
	fileUtil::WriteToFileBE(writer, static_cast<uint16_t>(1u));

	// reserved bytes
	const uint8_t zeroes[6] = { 0u, 0u, 0u, 0u, 0u, 0u };
	fileUtil::WriteToFile(writer, zeroes);

	// channel count
	const uint16_t documentChannelCount = static_cast<uint16_t>(document->colorMode + document->alphaChannelCount);
	fileUtil::WriteToFileBE(writer, documentChannelCount);

	// header
	const uint16_t mode = static_cast<uint16_t>(document->colorMode);
	fileUtil::WriteToFileBE(writer, document->height);
	fileUtil::WriteToFileBE(writer, document->width);
	fileUtil::WriteToFileBE(writer, document->bitsPerChannel);
	fileUtil::WriteToFileBE(writer, mode);

	if (document->bitsPerChannel == 32u)
	{
		// in 32-bit mode, Photoshop insists on having a color mode data section with magic info.
		// this whole section is undocumented. there's no information to be found on the web.
		// we write Photoshop's default values.
		const uint32_t colorModeSectionLength = 112u;
		fileUtil::WriteToFileBE(writer, colorModeSectionLength);
		{
			// tests suggest that this is some kind of HDR toning information
			const uint32_t key = util::Key<'h', 'd', 'r', 't'>::VALUE;
			fileUtil::WriteToFileBE(writer, key);

			fileUtil::WriteToFileBE(writer, static_cast<uint32_t>(3u));			// ?
			fileUtil::WriteToFileBE(writer, static_cast<float32_t>(0.23f));		// ?
			fileUtil::WriteToFileBE(writer, static_cast<uint32_t>(2u));			// ?

			fileUtil::WriteToFileBE(writer, static_cast<uint32_t>(8u));			// length of the following Unicode string
			fileUtil::WriteToFileBE(writer, static_cast<uint16_t>('D'));
			fileUtil::WriteToFileBE(writer, static_cast<uint16_t>('e'));
			fileUtil::WriteToFileBE(writer, static_cast<uint16_t>('f'));
			fileUtil::WriteToFileBE(writer, static_cast<uint16_t>('a'));
			fileUtil::WriteToFileBE(writer, static_cast<uint16_t>('u'));
			fileUtil::WriteToFileBE(writer, static_cast<uint16_t>('l'));
			fileUtil::WriteToFileBE(writer, static_cast<uint16_t>('t'));
			fileUtil::WriteToFileBE(writer, static_cast<uint16_t>('\0'));

			fileUtil::WriteToFileBE(writer, static_cast<uint16_t>(2u));			// ?
			fileUtil::WriteToFileBE(writer, static_cast<uint16_t>(2u));			// ?
			fileUtil::WriteToFileBE(writer, static_cast<uint16_t>(0u));			// ?
			fileUtil::WriteToFileBE(writer, static_cast<uint16_t>(0u));			// ?
			fileUtil::WriteToFileBE(writer, static_cast<uint16_t>(255u));		// ?
			fileUtil::WriteToFileBE(writer, static_cast<uint16_t>(255u));		// ?

			fileUtil::WriteToFileBE(writer, static_cast<uint8_t>(1u));			// ?
			fileUtil::WriteToFileBE(writer, static_cast<uint8_t>(1u));			// ?
			fileUtil::WriteToFileBE(writer, static_cast<uint32_t>(0u));			// ?
			fileUtil::WriteToFileBE(writer, static_cast<uint32_t>(0u));			// ?

			fileUtil::WriteToFileBE(writer, static_cast<float32_t>(16.0f));		// ?
			fileUtil::WriteToFileBE(writer, static_cast<uint32_t>(1u));			// ?
			fileUtil::WriteToFileBE(writer, static_cast<uint32_t>(1u));			// ?
			fileUtil::WriteToFileBE(writer, static_cast<float32_t>(1.0f));		// ?
		}
		{
			// HDR alpha information?
			const uint32_t key = util::Key<'h', 'd', 'r', 'a'>::VALUE;
			fileUtil::WriteToFileBE(writer, key);

			fileUtil::WriteToFileBE(writer, static_cast<uint32_t>(6u));			// number of following values
			fileUtil::WriteToFileBE(writer, static_cast<float32_t>(0.0f));		// ?
			fileUtil::WriteToFileBE(writer, static_cast<float32_t>(20.0f));		// ?
			fileUtil::WriteToFileBE(writer, static_cast<float32_t>(30.0f));		// ?
			fileUtil::WriteToFileBE(writer, static_cast<float32_t>(0.0f));		// ?
			fileUtil::WriteToFileBE(writer, static_cast<float32_t>(0.0f));		// ?
			fileUtil::WriteToFileBE(writer, static_cast<float32_t>(1.0f));		// ?

			fileUtil::WriteToFileBE(writer, static_cast<uint32_t>(0u));			// ?
			fileUtil::WriteToFileBE(writer, static_cast<uint16_t>(0u));			// ?
		}
	}
	else
	{
		// empty color mode data section
		fileUtil::WriteToFileBE(writer, static_cast<uint32_t>(0u));
	}

	// image resources
	{
		const bool hasMetaData = (document->attributeCount != 0u);
		const bool hasIccProfile = (document->iccProfile != nullptr);
		const bool hasExifData = (document->exifData != nullptr);
		const bool hasThumbnail = (document->thumbnail != nullptr);
		const bool hasAlphaChannels = (document->alphaChannelCount != 0u);
		const bool hasImageResources = (hasMetaData || hasIccProfile || hasExifData || hasThumbnail || hasAlphaChannels);

		// write image resources section with optional XMP meta data, ICC profile, EXIF data, thumbnail, alpha channels
		if (hasImageResources)
		{
			const uint32_t metaDataSize = hasMetaData ? GetMetaDataResourceSize(document) : 0u;
			const uint32_t iccProfileSize = hasIccProfile ? GetIccProfileResourceSize(document) : 0u;
			const uint32_t exifDataSize = hasExifData ? GetExifDataResourceSize(document) : 0u;
			const uint32_t thumbnailSize = hasThumbnail ? GetThumbnailResourceSize(document) : 0u;
			const uint32_t displayInfoSize = hasAlphaChannels ? GetDisplayInfoResourceSize(document) : 0u;
			const uint32_t channelNamesSize = hasAlphaChannels ? GetChannelNamesResourceSize(document) : 0u;
			const uint32_t unicodeChannelNamesSize = hasAlphaChannels ? GetUnicodeChannelNamesResourceSize(document) : 0u;

			uint32_t sectionLength = 0u;
			sectionLength += hasMetaData ? bitUtil::RoundUpToMultiple(GetImageResourceSize() + metaDataSize, 2u) : 0u;
			sectionLength += hasIccProfile ? bitUtil::RoundUpToMultiple(GetImageResourceSize() + iccProfileSize, 2u) : 0u;
			sectionLength += hasExifData ? bitUtil::RoundUpToMultiple(GetImageResourceSize() + exifDataSize, 2u) : 0u;
			sectionLength += hasThumbnail ? bitUtil::RoundUpToMultiple(GetImageResourceSize() + thumbnailSize, 2u) : 0u;
			sectionLength += hasAlphaChannels ? bitUtil::RoundUpToMultiple(GetImageResourceSize() + displayInfoSize, 2u) : 0u;
			sectionLength += hasAlphaChannels ? bitUtil::RoundUpToMultiple(GetImageResourceSize() + channelNamesSize, 2u) : 0u;
			sectionLength += hasAlphaChannels ? bitUtil::RoundUpToMultiple(GetImageResourceSize() + unicodeChannelNamesSize, 2u) : 0u;

			// image resource section starts with length of the whole section
			fileUtil::WriteToFileBE(writer, sectionLength);

			if (hasMetaData)
			{
				WriteImageResource(writer, imageResource::XMP_METADATA, metaDataSize);

				const uint64_t start = writer.GetPosition();
				{
					writer.Write(XMP_HEADER, sizeof(XMP_HEADER)-1u);
					for (unsigned int i = 0u; i < document->attributeCount; ++i)
					{
						writer.Write("<xmp:", 5u);
						writer.Write(document->attributes[i].name, static_cast<uint32_t>(strlen(document->attributes[i].name)));
						writer.Write(">", 1u);
						writer.Write(document->attributes[i].value, static_cast<uint32_t>(strlen(document->attributes[i].value)));
						writer.Write("</xmp:", 6u);
						writer.Write(document->attributes[i].name, static_cast<uint32_t>(strlen(document->attributes[i].name)));
						writer.Write(">\n", 2u);
					}
					writer.Write(XMP_FOOTER, sizeof(XMP_FOOTER)-1u);
				}
				const uint64_t bytesWritten = writer.GetPosition() - start;
				if (bytesWritten & 1ull)
				{
					// write padding byte
					fileUtil::WriteToFileBE(writer, static_cast<uint8_t>(0u));
				}
			}

			if (hasIccProfile)
			{
				WriteImageResource(writer, imageResource::ICC_PROFILE, iccProfileSize);

				const uint64_t start = writer.GetPosition();
				{
					writer.Write(document->iccProfile, document->sizeOfICCProfile);
				}
				const uint64_t bytesWritten = writer.GetPosition() - start;
				if (bytesWritten & 1ull)
				{
					// write padding byte
					fileUtil::WriteToFileBE(writer, static_cast<uint8_t>(0u));
				}
			}

			if (hasExifData)
			{
				WriteImageResource(writer, imageResource::EXIF_DATA, exifDataSize);

				const uint64_t start = writer.GetPosition();
				{
					writer.Write(document->exifData, document->sizeOfExifData);
				}
				const uint64_t bytesWritten = writer.GetPosition() - start;
				if (bytesWritten & 1ull)
				{
					// write padding byte
					fileUtil::WriteToFileBE(writer, static_cast<uint8_t>(0u));
				}
			}

			if (hasThumbnail)
			{
				WriteImageResource(writer, imageResource::THUMBNAIL_RESOURCE, thumbnailSize);

				const uint64_t start = writer.GetPosition();
				{
					const uint32_t format = 1u;				// format = kJpegRGB
					const uint16_t bitsPerPixel = 24u;
					const uint16_t planeCount = 1u;
					const uint32_t widthInBytes = (document->thumbnail->width * bitsPerPixel + 31u) / 32u * 4u;
					const uint32_t totalSize = widthInBytes * document->thumbnail->height * planeCount;

					fileUtil::WriteToFileBE(writer, format);
					fileUtil::WriteToFileBE(writer, document->thumbnail->width);
					fileUtil::WriteToFileBE(writer, document->thumbnail->height);
					fileUtil::WriteToFileBE(writer, widthInBytes);
					fileUtil::WriteToFileBE(writer, totalSize);
					fileUtil::WriteToFileBE(writer, document->thumbnail->binaryJpegSize);
					fileUtil::WriteToFileBE(writer, bitsPerPixel);
					fileUtil::WriteToFileBE(writer, planeCount);

					writer.Write(document->thumbnail->binaryJpeg, document->thumbnail->binaryJpegSize);
				}
				const uint64_t bytesWritten = writer.GetPosition() - start;
				if (bytesWritten & 1ull)
				{
					// write padding byte
					fileUtil::WriteToFileBE(writer, static_cast<uint8_t>(0u));
				}
			}

			if (hasAlphaChannels)
			{
				// write display info
				{
					WriteImageResource(writer, imageResource::DISPLAY_INFO, displayInfoSize);

					const uint64_t start = writer.GetPosition();

					// version
					fileUtil::WriteToFileBE(writer, static_cast<uint32_t>(1u));

					// per channel data
					for (unsigned int i=0u; i < document->alphaChannelCount; ++i)
					{
						AlphaChannel* channel = document->alphaChannels + i;
						fileUtil::WriteToFileBE(writer, channel->colorSpace);
						fileUtil::WriteToFileBE(writer, channel->color[0]);
						fileUtil::WriteToFileBE(writer, channel->color[1]);
						fileUtil::WriteToFileBE(writer, channel->color[2]);
						fileUtil::WriteToFileBE(writer, channel->color[3]);
						fileUtil::WriteToFileBE(writer, channel->opacity);
						fileUtil::WriteToFileBE(writer, channel->mode);
					}

					const uint64_t bytesWritten = writer.GetPosition() - start;
					if (bytesWritten & 1ull)
					{
						// write padding byte
						fileUtil::WriteToFileBE(writer, static_cast<uint8_t>(0u));
					}
				}

				// write channel names
				{
					WriteImageResource(writer, imageResource::ALPHA_CHANNEL_ASCII_NAMES, channelNamesSize);

					const uint64_t start = writer.GetPosition();

					for (unsigned int i = 0u; i < document->alphaChannelCount; ++i)
					{
						fileUtil::WriteToFileBE(writer, static_cast<uint8_t>(document->alphaChannels[i].asciiName.GetLength()));
						writer.Write(document->alphaChannels[i].asciiName.c_str(), static_cast<uint32_t>(document->alphaChannels[i].asciiName.GetLength()));
					}

					const uint64_t bytesWritten = writer.GetPosition() - start;
					if (bytesWritten & 1ull)
					{
						// write padding byte
						fileUtil::WriteToFileBE(writer, static_cast<uint8_t>(0u));
					}
				}

				// write unicode channel names
				{
					WriteImageResource(writer, imageResource::ALPHA_CHANNEL_UNICODE_NAMES, unicodeChannelNamesSize);

					const uint64_t start = writer.GetPosition();

					for (unsigned int i = 0u; i < document->alphaChannelCount; ++i)
					{
						// PSD expects UTF-16 strings, followed by a null terminator
						const size_t length = document->alphaChannels[i].asciiName.GetLength();
						fileUtil::WriteToFileBE(writer, static_cast<uint32_t>(length + 1u));

						const char* asciiStr = document->alphaChannels[i].asciiName.c_str();
						for (size_t j = 0u; j < length; ++j)
						{
							const uint16_t unicodeGlyph = asciiStr[j];
							fileUtil::WriteToFileBE(writer, unicodeGlyph);
						}

						fileUtil::WriteToFileBE(writer, uint16_t(0u));
					}

					const uint64_t bytesWritten = writer.GetPosition() - start;
					if (bytesWritten & 1ull)
					{
						// write padding byte
						fileUtil::WriteToFileBE(writer, static_cast<uint8_t>(0u));
					}
				}
			}
		}
		else
		{
			// no image resources
			fileUtil::WriteToFileBE(writer, static_cast<uint32_t>(0u));
		}
	}

	// layer mask section
	uint32_t layerInfoSectionLength = GetLayerInfoSectionLength(document);

	// layer info section must be padded to a multiple of 4
	const unsigned int paddingNeeded = bitUtil::RoundUpToMultiple(layerInfoSectionLength, 4u) - layerInfoSectionLength;
	layerInfoSectionLength += paddingNeeded;

	const bool is8BitData = (document->bitsPerChannel == 8u);
	if (is8BitData)
	{
		// 8-bit data
		// layer mask section length also includes global layer mask info marker. layer info follows directly after that
		const uint32_t layerMaskSectionLength = layerInfoSectionLength + 4u;
		fileUtil::WriteToFileBE(writer, layerMaskSectionLength);
	}
	else
	{
		// 16-bit and 32-bit layer data is stored in Additional Layer Information, so we leave the following layer info section empty
		const uint32_t layerMaskSectionLength = layerInfoSectionLength + 4u * 5u;
		fileUtil::WriteToFileBE(writer, layerMaskSectionLength);

		// empty layer info section
		fileUtil::WriteToFileBE(writer, static_cast<uint32_t>(0u));

		// empty global layer mask info
		fileUtil::WriteToFileBE(writer, static_cast<uint32_t>(0u));

		// additional layer information
		const uint32_t signature = util::Key<'8', 'B', 'I', 'M'>::VALUE;
		fileUtil::WriteToFileBE(writer, signature);

		if (document->bitsPerChannel == 16u)
		{
			const uint32_t key = util::Key<'L', 'r', '1', '6'>::VALUE;
			fileUtil::WriteToFileBE(writer, key);
		}
		else if (document->bitsPerChannel == 32u)
		{
			const uint32_t key = util::Key<'L', 'r', '3', '2'>::VALUE;
			fileUtil::WriteToFileBE(writer, key);
		}
	}

	fileUtil::WriteToFileBE(writer, layerInfoSectionLength);

	// layer count
	fileUtil::WriteToFileBE(writer, document->layerCount);

	// per-layer info
	for (unsigned int i = 0u; i < document->layerCount; ++i)
	{
		ExportLayer* layer = document->layers + i;
		fileUtil::WriteToFileBE(writer, layer->top);
		fileUtil::WriteToFileBE(writer, layer->left);
		fileUtil::WriteToFileBE(writer, layer->bottom);
		fileUtil::WriteToFileBE(writer, layer->right);

		const uint16_t channelCount = GetChannelCount(layer);
		fileUtil::WriteToFileBE(writer, channelCount);

		// per-channel info
		for (unsigned int j = 0u; j < ExportLayer::MAX_CHANNEL_COUNT; ++j)
		{
			if (layer->channelData[j])
			{
				const int16_t channelId = GetChannelId(j);
				fileUtil::WriteToFileBE(writer, channelId);

				// channel data always has a 2-byte compression type in front of the data
				const uint32_t channelDataSize = layer->channelSize[j] + 2u;
				fileUtil::WriteToFileBE(writer, channelDataSize);
			}
		}

		// blend mode signature
		fileUtil::WriteToFileBE(writer, util::Key<'8', 'B', 'I', 'M'>::VALUE);

		// blend mode data
		const uint8_t opacity = 255u;
		const uint8_t clipping = 0u;
		const uint8_t flags = 0u;
		const uint8_t filler = 0u;
		fileUtil::WriteToFileBE(writer, util::Key<'n', 'o', 'r', 'm'>::VALUE);
		fileUtil::WriteToFileBE(writer, opacity);
		fileUtil::WriteToFileBE(writer, clipping);
		fileUtil::WriteToFileBE(writer, flags);
		fileUtil::WriteToFileBE(writer, filler);

		// extra data, including layer name
		const uint32_t extraDataLength = GetExtraDataLength(layer);
		fileUtil::WriteToFileBE(writer, extraDataLength);

		const uint32_t layerMaskDataLength = 0u;
		fileUtil::WriteToFileBE(writer, layerMaskDataLength);

		const uint32_t layerBlendingRangesDataLength = 0u;
		fileUtil::WriteToFileBE(writer, layerBlendingRangesDataLength);

		// the layer name is stored as pascal string, padded to a multiple of 4
		const uint8_t nameLength = static_cast<uint8_t>(strlen(layer->name));
		const uint32_t paddedNameLength = bitUtil::RoundUpToMultiple(nameLength + 1u, 4u);
		fileUtil::WriteToFileBE(writer, nameLength);
		writer.Write(layer->name, paddedNameLength - 1u);
	}

	// per-layer data
	for (unsigned int i = 0u; i < document->layerCount; ++i)
	{
		ExportLayer* layer = document->layers + i;

		// per-channel data
		for (unsigned int j = 0u; j < ExportLayer::MAX_CHANNEL_COUNT; ++j)
		{
			if (layer->channelData[j])
			{
				fileUtil::WriteToFileBE(writer, layer->channelCompression[j]);
				writer.Write(layer->channelData[j], layer->channelSize[j]);
			}
		}
	}

	// add padding to align layer info section to multiple of 4
	if (paddingNeeded != 0u)
	{
		writer.Write(zeroes, paddingNeeded);
	}

	// global layer mask info
	const uint32_t globalLayerMaskInfoLength = 0u;
	fileUtil::WriteToFileBE(writer, globalLayerMaskInfoLength);

	// for some reason, Photoshop insists on having an (uncompressed) Image Data section for 32-bit files.
	// this is unfortunate, because it makes the files very large. don't think this is intentional, but rather a bug.
	// additionally, for documents of a certain size, Photoshop also expects merged data to be there.
	// hence we bite the bullet and just write the merged data section in all cases.

	// merged data section
	{
		const uint32_t size = document->width * document->height * document->bitsPerChannel / 8u;;
		uint8_t* emptyMemory = memoryUtil::AllocateArray<uint8_t>(allocator, size);
		memset(emptyMemory, 0, size);

		// write merged image
		fileUtil::WriteToFileBE(writer, static_cast<uint16_t>(compressionType::RAW));
		if (document->colorMode == exportColorMode::GRAYSCALE)
		{
			const void* dataGray = document->mergedImageData[0] ? document->mergedImageData[0] : emptyMemory;
			writer.Write(dataGray, size);
		}
		else if (document->colorMode == exportColorMode::RGB)
		{
			const void* dataR = document->mergedImageData[0] ? document->mergedImageData[0] : emptyMemory;
			const void* dataG = document->mergedImageData[1] ? document->mergedImageData[1] : emptyMemory;
			const void* dataB = document->mergedImageData[2] ? document->mergedImageData[2] : emptyMemory;
			writer.Write(dataR, size);
			writer.Write(dataG, size);
			writer.Write(dataB, size);
		}

		// write alpha channels
		for (unsigned int i = 0u; i < document->alphaChannelCount; ++i)
		{
			writer.Write(document->alphaChannelData[i], size);
		}

		memoryUtil::FreeArray(allocator, emptyMemory);
	}
}

PSD_NAMESPACE_END
