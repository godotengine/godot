// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#include "PsdPch.h"
#include "PsdParseLayerMaskSection.h"

#include "PsdDocument.h"
#include "PsdLayer.h"
#include "PsdChannel.h"
#include "PsdChannelType.h"
#include "PsdLayerMask.h"
#include "PsdVectorMask.h"
#include "PsdCompressionType.h"
#include "PsdLayerType.h"
#include "PsdFile.h"
#include "PsdLayerMaskSection.h"
#include "PsdKey.h"
#include "PsdBitUtil.h"
#include "PsdEndianConversion.h"
#include "PsdSyncFileReader.h"
#include "PsdSyncFileUtil.h"
#include "PsdMemoryUtil.h"
#include "PsdDecompressRle.h"
#include "PsdAllocator.h"
#include "Psdminiz.c"
#include "Psdinttypes.h"
#include "PsdLog.h"
#include <cstring>


PSD_NAMESPACE_BEGIN

namespace
{
	struct MaskData
	{
		int32_t top;
		int32_t left;
		int32_t bottom;
		int32_t right;
		uint8_t defaultColor;
		bool isVectorMask;
	};


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static int64_t ReadMaskRectangle(SyncFileReader& reader, MaskData& maskData)
	{
		maskData.top = fileUtil::ReadFromFileBE<int32_t>(reader);
		maskData.left = fileUtil::ReadFromFileBE<int32_t>(reader);
		maskData.bottom = fileUtil::ReadFromFileBE<int32_t>(reader);
		maskData.right = fileUtil::ReadFromFileBE<int32_t>(reader);

		return 4u*sizeof(int32_t);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static int64_t ReadMaskDensity(SyncFileReader& reader, uint8_t& density)
	{
		density = fileUtil::ReadFromFileBE<uint8_t>(reader);
		return sizeof(uint8_t);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static int64_t ReadMaskFeather(SyncFileReader& reader, float64_t& feather)
	{
		feather = fileUtil::ReadFromFileBE<float64_t>(reader);
		return sizeof(float64_t);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static int64_t ReadMaskParameters(SyncFileReader& reader, uint8_t& layerDensity, float64_t& layerFeather, uint8_t& vectorDensity, float64_t& vectorFeather)
	{
		int64_t bytesRead = 0;

		const uint8_t flags = fileUtil::ReadFromFileBE<uint8_t>(reader);
		bytesRead += sizeof(uint8_t);

		const bool hasUserDensity = (flags & (1u << 0)) != 0;
		const bool hasUserFeather = (flags & (1u << 1)) != 0;
		const bool hasVectorDensity = (flags & (1u << 2)) != 0;
		const bool hasVectorFeather = (flags & (1u << 3)) != 0;
		if (hasUserDensity)
		{
			bytesRead += ReadMaskDensity(reader, layerDensity);
		}
		if (hasUserFeather)
		{
			bytesRead += ReadMaskFeather(reader, layerFeather);
		}
		if (hasVectorDensity)
		{
			bytesRead += ReadMaskDensity(reader, vectorDensity);
		}
		if (hasVectorFeather)
		{
			bytesRead += ReadMaskFeather(reader, vectorFeather);
		}

		return bytesRead;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	static void ApplyMaskData(const MaskData& maskData, float64_t feather, uint8_t density, T* layerMask)
	{
		layerMask->top = maskData.top;
		layerMask->left = maskData.left;
		layerMask->bottom = maskData.bottom;
		layerMask->right = maskData.right;
		layerMask->feather = feather;
		layerMask->density = density;
		layerMask->defaultColor = maskData.defaultColor;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	static unsigned int GetWidth(const T* data)
	{
		if (data->right > data->left)
			return static_cast<unsigned int>(data->right - data->left);

		return 0u;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	static unsigned int GetHeight(const T* data)
	{
		if (data->bottom > data->top)
			return static_cast<unsigned int>(data->bottom - data->top);

		return 0u;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	static void GetExtents(const T* data, unsigned int& width, unsigned int& height)
	{
		width = GetWidth(data);
		height = GetHeight(data);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static void GetChannelExtents(const Layer* layer, const Channel* channel, unsigned int& width, unsigned int& height)
	{
		if (channel->type == channelType::TRANSPARENCY_MASK)
		{
			// the channel is the transparency mask, which has the same size as the layer
			return GetExtents(layer, width, height);
		}
		else if (channel->type == channelType::LAYER_OR_VECTOR_MASK)
		{
			// the channel is either the layer or vector mask, depending on how many masks there are in the layer.
			if (layer->vectorMask)
			{
				// a vector mask exists, so this always denotes a vector mask
				return GetExtents(layer->vectorMask, width, height);
			}
			else if (layer->layerMask)
			{
				// no vector mask exists, so the layer mask is the only mask left
				return GetExtents(layer->layerMask, width, height);
			}

			PSD_ASSERT(false, "The code failed to create a mask for this type internally. This should never happen.");
			width = 0;
			height = 0;
			return;
		}
		else if (channel->type == channelType::LAYER_MASK)
		{
			// this type is only valid when there are two masks stored, in which case this always denotes the layer mask
			return GetExtents(layer->layerMask, width, height);
		}

		// this is a color channel which has the same size as the layer
		return GetExtents(layer, width, height);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static uint8_t GetChannelDefaultColor(const Layer* layer, const Channel* channel)
	{
		if (channel->type == channelType::TRANSPARENCY_MASK)
		{
			return 0u;
		}
		else if (channel->type == channelType::LAYER_OR_VECTOR_MASK)
		{
			if (layer->vectorMask)
			{
				return layer->vectorMask->defaultColor;
			}
			else if (layer->layerMask)
			{
				return layer->layerMask->defaultColor;
			}

			PSD_ASSERT(false, "The code failed to create a mask for this type internally. This should never happen.");
			return 0u;
		}
		else if (channel->type == channelType::LAYER_MASK)
		{
			return layer->layerMask->defaultColor;
		}

		return 0u;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	static void MoveChannelToMask(Channel* channel, T* mask)
	{
		mask->data = channel->data;
		mask->fileOffset = channel->fileOffset;

		channel->data = nullptr;
		channel->type = channelType::INVALID;
		channel->fileOffset = 0ull;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	void EndianConvert(void* src, unsigned int width, unsigned int height)
	{
		PSD_ASSERT_NOT_NULL(src);

		T* data = static_cast<T*>(src);
		const unsigned int size = width*height;

		for (unsigned int i=0; i < size; ++i)
		{
			data[i] = endianUtil::BigEndianToNative(data[i]);
		}
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	static void* ReadChannelDataRaw(SyncFileReader& reader, Allocator* allocator, unsigned int width, unsigned int height)
	{
		const unsigned int size = width*height;
		if (size > 0)
		{
			void* planarData = allocator->Allocate(size*sizeof(T), 16u);
			reader.Read(planarData, size*sizeof(T));

			EndianConvert<T>(planarData, width, height);

			return planarData;
		}

		return nullptr;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	static void* ReadChannelDataRLE(SyncFileReader& reader, Allocator* allocator, unsigned int width, unsigned int height)
	{
		// the RLE-compressed data is preceded by a 2-byte data count for each scan line
		const unsigned int size = width*height;

		unsigned int rleDataSize = 0u;
		for (unsigned int i=0; i < height; ++i)
		{
			const uint16_t dataCount = fileUtil::ReadFromFileBE<uint16_t>(reader);
			rleDataSize += dataCount;
		}

		if (rleDataSize > 0)
		{
			void* planarData = allocator->Allocate(size*sizeof(T), 16u);

			// decompress RLE
			void* rleData = allocator->Allocate(rleDataSize, 4u);
			{
				reader.Read(rleData, rleDataSize);
				imageUtil::DecompressRle(static_cast<const uint8_t*>(rleData), rleDataSize, static_cast<uint8_t*>(planarData), width*height*sizeof(T));
			}
			allocator->Free(rleData);

			EndianConvert<T>(planarData, width, height);

			return planarData;
		}

		return nullptr;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	static void* ReadChannelDataZip(SyncFileReader& reader, Allocator* allocator, unsigned int width, unsigned int height, uint32_t channelSize)
	{
		if (channelSize > 0)
		{
			const unsigned int size = width*height;

			T* planarData = static_cast<T*>(allocator->Allocate(size*sizeof(T), 16));

			void* zipData = allocator->Allocate(channelSize, 4u);
			reader.Read(zipData, channelSize);

			// the zipped data stream has a zlib-header
			const size_t status = tinfl_decompress_mem_to_mem(planarData, size*sizeof(T), zipData, channelSize, TINFL_FLAG_PARSE_ZLIB_HEADER);
			if (status == TINFL_DECOMPRESS_MEM_TO_MEM_FAILED)
			{
				PSD_ERROR("PsdExtract", "Error while unzipping channel data.");
			}

			allocator->Free(zipData);

			EndianConvert<T>(planarData, width, height);

			return planarData;
		}

		return nullptr;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	static void ApplyPrediction(Allocator* allocator, void* PSD_RESTRICT planarData, unsigned int width, unsigned int height)
	{
		static_assert(sizeof(T) == -1, "Unknown data type.");
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <>
	void ApplyPrediction<uint8_t>(Allocator*, void* PSD_RESTRICT planarData, unsigned int width, unsigned int height)
	{
		uint8_t* buffer = static_cast<uint8_t*>(planarData);
		for (unsigned int y = 0; y < height; ++y)
		{
			++buffer;
			for (unsigned int x = 1; x < width; ++x)
			{
				const uint32_t previous = buffer[-1];
				const uint32_t current = buffer[0];
				const uint32_t value = current + previous;

				*buffer++ = static_cast<uint8_t>(value & 0xFFu);
			}
		}
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <>
	void ApplyPrediction<uint16_t>(Allocator*, void* PSD_RESTRICT planarData, unsigned int width, unsigned int height)
	{
		// 16-bit images are delta-encoded word-by-word.
		// the deltas are big-endian and must be reversed first for further processing. note that this is done
		// in-place with the delta-decoding.
		{
			uint16_t* buffer = static_cast<uint16_t*>(planarData);
			for (unsigned int y=0; y < height; ++y)
			{
				const uint16_t first = *buffer;
				*buffer++ = endianUtil::BigEndianToNative(first);
				for (unsigned int x=1; x < width; ++x)
				{
					buffer[0] = endianUtil::BigEndianToNative(buffer[0]);

					const uint32_t previous = buffer[-1];
					const uint32_t current = buffer[0];
					const uint32_t value = current + previous;

					// note that the data written here is now in little-endian format
					*buffer++ = static_cast<uint16_t>(value & 0xFFFFu);
				}
			}
		}
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <>
	void ApplyPrediction<float32_t>(Allocator* allocator, void* PSD_RESTRICT planarData, unsigned int width, unsigned int height)
	{
		// delta-decode row by row first
		{
			uint8_t* buffer = static_cast<uint8_t*>(planarData);
			for (unsigned int y=0; y < height; ++y)
			{
				++buffer;
				for (unsigned int x=1; x < width*4; ++x)
				{
					const uint32_t previous = buffer[-1];
					const uint32_t current = buffer[0];
					const uint32_t value = current + previous;

					*buffer++ = static_cast<uint8_t>(value & 0xFFu);
				}
			}
		}

		// the bytes of the 32-bit float are stored in planar fashion per row, big-endian format.
		// interleave the bytes, and store them in little-endian format at the same time.
		uint8_t* rowData = static_cast<uint8_t*>(allocator->Allocate(width*sizeof(float32_t), 16));
		{
			uint8_t* dest = static_cast<uint8_t*>(planarData);
			for (unsigned int y=0; y < height; ++y)
			{
				// copy first row of data to backup storage, because it will be overwritten inside our loop.
				// note that this operation cannot be done in-place, that's why we work row by row.
				memcpy(rowData, dest, width*sizeof(float32_t));

				const uint8_t* src0 = rowData;
				const uint8_t* src1 = rowData + 1*width;
				const uint8_t* src2 = rowData + 2*width;
				const uint8_t* src3 = rowData + 3*width;

				for (unsigned int x=0; x < width; ++x)
				{
					// write data in little-endian format
					dest[0] = *src3++;
					dest[1] = *src2++;
					dest[2] = *src1++;
					dest[3] = *src0++;
					dest += 4u;
				}
			}
		}

		allocator->Free(rowData);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	static void* ReadChannelDataZipPrediction(SyncFileReader& reader, Allocator* allocator, unsigned int width, unsigned int height, uint32_t channelSize)
	{
		if (channelSize > 0)
		{
			const unsigned int size = width*height;

			T* planarData = static_cast<T*>(allocator->Allocate(size*sizeof(T), 16));

			void* zipData = allocator->Allocate(channelSize, 4u);
			reader.Read(zipData, channelSize);

			// the zipped data stream has a zlib-header
			const size_t status = tinfl_decompress_mem_to_mem(planarData, size*sizeof(T), zipData, channelSize, TINFL_FLAG_PARSE_ZLIB_HEADER);
			if (status == TINFL_DECOMPRESS_MEM_TO_MEM_FAILED)
			{
				PSD_ERROR("PsdExtract", "Error while unzipping channel data.");
			}

			allocator->Free(zipData);

			// the data generated by applying the prediction data is already in little-endian format, so it doesn't have to be
			// endian converted further.
			ApplyPrediction<T>(allocator, planarData, width, height);

			return planarData;
		}

		return nullptr;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static LayerMaskSection* ParseLayer(const Document* document, SyncFileReader& reader, Allocator* allocator, uint64_t sectionOffset, uint32_t sectionLength, uint32_t layerLength)
	{
		LayerMaskSection* layerMaskSection = memoryUtil::Allocate<LayerMaskSection>(allocator);
		layerMaskSection->layers = nullptr;
		layerMaskSection->layerCount = 0u;
		layerMaskSection->overlayColorSpace = 0u;
		layerMaskSection->opacity = 0u;
		layerMaskSection->kind = 128u;
		layerMaskSection->hasTransparencyMask = false;

		if (layerLength != 0)
		{
			// read the layer count. if it is a negative number, its absolute value is the number of layers and the 
			// first alpha channel contains the transparency data for the merged result.
			// this will also be reflected in the channelCount of the document.
			int16_t layerCount = fileUtil::ReadFromFileBE<int16_t>(reader);
			layerMaskSection->hasTransparencyMask = (layerCount < 0);
			if (layerCount < 0)
				layerCount = -layerCount;

			layerMaskSection->layerCount = static_cast<unsigned int>(layerCount);
			layerMaskSection->layers = memoryUtil::AllocateArray<Layer>(allocator, layerMaskSection->layerCount);

			// read layer record for each layer
			for (unsigned int i=0; i < layerMaskSection->layerCount; ++i)
			{
				Layer* layer = &layerMaskSection->layers[i];
				layer->parent = nullptr;
				layer->utf16Name = nullptr;
				layer->layerMask = nullptr;
				layer->vectorMask = nullptr;
				layer->type = layerType::ANY;

				layer->top = fileUtil::ReadFromFileBE<int32_t>(reader);
				layer->left = fileUtil::ReadFromFileBE<int32_t>(reader);
				layer->bottom = fileUtil::ReadFromFileBE<int32_t>(reader);
				layer->right = fileUtil::ReadFromFileBE<int32_t>(reader);

				// number of channels in the layer.
				// this includes channels for transparency, layer, and vector masks, if any.
				const uint16_t channelCount = fileUtil::ReadFromFileBE<uint16_t>(reader);
				layer->channelCount = channelCount;
				layer->channels = memoryUtil::AllocateArray<Channel>(allocator, channelCount);

				// parse each channel
				for (unsigned int j=0; j < channelCount; ++j)
				{
					Channel* channel = &layer->channels[j];
					channel->fileOffset = 0ull;
					channel->data = nullptr;
					channel->type = fileUtil::ReadFromFileBE<int16_t>(reader);
					channel->size = fileUtil::ReadFromFileBE<uint32_t>(reader);
				}

				// blend mode signature must be '8BIM'
				const uint32_t blendModeSignature = fileUtil::ReadFromFileBE<uint32_t>(reader);
				if (blendModeSignature != util::Key<'8', 'B', 'I', 'M'>::VALUE)
				{
					PSD_ERROR("LayerMaskSection", "Layer mask info section seems to be corrupt, signature does not match \"8BIM\".");
					return layerMaskSection;
				}

				layer->blendModeKey = fileUtil::ReadFromFileBE<uint32_t>(reader);
				layer->opacity = fileUtil::ReadFromFileBE<uint8_t>(reader);
				layer->clipping = fileUtil::ReadFromFileBE<uint8_t>(reader);

				// extract flag information into layer struct
				{
					const uint8_t flags = fileUtil::ReadFromFileBE<uint8_t>(reader);
					layer->isVisible = !((flags & (1u << 1)) != 0);
				}

				// skip filler byte
				{
					const uint8_t filler = fileUtil::ReadFromFileBE<uint8_t>(reader);
					PSD_UNUSED(filler);
				}

				const uint32_t extraDataLength = fileUtil::ReadFromFileBE<uint32_t>(reader);
				const uint32_t layerMaskDataLength = fileUtil::ReadFromFileBE<uint32_t>(reader);

				// the layer mask data section is weird. it may contain extra data for masks, such as density and feather parameters.
				// there are 3 main possibilities:
				//	*) length == zero		->	skip this section
				//	*) length == [20, 28]	->	there is one mask, and that could be either a layer or vector mask.
				//								the mask flags give rise to mask parameters. they store the mask type, and additional parameters, if any.
				//								there might be some padding at the end of this section, and its size depends on which parameters are there.
				//	*) length == [36, 56]	->	there are two masks. the first mask has parameters, but does NOT store flags yet.
				//								instead, there comes a second section with the same info (flags, default color, rectangle), and
				//								the parameters follow after that. there is also padding at the end of this second section.
				if (layerMaskDataLength != 0)
				{
					// there can be at most two masks, one layer and one vector mask
					MaskData maskData[2] = {};
					unsigned int maskCount = 1u;

					float64_t layerFeather = 0.0;
					float64_t vectorFeather = 0.0;
					uint8_t layerDensity = 0;
					uint8_t vectorDensity = 0;

					int64_t toRead = layerMaskDataLength;

					// enclosing rectangle
					toRead -= ReadMaskRectangle(reader, maskData[0]);

					maskData[0].defaultColor = fileUtil::ReadFromFileBE<uint8_t>(reader);
					toRead -= sizeof(uint8_t);

					const uint8_t maskFlags = fileUtil::ReadFromFileBE<uint8_t>(reader);
					toRead -= sizeof(uint8_t);

					maskData[0].isVectorMask = (maskFlags & (1u << 3)) != 0;
					bool maskHasParameters = (maskFlags & (1u << 4)) != 0;
					if (maskHasParameters && (layerMaskDataLength <= 28))
					{
						toRead -= ReadMaskParameters(reader, layerDensity, layerFeather, vectorDensity, vectorFeather);
					}

					// check if there is enough data left for another section of mask data
					if (toRead >= 18)
					{
						// in case there is still data left to read, the following values are for the real layer mask.
						// the data we just read was for the vector mask.
						maskCount = 2u;

						const uint8_t realFlags = fileUtil::ReadFromFileBE<uint8_t>(reader);
						toRead -= sizeof(uint8_t);

						maskData[1].defaultColor = fileUtil::ReadFromFileBE<uint8_t>(reader);
						toRead -= sizeof(uint8_t);

						toRead -= ReadMaskRectangle(reader, maskData[1]);

						maskData[1].isVectorMask = (realFlags & (1u << 3)) != 0;

						// note the OR here. whether the following section has mask parameter data or not is influenced by
						// the availability of parameter data of the previous mask!
						maskHasParameters |= ((realFlags & (1u << 4)) != 0);
						if (maskHasParameters)
						{
							toRead -= ReadMaskParameters(reader, layerDensity, layerFeather, vectorDensity, vectorFeather);
						}
					}

					// skip the remaining padding bytes, if any
					PSD_ASSERT(toRead >= 0, "Parsing failed, %" PRId64 "bytes left.", toRead);
					reader.Skip(static_cast<uint64_t>(toRead));

					// apply mask data to our own data structures
					for (unsigned int mask=0; mask < maskCount; ++mask)
					{
						const bool isVectorMask = maskData[mask].isVectorMask;
						if (isVectorMask)
						{
							PSD_ASSERT(layer->vectorMask == nullptr, "A vector mask already exists.");
							layer->vectorMask = memoryUtil::Allocate<VectorMask>(allocator);
							layer->vectorMask->data = nullptr;
							layer->vectorMask->fileOffset = 0ull;
							ApplyMaskData(maskData[mask], vectorFeather, vectorDensity, layer->vectorMask);
						}
						else
						{
							PSD_ASSERT(layer->layerMask == nullptr, "A layer mask already exists.");
							layer->layerMask = memoryUtil::Allocate<LayerMask>(allocator);
							layer->layerMask->data = nullptr;
							layer->layerMask->fileOffset = 0ull;
							ApplyMaskData(maskData[mask], layerFeather, layerDensity, layer->layerMask);
						}
					}
				}

				// skip blending ranges data, we are not interested in that for now
				const uint32_t layerBlendingRangesDataLength = fileUtil::ReadFromFileBE<uint32_t>(reader);
				reader.Skip(layerBlendingRangesDataLength);

				// the layer name is stored as pascal string, padded to a multiple of 4
				char layerName[512] = {};
				const uint8_t nameLength = fileUtil::ReadFromFileBE<uint8_t>(reader);
				const uint32_t paddedNameLength = bitUtil::RoundUpToMultiple(nameLength + 1u, 4u);
				reader.Read(layerName, paddedNameLength - 1u);

				layer->name.Assign(layerName);

				// read Additional Layer Information that exists since Photoshop 4.0.
				// getting the size of this data is a bit awkward, because it's not stored explicitly somewhere. furthermore,
				// the PSD format sometimes includes the 4-byte length in its section size, and sometimes not.
				const uint32_t additionalLayerInfoSize = extraDataLength - layerMaskDataLength - layerBlendingRangesDataLength - paddedNameLength - 8u;
				int64_t toRead = additionalLayerInfoSize;
				while (toRead > 0)
				{
					const uint32_t signature = fileUtil::ReadFromFileBE<uint32_t>(reader);
					if (signature != util::Key<'8', 'B', 'I', 'M'>::VALUE)
					{
						PSD_ERROR("LayerMaskSection", "Additional Layer Information section seems to be corrupt, signature does not match \"8BIM\".");
						return layerMaskSection;
					}

					const uint32_t key = fileUtil::ReadFromFileBE<uint32_t>(reader);

					// length needs to be rounded to an even number
					uint32_t length = fileUtil::ReadFromFileBE<uint32_t>(reader);
					length = bitUtil::RoundUpToMultiple(length, 2u);

					// read "Section divider setting" to identify whether a layer is a group, or a section divider
					if (key == util::Key<'l', 's', 'c', 't'>::VALUE)
					{
						layer->type = fileUtil::ReadFromFileBE<uint32_t>(reader);

						// skip the rest of the data
						reader.Skip(length - 4u);
					}
					// read Unicode layer name
					else if (key == util::Key<'l', 'u', 'n', 'i'>::VALUE)
					{
						// PSD Unicode strings store 4 bytes for the number of characters, NOT bytes, followed by
						// 2-byte UTF16 Unicode data without the terminating null.
						const uint32_t characterCountWithoutNull = fileUtil::ReadFromFileBE<uint32_t>(reader);
						layer->utf16Name = memoryUtil::AllocateArray<uint16_t>(allocator, characterCountWithoutNull + 1u);

						for (uint32_t c = 0u; c < characterCountWithoutNull; ++c)
						{
							layer->utf16Name[c] = fileUtil::ReadFromFileBE<uint16_t>(reader);
						}
						layer->utf16Name[characterCountWithoutNull] = 0u;

						// skip possible padding bytes
						reader.Skip(length - 4u - characterCountWithoutNull * sizeof(uint16_t));
					}
					else
					{
						reader.Skip(length);
					}

					toRead -= 3*sizeof(uint32_t) + length;
				}
			}

			// walk through the layers and channels, but don't extract their data just yet. only save the file offset for extracting the
			// data later.
			for (unsigned int i=0; i < layerMaskSection->layerCount; ++i)
			{
				Layer* layer = &layerMaskSection->layers[i];
				const unsigned int channelCount = layer->channelCount;
				for (unsigned int j=0; j < channelCount; ++j)
				{
					Channel* channel = &layer->channels[j];
					channel->fileOffset = reader.GetPosition();
					reader.Skip(channel->size);
				}
			}
		}

		if (sectionLength > 0)
		{
			// start loading at the global layer mask info section, located after the Layer Information Section.
			// note that the 4 bytes that stored the length of the section are not included in the length itself.
			const uint64_t globalInfoSectionOffset = sectionOffset + layerLength + 4u;
			reader.SetPosition(globalInfoSectionOffset);

			// work out how many bytes are left to read at this point. we need that to figure out the size of the last
			// optional section, the Additional Layer Information.
			if (sectionOffset + sectionLength > globalInfoSectionOffset)
			{
				int64_t toRead = static_cast<int64_t>(sectionOffset + sectionLength - globalInfoSectionOffset);
				const uint32_t globalLayerMaskLength = fileUtil::ReadFromFileBE<uint32_t>(reader);
				toRead -= sizeof(uint32_t);

				if (globalLayerMaskLength != 0)
				{
					layerMaskSection->overlayColorSpace = fileUtil::ReadFromFileBE<uint16_t>(reader);

					// 4*2 byte color components
					reader.Skip(8);

					layerMaskSection->opacity = fileUtil::ReadFromFileBE<uint16_t>(reader);
					layerMaskSection->kind = fileUtil::ReadFromFileBE<uint8_t>(reader);

					toRead -= 2u*sizeof(uint16_t) + sizeof(uint8_t) + 8u;

					// filler bytes (zeroes)
					const uint32_t remaining = globalLayerMaskLength - 2u*sizeof(uint16_t) - sizeof(uint8_t) - 8u;
					reader.Skip(remaining);

					toRead -= remaining;
				}

				// are there still bytes left to read? then this is the Additional Layer Information that exists since Photoshop 4.0.
				while (toRead > 0)
				{
					const uint32_t signature = fileUtil::ReadFromFileBE<uint32_t>(reader);
					if (signature != util::Key<'8', 'B', 'I', 'M'>::VALUE)
					{
						PSD_ERROR("AdditionalLayerInfo", "Additional Layer Information section seems to be corrupt, signature does not match \"8BIM\".");
						return layerMaskSection;
					}

					const uint32_t key = fileUtil::ReadFromFileBE<uint32_t>(reader);

					// again, length is rounded to a multiple of 4
					uint32_t length = fileUtil::ReadFromFileBE<uint32_t>(reader);
					length = bitUtil::RoundUpToMultiple(length, 4u);

					if (key == util::Key<'L', 'r', '1', '6'>::VALUE)
					{
						const uint64_t offset = reader.GetPosition();
						DestroyLayerMaskSection(layerMaskSection, allocator);
						layerMaskSection = ParseLayer(document, reader, allocator, 0u, 0u, length);
						reader.SetPosition(offset + length);
					}
					else if (key == util::Key<'L', 'r', '3', '2'>::VALUE)
					{
						const uint64_t offset = reader.GetPosition();
						DestroyLayerMaskSection(layerMaskSection, allocator);
						layerMaskSection = ParseLayer(document, reader, allocator, 0u, 0u, length);
						reader.SetPosition(offset + length);
					}
					else if (key == util::Key<'v', 'm', 's', 'k'>::VALUE)
					{
						// TODO: could read extra vector mask data here
						reader.Skip(length);
					}
					else if (key == util::Key<'l', 'n', 'k', '2'>::VALUE)
					{
						// TODO: could read individual smart object layer data here
						reader.Skip(length);
					}
					else
					{
						reader.Skip(length);
					}

					toRead -= 3u*sizeof(uint32_t) + length;
				}
			}
		}

		return layerMaskSection;
	}
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
LayerMaskSection* ParseLayerMaskSection(const Document* document, File* file, Allocator* allocator)
{
	PSD_ASSERT_NOT_NULL(file);
	PSD_ASSERT_NOT_NULL(allocator);

	// if there are no layers or masks, this section is just 4 bytes: the length field, which is set to zero.
	const Section& section = document->layerMaskInfoSection;
	if (section.length == 0)
	{
		PSD_ERROR("PSD", "Document does not contain a layer mask section.");
		return nullptr;
	}

	SyncFileReader reader(file);
	reader.SetPosition(section.offset);

	const uint32_t layerInfoSectionLength = fileUtil::ReadFromFileBE<uint32_t>(reader);
	LayerMaskSection* layerMaskSection = ParseLayer(document, reader, allocator, section.offset, section.length, layerInfoSectionLength);

	// build the layer hierarchy
	if (layerMaskSection && layerMaskSection->layers)
	{
		Layer* layerStack[256] = {};
		layerStack[0] = nullptr;
		int stackIndex = 0;

		for (unsigned int i=0; i < layerMaskSection->layerCount; ++i)
		{
			// note that it is much easier to build the hierarchy by traversing the layers backwards
			Layer* layer = &layerMaskSection->layers[layerMaskSection->layerCount - i - 1u];

			PSD_ASSERT(stackIndex >= 0 && stackIndex < 256, "Stack index is out of bounds.");
			layer->parent = layerStack[stackIndex];

			unsigned int width = 0u;
			unsigned int height = 0u;
			GetExtents(layer, width, height);

			const bool isGroupStart = (layer->type == layerType::OPEN_FOLDER) || (layer->type == layerType::CLOSED_FOLDER);
			const bool isGroupEnd = (layer->type == layerType::SECTION_DIVIDER);
			if (isGroupEnd)
			{
				--stackIndex;
			}
			else if (isGroupStart)
			{
				++stackIndex;
				layerStack[stackIndex] = layer;
			}
		}
	}

	return layerMaskSection;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void ExtractLayer(const Document* document, File* file, Allocator* allocator, Layer* layer)
{
	PSD_ASSERT_NOT_NULL(file);
	PSD_ASSERT_NOT_NULL(allocator);
	PSD_ASSERT_NOT_NULL(layer);

	SyncFileReader reader(file);

	const unsigned int channelCount = layer->channelCount;
	for (unsigned int i=0; i < channelCount; ++i)
	{
		Channel* channel = &layer->channels[i];
		reader.SetPosition(channel->fileOffset);

		unsigned int width = 0u;
		unsigned int height = 0u;
		GetChannelExtents(layer, channel, width, height);

		// channel data is stored in 4 different formats, which is denoted by a 2-byte integer
		PSD_ASSERT(channel->data == nullptr, "Channel data has already been loaded.");
		const uint16_t compressionType = fileUtil::ReadFromFileBE<uint16_t>(reader);
		if (compressionType == compressionType::RAW)
		{
			if (document->bitsPerChannel == 8)
			{
				channel->data = ReadChannelDataRaw<uint8_t>(reader, allocator, width, height);
			}
			else if (document->bitsPerChannel == 16)
			{
				channel->data = ReadChannelDataRaw<uint16_t>(reader, allocator, width, height);
			}
			else if (document->bitsPerChannel == 32)
			{
				channel->data = ReadChannelDataRaw<float32_t>(reader, allocator, width, height);
			}
		}
		else if (compressionType == compressionType::RLE)
		{
			if (document->bitsPerChannel == 8)
			{
				channel->data = ReadChannelDataRLE<uint8_t>(reader, allocator, width, height);
			}
			else if (document->bitsPerChannel == 16)
			{
				channel->data = ReadChannelDataRLE<uint16_t>(reader, allocator, width, height);
			}
			else if (document->bitsPerChannel == 32)
			{
				channel->data = ReadChannelDataRLE<float32_t>(reader, allocator, width, height);
			}
		}
		else if (compressionType == compressionType::ZIP)
		{
			// note that we need to subtract 2 bytes from the channel data size because we already read the uint16_t
			// for the compression type.
			PSD_ASSERT(channel->size >= 2, "Invalid channel data size %d.", channel->size);
			const uint32_t channelDataSize = channel->size - 2u;
			if (document->bitsPerChannel == 8)
			{
				channel->data = ReadChannelDataZip<uint8_t>(reader, allocator, width, height, channelDataSize);
			}
			else if (document->bitsPerChannel == 16)
			{
				channel->data = ReadChannelDataZip<uint16_t>(reader, allocator, width, height, channelDataSize);
			}
			else if (document->bitsPerChannel == 32)
			{
				// note that this is NOT a bug.
				// in 32-bit mode, Photoshop always interprets ZIP compression as being ZIP_WITH_PREDICTION, presumably to get better compression when writing files.
				channel->data = ReadChannelDataZipPrediction<float32_t>(reader, allocator, width, height, channelDataSize);
			}
		}
		else if (compressionType == compressionType::ZIP_WITH_PREDICTION)
		{
			// note that we need to subtract 2 bytes from the channel data size because we already read the uint16_t
			// for the compression type.
			PSD_ASSERT(channel->size >= 2, "Invalid channel data size %d.", channel->size);
			const uint32_t channelDataSize = channel->size - 2u;
			if (document->bitsPerChannel == 8)
			{
				channel->data = ReadChannelDataZipPrediction<uint8_t>(reader, allocator, width, height, channelDataSize);
			}
			else if (document->bitsPerChannel == 16)
			{
				channel->data = ReadChannelDataZipPrediction<uint16_t>(reader, allocator, width, height, channelDataSize);
			}
			else if (document->bitsPerChannel == 32)
			{
				channel->data = ReadChannelDataZipPrediction<float32_t>(reader, allocator, width, height, channelDataSize);
			}
		}
		else
		{
			PSD_ASSERT(false, "Unsupported compression type %d", compressionType);
			return;
		}

		// if the channel doesn't have any data assigned to it, check if it is a mask channel of any kind.
		// layer masks sometimes don't have any planar data stored for them, because they are
		// e.g. pure black or white, which means they only get assigned a default color.
		if (!channel->data)
		{
			if (channel->type < 0)
			{
				// this is a layer mask, so create planar data for it
				const size_t dataSize = width * height * document->bitsPerChannel / 8u;
				void* channelData = allocator->Allocate(dataSize, 16u);
				memset(channelData, GetChannelDefaultColor(layer, channel), dataSize);
				channel->data = channelData;
			}
			else
			{
				// for layers like groups and group end markers ("</Layer group>") it is ok to not store any data
			}
		}
	}

	// now move channel data to our own data structures for layer and vector masks, invalidating the info stored in
	// that channel.
	for (unsigned int i=0; i < channelCount; ++i)
	{
		Channel* channel = &layer->channels[i];
		if (channel->type == channelType::LAYER_OR_VECTOR_MASK)
		{
			if (layer->vectorMask)
			{
				// layer has a vector mask, so this type always denotes the vector mask
				PSD_ASSERT(!layer->vectorMask->data, "Vector mask data has already been assigned.");
				MoveChannelToMask(channel, layer->vectorMask);
			}
			else if (layer->layerMask)
			{
				// we don't have a vector but a layer mask, so this type denotes the layer mask
				PSD_ASSERT(!layer->layerMask->data, "Layer mask data has already been assigned.");
				MoveChannelToMask(channel, layer->layerMask);
			}
			else
			{
				PSD_ASSERT(false, "The code failed to create a mask for this type internally. This should never happen.");
			}
		}
		else if (channel->type == channelType::LAYER_MASK)
		{
			PSD_ASSERT(layer->layerMask, "Layer mask must already exist.");
			PSD_ASSERT(!layer->layerMask->data, "Layer mask data has already been assigned.");
			MoveChannelToMask(channel, layer->layerMask);
		}
		else
		{
			// this channel is either a color channel, or the transparency mask. those should be stored in our channel array,
			// so there's nothing to do.
		}
	}
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void DestroyLayerMaskSection(LayerMaskSection*& section, Allocator* allocator)
{
	PSD_ASSERT_NOT_NULL(section);
	PSD_ASSERT_NOT_NULL(allocator);

	for (unsigned int i=0; i < section->layerCount; ++i)
	{
		Layer* layer = &section->layers[i];
		for (unsigned int j=0; j < layer->channelCount; ++j)
		{
			Channel* channel = &layer->channels[j];
			allocator->Free(channel->data);
		}

		memoryUtil::FreeArray(allocator, layer->utf16Name);

		memoryUtil::FreeArray(allocator, layer->channels);

		if (layer->layerMask)
		{
			allocator->Free(layer->layerMask->data);
		}
		memoryUtil::Free(allocator, layer->layerMask);

		if (layer->vectorMask)
		{
			allocator->Free(layer->vectorMask->data);
		}
		memoryUtil::Free(allocator, layer->vectorMask);
	}
	memoryUtil::FreeArray(allocator, section->layers);
	memoryUtil::Free(allocator, section);
}

PSD_NAMESPACE_END
