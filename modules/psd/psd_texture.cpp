/**************************************************************************/
/*  psd_texture.cpp                                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "psd_texture.h"

#include "core/io/file_access.h"


PSD_USING_NAMESPACE;

// helpers for reading PSDs
namespace
{
	static const unsigned int CHANNEL_NOT_FOUND = UINT_MAX;

	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T, typename DataHolder>
	static void* ExpandChannelToCanvas(Allocator* allocator, const DataHolder* layer, const void* data, unsigned int canvasWidth, unsigned int canvasHeight)
	{
		T* canvasData = static_cast<T*>(allocator->Allocate(sizeof(T) * canvasWidth * canvasHeight, 16u));
		memset(canvasData, 0u, sizeof(T) * canvasWidth * canvasHeight);

		imageUtil::CopyLayerData(static_cast<const T*>(data), canvasData, layer->left, layer->top, layer->right, layer->bottom, canvasWidth, canvasHeight);

		return canvasData;
	}

	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	static void* ExpandChannelToCanvas(const Document* document, Allocator* allocator, Layer* layer, Channel* channel)
	{
		if (document->bitsPerChannel == 8)
			return ExpandChannelToCanvas<uint8_t>(allocator, layer, channel->data, document->width, document->height);
		else if (document->bitsPerChannel == 16)
			return ExpandChannelToCanvas<uint16_t>(allocator, layer, channel->data, document->width, document->height);
		else if (document->bitsPerChannel == 32)
			return ExpandChannelToCanvas<float32_t>(allocator, layer, channel->data, document->width, document->height);

		return nullptr;
	}

	//canvasData[0] = ExpandChannelToCanvas(document, &allocator, layer, &layer->channels[indexR]);

	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	static void* ExpandMaskToCanvas(const Document* document, Allocator* allocator, T* mask)
	{
		if (document->bitsPerChannel == 8)
			return ExpandChannelToCanvas<uint8_t>(allocator, mask, mask->data, document->width, document->height);
		else if (document->bitsPerChannel == 16)
			return ExpandChannelToCanvas<uint16_t>(allocator, mask, mask->data, document->width, document->height);
		else if (document->bitsPerChannel == 32)
			return ExpandChannelToCanvas<float32_t>(allocator, mask, mask->data, document->width, document->height);

		return nullptr;
	}

	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	unsigned int FindChannel(Layer* layer, int16_t channelType)
	{
		for (unsigned int i = 0; i < layer->channelCount; ++i)
		{
			Channel* channel = &layer->channels[i];
			if (channel->data && channel->type == channelType)
				return i;
		}

		return CHANNEL_NOT_FOUND;
	}

	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	T* CreateInterleavedImage(Allocator* allocator, const void* srcR, const void* srcG, const void* srcB, unsigned int width, unsigned int height)
	{
		T* image = static_cast<T*>(allocator->Allocate(width * height * 4u * sizeof(T), 16u));

		const T* r = static_cast<const T*>(srcR);
		const T* g = static_cast<const T*>(srcG);
		const T* b = static_cast<const T*>(srcB);
		imageUtil::InterleaveRGB(r, g, b, T(0), image, width, height);

		return image;
	}

	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	T* CreateInterleavedImage(Allocator* allocator, const void* srcR, const void* srcG, const void* srcB, const void* srcA, unsigned int width, unsigned int height)
	{
		T* image = static_cast<T*>(allocator->Allocate(width * height * 4u * sizeof(T), 16u));

		const T* r = static_cast<const T*>(srcR);
		const T* g = static_cast<const T*>(srcG);
		const T* b = static_cast<const T*>(srcB);
		const T* a = static_cast<const T*>(srcA);
		imageUtil::InterleaveRGBA(r, g, b, a, image, width, height);

		return image;
	}
} // namespace

void PSDTexture::clear_data() {
    data.clear();
}

void PSDTexture::parse() {
	if (data.is_empty()) {
		return;
	}

	layers.clear();

	const std::wstring rawFile = L"";

	MallocAllocator allocator;
	NativeFile file(&allocator);



	file.OpenBuffer(data.ptr(), data_len);


	// try opening the file. if it fails, bail out.
	if (!file.OpenRead(rawFile.c_str()))
	{
		
		//ERR_FAIL_COND_V_MSG
	}


	// create a new document that can be used for extracting different sections from the PSD.
	// additionally, the document stores information like width, height, bits per pixel, etc.
	Document* document = CreateDocument(&file, &allocator);
	if (!document)
	{
		ERR_FAIL_MSG("PSD document initialization failed.");
	}

	ERR_FAIL_COND_MSG(document->colorMode != colorMode::RGB, "PSD uses unsupported color mode.");


	// extract image resources section.
	// this gives access to the ICC profile, EXIF data and XMP metadata.
	{
		ImageResourcesSection* imageResourcesSection = ParseImageResourcesSection(document, &file, &allocator);

		DestroyImageResourcesSection(imageResourcesSection, &allocator);
	}


	// extract all layers and masks.
	bool hasTransparencyMask = false;
	LayerMaskSection* layerMaskSection = ParseLayerMaskSection(document, &file, &allocator);
	if (layerMaskSection)
	{
		hasTransparencyMask = layerMaskSection->hasTransparencyMask;

		// extract all layers one by one. this should be done in parallel for maximum efficiency.
		for (unsigned int i = 0; i < layerMaskSection->layerCount; ++i)
		{
			Layer* layer = &layerMaskSection->layers[i];
			if (layer->type != layerType::ANY)
			{
				continue;
			}
			ExtractLayer(document, &file, &allocator, layer);

			// check availability of R, G, B, and A channels.
			// we need to determine the indices of channels individually, because there is no guarantee that R is the first channel,
			// G is the second, B is the third, and so on.
			const unsigned int indexR = FindChannel(layer, channelType::R);
			const unsigned int indexG = FindChannel(layer, channelType::G);
			const unsigned int indexB = FindChannel(layer, channelType::B);
			const unsigned int indexA = FindChannel(layer, channelType::TRANSPARENCY_MASK);

			unsigned int layerWidth = document->width;
			unsigned int layerHeight = document->height;
			if (cropToCanvas == false)
			{
				layerHeight = (unsigned int)(layer->bottom - layer->top);
				layerWidth = (unsigned int)(layer->right - layer->left);
			}

			// note that channel data is only as big as the layer it belongs to, e.g. it can be smaller or bigger than the canvas,
			// depending on where it is positioned. therefore, we use the provided utility functions to expand/shrink the channel data
			// to the canvas size. of course, you can work with the channel data directly if you need to.
			void* canvasData[4] = {};
			unsigned int channelCount = 0u;
			int channelType = -1;

			if ((indexR != CHANNEL_NOT_FOUND) && (indexG != CHANNEL_NOT_FOUND) && (indexB != CHANNEL_NOT_FOUND))
			{
				// RGB channels were found.
				if (cropToCanvas == true)
				{
					canvasData[0] = ExpandChannelToCanvas(document, &allocator, layer, &layer->channels[indexR]);
					canvasData[1] = ExpandChannelToCanvas(document, &allocator, layer, &layer->channels[indexG]);
					canvasData[2] = ExpandChannelToCanvas(document, &allocator, layer, &layer->channels[indexB]);
				}
				else
				{
					canvasData[0] = layer->channels[indexR].data;
					canvasData[1] = layer->channels[indexG].data;
					canvasData[2] = layer->channels[indexB].data;
				}
				channelCount = 3u;
				channelType = COLOR_SPACE_NAME::RGB;

				if (indexA != CHANNEL_NOT_FOUND)
				{
					// A channel was also found.
					if (cropToCanvas == true)
					{
						canvasData[3] = ExpandChannelToCanvas(document, &allocator, layer, &layer->channels[indexA]);
					}
					else
					{
						canvasData[3] = layer->channels[indexA].data;
					}
					channelCount = 4u;
					channelType = COLOR_SPACE_NAME::RGBA;
				}
			}
			

			// interleave the different pieces of planar canvas data into one RGB or RGBA image, depending on what channels
			// we found, and what color mode the document is stored in.
			uint8_t* image8 = nullptr;
			uint16_t* image16 = nullptr;
			float32_t* image32 = nullptr;
			if (channelCount == 3u)
			{
				if (document->bitsPerChannel == 8)
				{
					image8 = CreateInterleavedImage<uint8_t>(&allocator, canvasData[0], canvasData[1], canvasData[2], layerWidth, layerHeight);
				}
				else if (document->bitsPerChannel == 16)
				{
					image16 = CreateInterleavedImage<uint16_t>(&allocator, canvasData[0], canvasData[1], canvasData[2], layerWidth, layerHeight);
				}
				else if (document->bitsPerChannel == 32)
				{
					image32 = CreateInterleavedImage<float32_t>(&allocator, canvasData[0], canvasData[1], canvasData[2], layerWidth, layerHeight);
				}
			}
			else if (channelCount == 4u)
			{
				if (document->bitsPerChannel == 8)
				{
					image8 = CreateInterleavedImage<uint8_t>(&allocator, canvasData[0], canvasData[1], canvasData[2], canvasData[3], layerWidth, layerHeight);
				}
				else if (document->bitsPerChannel == 16)
				{
					image16 = CreateInterleavedImage<uint16_t>(&allocator, canvasData[0], canvasData[1], canvasData[2], canvasData[3], layerWidth, layerHeight);
				}
				else if (document->bitsPerChannel == 32)
				{
					image32 = CreateInterleavedImage<float32_t>(&allocator, canvasData[0], canvasData[1], canvasData[2], canvasData[3], layerWidth, layerHeight);
				}
			}

			// ONLY free canvasData if the channel was actually copied! Otherwise the channel data is already deleted here!
			if (cropToCanvas == true)
			{
				allocator.Free(canvasData[0]);
				allocator.Free(canvasData[1]);
				allocator.Free(canvasData[2]);
				allocator.Free(canvasData[3]);
			}

			// get the layer name.
			// Unicode data is preferred because it is not truncated by Photoshop, but unfortunately it is optional.
			// fall back to the ASCII name in case no Unicode name was found.
			std::wstringstream ssLayerName;
			if (layer->utf16Name)
			{
				ssLayerName << reinterpret_cast<wchar_t*>(layer->utf16Name);
			}
			else
			{
				ssLayerName << layer->name.c_str();
			}
			std::wstring wslayerName = ssLayerName.str();
			const wchar_t* layerName = wslayerName.c_str();

			// at this point, image8, image16 or image32 store either a 8-bit, 16-bit, or 32-bit image, respectively.
			// the image data is stored in interleaved RGB or RGBA, and has the size "document->width*document->height".
			// it is up to you to do whatever you want with the image data. in the sample, we simply write the image to a .TGA file.


			if (document->bitsPerChannel == 8u)
			{
				
				ExportLayer(layerName, layerWidth, layerHeight, image8, channelType);
			}


			allocator.Free(image8);
			allocator.Free(image16);
			allocator.Free(image32);
		}
		DestroyLayerMaskSection(layerMaskSection, &allocator);
	}

	// don't forget to destroy the document, and close the file.
	DestroyDocument(document, &allocator);
	file.Close();
}

void PSDTexture::set_data(const Vector<uint8_t> &p_data) {
    int src_data_len = p_data.size();
    const uint8_t *src_datar = p_data.ptr();

    clear_data();

    data.resize(src_data_len);
    memcpy(data.ptrw(), src_datar, src_data_len);
    data_len = src_data_len;
	
	parse();

}

void PSDTexture::ExportLayer(const wchar_t* name, unsigned int width, unsigned int height, const uint8_t *data, int channelType )
{
	if (channelType == -1) {
		layers[String(name)] = Ref<Texture>();

		return;
	}

	Vector<uint8_t> data_vector;

	int bytesPerPixel = 4;
	Image::Format imageFormat = Image::FORMAT_RGBA8;

	if (channelType == COLOR_SPACE_NAME::RGB) {
		bytesPerPixel = 3;
		imageFormat = Image::FORMAT_RGB8;
	}
	else if (channelType == COLOR_SPACE_NAME::RGBA) {
		bytesPerPixel = 4;
		imageFormat = Image::FORMAT_RGBA8;
	}
	else if (channelType == COLOR_SPACE_NAME::MONOCHROME) {
		bytesPerPixel = 1;
		imageFormat = Image::FORMAT_L8;
	}

	for (unsigned int i = 0; i < width * height * bytesPerPixel; i++) {
		data_vector.push_back(data[i]);
	}

	Ref<Image> image_layer = Image::create_from_data(width, height, false, imageFormat, data_vector);

	Ref<Texture> texture_layer = ImageTexture::create_from_image(image_layer);

	layers[String(name)] = texture_layer;

}

Array PSDTexture::get_layer_names() const {
	return layers.keys();
}

Ref<ImageTexture> PSDTexture::get_texture_layer(String p_name) const {
	return layers.get(p_name, Ref<ImageTexture>());
}


Vector<uint8_t> PSDTexture::get_data() const {
    return data;
}



void PSDTexture::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_data", "data"), &PSDTexture::set_data);
    ClassDB::bind_method(D_METHOD("get_data"), &PSDTexture::get_data);

	ClassDB::bind_method(D_METHOD("get_layer_names"), &PSDTexture::get_layer_names);
	ClassDB::bind_method(D_METHOD("get_texture_layer"), &PSDTexture::get_texture_layer);

    ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_data", "get_data");
    
}

PSDTexture::PSDTexture() {
	parse();
}

PSDTexture::~PSDTexture() {
    clear_data();
}
