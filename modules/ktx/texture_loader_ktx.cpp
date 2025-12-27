/**************************************************************************/
/*  texture_loader_ktx.cpp                                                */
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

#include "texture_loader_ktx.h"

#include "core/io/file_access.h"
#include "core/io/file_access_memory.h"
#include "scene/resources/image_texture.h"
#include "servers/rendering/rendering_server.h"

#include <ktx.h>
#include <vk_format.h>

KTX_error_code ktx_read(ktxStream *stream, void *dst, const ktx_size_t count) {
	Ref<FileAccess> *f = reinterpret_cast<Ref<FileAccess> *>(stream->data.custom_ptr.address);
	(*f)->get_buffer(reinterpret_cast<uint8_t *>(dst), count);
	return KTX_SUCCESS;
}

KTX_error_code ktx_skip(ktxStream *stream, const ktx_size_t count) {
	Ref<FileAccess> *f = reinterpret_cast<Ref<FileAccess> *>(stream->data.custom_ptr.address);
	for (ktx_size_t i = 0; i < count; ++i) {
		(*f)->get_8();
	}
	return KTX_SUCCESS;
}

KTX_error_code ktx_write(ktxStream *stream, const void *src, const ktx_size_t size, const ktx_size_t count) {
	Ref<FileAccess> *f = reinterpret_cast<Ref<FileAccess> *>(stream->data.custom_ptr.address);
	(*f)->store_buffer(reinterpret_cast<const uint8_t *>(src), size * count);
	return KTX_SUCCESS;
}

KTX_error_code ktx_getpos(ktxStream *stream, ktx_off_t *const offset) {
	Ref<FileAccess> *f = reinterpret_cast<Ref<FileAccess> *>(stream->data.custom_ptr.address);
	*offset = (*f)->get_position();
	return KTX_SUCCESS;
}

KTX_error_code ktx_setpos(ktxStream *stream, const ktx_off_t offset) {
	Ref<FileAccess> *f = reinterpret_cast<Ref<FileAccess> *>(stream->data.custom_ptr.address);
	(*f)->seek(offset);
	return KTX_SUCCESS;
}

KTX_error_code ktx_getsize(ktxStream *stream, ktx_size_t *const size) {
	Ref<FileAccess> *f = reinterpret_cast<Ref<FileAccess> *>(stream->data.custom_ptr.address);
	*size = (*f)->get_length();
	return KTX_SUCCESS;
}

void ktx_destruct(ktxStream *stream) {
	(void)stream;
}

static void init_ktx_stream(ktxStream &ktx_stream, Ref<FileAccess> *f) {
	ktx_stream.read = ktx_read;
	ktx_stream.skip = ktx_skip;
	ktx_stream.write = ktx_write;
	ktx_stream.getpos = ktx_getpos;
	ktx_stream.setpos = ktx_setpos;
	ktx_stream.getsize = ktx_getsize;
	ktx_stream.destruct = ktx_destruct;
	ktx_stream.type = eStreamTypeCustom;
	ktx_stream.data.custom_ptr.address = f;
	ktx_stream.data.custom_ptr.allocatorAddress = nullptr;
	ktx_stream.data.custom_ptr.size = 0;
	ktx_stream.readpos = 0;
	ktx_stream.closeOnDestruct = false;
}

static Ref<Resource> load_from_file_access(Ref<FileAccess> f, bool p_load_as_image, Error *r_error) {
	ktxStream ktx_stream;
	init_ktx_stream(ktx_stream, &f);
	ktxTexture *ktx_texture;
	KTX_error_code result = ktxTexture_CreateFromStream(&ktx_stream,
			KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT,
			&ktx_texture);
	if (result != KTX_SUCCESS) {
		ERR_FAIL_V_MSG(Ref<Resource>(), "Invalid or unsupported KTX texture file.");
	}

	if (p_load_as_image && (ktx_texture->numDimensions != 2u || ktx_texture->isCubemap || ktx_texture->isArray)) {
		ktxTexture_Destroy(ktx_texture);
		ERR_FAIL_V_MSG(Ref<Resource>(), "Unsupported non-2D KTX image file.");
	}

	if (ktx_texture->numDimensions < 2u || ktx_texture->numDimensions > 3u) {
		uint32_t dimensions = ktx_texture->numDimensions;
		ktxTexture_Destroy(ktx_texture);
		ERR_FAIL_V_MSG(Ref<Resource>(), "Unsupported " + itos(dimensions) + "-dimensional KTX texture file.");
	}

	if (ktx_texture->numFaces != (ktx_texture->isCubemap ? 6u : 1u)) {
		ktxTexture_Destroy(ktx_texture);
		ERR_FAIL_V_MSG(Ref<Resource>(), "Unsupported cube map KTX texture file.");
	}

	if (ktx_texture->isArray && ktx_texture->numDimensions != 2u) {
		ktxTexture_Destroy(ktx_texture);
		ERR_FAIL_V_MSG(Ref<Resource>(), "Unsupported non-2D array KTX texture file.");
	}

	uint32_t width = ktx_texture->baseWidth;
	uint32_t height = ktx_texture->baseHeight;
	uint32_t slices_or_faces = ktx_texture->isCubemap ? ktx_texture->numFaces : ktx_texture->baseDepth;
	uint32_t layers = ktx_texture->numLayers;
	uint32_t mipmaps = ktx_texture->numLevels;
	Image::Format format;

	switch (ktx_texture->classId) {
		case ktxTexture1_c: {
			ktxTexture1 *ktx_texture1 = reinterpret_cast<ktxTexture1 *>(ktx_texture);
			switch (ktx_texture1->glInternalformat) {
				case GL_LUMINANCE:
					format = Image::FORMAT_L8;
					break;
				case GL_LUMINANCE_ALPHA:
					format = Image::FORMAT_LA8;
					break;
				case GL_SRGB8:
					format = Image::FORMAT_RGB8;
					break;
				case GL_SRGB8_ALPHA8:
					format = Image::FORMAT_RGBA8;
					break;
				case GL_R8:
				case GL_R8UI:
					format = Image::FORMAT_R8;
					break;
				case GL_RG8:
					format = Image::FORMAT_RG8;
					break;
				case GL_RGB8:
					format = Image::FORMAT_RGB8;
					break;
				case GL_RGBA8:
					format = Image::FORMAT_RGBA8;
					break;
				case GL_RGBA4:
					format = Image::FORMAT_RGBA4444;
					break;
				case GL_RGB565:
					format = Image::FORMAT_RGB565;
					break;
				case GL_R32F:
					format = Image::FORMAT_RF;
					break;
				case GL_RG32F:
					format = Image::FORMAT_RGF;
					break;
				case GL_RGB32F:
					format = Image::FORMAT_RGBF;
					break;
				case GL_RGBA32F:
					format = Image::FORMAT_RGBAF;
					break;
				case GL_R16F:
					format = Image::FORMAT_RH;
					break;
				case GL_RG16F:
					format = Image::FORMAT_RGH;
					break;
				case GL_RGB16F:
					format = Image::FORMAT_RGBH;
					break;
				case GL_RGBA16F:
					format = Image::FORMAT_RGBAH;
					break;
				case GL_RGB9_E5:
					format = Image::FORMAT_RGBE9995;
					break;
				case GL_R16:
					format = Image::FORMAT_R16;
					break;
				case GL_RG16:
					format = Image::FORMAT_RG16;
					break;
				case GL_RGB16:
					format = Image::FORMAT_RGB16;
					break;
				case GL_RGBA16:
					format = Image::FORMAT_RGBA16;
					break;
				case GL_R16UI:
					format = Image::FORMAT_R16I;
					break;
				case GL_RG16UI:
					format = Image::FORMAT_RG16I;
					break;
				case GL_RGB16UI:
					format = Image::FORMAT_RGB16I;
					break;
				case GL_RGBA16UI:
					format = Image::FORMAT_RGBA16I;
					break;
				case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
				case GL_COMPRESSED_SRGB_S3TC_DXT1_EXT:
				case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
				case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT:
					format = Image::FORMAT_DXT1;
					break;
				case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
				case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT:
					format = Image::FORMAT_DXT3;
					break;
				case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:
				case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT:
					format = Image::FORMAT_DXT5;
					break;
				case GL_COMPRESSED_RED_RGTC1:
					format = Image::FORMAT_RGTC_R;
					break;
				case GL_COMPRESSED_RG_RGTC2:
					format = Image::FORMAT_RGTC_RG;
					break;
				case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
					format = Image::FORMAT_BPTC_RGBFU;
					break;
				case GL_COMPRESSED_RGBA_BPTC_UNORM:
				case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
					format = Image::FORMAT_BPTC_RGBA;
					break;
				case GL_ETC1_RGB8_OES:
					format = Image::FORMAT_ETC;
					break;
				case GL_COMPRESSED_R11_EAC:
					format = Image::FORMAT_ETC2_R11;
					break;
				// Decompression is not supported for this format.
				/*case GL_COMPRESSED_SIGNED_R11_EAC:
					format = Image::FORMAT_ETC2_R11S;
					break;*/
				case GL_COMPRESSED_RG11_EAC:
					format = Image::FORMAT_ETC2_RG11;
					break;
				// Decompression is not supported for this format.
				/*case GL_COMPRESSED_SIGNED_RG11_EAC:
					format = Image::FORMAT_ETC2_RG11S;
					break;*/
				case GL_COMPRESSED_RGB8_ETC2:
					format = Image::FORMAT_ETC2_RGB8;
					break;
				case GL_COMPRESSED_RGBA8_ETC2_EAC:
					format = Image::FORMAT_ETC2_RGBA8;
					break;
				// Decompression is not supported for this format.
				/*case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2:
					format = Image::FORMAT_ETC2_RGB8A1;
					break;*/
				case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:
					format = Image::FORMAT_ASTC_4x4;
					break;
				case GL_COMPRESSED_RGBA_ASTC_4x4_KHR:
					format = ktx_texture1->glType == GL_UNSIGNED_BYTE ? Image::FORMAT_ASTC_4x4 : Image::FORMAT_ASTC_4x4_HDR;
					break;
				case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:
					format = Image::FORMAT_ASTC_8x8;
					break;
				case GL_COMPRESSED_RGBA_ASTC_8x8_KHR:
					format = ktx_texture1->glType == GL_UNSIGNED_BYTE ? Image::FORMAT_ASTC_8x8 : Image::FORMAT_ASTC_8x8_HDR;
					break;
				default: {
					uint32_t gl_format = ktx_texture1->glInternalformat;
					ktxTexture_Destroy(ktx_texture);
					ERR_FAIL_V_MSG(Ref<Resource>(), "Unsupported format " + itos(gl_format) + " of KTX1 texture file.");
					break;
				}
			}
			break;
		}
		case ktxTexture2_c: {
			ktxTexture2 *ktx_texture2 = reinterpret_cast<ktxTexture2 *>(ktx_texture);
			if (ktx_texture2->vkFormat == VK_FORMAT_UNDEFINED) {
				if (!ktxTexture2_NeedsTranscoding(ktx_texture2)) {
					ktxTexture_Destroy(ktx_texture);
					ERR_FAIL_V_MSG(Ref<Resource>(), "Invalid VK_FORMAT_UNDEFINED of KTX2 texture file.");
				}
				ktx_transcode_fmt_e ktxfmt;
				switch (ktxTexture2_GetNumComponents(ktx_texture2)) {
					case 1: {
						if (ktxTexture2_GetOETF_e(ktx_texture2) == KHR_DF_TRANSFER_SRGB) {
							// sRGB isn't used by Godot in any 1-channel format
							ktxfmt = KTX_TTF_RGBA32;
						} else if (RS::get_singleton()->has_os_feature("rgtc")) {
							ktxfmt = KTX_TTF_BC4_R;
						} else {
							ktxfmt = KTX_TTF_RGBA32;
						}
						break;
					}
					case 2: {
						if (ktxTexture2_GetOETF_e(ktx_texture2) == KHR_DF_TRANSFER_SRGB) {
							// sRGB isn't used by Godot in any 2-channel format
							ktxfmt = KTX_TTF_RGBA32;
						} else if (RS::get_singleton()->has_os_feature("rgtc")) {
							ktxfmt = KTX_TTF_BC5_RG;
						} else {
							ktxfmt = KTX_TTF_RGBA32;
						}
						break;
					}
					case 3: {
						// Assume that sRGB textures have color information and non-sRGB textures have non-color information.
						// Godot doesn't handle color space per-texture, but rather per-use (albedo is treated as sRGB, normal maps aren't, for example).
						if (RS::get_singleton()->has_os_feature("bptc")) {
							ktxfmt = KTX_TTF_BC7_RGBA;
						} else if (RS::get_singleton()->has_os_feature("s3tc")) {
							ktxfmt = KTX_TTF_BC1_RGB;
						} else if (RS::get_singleton()->has_os_feature("etc2")) {
							ktxfmt = KTX_TTF_ETC1_RGB;
						} else {
							ktxfmt = KTX_TTF_RGBA32;
						}
						break;
					}
					case 4: {
						// Assume that sRGB textures have color information and non-sRGB textures have non-color information.
						// Godot doesn't handle color space per-texture, but rather per-use (albedo is treated as sRGB, normal maps aren't, for example).
						if (RS::get_singleton()->has_os_feature("astc")) {
							ktxfmt = KTX_TTF_ASTC_4x4_RGBA;
						} else if (RS::get_singleton()->has_os_feature("bptc")) {
							ktxfmt = KTX_TTF_BC7_RGBA;
						} else if (RS::get_singleton()->has_os_feature("s3tc")) {
							ktxfmt = KTX_TTF_BC3_RGBA;
						} else if (RS::get_singleton()->has_os_feature("etc2")) {
							ktxfmt = KTX_TTF_ETC2_RGBA;
						} else {
							ktxfmt = KTX_TTF_RGBA32;
						}
						break;
					}
					default: {
						uint32_t components = ktxTexture2_GetNumComponents(ktx_texture2);
						ktxTexture_Destroy(ktx_texture);
						ERR_FAIL_V_MSG(Ref<Resource>(), "Invalid " + itos(components) + "-component of KTX2 texture file.");
						break;
					}
				}
				result = ktxTexture2_TranscodeBasis(ktx_texture2, ktxfmt, 0);
				if (result != KTX_SUCCESS) {
					ktxTexture_Destroy(ktx_texture);
					ERR_FAIL_V_MSG(Ref<Resource>(), "Failed to transcode KTX2 texture file.");
				}
			}
			switch (ktx_texture2->vkFormat) {
				case VK_FORMAT_R8_UNORM:
					format = Image::FORMAT_L8;
					break;
				case VK_FORMAT_R8G8_UNORM:
					format = Image::FORMAT_LA8;
					break;
				case VK_FORMAT_R8G8B8_SRGB:
					format = Image::FORMAT_RGB8;
					break;
				case VK_FORMAT_R8G8B8A8_SRGB:
					format = Image::FORMAT_RGBA8;
					break;
				case VK_FORMAT_R8_UINT:
					format = Image::FORMAT_R8;
					break;
				case VK_FORMAT_R8G8_UINT:
					format = Image::FORMAT_RG8;
					break;
				case VK_FORMAT_R8G8B8_UINT:
					format = Image::FORMAT_RGB8;
					break;
				case VK_FORMAT_R8G8B8A8_UINT:
					format = Image::FORMAT_RGBA8;
					break;
				case VK_FORMAT_R4G4B4A4_UNORM_PACK16:
					format = Image::FORMAT_RGBA4444;
					break;
				case VK_FORMAT_R5G6B5_UNORM_PACK16:
					format = Image::FORMAT_RGB565;
					break;
				case VK_FORMAT_R32_SFLOAT:
					format = Image::FORMAT_RF;
					break;
				case VK_FORMAT_R32G32_SFLOAT:
					format = Image::FORMAT_RGF;
					break;
				case VK_FORMAT_R32G32B32_SFLOAT:
					format = Image::FORMAT_RGBF;
					break;
				case VK_FORMAT_R32G32B32A32_SFLOAT:
					format = Image::FORMAT_RGBAF;
					break;
				case VK_FORMAT_R16_SFLOAT:
					format = Image::FORMAT_RH;
					break;
				case VK_FORMAT_R16G16_SFLOAT:
					format = Image::FORMAT_RGH;
					break;
				case VK_FORMAT_R16G16B16_SFLOAT:
					format = Image::FORMAT_RGBH;
					break;
				case VK_FORMAT_R16G16B16A16_SFLOAT:
					format = Image::FORMAT_RGBAH;
					break;
				case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
					format = Image::FORMAT_RGBE9995;
					break;
				case VK_FORMAT_R16_UNORM:
					format = Image::FORMAT_R16;
					break;
				case VK_FORMAT_R16G16_UNORM:
					format = Image::FORMAT_RG16;
					break;
				case VK_FORMAT_R16G16B16_UNORM:
					format = Image::FORMAT_RGB16;
					break;
				case VK_FORMAT_R16G16B16A16_UNORM:
					format = Image::FORMAT_RGBA16;
					break;
				case VK_FORMAT_R16_UINT:
					format = Image::FORMAT_R16I;
					break;
				case VK_FORMAT_R16G16_UINT:
					format = Image::FORMAT_RG16I;
					break;
				case VK_FORMAT_R16G16B16_UINT:
					format = Image::FORMAT_RGB16I;
					break;
				case VK_FORMAT_R16G16B16A16_UINT:
					format = Image::FORMAT_RGBA16I;
					break;
				case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
				case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
				case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
				case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
					format = Image::FORMAT_DXT1;
					break;
				case VK_FORMAT_BC2_UNORM_BLOCK:
				case VK_FORMAT_BC2_SRGB_BLOCK:
					format = Image::FORMAT_DXT3;
					break;
				case VK_FORMAT_BC3_UNORM_BLOCK:
				case VK_FORMAT_BC3_SRGB_BLOCK:
					format = Image::FORMAT_DXT5;
					break;
				case VK_FORMAT_BC4_UNORM_BLOCK:
					format = Image::FORMAT_RGTC_R;
					break;
				case VK_FORMAT_BC5_UNORM_BLOCK:
					format = Image::FORMAT_RGTC_RG;
					break;
				case VK_FORMAT_BC6H_UFLOAT_BLOCK:
					format = Image::FORMAT_BPTC_RGBFU;
					break;
				case VK_FORMAT_BC6H_SFLOAT_BLOCK:
					format = Image::FORMAT_BPTC_RGBF;
					break;
				case VK_FORMAT_BC7_UNORM_BLOCK:
				case VK_FORMAT_BC7_SRGB_BLOCK:
					format = Image::FORMAT_BPTC_RGBA;
					break;
				case VK_FORMAT_EAC_R11_UNORM_BLOCK:
					format = Image::FORMAT_ETC2_R11;
					break;
				// Decompression is not supported for this format.
				/*case VK_FORMAT_EAC_R11_SNORM_BLOCK:
					format = Image::FORMAT_ETC2_R11S;
					break;*/
				case VK_FORMAT_EAC_R11G11_UNORM_BLOCK:
					format = Image::FORMAT_ETC2_RG11;
					break;
				// Decompression is not supported for this format.
				/*case VK_FORMAT_EAC_R11G11_SNORM_BLOCK:
					format = Image::FORMAT_ETC2_RG11S;
					break;*/
				case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:
				case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
					format = Image::FORMAT_ETC2_RGB8;
					break;
				case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK:
				case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK:
					format = Image::FORMAT_ETC2_RGBA8;
					break;
				// Decompression is not supported for this format.
				/*case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK:
					format = Image::FORMAT_ETC2_RGB8A1;
					break;*/
				case VK_FORMAT_ASTC_4x4_UNORM_BLOCK:
				case VK_FORMAT_ASTC_4x4_SRGB_BLOCK:
					format = Image::FORMAT_ASTC_4x4;
					break;
				case VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK:
					format = Image::FORMAT_ASTC_4x4_HDR;
					break;
				case VK_FORMAT_ASTC_8x8_UNORM_BLOCK:
				case VK_FORMAT_ASTC_8x8_SRGB_BLOCK:
					format = Image::FORMAT_ASTC_8x8;
					break;
				case VK_FORMAT_ASTC_8x8_SFLOAT_BLOCK:
					format = Image::FORMAT_ASTC_8x8_HDR;
					break;
				default: {
					uint32_t vk_format = ktx_texture2->vkFormat;
					ktxTexture_Destroy(ktx_texture);
					ERR_FAIL_V_MSG(Ref<Resource>(), "Unsupported Vulkan format " + itos(vk_format) + " of KTX2 texture file.");
					break;
				}
			}
			break;
		}
		default: {
			uint32_t version = ktx_texture->classId;
			ktxTexture_Destroy(ktx_texture);
			ERR_FAIL_V_MSG(Ref<Resource>(), "Unsupported version " + itos(version) + " KTX texture file.");
			break;
		}
	}

	// KTX use 4-bytes padding, don't use mipmaps if padding is effective
	// TODO: unpad dynamically
	int pixel_size = Image::get_format_pixel_size(format);
	int pixel_rshift = Image::get_format_pixel_rshift(format);
	int block = Image::get_format_block_size(format);
	int minw, minh;
	Image::get_format_min_pixel_size(format, minw, minh);
	int w = width;
	int h = height;
	int all_mipmaps_size = 0;
	for (uint32_t i = 0; i < mipmaps; ++i) {
		ktx_size_t mip_size = ktxTexture_GetImageSize(ktx_texture, i);
		size_t bw = w % block != 0 ? w + (block - w % block) : w;
		size_t bh = h % block != 0 ? h + (block - h % block) : h;
		size_t s = bw * bh;
		s *= pixel_size;
		s >>= pixel_rshift;
		all_mipmaps_size += s;
		if (mip_size != static_cast<ktx_size_t>(s)) {
			if (!i) {
				ktxTexture_Destroy(ktx_texture);
				ERR_FAIL_V_MSG(Ref<Resource>(), "Unsupported padded KTX texture file.");
			}
			mipmaps = 1;
			all_mipmaps_size = ktxTexture_GetImageSize(ktx_texture, 0);
			break;
		}
		w = MAX(minw, w >> 1);
		h = MAX(minh, h >> 1);
	}

	Vector<Ref<Image>> layers_slices_faces;
	layers_slices_faces.reserve(layers * slices_or_faces);
	for (uint32_t layer = 0; layer < layers; ++layer) {
		for (uint32_t face_slice = 0; face_slice < slices_or_faces; ++face_slice) {
			Vector<uint8_t> src_data;
			src_data.reserve(all_mipmaps_size);

			for (uint32_t i = 0; i < mipmaps; ++i) {
				ktx_size_t mip_size = ktxTexture_GetImageSize(ktx_texture, i);
				ktx_size_t offset;
				if (ktxTexture_GetImageOffset(ktx_texture, i, layer, face_slice, &offset) != KTX_SUCCESS) {
					ktxTexture_Destroy(ktx_texture);
					ERR_FAIL_V_MSG(Ref<Resource>(), "Invalid KTX texture file.");
				}
				int prev_size = src_data.size();
				src_data.resize(prev_size + mip_size);
				memcpy(src_data.ptrw() + prev_size, ktxTexture_GetData(ktx_texture) + offset, mip_size);
			}

			Ref<Image> img = memnew(Image(width, height, mipmaps > 1u, format, src_data));

			layers_slices_faces.append(img);
		}
	}

	if (r_error) {
		*r_error = OK;
	}

	if (p_load_as_image) {
		ktxTexture_Destroy(ktx_texture);
		return layers_slices_faces[0];
	}

	if (ktx_texture->numDimensions == 3) {
		ktxTexture_Destroy(ktx_texture);

		Ref<ImageTexture3D> texture;
		texture.instantiate();

		Error err = texture->create(format, width, height, slices_or_faces, mipmaps > 1u, layers_slices_faces);
		if (r_error) {
			*r_error = err;
		}

		return texture;
	}

	if (ktx_texture->isCubemap && ktx_texture->isArray) {
		ktxTexture_Destroy(ktx_texture);

		Ref<CubemapArray> texture;
		texture.instantiate();

		Error err = texture->create_from_images(layers_slices_faces);
		if (r_error) {
			*r_error = err;
		}

		return texture;
	}

	if (ktx_texture->isArray) {
		ktxTexture_Destroy(ktx_texture);

		Ref<Texture2DArray> texture;
		texture.instantiate();

		Error err = texture->create_from_images(layers_slices_faces);
		if (r_error) {
			*r_error = err;
		}

		return texture;
	}

	if (ktx_texture->isCubemap) {
		ktxTexture_Destroy(ktx_texture);

		Ref<Cubemap> texture;
		texture.instantiate();

		Error err = texture->create_from_images(layers_slices_faces);
		if (r_error) {
			*r_error = err;
		}

		return texture;
	}

	DEV_ASSERT(layers_slices_faces.size() == 1);
	ktxTexture_Destroy(ktx_texture);
	return ImageTexture::create_from_image(layers_slices_faces[0]);
}

Ref<Resource> ResourceFormatKTX::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Error err;
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ, &err);
	if (f.is_null()) {
		if (r_error) {
			*r_error = err;
		}

		return Ref<Resource>();
	}

	if (r_error) {
		*r_error = ERR_FILE_CORRUPT;
	}

	ERR_FAIL_COND_V_MSG(err != OK, Ref<Resource>(), "Unable to open KTX texture file '" + p_path + "'.");
	return load_from_file_access(f, false, r_error);
}

static Ref<Image> _ktx_mem_loader_func(const uint8_t *p_ktx, int p_size) {
	Ref<FileAccessMemory> f;
	f.instantiate();
	f->open_custom(p_ktx, p_size);

	Error err;
	Ref<Image> img = load_from_file_access(f, true, &err);
	ERR_FAIL_COND_V(err, Ref<Image>());
	return img;
}

void ResourceFormatKTX::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("ktx");
	p_extensions->push_back("ktx2");
}

bool ResourceFormatKTX::handles_type(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "Texture");
}

String ResourceFormatKTX::get_resource_type(const String &p_path) const {
	if (p_path.has_extension("ktx") || p_path.has_extension("ktx2")) {
		Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
		if (unlikely(f.is_null())) {
			return "Texture";
		}

		ktxStream ktx_stream;
		init_ktx_stream(ktx_stream, &f);

		ktxTexture *ktx_texture;
		ktx_error_code_e err = ktxTexture_CreateFromStream(&ktx_stream, KTX_TEXTURE_CREATE_SKIP_KVDATA_BIT, &ktx_texture);
		if (unlikely(err != KTX_SUCCESS)) {
			return "Texture";
		}

		uint32_t dimensions = ktx_texture->numDimensions;
		bool array = ktx_texture->isArray;
		bool cubemap = ktx_texture->isCubemap;

		ktxTexture_Destroy(ktx_texture);

		if (dimensions == 3 && !array && !cubemap) {
			return "ImageTexture3D";
		}

		ERR_FAIL_COND_V_MSG(dimensions != 2, "Texture", "Unhandled " + itos(dimensions) + " dimensional KTX texture: " + p_path);

		if (cubemap) {
			return array ? "CubemapArray" : "Cubemap";
		}

		return array ? "Texture2DArray" : "ImageTexture";
	}

	return "";
}

ResourceFormatKTX::ResourceFormatKTX() {
	Image::_ktx_mem_loader_func = _ktx_mem_loader_func;
}
