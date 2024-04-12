/**************************************************************************/
/*  texture_loader_dds.cpp                                                */
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

#include "texture_loader_dds.h"

#include "core/io/file_access.h"
#include "scene/resources/image_texture.h"

#define PF_FOURCC(s) ((uint32_t)(((s)[3] << 24U) | ((s)[2] << 16U) | ((s)[1] << 8U) | ((s)[0])))

// Reference: https://docs.microsoft.com/en-us/windows/win32/direct3ddds/dds-header

enum {
	DDS_MAGIC = 0x20534444,
	DDSD_PITCH = 0x00000008,
	DDSD_LINEARSIZE = 0x00080000,
	DDSD_MIPMAPCOUNT = 0x00020000,
	DDPF_FOURCC = 0x00000004,
	DDPF_ALPHAPIXELS = 0x00000001,
	DDPF_RGB = 0x00000040
};

enum DDSFourCC {
	DDFCC_DXT1 = PF_FOURCC("DXT1"),
	DDFCC_DXT3 = PF_FOURCC("DXT3"),
	DDFCC_DXT5 = PF_FOURCC("DXT5"),
	DDFCC_ATI1 = PF_FOURCC("ATI1"),
	DDFCC_BC4U = PF_FOURCC("BC4U"),
	DDFCC_ATI2 = PF_FOURCC("ATI2"),
	DDFCC_BC5U = PF_FOURCC("BC5U"),
	DDFCC_A2XY = PF_FOURCC("A2XY"),
	DDFCC_DX10 = PF_FOURCC("DX10"),
	DDFCC_R16F = 111,
	DDFCC_RG16F = 112,
	DDFCC_RGBA16F = 113,
	DDFCC_R32F = 114,
	DDFCC_RG32F = 115,
	DDFCC_RGBA32F = 116
};

// Reference: https://learn.microsoft.com/en-us/windows/win32/api/dxgiformat/ne-dxgiformat-dxgi_format
enum DXGIFormat {
	DXGI_R32G32B32A32_FLOAT = 2,
	DXGI_R16G16B16A16_FLOAT = 10,
	DXGI_R32G32_FLOAT = 16,
	DXGI_R10G10B10A2_UNORM = 24,
	DXGI_R8G8B8A8_UNORM = 28,
	DXGI_R16G16_FLOAT = 34,
	DXGI_R32_FLOAT = 41,
	DXGI_R16_FLOAT = 54,
	DXGI_R9G9B9E5 = 67,
	DXGI_BC1_UNORM = 71,
	DXGI_BC2_UNORM = 74,
	DXGI_BC3_UNORM = 77,
	DXGI_BC4_UNORM = 80,
	DXGI_BC5_UNORM = 83,
	DXGI_B5G6R5_UNORM = 85,
	DXGI_B5G5R5A1_UNORM = 86,
	DXGI_B8G8R8A8_UNORM = 87,
	DXGI_BC6H_UF16 = 95,
	DXGI_BC6H_SF16 = 96,
	DXGI_BC7_UNORM = 98,
	DXGI_B4G4R4A4_UNORM = 115
};

// The legacy bitmasked format names here represent the actual data layout in the files,
// while their official names are flipped (e.g. RGBA8 layout is officially called ABGR8).
enum DDSFormat {
	DDS_DXT1,
	DDS_DXT3,
	DDS_DXT5,
	DDS_ATI1,
	DDS_ATI2,
	DDS_BC6U,
	DDS_BC6S,
	DDS_BC7U,
	DDS_R16F,
	DDS_RG16F,
	DDS_RGBA16F,
	DDS_R32F,
	DDS_RG32F,
	DDS_RGBA32F,
	DDS_RGB9E5,
	DDS_BGRA8,
	DDS_BGR8,
	DDS_RGBA8,
	DDS_RGB8,
	DDS_BGR5A1,
	DDS_BGR565,
	DDS_BGR10A2,
	DDS_RGB10A2,
	DDS_BGRA4,
	DDS_LUMINANCE,
	DDS_LUMINANCE_ALPHA,
	DDS_MAX
};

struct DDSFormatInfo {
	const char *name = nullptr;
	bool compressed = false;
	uint32_t divisor = 0;
	uint32_t block_size = 0;
	Image::Format format = Image::Format::FORMAT_BPTC_RGBA;
};

static const DDSFormatInfo dds_format_info[DDS_MAX] = {
	{ "DXT1/BC1", true, 4, 8, Image::FORMAT_DXT1 },
	{ "DXT3/BC2", true, 4, 16, Image::FORMAT_DXT3 },
	{ "DXT5/BC3", true, 4, 16, Image::FORMAT_DXT5 },
	{ "ATI1/BC4", true, 4, 8, Image::FORMAT_RGTC_R },
	{ "ATI2/A2XY/BC5", true, 4, 16, Image::FORMAT_RGTC_RG },
	{ "BC6U", true, 4, 16, Image::FORMAT_BPTC_RGBFU },
	{ "BC6S", true, 4, 16, Image::FORMAT_BPTC_RGBF },
	{ "BC7U", true, 4, 16, Image::FORMAT_BPTC_RGBA },
	{ "R16F", false, 1, 2, Image::FORMAT_RH },
	{ "RG16F", false, 1, 4, Image::FORMAT_RGH },
	{ "RGBA16F", false, 1, 8, Image::FORMAT_RGBAH },
	{ "R32F", false, 1, 4, Image::FORMAT_RF },
	{ "RG32F", false, 1, 8, Image::FORMAT_RGF },
	{ "RGBA32F", false, 1, 16, Image::FORMAT_RGBAF },
	{ "RGB9E5", false, 1, 4, Image::FORMAT_RGBE9995 },
	{ "BGRA8", false, 1, 4, Image::FORMAT_RGBA8 },
	{ "BGR8", false, 1, 3, Image::FORMAT_RGB8 },
	{ "RGBA8", false, 1, 4, Image::FORMAT_RGBA8 },
	{ "RGB8", false, 1, 3, Image::FORMAT_RGB8 },
	{ "BGR5A1", false, 1, 2, Image::FORMAT_RGBA8 },
	{ "BGR565", false, 1, 2, Image::FORMAT_RGB8 },
	{ "BGR10A2", false, 1, 4, Image::FORMAT_RGBA8 },
	{ "RGB10A2", false, 1, 4, Image::FORMAT_RGBA8 },
	{ "BGRA4", false, 1, 2, Image::FORMAT_RGBA8 },
	{ "GRAYSCALE", false, 1, 1, Image::FORMAT_L8 },
	{ "GRAYSCALE_ALPHA", false, 1, 2, Image::FORMAT_LA8 }
};

static DDSFormat dxgi_to_dds_format(uint32_t p_dxgi_format) {
	switch (p_dxgi_format) {
		case DXGI_R32G32B32A32_FLOAT: {
			return DDS_RGBA32F;
		}
		case DXGI_R16G16B16A16_FLOAT: {
			return DDS_RGBA16F;
		}
		case DXGI_R32G32_FLOAT: {
			return DDS_RG32F;
		}
		case DXGI_R10G10B10A2_UNORM: {
			return DDS_RGB10A2;
		}
		case DXGI_R8G8B8A8_UNORM: {
			return DDS_RGBA8;
		}
		case DXGI_R16G16_FLOAT: {
			return DDS_RG16F;
		}
		case DXGI_R32_FLOAT: {
			return DDS_R32F;
		}
		case DXGI_R16_FLOAT: {
			return DDS_R16F;
		}
		case DXGI_R9G9B9E5: {
			return DDS_RGB9E5;
		}
		case DXGI_BC1_UNORM: {
			return DDS_DXT1;
		}
		case DXGI_BC2_UNORM: {
			return DDS_DXT3;
		}
		case DXGI_BC3_UNORM: {
			return DDS_DXT5;
		}
		case DXGI_BC4_UNORM: {
			return DDS_ATI1;
		}
		case DXGI_BC5_UNORM: {
			return DDS_ATI2;
		}
		case DXGI_B5G6R5_UNORM: {
			return DDS_BGR565;
		}
		case DXGI_B5G5R5A1_UNORM: {
			return DDS_BGR5A1;
		}
		case DXGI_B8G8R8A8_UNORM: {
			return DDS_BGRA8;
		}
		case DXGI_BC6H_UF16: {
			return DDS_BC6U;
		}
		case DXGI_BC6H_SF16: {
			return DDS_BC6S;
		}
		case DXGI_BC7_UNORM: {
			return DDS_BC7U;
		}
		case DXGI_B4G4R4A4_UNORM: {
			return DDS_BGRA4;
		}

		default: {
			return DDS_MAX;
		}
	}
}

Ref<Resource> ResourceFormatDDS::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	if (r_error) {
		*r_error = ERR_CANT_OPEN;
	}

	Error err;
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ, &err);
	if (f.is_null()) {
		return Ref<Resource>();
	}

	Ref<FileAccess> fref(f);
	if (r_error) {
		*r_error = ERR_FILE_CORRUPT;
	}

	ERR_FAIL_COND_V_MSG(err != OK, Ref<Resource>(), "Unable to open DDS texture file '" + p_path + "'.");

	uint32_t magic = f->get_32();
	uint32_t hsize = f->get_32();
	uint32_t flags = f->get_32();
	uint32_t height = f->get_32();
	uint32_t width = f->get_32();
	uint32_t pitch = f->get_32();
	/* uint32_t depth = */ f->get_32();
	uint32_t mipmaps = f->get_32();

	// Skip reserved.
	for (int i = 0; i < 11; i++) {
		f->get_32();
	}

	// Validate.
	// We don't check DDSD_CAPS or DDSD_PIXELFORMAT, as they're mandatory when writing,
	// but non-mandatory when reading (as some writers don't set them).
	if (magic != DDS_MAGIC || hsize != 124) {
		ERR_FAIL_V_MSG(Ref<Resource>(), "Invalid or unsupported DDS texture file '" + p_path + "'.");
	}

	/* uint32_t format_size = */ f->get_32();
	uint32_t format_flags = f->get_32();
	uint32_t format_fourcc = f->get_32();
	uint32_t format_rgb_bits = f->get_32();
	uint32_t format_red_mask = f->get_32();
	uint32_t format_green_mask = f->get_32();
	uint32_t format_blue_mask = f->get_32();
	uint32_t format_alpha_mask = f->get_32();

	/* uint32_t caps_1 = */ f->get_32();
	/* uint32_t caps_2 = */ f->get_32();
	/* uint32_t caps_3 = */ f->get_32();
	/* uint32_t caps_4 = */ f->get_32();

	// Skip reserved.
	f->get_32();

	if (f->get_position() < 128) {
		f->seek(128);
	}

	DDSFormat dds_format = DDS_MAX;

	if (format_flags & DDPF_FOURCC) {
		// FourCC formats.
		switch (format_fourcc) {
			case DDFCC_DXT1: {
				dds_format = DDS_DXT1;
			} break;
			case DDFCC_DXT3: {
				dds_format = DDS_DXT3;
			} break;
			case DDFCC_DXT5: {
				dds_format = DDS_DXT5;
			} break;
			case DDFCC_ATI1:
			case DDFCC_BC4U: {
				dds_format = DDS_ATI1;
			} break;
			case DDFCC_ATI2:
			case DDFCC_BC5U:
			case DDFCC_A2XY: {
				dds_format = DDS_ATI2;
			} break;
			case DDFCC_R16F: {
				dds_format = DDS_R16F;
			} break;
			case DDFCC_RG16F: {
				dds_format = DDS_RG16F;
			} break;
			case DDFCC_RGBA16F: {
				dds_format = DDS_RGBA16F;
			} break;
			case DDFCC_R32F: {
				dds_format = DDS_R32F;
			} break;
			case DDFCC_RG32F: {
				dds_format = DDS_RG32F;
			} break;
			case DDFCC_RGBA32F: {
				dds_format = DDS_RGBA32F;
			} break;
			case DDFCC_DX10: {
				uint32_t dxgi_format = f->get_32();
				/* uint32_t dimension = */ f->get_32();
				/* uint32_t misc_flags_1 = */ f->get_32();
				/* uint32_t array_size = */ f->get_32();
				/* uint32_t misc_flags_2 = */ f->get_32();

				dds_format = dxgi_to_dds_format(dxgi_format);
			} break;

			default: {
				ERR_FAIL_V_MSG(Ref<Resource>(), "Unrecognized or unsupported FourCC in DDS '" + p_path + "'.");
			}
		}

	} else if (format_flags & DDPF_RGB) {
		// Channel-bitmasked formats.
		if (format_flags & DDPF_ALPHAPIXELS) {
			// With alpha.
			if (format_rgb_bits == 32 && format_red_mask == 0xff0000 && format_green_mask == 0xff00 && format_blue_mask == 0xff && format_alpha_mask == 0xff000000) {
				dds_format = DDS_BGRA8;
			} else if (format_rgb_bits == 32 && format_red_mask == 0xff && format_green_mask == 0xff00 && format_blue_mask == 0xff0000 && format_alpha_mask == 0xff000000) {
				dds_format = DDS_RGBA8;
			} else if (format_rgb_bits == 16 && format_red_mask == 0x00007c00 && format_green_mask == 0x000003e0 && format_blue_mask == 0x0000001f && format_alpha_mask == 0x00008000) {
				dds_format = DDS_BGR5A1;
			} else if (format_rgb_bits == 32 && format_red_mask == 0x3ff00000 && format_green_mask == 0xffc00 && format_blue_mask == 0x3ff && format_alpha_mask == 0xc0000000) {
				dds_format = DDS_BGR10A2;
			} else if (format_rgb_bits == 32 && format_red_mask == 0x3ff && format_green_mask == 0xffc00 && format_blue_mask == 0x3ff00000 && format_alpha_mask == 0xc0000000) {
				dds_format = DDS_RGB10A2;
			} else if (format_rgb_bits == 16 && format_red_mask == 0xf00 && format_green_mask == 0xf0 && format_blue_mask == 0xf && format_alpha_mask == 0xf000) {
				dds_format = DDS_BGRA4;
			}

		} else {
			// Without alpha.
			if (format_rgb_bits == 24 && format_red_mask == 0xff0000 && format_green_mask == 0xff00 && format_blue_mask == 0xff) {
				dds_format = DDS_BGR8;
			} else if (format_rgb_bits == 24 && format_red_mask == 0xff && format_green_mask == 0xff00 && format_blue_mask == 0xff0000) {
				dds_format = DDS_RGB8;
			} else if (format_rgb_bits == 16 && format_red_mask == 0x0000f800 && format_green_mask == 0x000007e0 && format_blue_mask == 0x0000001f) {
				dds_format = DDS_BGR565;
			}
		}

	} else {
		// Other formats.
		if (format_flags & DDPF_ALPHAPIXELS && format_rgb_bits == 16 && format_red_mask == 0xff && format_alpha_mask == 0xff00) {
			dds_format = DDS_LUMINANCE_ALPHA;
		} else if (!(format_flags & DDPF_ALPHAPIXELS) && format_rgb_bits == 8 && format_red_mask == 0xff) {
			dds_format = DDS_LUMINANCE;
		}
	}

	// No format detected, error.
	if (dds_format == DDS_MAX) {
		ERR_FAIL_V_MSG(Ref<Resource>(), "Unrecognized or unsupported color layout in DDS '" + p_path + "'.");
	}

	if (!(flags & DDSD_MIPMAPCOUNT)) {
		mipmaps = 1;
	}

	Vector<uint8_t> src_data;

	const DDSFormatInfo &info = dds_format_info[dds_format];
	uint32_t w = width;
	uint32_t h = height;

	if (info.compressed) {
		// BC compressed.
		uint32_t size = MAX(info.divisor, w) / info.divisor * MAX(info.divisor, h) / info.divisor * info.block_size;

		if (flags & DDSD_LINEARSIZE) {
			ERR_FAIL_COND_V_MSG(size != pitch, Ref<Resource>(), "DDS header flags specify that a linear size of the top-level image is present, but the specified size does not match the expected value.");
		} else {
			ERR_FAIL_COND_V_MSG(pitch != 0, Ref<Resource>(), "DDS header flags specify that no linear size will given for the top-level image, but a non-zero linear size value is present in the header.");
		}

		for (uint32_t i = 1; i < mipmaps; i++) {
			w = MAX(1u, w >> 1);
			h = MAX(1u, h >> 1);

			uint32_t bsize = MAX(info.divisor, w) / info.divisor * MAX(info.divisor, h) / info.divisor * info.block_size;
			size += bsize;
		}

		src_data.resize(size);
		uint8_t *wb = src_data.ptrw();
		f->get_buffer(wb, size);

	} else {
		// Generic uncompressed.
		uint32_t size = width * height * info.block_size;

		for (uint32_t i = 1; i < mipmaps; i++) {
			w = (w + 1) >> 1;
			h = (h + 1) >> 1;
			size += w * h * info.block_size;
		}

		// Calculate the space these formats will take up after decoding.
		if (dds_format == DDS_BGR565) {
			size = size * 3 / 2;
		} else if (dds_format == DDS_BGR5A1 || dds_format == DDS_BGRA4) {
			size = size * 2;
		}

		src_data.resize(size);
		uint8_t *wb = src_data.ptrw();
		f->get_buffer(wb, size);

		// Decode nonstandard formats.
		switch (dds_format) {
			case DDS_BGR5A1: {
				// To RGBA8.
				int colcount = size / 4;

				for (int i = colcount - 1; i >= 0; i--) {
					int src_ofs = i * 2;
					int dst_ofs = i * 4;

					uint8_t a = wb[src_ofs + 1] & 0x80;
					uint8_t b = wb[src_ofs] & 0x1F;
					uint8_t g = (wb[src_ofs] >> 5) | ((wb[src_ofs + 1] & 0x3) << 3);
					uint8_t r = (wb[src_ofs + 1] >> 2) & 0x1F;

					wb[dst_ofs + 0] = r << 3;
					wb[dst_ofs + 1] = g << 3;
					wb[dst_ofs + 2] = b << 3;
					wb[dst_ofs + 3] = a ? 255 : 0;
				}

			} break;
			case DDS_BGR565: {
				// To RGB8.
				int colcount = size / 3;

				for (int i = colcount - 1; i >= 0; i--) {
					int src_ofs = i * 2;
					int dst_ofs = i * 3;

					uint8_t b = wb[src_ofs] & 0x1F;
					uint8_t g = (wb[src_ofs] >> 5) | ((wb[src_ofs + 1] & 0x7) << 3);
					uint8_t r = wb[src_ofs + 1] >> 3;

					wb[dst_ofs + 0] = r << 3;
					wb[dst_ofs + 1] = g << 2;
					wb[dst_ofs + 2] = b << 3;
				}

			} break;
			case DDS_BGRA4: {
				// To RGBA8.
				int colcount = size / 4;

				for (int i = colcount - 1; i >= 0; i--) {
					int src_ofs = i * 2;
					int dst_ofs = i * 4;

					uint8_t b = wb[src_ofs] & 0x0F;
					uint8_t g = wb[src_ofs] & 0xF0;
					uint8_t r = wb[src_ofs + 1] & 0x0F;
					uint8_t a = wb[src_ofs + 1] & 0xF0;

					wb[dst_ofs] = (r << 4) | r;
					wb[dst_ofs + 1] = g | (g >> 4);
					wb[dst_ofs + 2] = (b << 4) | b;
					wb[dst_ofs + 3] = a | (a >> 4);
				}

			} break;
			case DDS_RGB10A2: {
				// To RGBA8.
				int colcount = size / 4;

				for (int i = 0; i < colcount; i++) {
					int ofs = i * 4;

					uint32_t w32 = uint32_t(wb[ofs + 0]) | (uint32_t(wb[ofs + 1]) << 8) | (uint32_t(wb[ofs + 2]) << 16) | (uint32_t(wb[ofs + 3]) << 24);

					// This method follows the 'standard' way of decoding 10-bit dds files,
					// which means the ones created with DirectXTex will be loaded incorrectly.
					uint8_t a = (w32 & 0xc0000000) >> 24;
					uint8_t r = (w32 & 0x3ff) >> 2;
					uint8_t g = (w32 & 0xffc00) >> 12;
					uint8_t b = (w32 & 0x3ff00000) >> 22;

					wb[ofs + 0] = r;
					wb[ofs + 1] = g;
					wb[ofs + 2] = b;
					wb[ofs + 3] = a == 0xc0 ? 255 : a; // 0xc0 should be opaque.
				}

			} break;
			case DDS_BGR10A2: {
				// To RGBA8.
				int colcount = size / 4;

				for (int i = 0; i < colcount; i++) {
					int ofs = i * 4;

					uint32_t w32 = uint32_t(wb[ofs + 0]) | (uint32_t(wb[ofs + 1]) << 8) | (uint32_t(wb[ofs + 2]) << 16) | (uint32_t(wb[ofs + 3]) << 24);

					// This method follows the 'standard' way of decoding 10-bit dds files,
					// which means the ones created with DirectXTex will be loaded incorrectly.
					uint8_t a = (w32 & 0xc0000000) >> 24;
					uint8_t r = (w32 & 0x3ff00000) >> 22;
					uint8_t g = (w32 & 0xffc00) >> 12;
					uint8_t b = (w32 & 0x3ff) >> 2;

					wb[ofs + 0] = r;
					wb[ofs + 1] = g;
					wb[ofs + 2] = b;
					wb[ofs + 3] = a == 0xc0 ? 255 : a; // 0xc0 should be opaque.
				}

			} break;
			case DDS_BGRA8: {
				// To RGBA8.
				int colcount = size / 4;

				for (int i = 0; i < colcount; i++) {
					SWAP(wb[i * 4 + 0], wb[i * 4 + 2]);
				}

			} break;
			case DDS_BGR8: {
				// To RGB8.
				int colcount = size / 3;

				for (int i = 0; i < colcount; i++) {
					SWAP(wb[i * 3 + 0], wb[i * 3 + 2]);
				}

			} break;

			default: {
			}
		}
	}

	Ref<Image> img = memnew(Image(width, height, mipmaps - 1, info.format, src_data));
	Ref<ImageTexture> texture = ImageTexture::create_from_image(img);

	if (r_error) {
		*r_error = OK;
	}

	return texture;
}

void ResourceFormatDDS::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("dds");
}

bool ResourceFormatDDS::handles_type(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "Texture2D");
}

String ResourceFormatDDS::get_resource_type(const String &p_path) const {
	if (p_path.get_extension().to_lower() == "dds") {
		return "ImageTexture";
	}
	return "";
}
