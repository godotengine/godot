/*************************************************************************/
/*  texture_loader_dds.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "texture_loader_dds.h"
#include "os/file_access.h"

enum {
	DDS_MAGIC = 0x20534444,
	DDSD_CAPS = 0x00000001,
	DDSD_PIXELFORMAT = 0x00001000,
	DDSD_PITCH = 0x00000008,
	DDSD_LINEARSIZE = 0x00080000,
	DDSD_MIPMAPCOUNT = 0x00020000,
	DDPF_FOURCC = 0x00000004,
	DDPF_ALPHAPIXELS = 0x00000001,
	DDPF_INDEXED = 0x00000020,
	DDPF_RGB = 0x00000040,
};

enum DDSFormat {

	DDS_DXT1,
	DDS_DXT3,
	DDS_DXT5,
	DDS_ATI1,
	DDS_ATI2,
	DDS_BGRA8,
	DDS_BGR8,
	DDS_RGBA8, //flipped in dds
	DDS_RGB8, //flipped in dds
	DDS_BGR5A1,
	DDS_BGR565,
	DDS_BGR10A2,
	DDS_INDEXED,
	DDS_LUMINANCE,
	DDS_LUMINANCE_ALPHA,
	DDS_MAX
};

struct DDSFormatInfo {
	const char *name;
	bool compressed;
	bool palette;
	uint32_t divisor;
	uint32_t block_size;
	Image::Format format;
};

static const DDSFormatInfo dds_format_info[DDS_MAX] = {
	{ "DXT1", true, false, 4, 8, Image::FORMAT_DXT1 },
	{ "DXT3", true, false, 4, 16, Image::FORMAT_DXT3 },
	{ "DXT5", true, false, 4, 16, Image::FORMAT_DXT5 },
	{ "BGRA8", false, false, 1, 4, Image::FORMAT_RGBA8 },
	{ "BGR8", false, false, 1, 3, Image::FORMAT_RGB8 },
	{ "RGBA8", false, false, 1, 4, Image::FORMAT_RGBA8 },
	{ "RGB8", false, false, 1, 3, Image::FORMAT_RGB8 },
	{ "BGR5A1", false, false, 1, 2, Image::FORMAT_RGBA8 },
	{ "BGR565", false, false, 1, 2, Image::FORMAT_RGB8 },
	{ "BGR10A2", false, false, 1, 4, Image::FORMAT_RGBA8 },
	{ "GRAYSCALE", false, false, 1, 1, Image::FORMAT_L8 },
	{ "GRAYSCALE_ALPHA", false, false, 1, 2, Image::FORMAT_LA8 }
};

RES ResourceFormatDDS::load(const String &p_path, const String &p_original_path, Error *r_error) {

	if (r_error)
		*r_error = ERR_CANT_OPEN;

	Error err;
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ, &err);
	if (!f)
		return RES();

	FileAccessRef fref(f);
	if (r_error)
		*r_error = ERR_FILE_CORRUPT;

	ERR_EXPLAIN("Unable to open DDS texture file: " + p_path);
	ERR_FAIL_COND_V(err != OK, RES());

	uint32_t magic = f->get_32();
	uint32_t hsize = f->get_32();
	uint32_t flags = f->get_32();
	uint32_t width = f->get_32();
	uint32_t height = f->get_32();
	uint32_t pitch = f->get_32();
	/* uint32_t depth = */ f->get_32();
	uint32_t mipmaps = f->get_32();

	//skip 11
	for (int i = 0; i < 11; i++)
		f->get_32();

	//validate

	if (magic != DDS_MAGIC || hsize != 124 || !(flags & DDSD_PIXELFORMAT) || !(flags & DDSD_CAPS)) {

		ERR_EXPLAIN("Invalid or Unsupported DDS texture file: " + p_path);
		ERR_FAIL_V(RES());
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
	/* uint32_t caps_ddsx = */ f->get_32();

	//reserved skip
	f->get_32();
	f->get_32();

	/*
	print_line("DDS width: "+itos(width));
	print_line("DDS height: "+itos(height));
	print_line("DDS mipmaps: "+itos(mipmaps));

	printf("fourcc: %x fflags: %x, rgbbits: %x, fsize: %x\n",format_fourcc,format_flags,format_rgb_bits,format_size);
	printf("rmask: %x gmask: %x, bmask: %x, amask: %x\n",format_red_mask,format_green_mask,format_blue_mask,format_alpha_mask);
	*/

	//must avoid this later
	while (f->get_position() < 128)
		f->get_8();

	DDSFormat dds_format;

	if (format_flags & DDPF_FOURCC && format_fourcc == 0x31545844) { //'1TXD'

		dds_format = DDS_DXT1;
	} else if (format_flags & DDPF_FOURCC && format_fourcc == 0x33545844) { //'3TXD'

		dds_format = DDS_DXT3;

	} else if (format_flags & DDPF_FOURCC && format_fourcc == 0x35545844) { //'5TXD'

		dds_format = DDS_DXT5;
	} else if (format_flags & DDPF_FOURCC && format_fourcc == 0x31495441) { //'1ITA'

		dds_format = DDS_ATI1;
	} else if (format_flags & DDPF_FOURCC && format_fourcc == 0x32495441) { //'2ITA'

		dds_format = DDS_ATI2;

	} else if (format_flags & DDPF_RGB && format_flags & DDPF_ALPHAPIXELS && format_rgb_bits == 32 && format_red_mask == 0xff0000 && format_green_mask == 0xff00 && format_blue_mask == 0xff && format_alpha_mask == 0xff000000) {

		dds_format = DDS_BGRA8;
	} else if (format_flags & DDPF_RGB && !(format_flags & DDPF_ALPHAPIXELS) && format_rgb_bits == 24 && format_red_mask == 0xff0000 && format_green_mask == 0xff00 && format_blue_mask == 0xff) {

		dds_format = DDS_BGR8;
	} else if (format_flags & DDPF_RGB && format_flags & DDPF_ALPHAPIXELS && format_rgb_bits == 32 && format_red_mask == 0xff && format_green_mask == 0xff00 && format_blue_mask == 0xff0000 && format_alpha_mask == 0xff000000) {

		dds_format = DDS_RGBA8;
	} else if (format_flags & DDPF_RGB && !(format_flags & DDPF_ALPHAPIXELS) && format_rgb_bits == 24 && format_red_mask == 0xff && format_green_mask == 0xff00 && format_blue_mask == 0xff0000) {

		dds_format = DDS_RGB8;

	} else if (format_flags & DDPF_RGB && format_flags & DDPF_ALPHAPIXELS && format_rgb_bits == 16 && format_red_mask == 0x00007c00 && format_green_mask == 0x000003e0 && format_blue_mask == 0x0000001f && format_alpha_mask == 0x00008000) {

		dds_format = DDS_BGR5A1;
	} else if (format_flags & DDPF_RGB && format_flags & DDPF_ALPHAPIXELS && format_rgb_bits == 32 && format_red_mask == 0x3ff00000 && format_green_mask == 0xffc00 && format_blue_mask == 0x3ff && format_alpha_mask == 0xc0000000) {

		dds_format = DDS_BGR10A2;
	} else if (format_flags & DDPF_RGB && !(format_flags & DDPF_ALPHAPIXELS) && format_rgb_bits == 16 && format_red_mask == 0x0000f800 && format_green_mask == 0x000007e0 && format_blue_mask == 0x0000001f) {

		dds_format = DDS_BGR565;
	} else if (!(format_flags & DDPF_ALPHAPIXELS) && format_rgb_bits == 8 && format_red_mask == 0xff && format_green_mask == 0xff && format_blue_mask == 0xff) {

		dds_format = DDS_LUMINANCE;
	} else if ((format_flags & DDPF_ALPHAPIXELS) && format_rgb_bits == 16 && format_red_mask == 0xff && format_green_mask == 0xff && format_blue_mask == 0xff && format_alpha_mask == 0xff00) {

		dds_format = DDS_LUMINANCE_ALPHA;
	} else if (format_flags & DDPF_INDEXED && format_rgb_bits == 8) {

		dds_format = DDS_BGR565;
	} else {

		printf("unrecognized fourcc %x format_flags: %x - rgbbits %i - red_mask %x green mask %x blue mask %x alpha mask %x\n", format_fourcc, format_flags, format_rgb_bits, format_red_mask, format_green_mask, format_blue_mask, format_alpha_mask);
		ERR_EXPLAIN("Unrecognized or Unsupported color layout in DDS: " + p_path);

		ERR_FAIL_V(RES());
	}

	if (!(flags & DDSD_MIPMAPCOUNT))
		mipmaps = 1;

	//print_line("found format: "+String(dds_format_info[dds_format].name));

	PoolVector<uint8_t> src_data;

	const DDSFormatInfo &info = dds_format_info[dds_format];
	uint32_t w = width;
	uint32_t h = height;

	if (info.compressed) {
		//compressed bc

		uint32_t size = MAX(info.divisor, w) / info.divisor * MAX(info.divisor, h) / info.divisor * info.block_size;
		ERR_FAIL_COND_V(size != pitch, RES());
		ERR_FAIL_COND_V(!(flags & DDSD_LINEARSIZE), RES());

		for (uint32_t i = 1; i < mipmaps; i++) {

			w = MAX(1, w >> 1);
			h = MAX(1, h >> 1);
			uint32_t bsize = MAX(info.divisor, w) / info.divisor * MAX(info.divisor, h) / info.divisor * info.block_size;
			//printf("%i x %i - block: %i\n",w,h,bsize);
			size += bsize;
		}

		src_data.resize(size);
		PoolVector<uint8_t>::Write wb = src_data.write();
		f->get_buffer(wb.ptr(), size);
		wb = PoolVector<uint8_t>::Write();

	} else if (info.palette) {

		//indexed
		ERR_FAIL_COND_V(!(flags & DDSD_PITCH), RES());
		ERR_FAIL_COND_V(format_rgb_bits != 8, RES());

		uint32_t size = pitch * height;
		ERR_FAIL_COND_V(size != width * height * info.block_size, RES());

		uint8_t palette[256 * 4];
		f->get_buffer(palette, 256 * 4);

		int colsize = 3;
		for (int i = 0; i < 256; i++) {

			if (palette[i * 4 + 3] < 255)
				colsize = 4;
		}

		int w = width;
		int h = height;

		for (uint32_t i = 1; i < mipmaps; i++) {

			w = (w + 1) >> 1;
			h = (h + 1) >> 1;
			size += w * h * info.block_size;
		}

		src_data.resize(size + 256 * colsize);
		PoolVector<uint8_t>::Write wb = src_data.write();
		f->get_buffer(wb.ptr(), size);

		for (int i = 0; i < 256; i++) {

			int dst_ofs = size + i * colsize;
			int src_ofs = i * 4;
			wb[dst_ofs + 0] = palette[src_ofs + 2];
			wb[dst_ofs + 1] = palette[src_ofs + 1];
			wb[dst_ofs + 2] = palette[src_ofs + 0];
			if (colsize == 4)
				wb[dst_ofs + 3] = palette[src_ofs + 3];
		}

		wb = PoolVector<uint8_t>::Write();
	} else {
		//uncompressed generic...

		uint32_t size = width * height * info.block_size;

		for (uint32_t i = 1; i < mipmaps; i++) {

			w = (w + 1) >> 1;
			h = (h + 1) >> 1;
			size += w * h * info.block_size;
		}

		if (dds_format == DDS_BGR565)
			size = size * 3 / 2;
		else if (dds_format == DDS_BGR5A1)
			size = size * 2;

		src_data.resize(size);
		PoolVector<uint8_t>::Write wb = src_data.write();
		f->get_buffer(wb.ptr(), size);

		switch (dds_format) {

			case DDS_BGR5A1: {

				// TO RGBA
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

				int colcount = size / 3;

				for (int i = colcount - 1; i >= 0; i--) {

					int src_ofs = i * 2;
					int dst_ofs = i * 3;

					uint8_t b = wb[src_ofs] & 0x1F;
					uint8_t g = (wb[src_ofs] >> 5) | ((wb[src_ofs + 1] & 0x7) << 3);
					uint8_t r = wb[src_ofs + 1] >> 3;
					wb[dst_ofs + 0] = r << 3;
					wb[dst_ofs + 1] = g << 2;
					wb[dst_ofs + 2] = b << 3; //b<<3;
				}

			} break;
			case DDS_BGR10A2: {

				// TO RGBA
				int colcount = size / 4;

				for (int i = colcount - 1; i >= 0; i--) {

					int ofs = i * 4;

					uint32_t w32 = uint32_t(wb[ofs + 0]) | (uint32_t(wb[ofs + 1]) << 8) | (uint32_t(wb[ofs + 2]) << 16) | (uint32_t(wb[ofs + 3]) << 24);

					uint8_t a = (w32 & 0xc0000000) >> 24;
					uint8_t r = (w32 & 0x3ff00000) >> 22;
					uint8_t g = (w32 & 0xffc00) >> 12;
					uint8_t b = (w32 & 0x3ff) >> 2;

					wb[ofs + 0] = r;
					wb[ofs + 1] = g;
					wb[ofs + 2] = b;
					wb[ofs + 3] = a == 0xc0 ? 255 : a; //0xc0 should be opaque
				}
			} break;
			case DDS_BGRA8: {

				int colcount = size / 4;

				for (int i = 0; i < colcount; i++) {

					SWAP(wb[i * 4 + 0], wb[i * 4 + 2]);
				}

			} break;
			case DDS_BGR8: {

				int colcount = size / 3;

				for (int i = 0; i < colcount; i++) {

					SWAP(wb[i * 3 + 0], wb[i * 3 + 2]);
				}
			} break;
			case DDS_RGBA8: {

				/* do nothing either
				int colcount = size/4;

				for(int i=0;i<colcount;i++) {

					uint8_t r = wb[i*4+1];
					uint8_t g = wb[i*4+2];
					uint8_t b = wb[i*4+3];
					uint8_t a = wb[i*4+0];

					wb[i*4+0]=r;
					wb[i*4+1]=g;
					wb[i*4+2]=b;
					wb[i*4+3]=a;
				}
				*/
			} break;
			case DDS_RGB8: {

				// do nothing
				/*
				int colcount = size/3;

				for(int i=0;i<colcount;i++) {

					SWAP( wb[i*3+0],wb[i*3+2] );
				}*/
			} break;
			case DDS_LUMINANCE: {

				// do nothing i guess?

			} break;
			case DDS_LUMINANCE_ALPHA: {

				// do nothing i guess?

			} break;

			default: {}
		}

		wb = PoolVector<uint8_t>::Write();
	}

	Ref<Image> img = memnew(Image(width, height, mipmaps - 1, info.format, src_data));

	Ref<ImageTexture> texture = memnew(ImageTexture);
	texture->create_from_image(img);

	if (r_error)
		*r_error = OK;

	return texture;
}

void ResourceFormatDDS::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("dds");
}

bool ResourceFormatDDS::handles_type(const String &p_type) const {

	return ClassDB::is_parent_class(p_type, "Texture");
}

String ResourceFormatDDS::get_resource_type(const String &p_path) const {

	if (p_path.get_extension().to_lower() == "dds")
		return "ImageTexture";
	return "";
}
