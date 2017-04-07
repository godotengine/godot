/*************************************************************************/
/*  image.h                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#ifndef IMAGE_H
#define IMAGE_H

#include "color.h"
#include "dvector.h"
#include "math_2d.h"
/**
 *	@author Juan Linietsky <reduzio@gmail.com>
 *
 * Image storage class. This is used to store an image in user memory, as well as
 * providing some basic methods for image manipulation.
 * Images can be loaded from a file, or registered into the Render object as textures.
*/

class Image;

typedef Error (*SavePNGFunc)(const String &p_path, Image &p_img);

class Image {

	enum {
		MAX_WIDTH = 16384, // force a limit somehow
		MAX_HEIGHT = 16384 // force a limit somehow
	};

public:
	static SavePNGFunc save_png_func;

	enum Format {

		FORMAT_L8, //luminance
		FORMAT_LA8, //luminance-alpha
		FORMAT_R8,
		FORMAT_RG8,
		FORMAT_RGB8,
		FORMAT_RGBA8,
		FORMAT_RGB565, //16 bit
		FORMAT_RGBA4444,
		FORMAT_RGBA5551,
		FORMAT_RF, //float
		FORMAT_RGF,
		FORMAT_RGBF,
		FORMAT_RGBAF,
		FORMAT_RH, //half float
		FORMAT_RGH,
		FORMAT_RGBH,
		FORMAT_RGBAH,
		FORMAT_DXT1, //s3tc bc1
		FORMAT_DXT3, //bc2
		FORMAT_DXT5, //bc3
		FORMAT_ATI1, //bc4
		FORMAT_ATI2, //bc5
		FORMAT_BPTC_RGBA, //btpc bc6h
		FORMAT_BPTC_RGBF, //float /
		FORMAT_BPTC_RGBFU, //unsigned float
		FORMAT_PVRTC2, //pvrtc
		FORMAT_PVRTC2A,
		FORMAT_PVRTC4,
		FORMAT_PVRTC4A,
		FORMAT_ETC, //etc1
		FORMAT_ETC2_R11, //etc2
		FORMAT_ETC2_R11S, //signed, NOT srgb.
		FORMAT_ETC2_RG11,
		FORMAT_ETC2_RG11S,
		FORMAT_ETC2_RGB8,
		FORMAT_ETC2_RGBA8,
		FORMAT_ETC2_RGB8A1,
		FORMAT_MAX
	};

	static const char *format_names[FORMAT_MAX];
	enum Interpolation {

		INTERPOLATE_NEAREST,
		INTERPOLATE_BILINEAR,
		INTERPOLATE_CUBIC,
		/* INTERPOLATE GAUSS */
	};

	//some functions provided by something else

	static Image (*_png_mem_loader_func)(const uint8_t *p_png, int p_size);
	static Image (*_jpg_mem_loader_func)(const uint8_t *p_png, int p_size);

	static void (*_image_compress_bc_func)(Image *);
	static void (*_image_compress_pvrtc2_func)(Image *);
	static void (*_image_compress_pvrtc4_func)(Image *);
	static void (*_image_compress_etc_func)(Image *);
	static void (*_image_compress_etc2_func)(Image *);

	static void (*_image_decompress_pvrtc)(Image *);
	static void (*_image_decompress_bc)(Image *);
	static void (*_image_decompress_etc)(Image *);
	static void (*_image_decompress_etc2)(Image *);

	Error _decompress_bc();

	static PoolVector<uint8_t> (*lossy_packer)(const Image &p_image, float p_quality);
	static Image (*lossy_unpacker)(const PoolVector<uint8_t> &p_buffer);
	static PoolVector<uint8_t> (*lossless_packer)(const Image &p_image);
	static Image (*lossless_unpacker)(const PoolVector<uint8_t> &p_buffer);

private:
	Format format;
	PoolVector<uint8_t> data;
	int width, height;
	bool mipmaps;

	_FORCE_INLINE_ void _get_mipmap_offset_and_size(int p_mipmap, int &r_offset, int &r_width, int &r_height) const; //get where the mipmap begins in data

	static int _get_dst_image_size(int p_width, int p_height, Format p_format, int &r_mipmaps, int p_mipmaps = -1);
	bool _can_modify(Format p_format) const;

	_FORCE_INLINE_ void _put_pixelb(int p_x, int p_y, uint32_t p_pixelsize, uint8_t *p_dst, const uint8_t *p_src);
	_FORCE_INLINE_ void _get_pixelb(int p_x, int p_y, uint32_t p_pixelsize, const uint8_t *p_src, uint8_t *p_dst);

public:
	int get_width() const; ///< Get image width
	int get_height() const; ///< Get image height
	bool has_mipmaps() const;
	int get_mipmap_count() const;

	/**
	 * Convert the image to another format, conversion only to raw byte format
	 */
	void convert(Format p_new_format);

	Image converted(int p_new_format) {
		ERR_FAIL_INDEX_V(p_new_format, FORMAT_MAX, Image());

		Image ret = *this;
		ret.convert((Format)p_new_format);
		return ret;
	};

	/**
	 * Get the current image format.
	 */
	Format get_format() const;

	int get_mipmap_offset(int p_mipmap) const; //get where the mipmap begins in data
	void get_mipmap_offset_and_size(int p_mipmap, int &r_ofs, int &r_size) const; //get where the mipmap begins in data
	void get_mipmap_offset_size_and_dimensions(int p_mipmap, int &r_ofs, int &r_size, int &w, int &h) const; //get where the mipmap begins in data

	/**
	 * Resize the image, using the preferred interpolation method.
	 * Indexed-Color images always use INTERPOLATE_NEAREST.
	 */

	void resize_to_po2(bool p_square = false);
	void resize(int p_width, int p_height, Interpolation p_interpolation = INTERPOLATE_BILINEAR);
	Image resized(int p_width, int p_height, int p_interpolation = INTERPOLATE_BILINEAR);
	void shrink_x2();
	void expand_x2_hq2x();
	/**
	 * Crop the image to a specific size, if larger, then the image is filled by black
	 */
	void crop(int p_width, int p_height);

	void flip_x();
	void flip_y();

	/**
	 * Generate a mipmap to an image (creates an image 1/4 the size, with averaging of 4->1)
	 */
	Error generate_mipmaps();

	void clear_mipmaps();

	/**
	 * Create a new image of a given size and format. Current image will be lost
	 */
	void create(int p_width, int p_height, bool p_use_mipmaps, Format p_format);
	void create(int p_width, int p_height, bool p_use_mipmaps, Format p_format, const PoolVector<uint8_t> &p_data);

	void create(const char **p_xpm);
	/**
	 * returns true when the image is empty (0,0) in size
	 */
	bool empty() const;

	PoolVector<uint8_t> get_data() const;

	Error load(const String &p_path);
	Error save_png(const String &p_path);

	/**
	 * create an empty image
	 */
	Image();
	/**
	 * create an empty image of a specific size and format
	 */
	Image(int p_width, int p_height, bool p_use_mipmaps, Format p_format);
	/**
	 * import an image of a specific size and format from a pointer
	 */
	Image(int p_width, int p_height, bool p_mipmaps, Format p_format, const PoolVector<uint8_t> &p_data);

	enum AlphaMode {
		ALPHA_NONE,
		ALPHA_BIT,
		ALPHA_BLEND
	};

	AlphaMode detect_alpha() const;
	bool is_invisible() const;

	static int get_format_pixel_size(Format p_format);
	static int get_format_pixel_rshift(Format p_format);
	static void get_format_min_pixel_size(Format p_format, int &r_w, int &r_h);

	static int get_image_data_size(int p_width, int p_height, Format p_format, int p_mipmaps = 0);
	static int get_image_required_mipmaps(int p_width, int p_height, Format p_format);

	bool operator==(const Image &p_image) const;

	enum CompressMode {
		COMPRESS_16BIT,
		COMPRESS_S3TC,
		COMPRESS_PVRTC2,
		COMPRESS_PVRTC4,
		COMPRESS_ETC,
		COMPRESS_ETC2
	};

	Error compress(CompressMode p_mode = COMPRESS_S3TC);
	Image compressed(int p_mode); /* from the Image::CompressMode enum */
	Error decompress();
	Image decompressed() const;
	bool is_compressed() const;

	void fix_alpha_edges();
	void premultiply_alpha();
	void srgb_to_linear();
	void normalmap_to_xy();

	void blit_rect(const Image &p_src, const Rect2 &p_src_rect, const Point2 &p_dest);

	Rect2 get_used_rect() const;
	Image get_rect(const Rect2 &p_area) const;

	static void set_compress_bc_func(void (*p_compress_func)(Image *));
	static String get_format_name(Format p_format);

	Image(const uint8_t *p_mem_png_jpg, int p_len = -1);
	Image(const char **p_xpm);
	~Image();
};

#endif
