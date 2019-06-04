/*************************************************************************/
/*  image.h                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
		FORMAT_GRAYSCALE, ///< one byte per pixel, 0-255
		FORMAT_INTENSITY, ///< one byte per pixel, 0-255
		FORMAT_GRAYSCALE_ALPHA, ///< two bytes per pixel, 0-255. alpha 0-255
		FORMAT_RGB, ///< one byte R, one byte G, one byte B
		FORMAT_RGBA, ///< one byte R, one byte G, one byte B, one byte A
		FORMAT_INDEXED, ///< index byte 0-256, and after image end, 256*3 bytes of palette
		FORMAT_INDEXED_ALPHA, ///< index byte 0-256, and after image end, 256*4 bytes of palette (alpha)
		FORMAT_YUV_422,
		FORMAT_YUV_444,
		FORMAT_BC1, // DXT1
		FORMAT_BC2, // DXT3
		FORMAT_BC3, // DXT5
		FORMAT_BC4, // ATI1
		FORMAT_BC5, // ATI2
		FORMAT_PVRTC2,
		FORMAT_PVRTC2_ALPHA,
		FORMAT_PVRTC4,
		FORMAT_PVRTC4_ALPHA,
		FORMAT_ETC, // regular ETC, no transparency
		FORMAT_ATC,
		FORMAT_ATC_ALPHA_EXPLICIT,
		FORMAT_ATC_ALPHA_INTERPOLATED,
		/*FORMAT_ETC2_R, for the future..
		FORMAT_ETC2_RG,
		FORMAT_ETC2_RGB,
		FORMAT_ETC2_RGBA1,
		FORMAT_ETC2_RGBA,*/
		FORMAT_CUSTOM,

		FORMAT_MAX
	};

	static const char *format_names[FORMAT_MAX];
	enum Interpolation {

		INTERPOLATE_NEAREST,
		INTERPOLATE_BILINEAR,
		INTERPOLATE_CUBIC,
		/* INTERPOLATE GAUSS */
	};

	static Image (*_png_mem_loader_func)(const uint8_t *p_png, int p_size);
	static Image (*_jpg_mem_loader_func)(const uint8_t *p_png, int p_size);
	static void (*_image_compress_bc_func)(Image *);
	static void (*_image_compress_pvrtc2_func)(Image *);
	static void (*_image_compress_pvrtc4_func)(Image *);
	static void (*_image_compress_etc_func)(Image *);
	static void (*_image_decompress_pvrtc)(Image *);
	static void (*_image_decompress_bc)(Image *);
	static void (*_image_decompress_etc)(Image *);

	Error _decompress_bc();

	static DVector<uint8_t> (*lossy_packer)(const Image &p_image, float p_quality);
	static Image (*lossy_unpacker)(const DVector<uint8_t> &p_buffer);
	static DVector<uint8_t> (*lossless_packer)(const Image &p_image);
	static Image (*lossless_unpacker)(const DVector<uint8_t> &p_buffer);

private:
	//internal byte based color
	struct BColor {
		union {
			uint8_t col[4];
			struct {
				uint8_t r, g, b, a;
			};
		};

		bool operator==(const BColor &p_color) const {
			for (int i = 0; i < 4; i++) {
				if (col[i] != p_color.col[i]) return false;
			}
			return true;
		}
		_FORCE_INLINE_ uint8_t gray() const { return (uint16_t(col[0]) + uint16_t(col[1]) + uint16_t(col[2])) / 3; }
		_FORCE_INLINE_ BColor() {}
		BColor(uint8_t p_r, uint8_t p_g, uint8_t p_b, uint8_t p_a = 255) {
			col[0] = p_r;
			col[1] = p_g;
			col[2] = p_b;
			col[3] = p_a;
		}
	};

	//median cut classes

	struct BColorPos {

		uint32_t index;
		BColor color;
		struct SortR {

			bool operator()(const BColorPos &ca, const BColorPos &cb) const { return ca.color.r < cb.color.r; }
		};

		struct SortG {

			bool operator()(const BColorPos &ca, const BColorPos &cb) const { return ca.color.g < cb.color.g; }
		};

		struct SortB {

			bool operator()(const BColorPos &ca, const BColorPos &cb) const { return ca.color.b < cb.color.b; }
		};

		struct SortA {

			bool operator()(const BColorPos &ca, const BColorPos &cb) const { return ca.color.a < cb.color.a; }
		};
	};

	struct SPTree {

		bool leaf;
		uint8_t split_plane;
		uint8_t split_value;
		union {
			int left;
			int color;
		};
		int right;
		SPTree() {
			leaf = true;
			left = -1;
			right = -1;
		}
	};

	struct MCBlock {

		BColorPos min_color, max_color;
		BColorPos *colors;
		int sp_idx;
		int color_count;
		int get_longest_axis_index() const;
		int get_longest_axis_length() const;
		bool operator<(const MCBlock &p_block) const;
		void shrink();
		MCBlock();
		MCBlock(BColorPos *p_colors, int p_color_count);
	};

	Format format;
	DVector<uint8_t> data;
	int width, height, mipmaps;

	_FORCE_INLINE_ BColor _get_pixel(int p_x, int p_y, const unsigned char *p_data, int p_data_size) const;
	_FORCE_INLINE_ BColor _get_pixelw(int p_x, int p_y, int p_width, const unsigned char *p_data, int p_data_size) const;
	_FORCE_INLINE_ void _put_pixelw(int p_x, int p_y, int p_width, const BColor &p_color, unsigned char *p_data);
	_FORCE_INLINE_ void _put_pixel(int p_x, int p_y, const BColor &p_color, unsigned char *p_data);
	_FORCE_INLINE_ void _get_mipmap_offset_and_size(int p_mipmap, int &r_offset, int &r_width, int &r_height) const; //get where the mipmap begins in data
	_FORCE_INLINE_ static void _get_format_min_data_size(Format p_format, int &r_w, int &r_h);

	static int _get_dst_image_size(int p_width, int p_height, Format p_format, int &r_mipmaps, int p_mipmaps = -1);
	bool _can_modify(Format p_format) const;

public:
	int get_width() const; ///< Get image width
	int get_height() const; ///< Get image height
	int get_mipmaps() const;

	/**
	 * Get a pixel from the image. for grayscale or indexed formats, use Color::gray to obtain the actual
	 * value.
	 */
	Color get_pixel(int p_x, int p_y, int p_mipmap = 0) const;
	/**
	 * Set a pixel into the image. for grayscale or indexed formats, a suitable Color constructor.
	 */
	void put_pixel(int p_x, int p_y, const Color &p_color, int p_mipmap = 0); /* alpha and index are averaged */

	/**
	 * Convert the image to another format, as close as it can be done.
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
	 * Resize the image, using the prefered interpolation method.
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
	Error generate_mipmaps(int p_amount = -1, bool p_keep_existing = false);

	void clear_mipmaps();

	/**
	 * Generate a normal map from a grayscale image
	 */

	void make_normalmap(float p_height_scale = 1.0);

	/**
	 * Create a new image of a given size and format. Current image will be lost
	 */
	void create(int p_width, int p_height, bool p_use_mipmaps, Format p_format);
	void create(int p_width, int p_height, int p_mipmaps, Format p_format, const DVector<uint8_t> &p_data);

	void create(const char **p_xpm);
	/**
	 * returns true when the image is empty (0,0) in size
	 */
	bool empty() const;

	DVector<uint8_t> get_data() const;

	Error load(const String &p_path);
	Error save_png(const String &p_path) const;

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
	Image(int p_width, int p_height, int p_mipmaps, Format p_format, const DVector<uint8_t> &p_data);

	enum AlphaMode {
		ALPHA_NONE,
		ALPHA_BIT,
		ALPHA_BLEND
	};

	AlphaMode detect_alpha() const;
	bool is_invisible() const;

	void put_indexed_pixel(int p_x, int p_y, uint8_t p_idx, int p_mipmap = 0);
	uint8_t get_indexed_pixel(int p_x, int p_y, int p_mipmap = 0) const;
	void set_pallete(const DVector<uint8_t> &p_data);

	static int get_format_pixel_size(Format p_format);
	static int get_format_pixel_rshift(Format p_format);
	static int get_format_pallete_size(Format p_format);
	static int get_image_data_size(int p_width, int p_height, Format p_format, int p_mipmaps = 0);
	static int get_image_required_mipmaps(int p_width, int p_height, Format p_format);

	bool operator==(const Image &p_image) const;

	void quantize();

	enum CompressMode {
		COMPRESS_BC,
		COMPRESS_PVRTC2,
		COMPRESS_PVRTC4,
		COMPRESS_ETC
	};

	Error compress(CompressMode p_mode = COMPRESS_BC);
	Image compressed(int p_mode); /* from the Image::CompressMode enum */
	Error decompress();
	Image decompressed() const;
	bool is_compressed() const;

	void fix_alpha_edges();
	void premultiply_alpha();
	void srgb_to_linear();
	void normalmap_to_xy();

	void blit_rect(const Image &p_src, const Rect2 &p_src_rect, const Point2 &p_dest);
	void blit_rect_mask(const Image &p_src, const Image &p_mask, const Rect2 &p_src_rect, const Point2 &p_dest);
	void blend_rect(const Image &p_src, const Rect2 &p_src_rect, const Point2 &p_dest);
	void blend_rect_mask(const Image &p_src, const Image &p_mask, const Rect2 &p_src_rect, const Point2 &p_dest);

	void fill(const Color &p_color);

	void brush_transfer(const Image &p_src, const Image &p_brush, const Point2 &p_dest);
	Image brushed(const Image &p_src, const Image &p_brush, const Point2 &p_dest) const;

	Rect2 get_used_rect() const;
	Image get_rect(const Rect2 &p_area) const;

	static void set_compress_bc_func(void (*p_compress_func)(Image *));
	static String get_format_name(Format p_format);

	Image(const uint8_t *p_mem_png_jpg, int p_len = -1);
	Image(const char **p_xpm);
	~Image();
};

#endif
