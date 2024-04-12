/**************************************************************************/
/*  image.h                                                               */
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

#ifndef IMAGE_H
#define IMAGE_H

#include "core/io/resource.h"
#include "core/math/color.h"
#include "core/math/rect2.h"

/**
 * Image storage class. This is used to store an image in user memory, as well as
 * providing some basic methods for image manipulation.
 * Images can be loaded from a file, or registered into the Render object as textures.
 */

class Image;

typedef Error (*SavePNGFunc)(const String &p_path, const Ref<Image> &p_img);
typedef Vector<uint8_t> (*SavePNGBufferFunc)(const Ref<Image> &p_img);
typedef Error (*SaveJPGFunc)(const String &p_path, const Ref<Image> &p_img, float p_quality);
typedef Vector<uint8_t> (*SaveJPGBufferFunc)(const Ref<Image> &p_img, float p_quality);
typedef Ref<Image> (*ImageMemLoadFunc)(const uint8_t *p_png, int p_size);
typedef Ref<Image> (*ScalableImageMemLoadFunc)(const uint8_t *p_data, int p_size, float p_scale);
typedef Error (*SaveWebPFunc)(const String &p_path, const Ref<Image> &p_img, const bool p_lossy, const float p_quality);
typedef Vector<uint8_t> (*SaveWebPBufferFunc)(const Ref<Image> &p_img, const bool p_lossy, const float p_quality);

typedef Error (*SaveEXRFunc)(const String &p_path, const Ref<Image> &p_img, bool p_grayscale);
typedef Vector<uint8_t> (*SaveEXRBufferFunc)(const Ref<Image> &p_img, bool p_grayscale);

class Image : public Resource {
	GDCLASS(Image, Resource);

public:
	static SavePNGFunc save_png_func;
	static SaveJPGFunc save_jpg_func;
	static SaveEXRFunc save_exr_func;
	static SavePNGBufferFunc save_png_buffer_func;
	static SaveEXRBufferFunc save_exr_buffer_func;
	static SaveJPGBufferFunc save_jpg_buffer_func;
	static SaveWebPFunc save_webp_func;
	static SaveWebPBufferFunc save_webp_buffer_func;

	enum {
		MAX_WIDTH = (1 << 24), // force a limit somehow
		MAX_HEIGHT = (1 << 24), // force a limit somehow
		MAX_PIXELS = 268435456
	};

	enum Format {
		FORMAT_L8, //luminance
		FORMAT_LA8, //luminance-alpha
		FORMAT_R8,
		FORMAT_RG8,
		FORMAT_RGB8,
		FORMAT_RGBA8,
		FORMAT_RGBA4444,
		FORMAT_RGB565,
		FORMAT_RF, //float
		FORMAT_RGF,
		FORMAT_RGBF,
		FORMAT_RGBAF,
		FORMAT_RH, //half float
		FORMAT_RGH,
		FORMAT_RGBH,
		FORMAT_RGBAH,
		FORMAT_RGBE9995,
		FORMAT_DXT1, //s3tc bc1
		FORMAT_DXT3, //bc2
		FORMAT_DXT5, //bc3
		FORMAT_RGTC_R,
		FORMAT_RGTC_RG,
		FORMAT_BPTC_RGBA, //btpc bc7
		FORMAT_BPTC_RGBF, //float bc6h
		FORMAT_BPTC_RGBFU, //unsigned float bc6hu
		FORMAT_ETC, //etc1
		FORMAT_ETC2_R11, //etc2
		FORMAT_ETC2_R11S, //signed, NOT srgb.
		FORMAT_ETC2_RG11,
		FORMAT_ETC2_RG11S,
		FORMAT_ETC2_RGB8,
		FORMAT_ETC2_RGBA8,
		FORMAT_ETC2_RGB8A1,
		FORMAT_ETC2_RA_AS_RG, //used to make basis universal happy
		FORMAT_DXT5_RA_AS_RG, //used to make basis universal happy
		FORMAT_ASTC_4x4,
		FORMAT_ASTC_4x4_HDR,
		FORMAT_ASTC_8x8,
		FORMAT_ASTC_8x8_HDR,
		FORMAT_MAX
	};

	static const char *format_names[FORMAT_MAX];
	enum Interpolation {
		INTERPOLATE_NEAREST,
		INTERPOLATE_BILINEAR,
		INTERPOLATE_CUBIC,
		INTERPOLATE_TRILINEAR,
		INTERPOLATE_LANCZOS,
		/* INTERPOLATE_TRICUBIC, */
		/* INTERPOLATE GAUSS */
	};

	//this is used for compression
	enum UsedChannels {
		USED_CHANNELS_L,
		USED_CHANNELS_LA,
		USED_CHANNELS_R,
		USED_CHANNELS_RG,
		USED_CHANNELS_RGB,
		USED_CHANNELS_RGBA,
	};
	//some functions provided by something else

	enum ASTCFormat {
		ASTC_FORMAT_4x4,
		ASTC_FORMAT_8x8,
	};

	static ImageMemLoadFunc _png_mem_loader_func;
	static ImageMemLoadFunc _png_mem_unpacker_func;
	static ImageMemLoadFunc _jpg_mem_loader_func;
	static ImageMemLoadFunc _webp_mem_loader_func;
	static ImageMemLoadFunc _tga_mem_loader_func;
	static ImageMemLoadFunc _bmp_mem_loader_func;
	static ScalableImageMemLoadFunc _svg_scalable_mem_loader_func;
	static ImageMemLoadFunc _ktx_mem_loader_func;

	static void (*_image_compress_bc_func)(Image *, UsedChannels p_channels);
	static void (*_image_compress_bptc_func)(Image *, UsedChannels p_channels);
	static void (*_image_compress_etc1_func)(Image *);
	static void (*_image_compress_etc2_func)(Image *, UsedChannels p_channels);
	static void (*_image_compress_astc_func)(Image *, ASTCFormat p_format);

	static void (*_image_decompress_bc)(Image *);
	static void (*_image_decompress_bptc)(Image *);
	static void (*_image_decompress_etc1)(Image *);
	static void (*_image_decompress_etc2)(Image *);
	static void (*_image_decompress_astc)(Image *);

	static Vector<uint8_t> (*webp_lossy_packer)(const Ref<Image> &p_image, float p_quality);
	static Vector<uint8_t> (*webp_lossless_packer)(const Ref<Image> &p_image);
	static Ref<Image> (*webp_unpacker)(const Vector<uint8_t> &p_buffer);
	static Vector<uint8_t> (*png_packer)(const Ref<Image> &p_image);
	static Ref<Image> (*png_unpacker)(const Vector<uint8_t> &p_buffer);
	static Vector<uint8_t> (*basis_universal_packer)(const Ref<Image> &p_image, UsedChannels p_channels);
	static Ref<Image> (*basis_universal_unpacker)(const Vector<uint8_t> &p_buffer);
	static Ref<Image> (*basis_universal_unpacker_ptr)(const uint8_t *p_data, int p_size);

	_FORCE_INLINE_ Color _get_color_at_ofs(const uint8_t *ptr, uint32_t ofs) const;
	_FORCE_INLINE_ void _set_color_at_ofs(uint8_t *ptr, uint32_t ofs, const Color &p_color);

protected:
	static void _bind_methods();

private:
	Format format = FORMAT_L8;
	Vector<uint8_t> data;
	int width = 0;
	int height = 0;
	bool mipmaps = false;

	void _copy_internals_from(const Image &p_image) {
		format = p_image.format;
		width = p_image.width;
		height = p_image.height;
		mipmaps = p_image.mipmaps;
		data = p_image.data;
	}

	_FORCE_INLINE_ void _get_mipmap_offset_and_size(int p_mipmap, int &r_offset, int &r_width, int &r_height) const; //get where the mipmap begins in data

	static int _get_dst_image_size(int p_width, int p_height, Format p_format, int &r_mipmaps, int p_mipmaps = -1, int *r_mm_width = nullptr, int *r_mm_height = nullptr);
	bool _can_modify(Format p_format) const;

	_FORCE_INLINE_ void _get_clipped_src_and_dest_rects(const Ref<Image> &p_src, const Rect2i &p_src_rect, const Point2i &p_dest, Rect2i &r_clipped_src_rect, Rect2i &r_clipped_dest_rect) const;

	_FORCE_INLINE_ void _put_pixelb(int p_x, int p_y, uint32_t p_pixel_size, uint8_t *p_data, const uint8_t *p_pixel);
	_FORCE_INLINE_ void _get_pixelb(int p_x, int p_y, uint32_t p_pixel_size, const uint8_t *p_data, uint8_t *p_pixel);

	_FORCE_INLINE_ void _repeat_pixel_over_subsequent_memory(uint8_t *p_pixel, int p_pixel_size, int p_count);

	void _set_data(const Dictionary &p_data);
	Dictionary _get_data() const;

	Error _load_from_buffer(const Vector<uint8_t> &p_array, ImageMemLoadFunc p_loader);

	static void average_4_uint8(uint8_t &p_out, const uint8_t &p_a, const uint8_t &p_b, const uint8_t &p_c, const uint8_t &p_d);
	static void average_4_float(float &p_out, const float &p_a, const float &p_b, const float &p_c, const float &p_d);
	static void average_4_half(uint16_t &p_out, const uint16_t &p_a, const uint16_t &p_b, const uint16_t &p_c, const uint16_t &p_d);
	static void average_4_rgbe9995(uint32_t &p_out, const uint32_t &p_a, const uint32_t &p_b, const uint32_t &p_c, const uint32_t &p_d);
	static void renormalize_uint8(uint8_t *p_rgb);
	static void renormalize_float(float *p_rgb);
	static void renormalize_half(uint16_t *p_rgb);
	static void renormalize_rgbe9995(uint32_t *p_rgb);

public:
	int get_width() const; ///< Get image width
	int get_height() const; ///< Get image height
	Size2i get_size() const;
	bool has_mipmaps() const;
	int get_mipmap_count() const;

	/**
	 * Convert the image to another format, conversion only to raw byte format
	 */
	void convert(Format p_new_format);

	/**
	 * Get the current image format.
	 */
	Format get_format() const;

	int get_mipmap_byte_size(int p_mipmap) const; //get where the mipmap begins in data
	int get_mipmap_offset(int p_mipmap) const; //get where the mipmap begins in data
	void get_mipmap_offset_and_size(int p_mipmap, int &r_ofs, int &r_size) const; //get where the mipmap begins in data
	void get_mipmap_offset_size_and_dimensions(int p_mipmap, int &r_ofs, int &r_size, int &w, int &h) const; //get where the mipmap begins in data

	enum Image3DValidateError {
		VALIDATE_3D_OK,
		VALIDATE_3D_ERR_IMAGE_EMPTY,
		VALIDATE_3D_ERR_MISSING_IMAGES,
		VALIDATE_3D_ERR_EXTRA_IMAGES,
		VALIDATE_3D_ERR_IMAGE_SIZE_MISMATCH,
		VALIDATE_3D_ERR_IMAGE_FORMAT_MISMATCH,
		VALIDATE_3D_ERR_IMAGE_HAS_MIPMAPS,
	};

	static Image3DValidateError validate_3d_image(Format p_format, int p_width, int p_height, int p_depth, bool p_mipmaps, const Vector<Ref<Image>> &p_images);
	static String get_3d_image_validation_error_text(Image3DValidateError p_error);

	/**
	 * Resize the image, using the preferred interpolation method.
	 */
	void resize_to_po2(bool p_square = false, Interpolation p_interpolation = INTERPOLATE_BILINEAR);
	void resize(int p_width, int p_height, Interpolation p_interpolation = INTERPOLATE_BILINEAR);
	void shrink_x2();
	bool is_size_po2() const;
	/**
	 * Crop the image to a specific size, if larger, then the image is filled by black
	 */
	void crop_from_point(int p_x, int p_y, int p_width, int p_height);
	void crop(int p_width, int p_height);

	void rotate_90(ClockDirection p_direction);
	void rotate_180();

	void flip_x();
	void flip_y();

	/**
	 * Generate a mipmap to an image (creates an image 1/4 the size, with averaging of 4->1)
	 */
	Error generate_mipmaps(bool p_renormalize = false);

	enum RoughnessChannel {
		ROUGHNESS_CHANNEL_R,
		ROUGHNESS_CHANNEL_G,
		ROUGHNESS_CHANNEL_B,
		ROUGHNESS_CHANNEL_A,
		ROUGHNESS_CHANNEL_L,
	};

	Error generate_mipmap_roughness(RoughnessChannel p_roughness_channel, const Ref<Image> &p_normal_map);

	void clear_mipmaps();
	void normalize(); //for normal maps

	/**
	 * Creates new internal image data of a given size and format. Current image will be lost.
	 */
	void initialize_data(int p_width, int p_height, bool p_use_mipmaps, Format p_format);
	void initialize_data(int p_width, int p_height, bool p_use_mipmaps, Format p_format, const Vector<uint8_t> &p_data);
	void initialize_data(const char **p_xpm);

	/**
	 * returns true when the image is empty (0,0) in size
	 */
	bool is_empty() const;

	Vector<uint8_t> get_data() const;

	Error load(const String &p_path);
	static Ref<Image> load_from_file(const String &p_path);
	Error save_png(const String &p_path) const;
	Error save_jpg(const String &p_path, float p_quality = 0.75) const;
	Vector<uint8_t> save_png_to_buffer() const;
	Vector<uint8_t> save_jpg_to_buffer(float p_quality = 0.75) const;
	Vector<uint8_t> save_exr_to_buffer(bool p_grayscale = false) const;
	Error save_exr(const String &p_path, bool p_grayscale = false) const;
	Error save_webp(const String &p_path, const bool p_lossy = false, const float p_quality = 0.75f) const;
	Vector<uint8_t> save_webp_to_buffer(const bool p_lossy = false, const float p_quality = 0.75f) const;

	static Ref<Image> create_empty(int p_width, int p_height, bool p_use_mipmaps, Format p_format);
	static Ref<Image> create_from_data(int p_width, int p_height, bool p_use_mipmaps, Format p_format, const Vector<uint8_t> &p_data);
	void set_data(int p_width, int p_height, bool p_use_mipmaps, Format p_format, const Vector<uint8_t> &p_data);

	/**
	 * create an empty image
	 */
	Image() {}
	/**
	 * create an empty image of a specific size and format
	 */
	Image(int p_width, int p_height, bool p_use_mipmaps, Format p_format);
	/**
	 * import an image of a specific size and format from a pointer
	 */
	Image(int p_width, int p_height, bool p_mipmaps, Format p_format, const Vector<uint8_t> &p_data);

	~Image() {}

	enum AlphaMode {
		ALPHA_NONE,
		ALPHA_BIT,
		ALPHA_BLEND
	};

	AlphaMode detect_alpha() const;
	bool is_invisible() const;

	static int get_format_pixel_size(Format p_format);
	static int get_format_pixel_rshift(Format p_format);
	static int get_format_block_size(Format p_format);
	static void get_format_min_pixel_size(Format p_format, int &r_w, int &r_h);

	static int get_image_data_size(int p_width, int p_height, Format p_format, bool p_mipmaps = false);
	static int get_image_required_mipmaps(int p_width, int p_height, Format p_format);
	static Size2i get_image_mipmap_size(int p_width, int p_height, Format p_format, int p_mipmap);
	static int get_image_mipmap_offset(int p_width, int p_height, Format p_format, int p_mipmap);
	static int get_image_mipmap_offset_and_dimensions(int p_width, int p_height, Format p_format, int p_mipmap, int &r_w, int &r_h);

	enum CompressMode {
		COMPRESS_S3TC,
		COMPRESS_ETC,
		COMPRESS_ETC2,
		COMPRESS_BPTC,
		COMPRESS_ASTC,
		COMPRESS_MAX,
	};
	enum CompressSource {
		COMPRESS_SOURCE_GENERIC,
		COMPRESS_SOURCE_SRGB,
		COMPRESS_SOURCE_NORMAL,
		COMPRESS_SOURCE_MAX,
	};

	Error compress(CompressMode p_mode, CompressSource p_source = COMPRESS_SOURCE_GENERIC, ASTCFormat p_astc_format = ASTC_FORMAT_4x4);
	Error compress_from_channels(CompressMode p_mode, UsedChannels p_channels, ASTCFormat p_astc_format = ASTC_FORMAT_4x4);
	Error decompress();
	bool is_compressed() const;

	void fix_alpha_edges();
	void premultiply_alpha();
	void srgb_to_linear();
	void normal_map_to_xy();
	Ref<Image> rgbe_to_srgb();
	Ref<Image> get_image_from_mipmap(int p_mipamp) const;
	void bump_map_to_normal_map(float bump_scale = 1.0);

	void blit_rect(const Ref<Image> &p_src, const Rect2i &p_src_rect, const Point2i &p_dest);
	void blit_rect_mask(const Ref<Image> &p_src, const Ref<Image> &p_mask, const Rect2i &p_src_rect, const Point2i &p_dest);
	void blend_rect(const Ref<Image> &p_src, const Rect2i &p_src_rect, const Point2i &p_dest);
	void blend_rect_mask(const Ref<Image> &p_src, const Ref<Image> &p_mask, const Rect2i &p_src_rect, const Point2i &p_dest);
	void fill(const Color &p_color);
	void fill_rect(const Rect2i &p_rect, const Color &p_color);

	Rect2i get_used_rect() const;
	Ref<Image> get_region(const Rect2i &p_area) const;

	static void set_compress_bc_func(void (*p_compress_func)(Image *, UsedChannels));
	static void set_compress_bptc_func(void (*p_compress_func)(Image *, UsedChannels));
	static String get_format_name(Format p_format);

	Error load_png_from_buffer(const Vector<uint8_t> &p_array);
	Error load_jpg_from_buffer(const Vector<uint8_t> &p_array);
	Error load_webp_from_buffer(const Vector<uint8_t> &p_array);
	Error load_tga_from_buffer(const Vector<uint8_t> &p_array);
	Error load_bmp_from_buffer(const Vector<uint8_t> &p_array);
	Error load_ktx_from_buffer(const Vector<uint8_t> &p_array);

	Error load_svg_from_buffer(const Vector<uint8_t> &p_array, float scale = 1.0);
	Error load_svg_from_string(const String &p_svg_str, float scale = 1.0);

	void convert_rg_to_ra_rgba8();
	void convert_ra_rgba8_to_rg();
	void convert_rgba8_to_bgra8();

	Image(const uint8_t *p_mem_png_jpg, int p_len = -1);
	Image(const char **p_xpm);

	virtual Ref<Resource> duplicate(bool p_subresources = false) const override;

	UsedChannels detect_used_channels(CompressSource p_source = COMPRESS_SOURCE_GENERIC) const;
	void optimize_channels();

	Color get_pixelv(const Point2i &p_point) const;
	Color get_pixel(int p_x, int p_y) const;
	void set_pixelv(const Point2i &p_point, const Color &p_color);
	void set_pixel(int p_x, int p_y, const Color &p_color);

	const uint8_t *ptr() const;
	uint8_t *ptrw();
	int64_t data_size() const;

	void adjust_bcs(float p_brightness, float p_contrast, float p_saturation);

	void set_as_black();

	void copy_internals_from(const Ref<Image> &p_image) {
		ERR_FAIL_COND_MSG(p_image.is_null(), "Cannot copy image internals: invalid Image object.");
		format = p_image->format;
		width = p_image->width;
		height = p_image->height;
		mipmaps = p_image->mipmaps;
		data = p_image->data;
	}

	Dictionary compute_image_metrics(const Ref<Image> p_compared_image, bool p_luma_metric = true);
};

VARIANT_ENUM_CAST(Image::Format)
VARIANT_ENUM_CAST(Image::Interpolation)
VARIANT_ENUM_CAST(Image::CompressMode)
VARIANT_ENUM_CAST(Image::CompressSource)
VARIANT_ENUM_CAST(Image::UsedChannels)
VARIANT_ENUM_CAST(Image::AlphaMode)
VARIANT_ENUM_CAST(Image::RoughnessChannel)
VARIANT_ENUM_CAST(Image::ASTCFormat)

#endif // IMAGE_H
