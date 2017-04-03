#ifndef GODOT_DLSCRIPT_IMAGE_H
#define GODOT_DLSCRIPT_IMAGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifndef GODOT_CORE_API_GODOT_IMAGE_TYPE_DEFINED
typedef struct godot_image {
	uint8_t _dont_touch_that[32];
} godot_image;
#endif

#include "godot_pool_arrays.h"

#include "../godot.h"

// This is a copypasta of the C++ enum inside the Image class
// There's no neat way of automatically updating the C enum / using the C++ enum directly
// if somebody knows a way feel free to open a PR or open an issue (or ask for Karroffel or bojidar-bg on IRC)

enum godot_image_format {

	GODOT_IMAGE_FORMAT_L8, //luminance
	GODOT_IMAGE_FORMAT_LA8, //luminance-alpha
	GODOT_IMAGE_FORMAT_R8,
	GODOT_IMAGE_FORMAT_RG8,
	GODOT_IMAGE_FORMAT_RGB8,
	GODOT_IMAGE_FORMAT_RGBA8,
	GODOT_IMAGE_FORMAT_RGB565, //16 bit
	GODOT_IMAGE_FORMAT_RGBA4444,
	GODOT_IMAGE_FORMAT_RGBA5551,
	GODOT_IMAGE_FORMAT_RF, //float
	GODOT_IMAGE_FORMAT_RGF,
	GODOT_IMAGE_FORMAT_RGBF,
	GODOT_IMAGE_FORMAT_RGBAF,
	GODOT_IMAGE_FORMAT_RH, //half float
	GODOT_IMAGE_FORMAT_RGH,
	GODOT_IMAGE_FORMAT_RGBH,
	GODOT_IMAGE_FORMAT_RGBAH,
	GODOT_IMAGE_FORMAT_DXT1, //s3tc bc1
	GODOT_IMAGE_FORMAT_DXT3, //bc2
	GODOT_IMAGE_FORMAT_DXT5, //bc3
	GODOT_IMAGE_FORMAT_ATI1, //bc4
	GODOT_IMAGE_FORMAT_ATI2, //bc5
	GODOT_IMAGE_FORMAT_BPTC_RGBA, //btpc bc6h
	GODOT_IMAGE_FORMAT_BPTC_RGBF, //float /
	GODOT_IMAGE_FORMAT_BPTC_RGBFU, //unsigned float
	GODOT_IMAGE_FORMAT_PVRTC2, //pvrtc
	GODOT_IMAGE_FORMAT_PVRTC2A,
	GODOT_IMAGE_FORMAT_PVRTC4,
	GODOT_IMAGE_FORMAT_PVRTC4A,
	GODOT_IMAGE_FORMAT_ETC, //etc1
	GODOT_IMAGE_FORMAT_ETC2_R11, //etc2
	GODOT_IMAGE_FORMAT_ETC2_R11S, //signed, NOT srgb.
	GODOT_IMAGE_FORMAT_ETC2_RG11,
	GODOT_IMAGE_FORMAT_ETC2_RG11S,
	GODOT_IMAGE_FORMAT_ETC2_RGB8,
	GODOT_IMAGE_FORMAT_ETC2_RGBA8,
	GODOT_IMAGE_FORMAT_ETC2_RGB8A1,
	GODOT_IMAGE_FORMAT_MAX
};
typedef enum godot_image_format godot_image_format;

void GDAPI godot_image_new(godot_image *p_img);
// p_len can be -1
void GDAPI godot_image_new_with_png_jpg(godot_image *p_img, const uint8_t *p_mem_png_jpg, int p_len);
void GDAPI godot_image_new_with_xpm(godot_image *p_img, const char **p_xpm);

void GDAPI godot_image_new_with_size_format(godot_image *p_img, int p_width, int p_height, bool p_use_mipmaps, godot_image_format p_format);
void GDAPI godot_image_new_with_size_format_data(godot_image *p_img, int p_width, int p_height, bool p_use_mipmaps, godot_image_format p_format, godot_pool_byte_array *p_data);

godot_pool_byte_array GDAPI godot_image_get_data(godot_image *p_img);

godot_error GDAPI godot_image_load(godot_image *p_img, const godot_string *p_path);
godot_error GDAPI godot_image_save_png(godot_image *p_img, const godot_string *p_path);

int GDAPI godot_image_get_width(const godot_image *p_img);
int GDAPI godot_image_get_height(const godot_image *p_img);
godot_bool GDAPI godot_image_has_mipmaps(const godot_image *p_img);
int GDAPI godot_image_get_mipmap_count(const godot_image *p_img);

// @Incomplete
// I think it's too complex for the binding authors to implement the image class anew, so we should definitely
// export all methods here. That takes a while so it's on my @Todo list

void GDAPI godot_image_destroy(godot_image *p_img);

#ifdef __cplusplus
}
#endif

#endif // GODOT_DLSCRIPT_IMAGE_H
