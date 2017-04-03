#include "godot_image.h"

#include "image.h"

#ifdef __cplusplus
extern "C" {
#endif

void _image_api_anchor() {
}

#define memnew_placement_custom(m_placement, m_class, m_constr) _post_initialize(new (m_placement, sizeof(m_class), "") m_constr)

void GDAPI godot_image_new(godot_image *p_img) {
	Image *img = (Image *)p_img;
	memnew_placement_custom(img, Image, Image());
}

void GDAPI godot_image_new_with_png_jpg(godot_image *p_img, const uint8_t *p_mem_png_jpg, int p_len) {
	Image *img = (Image *)p_img;
	memnew_placement_custom(img, Image, Image(p_mem_png_jpg, p_len));
}

void GDAPI godot_image_new_with_xpm(godot_image *p_img, const char **p_xpm) {
	Image *img = (Image *)p_img;
	memnew_placement_custom(img, Image, Image(p_xpm));
}

void GDAPI godot_image_new_with_size_format(godot_image *p_img, int p_width, int p_height, bool p_use_mipmaps, godot_image_format p_format) {
	Image *img = (Image *)p_img;
	memnew_placement_custom(img, Image, Image(p_width, p_height, p_use_mipmaps, (Image::Format)p_format));
}

void GDAPI godot_image_new_with_size_format_data(godot_image *p_img, int p_width, int p_height, bool p_use_mipmaps, godot_image_format p_format, godot_pool_byte_array *p_data) {
	Image *img = (Image *)p_img;
	PoolVector<uint8_t> *data = (PoolVector<uint8_t> *)p_data;
	memnew_placement_custom(img, Image, Image(p_width, p_height, p_use_mipmaps, (Image::Format)p_format, *data));
}

godot_pool_byte_array GDAPI godot_image_get_data(godot_image *p_img) {
	Image *img = (Image *)p_img;
	PoolVector<uint8_t> cpp_data = img->get_data();
	godot_pool_byte_array *data = (godot_pool_byte_array *)&cpp_data;
	return *data;
}

godot_error GDAPI godot_image_load(godot_image *p_img, const godot_string *p_path) {
	Image *img = (Image *)p_img;
	String *path = (String *)p_path;
	return (godot_error)img->load(*path);
}

godot_error GDAPI godot_image_save_png(godot_image *p_img, const godot_string *p_path) {
	Image *img = (Image *)p_img;
	String *path = (String *)p_path;
	return (godot_error)img->save_png(*path);
}

int GDAPI godot_image_get_width(const godot_image *p_img) {
	Image *img = (Image *)p_img;
	return img->get_width();
}

int GDAPI godot_image_get_height(const godot_image *p_img) {
	Image *img = (Image *)p_img;
	return img->get_height();
}

godot_bool GDAPI godot_image_has_mipmaps(const godot_image *p_img) {
	Image *img = (Image *)p_img;
	return img->has_mipmaps();
}

int GDAPI godot_image_get_mipmap_count(const godot_image *p_img) {
	Image *img = (Image *)p_img;
	return img->get_mipmap_count();
}

void GDAPI godot_image_destroy(godot_image *p_img) {
	((Image *)p_img)->~Image();
}

#ifdef __cplusplus
}
#endif
