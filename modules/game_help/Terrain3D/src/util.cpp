// Copyright © 2023 Cory Petkovsek, Roope Palmroos, and Contributors.

#include "util.h"
#include "logger.h"

///////////////////////////
// Public Functions
///////////////////////////

void Util::print_dict(String p_name, const Dictionary &p_dict, int level) {
	LOG(level, "Printing Dictionary: ", p_name);
	Array keys = p_dict.keys();
	for (int i = 0; i < keys.size(); i++) {
		LOG(level, "Key: ", keys[i], ", Value: ", p_dict[keys[i]]);
	}
}

void Util::dump_gen(GeneratedTex p_gen, String p_name) {
	LOG(DEBUG, "Generated ", p_name, " RID: ", p_gen.get_rid(), ", dirty: ", p_gen.is_dirty(), ", image: ", p_gen.get_image());
}

void Util::dump_maps(const TypedArray<Ref<Image>> p_maps, String p_name) {
	LOG(DEBUG, "Dumping ", p_name, " map array. Size: ", p_maps.size());
	for (int i = 0; i < p_maps.size(); i++) {
		Ref<Image> img = p_maps[i];
		LOG(DEBUG, "[", i, "]: Map size: ", img->get_size(), " format: ", img->get_format(), " ", img);
	}
}

/**
 * Returns the minimum and maximum values for a heightmap (red channel only)
 */
Vector2 Util::get_min_max(const Ref<Image> p_image) {
	if (p_image.is_null()) {
		LOG(ERROR, "Provided image is not valid. Nothing to analyze");
		return Vector2(INFINITY, INFINITY);
	} else if (p_image->is_empty()) {
		LOG(ERROR, "Provided image is empty. Nothing to analyze");
		return Vector2(INFINITY, INFINITY);
	}

	Vector2 min_max = Vector2(0, 0);

	for (int y = 0; y < p_image->get_height(); y++) {
		for (int x = 0; x < p_image->get_width(); x++) {
			Color col = p_image->get_pixel(x, y);
			if (col.r < min_max.x) {
				min_max.x = col.r;
			}
			if (col.r > min_max.y) {
				min_max.y = col.r;
			}
		}
	}

	LOG(INFO, "Calculating minimum and maximum values of the image: ", min_max);
	return min_max;
}

/**
 * Returns a Image of a float heightmap normalized to RGB8 greyscale and scaled
 * Minimum of 8x8
 */
Ref<Image> Util::get_thumbnail(const Ref<Image> p_image, Vector2i p_size) {
	if (p_image.is_null()) {
		LOG(ERROR, "Provided image is not valid. Nothing to process.");
		return Ref<Image>();
	} else if (p_image->is_empty()) {
		LOG(ERROR, "Provided image is empty. Nothing to process.");
		return Ref<Image>();
	}
	p_size.x = CLAMP(p_size.x, 8, 16384);
	p_size.y = CLAMP(p_size.y, 8, 16384);

	LOG(INFO, "Drawing a thumbnail sized: ", p_size);
	// Create a temporary work image scaled to desired width
	Ref<Image> img;
	img.instantiate();
	img->copy_from(p_image);
	img->resize(p_size.x, p_size.y, Image::INTERPOLATE_LANCZOS);

	// Get minimum and maximum height values on the scaled image
	Vector2 minmax = get_min_max(img);
	real_t hmin = minmax.x;
	real_t hmax = minmax.y;
	// Define maximum range
	hmin = abs(hmin);
	hmax = abs(hmax) + hmin;
	// Avoid divide by zero
	hmax = (hmax == 0) ? 0.001 : hmax;

	// Create a new image w / normalized values
	Ref<Image> thumb = memnew(Image(p_size.x, p_size.y, false, Image::FORMAT_RGB8));
	for (int y = 0; y < thumb->get_height(); y++) {
		for (int x = 0; x < thumb->get_width(); x++) {
			Color col = img->get_pixel(x, y);
			col.r = (col.r + hmin) / hmax;
			col.g = col.r;
			col.b = col.r;
			thumb->set_pixel(x, y, col);
		}
	}
	return thumb;
}

/* Get image filled with your desired color and format
 * If alpha < 0, fill with checkered pattern multiplied by rgb
 */
Ref<Image> Util::get_filled_image(Vector2i p_size, Color p_color, bool p_create_mipmaps, Image::Format p_format) {
	Ref<Image> img = memnew(Image(p_size.x, p_size.y, p_create_mipmaps, p_format));
	if (p_color.a < 0.0f) {
		p_color.a = 1.0f;
		Color col_a = Color(0.8f, 0.8f, 0.8f, 1.0) * p_color;
		Color col_b = Color(0.5f, 0.5f, 0.5f, 1.0) * p_color;
		img->fill_rect(Rect2i(Vector2i(0, 0), p_size / 2), col_a);
		img->fill_rect(Rect2i(p_size / 2, p_size / 2), col_a);
		img->fill_rect(Rect2i(Vector2(p_size.x, 0) / 2, p_size / 2), col_b);
		img->fill_rect(Rect2i(Vector2(0, p_size.y) / 2, p_size / 2), col_b);
	} else {
		img->fill(p_color);
	}
	if (p_create_mipmaps) {
		img->generate_mipmaps();
	}
	return img;
}
