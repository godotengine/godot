/*************************************************************************/
/*  test_image.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TEST_IMAGE_H
#define TEST_IMAGE_H

#include "core/io/file_access_pack.h"
#include "core/io/image.h"
#include "test_utils.h"

#include "thirdparty/doctest/doctest.h"

namespace TestImage {

TEST_CASE("[Image] Instantiation") {
	Ref<Image> image = memnew(Image(8, 4, false, Image::FORMAT_RGBA8));
	CHECK_MESSAGE(
			!image->is_empty(),
			"An image created with specified size and format should not be empty at first.");
	CHECK_MESSAGE(
			image->is_invisible(),
			"A newly created image should be invisible.");
	CHECK_MESSAGE(
			!image->is_compressed(),
			"A newly created image should not be compressed.");
	CHECK(!image->has_mipmaps());

	Ref<Image> image_copy = memnew(Image());
	CHECK_MESSAGE(
			image_copy->is_empty(),
			"An image created without any specified size and format be empty at first.");
	image_copy->copy_internals_from(image);

	CHECK_MESSAGE(
			image->get_data() == image_copy->get_data(),
			"Duplicated images should have the same data.");

	PackedByteArray image_data = image->get_data();
	Ref<Image> image_from_data = memnew(Image(8, 4, false, Image::FORMAT_RGBA8, image_data));
	CHECK_MESSAGE(
			image->get_data() == image_from_data->get_data(),
			"An image created from data of another image should have the same data of the original image.");
}

TEST_CASE("[Image] Saving and loading") {
	Ref<Image> image = memnew(Image(4, 4, false, Image::FORMAT_RGBA8));
	const String save_path_png = OS::get_singleton()->get_cache_path().plus_file("image.png");
	const String save_path_exr = OS::get_singleton()->get_cache_path().plus_file("image.exr");

	// Save PNG
	Error err;
	err = image->save_png(save_path_png);
	CHECK_MESSAGE(
			err == OK,
			"The image should be saved successfully as a .png file.");

	// Save EXR
	err = image->save_exr(save_path_exr, false);
	CHECK_MESSAGE(
			err == OK,
			"The image should be saved successfully as an .exr file.");

	// Load using load()
	Ref<Image> image_load = memnew(Image());
	err = image_load->load(save_path_png);
	CHECK_MESSAGE(
			err == OK,
			"The image should load successfully using load().");
	CHECK_MESSAGE(
			image->get_data() == image_load->get_data(),
			"The loaded image should have the same data as the one that got saved.");

	// Load BMP
	Ref<Image> image_bmp = memnew(Image());
	FileAccessRef f_bmp = FileAccess::open(TestUtils::get_data_path("images/icon.bmp"), FileAccess::READ, &err);
	PackedByteArray data_bmp;
	data_bmp.resize(f_bmp->get_length() + 1);
	f_bmp->get_buffer(data_bmp.ptrw(), f_bmp->get_length());
	CHECK_MESSAGE(
			image_bmp->load_bmp_from_buffer(data_bmp) == OK,
			"The BMP image should load successfully.");

	// Load JPG
	Ref<Image> image_jpg = memnew(Image());
	FileAccessRef f_jpg = FileAccess::open(TestUtils::get_data_path("images/icon.jpg"), FileAccess::READ, &err);
	PackedByteArray data_jpg;
	data_jpg.resize(f_jpg->get_length() + 1);
	f_jpg->get_buffer(data_jpg.ptrw(), f_jpg->get_length());
	CHECK_MESSAGE(
			image_jpg->load_jpg_from_buffer(data_jpg) == OK,
			"The JPG image should load successfully.");

	// Load WEBP
	Ref<Image> image_webp = memnew(Image());
	FileAccessRef f_webp = FileAccess::open(TestUtils::get_data_path("images/icon.webp"), FileAccess::READ, &err);
	PackedByteArray data_webp;
	data_webp.resize(f_webp->get_length() + 1);
	f_webp->get_buffer(data_webp.ptrw(), f_webp->get_length());
	CHECK_MESSAGE(
			image_webp->load_webp_from_buffer(data_webp) == OK,
			"The WEBP image should load successfully.");

	// Load PNG
	Ref<Image> image_png = memnew(Image());
	FileAccessRef f_png = FileAccess::open(TestUtils::get_data_path("images/icon.png"), FileAccess::READ, &err);
	PackedByteArray data_png;
	data_png.resize(f_png->get_length() + 1);
	f_png->get_buffer(data_png.ptrw(), f_png->get_length());
	CHECK_MESSAGE(
			image_png->load_png_from_buffer(data_png) == OK,
			"The PNG image should load successfully.");

	// Load TGA
	Ref<Image> image_tga = memnew(Image());
	FileAccessRef f_tga = FileAccess::open(TestUtils::get_data_path("images/icon.tga"), FileAccess::READ, &err);
	PackedByteArray data_tga;
	data_tga.resize(f_tga->get_length() + 1);
	f_tga->get_buffer(data_tga.ptrw(), f_tga->get_length());
	CHECK_MESSAGE(
			image_tga->load_tga_from_buffer(data_tga) == OK,
			"The TGA image should load successfully.");
}

TEST_CASE("[Image] Basic getters") {
	Ref<Image> image = memnew(Image(8, 4, false, Image::FORMAT_LA8));
	CHECK(image->get_width() == 8);
	CHECK(image->get_height() == 4);
	CHECK(image->get_size() == Vector2(8, 4));
	CHECK(image->get_format() == Image::FORMAT_LA8);
	CHECK(image->get_used_rect() == Rect2(0, 0, 0, 0));
	Ref<Image> image_get_rect = image->get_rect(Rect2(0, 0, 2, 1));
	CHECK(image_get_rect->get_size() == Vector2(2, 1));
}

TEST_CASE("[Image] Resizing") {
	Ref<Image> image = memnew(Image(8, 8, false, Image::FORMAT_RGBA8));
	// Crop
	image->crop(4, 4);
	CHECK_MESSAGE(
			image->get_size() == Vector2(4, 4),
			"get_size() should return the correct size after cropping.");
	image->set_pixel(0, 0, Color(1, 1, 1, 1));

	// Resize
	for (int i = 0; i < 5; i++) {
		Ref<Image> image_resized = memnew(Image());
		image_resized->copy_internals_from(image);
		Image::Interpolation interpolation = static_cast<Image::Interpolation>(i);
		image_resized->resize(8, 8, interpolation);
		CHECK_MESSAGE(
				image_resized->get_size() == Vector2(8, 8),
				"get_size() should return the correct size after resizing.");
		CHECK_MESSAGE(
				image_resized->get_pixel(1, 1).a > 0,
				"Resizing an image should also affect its content.");
	}

	// shrink_x2()
	image->shrink_x2();
	CHECK_MESSAGE(
			image->get_size() == Vector2(2, 2),
			"get_size() should return the correct size after shrink_x2().");

	// resize_to_po2()
	Ref<Image> image_po_2 = memnew(Image(14, 28, false, Image::FORMAT_RGBA8));
	image_po_2->resize_to_po2();
	CHECK_MESSAGE(
			image_po_2->get_size() == Vector2(16, 32),
			"get_size() should return the correct size after resize_to_po2().");
}

TEST_CASE("[Image] Modifying pixels of an image") {
	Ref<Image> image = memnew(Image(3, 3, false, Image::FORMAT_RGBA8));
	image->set_pixel(0, 0, Color(1, 1, 1, 1));
	CHECK_MESSAGE(
			!image->is_invisible(),
			"Image should not be invisible after drawing on it.");
	CHECK_MESSAGE(
			image->get_pixelv(Vector2(0, 0)).is_equal_approx(Color(1, 1, 1, 1)),
			"Image's get_pixel() should return the same color value as the one being set with set_pixel() in the same position.");
	CHECK_MESSAGE(
			image->get_used_rect() == Rect2(0, 0, 1, 1),
			"Image's get_used_rect should return the expected value, larger than Rect2(0, 0, 0, 0) if it's visible.");

	image->set_pixelv(Vector2(0, 0), Color(0.5, 0.5, 0.5, 0.5));
	Ref<Image> image2 = memnew(Image(3, 3, false, Image::FORMAT_RGBA8));

	// Fill image with color
	image2->fill(Color(0.5, 0.5, 0.5, 0.5));
	for (int x = 0; x < image2->get_width(); x++) {
		for (int y = 0; y < image2->get_height(); y++) {
			CHECK_MESSAGE(
					image2->get_pixel(x, y).r > 0.49,
					"fill() should colorize all pixels of the image.");
		}
	}

	// Blend two images together
	image->blend_rect(image2, Rect2(Vector2(0, 0), image2->get_size()), Vector2(0, 0));
	CHECK_MESSAGE(
			image->get_pixel(0, 0).a > 0.7,
			"blend_rect() should blend the alpha values of the two images.");
	CHECK_MESSAGE(
			image->get_used_rect().size == image->get_size(),
			"get_used_rect() should return the expected value, its Rect size should be the same as get_size() if there are no transparent pixels.");

	Ref<Image> image3 = memnew(Image(2, 2, false, Image::FORMAT_RGBA8));
	image3->set_pixel(0, 0, Color(0, 1, 0, 1));

	//blit_rect() two images together
	image->blit_rect(image3, Rect2(Vector2(0, 0), image3->get_size()), Vector2(0, 0));
	CHECK_MESSAGE(
			image->get_pixel(0, 0).is_equal_approx(Color(0, 1, 0, 1)),
			"blit_rect() should replace old colors and not blend them.");
	CHECK_MESSAGE(
			!image->get_pixel(2, 2).is_equal_approx(Color(0, 1, 0, 1)),
			"blit_rect() should not affect the area of the image that is outside src_rect.");

	// Flip image
	image3->flip_x();
	CHECK(image3->get_pixel(1, 0).is_equal_approx(Color(0, 1, 0, 1)));
	CHECK_MESSAGE(
			image3->get_pixel(0, 0).is_equal_approx(Color(0, 0, 0, 0)),
			"flip_x() should not leave old pixels behind.");
	image3->flip_y();
	CHECK(image3->get_pixel(1, 1).is_equal_approx(Color(0, 1, 0, 1)));
	CHECK_MESSAGE(
			image3->get_pixel(1, 0).is_equal_approx(Color(0, 0, 0, 0)),
			"flip_y() should not leave old pixels behind.");
}
} // namespace TestImage
#endif // TEST_IMAGE_H
