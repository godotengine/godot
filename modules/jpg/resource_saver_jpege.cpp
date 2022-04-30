/*************************************************************************/
/*  resource_saver_jpege.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "resource_saver_jpege.h"

#include "core/io/file_access.h"
#include "scene/resources/texture.h"

Error ResourceSaverJPG::save(const String &p_path, const RES &p_resource, uint32_t p_flags) {
	Ref<ImageTexture> texture = p_resource;

	ERR_FAIL_COND_V_MSG(!texture.is_valid(), ERR_INVALID_PARAMETER, "Can't save invalid texture as JPEG.");
	ERR_FAIL_COND_V_MSG(!texture->get_width(), ERR_INVALID_PARAMETER, "Can't save empty texture as JPEG.");
	return save_image(p_path, texture->get_image());
}

Error ResourceSaverJPG::save_image(const String &p_path, const Ref<Image> &p_image) {
	ERR_FAIL_COND_V_MSG(p_image.is_null(), ERR_INVALID_PARAMETER, "Image is null.");
	const Vector<uint8_t> compressed = save_jpg_to_buffer(p_image);
	ERR_FAIL_COND_V_MSG(compressed.size() == 0, FAILED, "Can't convert image to JPEG");
	Error err;
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE, &err);
	ERR_FAIL_COND_V_MSG(err, err, vformat("Can't save JPEG at path: '%s'.", p_path));

	file->store_buffer(compressed.ptr(), compressed.size());
	if (file->get_error() != OK && file->get_error() != ERR_FILE_EOF) {
		return ERR_CANT_CREATE;
	}
	return OK;
}

Vector<uint8_t> ResourceSaverJPG::save_jpg_to_buffer(Ref<Image> p_image) {
	Vector<uint8_t> out;
	Ref<Image> source_image = p_image->duplicate();
	if (source_image->is_compressed()) {
		source_image->decompress();
	}

	ERR_FAIL_COND_V(source_image->is_compressed(), out);

	source_image->convert(Image::FORMAT_RGB8);
	const int width = source_image->get_width();
	const int height = source_image->get_height();
	const Vector<uint8_t> image_data = source_image->get_data();
	int size = image_data.size();
	out.resize(size);

	struct jpge::params params;
	bool ret = jpge::compress_image_to_jpeg_file_in_memory(out.ptrw(), size, width, height, 3, image_data.ptr(), params);
	ERR_FAIL_COND_V(!ret, out);
	return out;
}

bool ResourceSaverJPG::recognize(const RES &p_resource) const {
	return (p_resource.is_valid() && p_resource->is_class("ImageTexture"));
}

void ResourceSaverJPG::get_recognized_extensions(const RES &p_resource, List<String> *p_extensions) const {
	if (Object::cast_to<ImageTexture>(*p_resource)) {
		p_extensions->push_back("jpg");
		p_extensions->push_back("jpeg");
	}
}
