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

void ResourceSaverJPG::_bind_methods() {
	ClassDB::bind_method(D_METHOD("save_image", "path", "image"), &ResourceSaverJPG::save_image);
	ClassDB::bind_method(D_METHOD("save_image_to_buffer", "image"), &ResourceSaverJPG::save_jpg_to_buffer);
	ClassDB::bind_method(D_METHOD("set_encode_options", "options"), &ResourceSaverJPG::set_encode_options);
	BIND_ENUM_CONSTANT(SUBSAPLING_Y_ONLY);
	BIND_ENUM_CONSTANT(SUBSAPLING_H1V1);
	BIND_ENUM_CONSTANT(SUBSAPLING_H2V1);
	BIND_ENUM_CONSTANT(SUBSAPLING_H2V2);
	BIND_ENUM_CONSTANT(SUBSAPLING_MAX);
}

#define SET_VAL(p_key, p_type)                                                                                                                                    \
	if (p_config.has(#p_key)) {                                                                                                                                   \
		ERR_FAIL_COND_V_MSG(p_config[#p_key].get_type() != p_type, ERR_INVALID_PARAMETER, vformat("Invalid property type for: %s, expected %d", #p_key, p_type)); \
		p_params.m_##p_key = p_config[#p_key];                                                                                                                    \
	}
Error ResourceSaverJPG::_configure_jpge_parameters(const Dictionary &p_config, jpge::params &p_params) {
	SET_VAL(quality, Variant::INT);
	SET_VAL(no_chroma_discrim_flag, Variant::BOOL);
	SET_VAL(two_pass_flag, Variant::BOOL);
	SET_VAL(use_std_tables, Variant::BOOL);
	if (p_config.has("subsampling") && p_config["subsampling"].get_type() == Variant::INT) {
		SubSamplingFactor factor = (SubSamplingFactor)(p_config["subsampling"].operator int());
		switch (factor) {
			case SUBSAPLING_Y_ONLY:
				p_params.m_subsampling = jpge::Y_ONLY;
				break;
			case SUBSAPLING_H1V1:
				p_params.m_subsampling = jpge::H1V1;
				break;
			case SUBSAPLING_H2V1:
				p_params.m_subsampling = jpge::H2V1;
				break;
			case SUBSAPLING_H2V2:
				p_params.m_subsampling = jpge::H2V2;
				break;
			default:
				ERR_FAIL_V_MSG(ERR_INVALID_PARAMETER, "Invalid subsampling factor");
		}
	}
	return OK;
}
#undef SET_VAL

void ResourceSaverJPG::set_encode_options(const Dictionary &p_options) {
	// Validate options.
	struct jpge::params params;
	Error err = _configure_jpge_parameters(p_options, params);
	ERR_FAIL_COND(err != OK);

	jpge_options = p_options;
}

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

	struct jpge::params params;
	Error err = _configure_jpge_parameters(jpge_options, params);
	ERR_FAIL_COND_V(err != OK, out);

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
