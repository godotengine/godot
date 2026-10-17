/**************************************************************************/
/*  compressed_texture_resource_format.cpp                                */
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

#include "compressed_texture_resource_format.h"

#include "scene/resources/compressed_texture.h"

Ref<Resource> ResourceFormatLoaderCompressedTexture2D::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Ref<CompressedTexture2D> st;
	st.instantiate();
	Error err = st->load(p_path);
	if (r_error) {
		*r_error = err;
	}
	if (err != OK) {
		return Ref<Resource>();
	}

	return st;
}

void ResourceFormatLoaderCompressedTexture2D::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("ctex");
}

bool ResourceFormatLoaderCompressedTexture2D::handles_type(const String &p_type) const {
	return p_type == "CompressedTexture2D";
}

String ResourceFormatLoaderCompressedTexture2D::get_resource_type(const String &p_path) const {
	if (p_path.has_extension("ctex")) {
		return "CompressedTexture2D";
	}
	return "";
}

Ref<Resource> ResourceFormatLoaderCompressedTextureLayered::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Ref<CompressedTextureLayered> ct;
	if (p_path.has_extension("ctexarray")) {
		Ref<CompressedTexture2DArray> c;
		c.instantiate();
		ct = c;
	} else if (p_path.has_extension("ccube")) {
		Ref<CompressedCubemap> c;
		c.instantiate();
		ct = c;
	} else if (p_path.has_extension("ccubearray")) {
		Ref<CompressedCubemapArray> c;
		c.instantiate();
		ct = c;
	} else {
		if (r_error) {
			*r_error = ERR_FILE_UNRECOGNIZED;
		}
		return Ref<Resource>();
	}
	Error err = ct->load(p_path);
	if (r_error) {
		*r_error = err;
	}
	if (err != OK) {
		return Ref<Resource>();
	}

	return ct;
}

void ResourceFormatLoaderCompressedTextureLayered::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("ctexarray");
	p_extensions->push_back("ccube");
	p_extensions->push_back("ccubearray");
}

bool ResourceFormatLoaderCompressedTextureLayered::handles_type(const String &p_type) const {
	return p_type == "CompressedTexture2DArray" || p_type == "CompressedCubemap" || p_type == "CompressedCubemapArray";
}

String ResourceFormatLoaderCompressedTextureLayered::get_resource_type(const String &p_path) const {
	if (p_path.has_extension("ctexarray")) {
		return "CompressedTexture2DArray";
	}
	if (p_path.has_extension("ccube")) {
		return "CompressedCubemap";
	}
	if (p_path.has_extension("ccubearray")) {
		return "CompressedCubemapArray";
	}
	return "";
}

Ref<Resource> ResourceFormatLoaderCompressedTexture3D::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Ref<CompressedTexture3D> st;
	st.instantiate();
	Error err = st->load(p_path);
	if (r_error) {
		*r_error = err;
	}
	if (err != OK) {
		return Ref<Resource>();
	}

	return st;
}

void ResourceFormatLoaderCompressedTexture3D::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("ctex3d");
}

bool ResourceFormatLoaderCompressedTexture3D::handles_type(const String &p_type) const {
	return p_type == "CompressedTexture3D";
}

String ResourceFormatLoaderCompressedTexture3D::get_resource_type(const String &p_path) const {
	if (p_path.has_extension("ctex3d")) {
		return "CompressedTexture3D";
	}
	return "";
}
