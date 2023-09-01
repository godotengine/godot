/**************************************************************************/
/*  gltf_document_extension_texture_ktx.cpp                               */
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

#include "gltf_document_extension_texture_ktx.h"

// Import process.
Error GLTFDocumentExtensionTextureKTX::import_preflight(Ref<GLTFState> p_state, Vector<String> p_extensions) {
	if (!p_extensions.has("KHR_texture_basisu")) {
		return ERR_SKIP;
	}
	return OK;
}

Vector<String> GLTFDocumentExtensionTextureKTX::get_supported_extensions() {
	Vector<String> ret;
	ret.push_back("KHR_texture_basisu");
	return ret;
}

Error GLTFDocumentExtensionTextureKTX::parse_image_data(Ref<GLTFState> p_state, const PackedByteArray &p_image_data, const String &p_mime_type, Ref<Image> r_image) {
	if (p_mime_type == "image/ktx2") {
		return r_image->load_ktx_from_buffer(p_image_data);
	}
	return OK;
}

Error GLTFDocumentExtensionTextureKTX::parse_texture_json(Ref<GLTFState> p_state, const Dictionary &p_texture_json, Ref<GLTFTexture> r_gltf_texture) {
	if (!p_texture_json.has("extensions")) {
		return OK;
	}
	const Dictionary &extensions = p_texture_json["extensions"];
	if (!extensions.has("KHR_texture_basisu")) {
		return OK;
	}
	const Dictionary &texture_ktx = extensions["KHR_texture_basisu"];
	ERR_FAIL_COND_V(!texture_ktx.has("source"), ERR_PARSE_ERROR);
	r_gltf_texture->set_src_image(texture_ktx["source"]);
	return OK;
}
