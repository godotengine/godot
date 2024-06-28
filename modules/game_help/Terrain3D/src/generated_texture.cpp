// Copyright © 2023 Cory Petkovsek, Roope Palmroos, and Contributors.

#include "servers/rendering_server.h"

#include "generated_texture.h"
#include "logger.h"

///////////////////////////
// Public Functions
///////////////////////////

static Vector<Ref<Image>> _get_imgvec(const TypedArray<Image> &p_layers) {
	Vector<Ref<Image>> images;
	images.resize(p_layers.size());
	for (int i = 0; i < p_layers.size(); i++) {
		images.write[i] = p_layers[i];
	}
	return images;
}
RID GeneratedTexture::create(const TypedArray<Image> &p_layers) {
	if (!p_layers.is_empty()) {
		if (Terrain3D::debug_level >= DEBUG) {
			LOG(DEBUG_CONT, "RenderingServer creating Texture2DArray, layers size: ", p_layers.size());
			for (int i = 0; i < p_layers.size(); i++) {
				Ref<Image> img = p_layers[i];
				LOG(DEBUG_CONT, i, ": ", img, ", empty: ", img->is_empty(), ", size: ", img->get_size(), ", format: ", img->get_format());
			}
		}
		_rid = RS::get_singleton()->texture_2d_layered_create(_get_imgvec(p_layers), RenderingServer::TEXTURE_LAYERED_2D_ARRAY);
		_dirty = false;
	} else {
		clear();
	}
	return _rid;
}

RID GeneratedTexture::create(const Ref<Image> &p_image) {
	LOG(DEBUG_CONT, "RenderingServer creating Texture2D");
	_image = p_image;
	_rid = RS::get_singleton()->texture_2d_create(_image);
	_dirty = false;
	return _rid;
}

void GeneratedTexture::clear() {
	if (_rid.is_valid()) {
		LOG(DEBUG_CONT, "GeneratedTexture freeing ", _rid);
		RS::get_singleton()->free(_rid);
	}
	if (_image.is_valid()) {
		LOG(DEBUG_CONT, "GeneratedTexture unref image", _image);
		_image.unref();
	}
	_rid = RID();
	_dirty = true;
}