// Copyright © 2023 Cory Petkovsek, Roope Palmroos, and Contributors.

//#include <godot_cpp/classes/rendering_server.hpp>

#include "servers/rendering_server.h"

#include "generated_tex.h"
#include "logger.h"

///////////////////////////
// Public Functions
///////////////////////////

void GeneratedTex::create(const TypedArray<Ref<Image>> &p_layers) {
	if (!p_layers.is_empty()) {
		if (Terrain3D::debug_level >= DEBUG) {
			LOG(DEBUG, "RenderingServer creating Texture2DArray, layers size: ", p_layers.size());
			for (int i = 0; i < p_layers.size(); i++) {
				Ref<Image> img = p_layers[i];
				LOG(DEBUG, i, ": ", img, ", empty: ", img->is_empty(), ", size: ", img->get_size(), ", format: ", img->get_format());
			}
		}
		_rid = RSS->texture_2d_layered_create(p_layers.get_vector<Ref<Image>>(), RenderingServer::TEXTURE_LAYERED_2D_ARRAY);
		_dirty = false;
	} else {
		clear();
	}
}

void GeneratedTex::create(const Ref<Image> &p_image) {
	LOG(DEBUG, "RenderingServer creating Texture2D");
	_image = p_image;
	_rid = RSS->texture_2d_create(_image);
	_dirty = false;
}

void GeneratedTex::clear() {
	if (_rid.is_valid()) {
		LOG(DEBUG, "GeneratedTex freeing ", _rid);
		RSS->free(_rid);
	}
	if (_image.is_valid()) {
		LOG(DEBUG, "GeneratedTex unref image", _image);
		_image.unref();
	}
	_rid = RID();
	_dirty = true;
}