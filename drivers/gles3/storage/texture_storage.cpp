/*************************************************************************/
/*  texture_storage.cpp                                                  */
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

#ifdef GLES3_ENABLED

#include "texture_storage.h"
#include "config.h"

using namespace GLES3;

TextureStorage *TextureStorage::singleton = nullptr;

TextureStorage *TextureStorage::get_singleton() {
	return singleton;
}

TextureStorage::TextureStorage() {
	singleton = this;
}

TextureStorage::~TextureStorage() {
	singleton = nullptr;
}

void TextureStorage::set_main_thread_id(Thread::ID p_id) {
	_main_thread_id = p_id;
}

bool TextureStorage::_is_main_thread() {
	//#if defined DEBUG_ENABLED && defined TOOLS_ENABLED
	// must be called from main thread in OpenGL
	bool is_main_thread = _main_thread_id == Thread::get_caller_id();
	//#endif
	return is_main_thread;
}

bool TextureStorage::can_create_resources_async() const {
	return false;
}

static const GLenum _cube_side_enum[6] = {
	GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
	GL_TEXTURE_CUBE_MAP_POSITIVE_X,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
};

Ref<Image> TextureStorage::_get_gl_image_and_format(const Ref<Image> &p_image, Image::Format p_format, uint32_t p_flags, Image::Format &r_real_format, GLenum &r_gl_format, GLenum &r_gl_internal_format, GLenum &r_gl_type, bool &r_compressed, bool p_force_decompress) const {
	Config *config = Config::get_singleton();
	r_gl_format = 0;
	Ref<Image> image = p_image;
	r_compressed = false;
	r_real_format = p_format;

	bool need_decompress = false;

	switch (p_format) {
		case Image::FORMAT_L8: {
#ifdef GLES_OVER_GL
			r_gl_internal_format = GL_R8;
			r_gl_format = GL_RED;
			r_gl_type = GL_UNSIGNED_BYTE;
#else
			r_gl_internal_format = GL_LUMINANCE;
			r_gl_format = GL_LUMINANCE;
			r_gl_type = GL_UNSIGNED_BYTE;
#endif
		} break;
		case Image::FORMAT_LA8: {
#ifdef GLES_OVER_GL
			r_gl_internal_format = GL_RG8;
			r_gl_format = GL_RG;
			r_gl_type = GL_UNSIGNED_BYTE;
#else
			r_gl_internal_format = GL_LUMINANCE_ALPHA;
			r_gl_format = GL_LUMINANCE_ALPHA;
			r_gl_type = GL_UNSIGNED_BYTE;
#endif
		} break;
		case Image::FORMAT_R8: {
			r_gl_internal_format = GL_R8;
			r_gl_format = GL_RED;
			r_gl_type = GL_UNSIGNED_BYTE;

		} break;
		case Image::FORMAT_RG8: {
			r_gl_internal_format = GL_RG8;
			r_gl_format = GL_RG;
			r_gl_type = GL_UNSIGNED_BYTE;

		} break;
		case Image::FORMAT_RGB8: {
			r_gl_internal_format = GL_RGB8;
			r_gl_format = GL_RGB;
			r_gl_type = GL_UNSIGNED_BYTE;
			//r_srgb = true;

		} break;
		case Image::FORMAT_RGBA8: {
			r_gl_format = GL_RGBA;
			r_gl_internal_format = GL_RGBA8;
			r_gl_type = GL_UNSIGNED_BYTE;
			//r_srgb = true;

		} break;
		case Image::FORMAT_RGBA4444: {
			r_gl_internal_format = GL_RGBA4;
			r_gl_format = GL_RGBA;
			r_gl_type = GL_UNSIGNED_SHORT_4_4_4_4;

		} break;
			//case Image::FORMAT_RGBA5551: {
			//	r_gl_internal_format = GL_RGB5_A1;
			//	r_gl_format = GL_RGBA;
			//	r_gl_type = GL_UNSIGNED_SHORT_5_5_5_1;
			//
			//} break;
		case Image::FORMAT_RF: {
			r_gl_internal_format = GL_R32F;
			r_gl_format = GL_RED;
			r_gl_type = GL_FLOAT;

		} break;
		case Image::FORMAT_RGF: {
			r_gl_internal_format = GL_RG32F;
			r_gl_format = GL_RG;
			r_gl_type = GL_FLOAT;

		} break;
		case Image::FORMAT_RGBF: {
			r_gl_internal_format = GL_RGB32F;
			r_gl_format = GL_RGB;
			r_gl_type = GL_FLOAT;

		} break;
		case Image::FORMAT_RGBAF: {
			r_gl_internal_format = GL_RGBA32F;
			r_gl_format = GL_RGBA;
			r_gl_type = GL_FLOAT;

		} break;
		case Image::FORMAT_RH: {
			r_gl_internal_format = GL_R16F;
			r_gl_format = GL_RED;
			r_gl_type = GL_HALF_FLOAT;
		} break;
		case Image::FORMAT_RGH: {
			r_gl_internal_format = GL_RG16F;
			r_gl_format = GL_RG;
			r_gl_type = GL_HALF_FLOAT;

		} break;
		case Image::FORMAT_RGBH: {
			r_gl_internal_format = GL_RGB16F;
			r_gl_format = GL_RGB;
			r_gl_type = GL_HALF_FLOAT;

		} break;
		case Image::FORMAT_RGBAH: {
			r_gl_internal_format = GL_RGBA16F;
			r_gl_format = GL_RGBA;
			r_gl_type = GL_HALF_FLOAT;

		} break;
		case Image::FORMAT_RGBE9995: {
			r_gl_internal_format = GL_RGB9_E5;
			r_gl_format = GL_RGB;
			r_gl_type = GL_UNSIGNED_INT_5_9_9_9_REV;

		} break;
		case Image::FORMAT_DXT1: {
			if (config->s3tc_supported) {
				r_gl_internal_format = _EXT_COMPRESSED_RGBA_S3TC_DXT1_EXT;
				r_gl_format = GL_RGBA;
				r_gl_type = GL_UNSIGNED_BYTE;
				r_compressed = true;
				//r_srgb = true;

			} else {
				need_decompress = true;
			}
		} break;
		case Image::FORMAT_DXT3: {
			if (config->s3tc_supported) {
				r_gl_internal_format = _EXT_COMPRESSED_RGBA_S3TC_DXT3_EXT;
				r_gl_format = GL_RGBA;
				r_gl_type = GL_UNSIGNED_BYTE;
				r_compressed = true;
				//r_srgb = true;

			} else {
				need_decompress = true;
			}
		} break;
		case Image::FORMAT_DXT5: {
			if (config->s3tc_supported) {
				r_gl_internal_format = _EXT_COMPRESSED_RGBA_S3TC_DXT5_EXT;
				r_gl_format = GL_RGBA;
				r_gl_type = GL_UNSIGNED_BYTE;
				r_compressed = true;
				//r_srgb = true;

			} else {
				need_decompress = true;
			}
		} break;
		case Image::FORMAT_RGTC_R: {
			if (config->rgtc_supported) {
				r_gl_internal_format = _EXT_COMPRESSED_RED_RGTC1_EXT;
				r_gl_format = GL_RGBA;
				r_gl_type = GL_UNSIGNED_BYTE;
				r_compressed = true;

			} else {
				need_decompress = true;
			}
		} break;
		case Image::FORMAT_RGTC_RG: {
			if (config->rgtc_supported) {
				r_gl_internal_format = _EXT_COMPRESSED_RED_GREEN_RGTC2_EXT;
				r_gl_format = GL_RGBA;
				r_gl_type = GL_UNSIGNED_BYTE;
				r_compressed = true;
			} else {
				need_decompress = true;
			}
		} break;
		case Image::FORMAT_BPTC_RGBA: {
			if (config->bptc_supported) {
				r_gl_internal_format = _EXT_COMPRESSED_RGBA_BPTC_UNORM;
				r_gl_format = GL_RGBA;
				r_gl_type = GL_UNSIGNED_BYTE;
				r_compressed = true;
				//r_srgb = true;

			} else {
				need_decompress = true;
			}
		} break;
		case Image::FORMAT_BPTC_RGBF: {
			if (config->bptc_supported) {
				r_gl_internal_format = _EXT_COMPRESSED_RGB_BPTC_SIGNED_FLOAT;
				r_gl_format = GL_RGB;
				r_gl_type = GL_FLOAT;
				r_compressed = true;
			} else {
				need_decompress = true;
			}
		} break;
		case Image::FORMAT_BPTC_RGBFU: {
			if (config->bptc_supported) {
				r_gl_internal_format = _EXT_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT;
				r_gl_format = GL_RGB;
				r_gl_type = GL_FLOAT;
				r_compressed = true;
			} else {
				need_decompress = true;
			}
		} break;
		case Image::FORMAT_ETC: {
			if (config->etc_supported) {
				r_gl_internal_format = _EXT_ETC1_RGB8_OES;
				r_gl_format = GL_RGBA;
				r_gl_type = GL_UNSIGNED_BYTE;
				r_compressed = true;

			} else {
				need_decompress = true;
			}

		} break;
		/*
		case Image::FORMAT_ETC2_R11: {
			if (config->etc2_supported) {
				r_gl_internal_format = _EXT_COMPRESSED_R11_EAC;
				r_gl_format = GL_RED;
				r_gl_type = GL_UNSIGNED_BYTE;
				r_compressed = true;

			} else {
				need_decompress = true;
			}
		} break;
		case Image::FORMAT_ETC2_R11S: {
			if (config->etc2_supported) {
				r_gl_internal_format = _EXT_COMPRESSED_SIGNED_R11_EAC;
				r_gl_format = GL_RED;
				r_gl_type = GL_UNSIGNED_BYTE;
				r_compressed = true;

			} else {
				need_decompress = true;
			}
		} break;
		case Image::FORMAT_ETC2_RG11: {
			if (config->etc2_supported) {
				r_gl_internal_format = _EXT_COMPRESSED_RG11_EAC;
				r_gl_format = GL_RG;
				r_gl_type = GL_UNSIGNED_BYTE;
				r_compressed = true;

			} else {
				need_decompress = true;
			}
		} break;
		case Image::FORMAT_ETC2_RG11S: {
			if (config->etc2_supported) {
				r_gl_internal_format = _EXT_COMPRESSED_SIGNED_RG11_EAC;
				r_gl_format = GL_RG;
				r_gl_type = GL_UNSIGNED_BYTE;
				r_compressed = true;

			} else {
				need_decompress = true;
			}
		} break;
		case Image::FORMAT_ETC2_RGB8: {
			if (config->etc2_supported) {
				r_gl_internal_format = _EXT_COMPRESSED_RGB8_ETC2;
				r_gl_format = GL_RGB;
				r_gl_type = GL_UNSIGNED_BYTE;
				r_compressed = true;
				//r_srgb = true;

			} else {
				need_decompress = true;
			}
		} break;
		case Image::FORMAT_ETC2_RGBA8: {
			if (config->etc2_supported) {
				r_gl_internal_format = _EXT_COMPRESSED_RGBA8_ETC2_EAC;
				r_gl_format = GL_RGBA;
				r_gl_type = GL_UNSIGNED_BYTE;
				r_compressed = true;
				//r_srgb = true;

			} else {
				need_decompress = true;
			}
		} break;
		case Image::FORMAT_ETC2_RGB8A1: {
			if (config->etc2_supported) {
				r_gl_internal_format = _EXT_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2;
				r_gl_format = GL_RGBA;
				r_gl_type = GL_UNSIGNED_BYTE;
				r_compressed = true;
				//r_srgb = true;

			} else {
				need_decompress = true;
			}
		} break;
		*/
		default: {
			ERR_FAIL_V(Ref<Image>());
		}
	}

	if (need_decompress || p_force_decompress) {
		if (!image.is_null()) {
			image = image->duplicate();
			image->decompress();
			ERR_FAIL_COND_V(image->is_compressed(), image);
			switch (image->get_format()) {
				case Image::FORMAT_RGB8: {
					r_gl_format = GL_RGB;
					r_gl_internal_format = GL_RGB;
					r_gl_type = GL_UNSIGNED_BYTE;
					r_real_format = Image::FORMAT_RGB8;
					r_compressed = false;
				} break;
				case Image::FORMAT_RGBA8: {
					r_gl_format = GL_RGBA;
					r_gl_internal_format = GL_RGBA;
					r_gl_type = GL_UNSIGNED_BYTE;
					r_real_format = Image::FORMAT_RGBA8;
					r_compressed = false;
				} break;
				default: {
					image->convert(Image::FORMAT_RGBA8);
					r_gl_format = GL_RGBA;
					r_gl_internal_format = GL_RGBA;
					r_gl_type = GL_UNSIGNED_BYTE;
					r_real_format = Image::FORMAT_RGBA8;
					r_compressed = false;

				} break;
			}
		}

		return image;
	}

	return p_image;
}

void TextureStorage::_texture_set_state_from_flags(Texture *p_tex) {
	// Config *config = Config::get_singleton();

	if ((p_tex->flags & TEXTURE_FLAG_MIPMAPS) && !p_tex->ignore_mipmaps) {
		if (p_tex->flags & TEXTURE_FLAG_FILTER) {
			// these do not exactly correspond ...
			p_tex->GLSetFilter(p_tex->target, RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS);
			//texture->glTexParam_MinFilter(texture->target, config->use_fast_texture_filter ? GL_LINEAR_MIPMAP_NEAREST : GL_LINEAR_MIPMAP_LINEAR);
		} else {
			p_tex->GLSetFilter(p_tex->target, RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS);
			//texture->glTexParam_MinFilter(texture->target, config->use_fast_texture_filter ? GL_NEAREST_MIPMAP_NEAREST : GL_NEAREST_MIPMAP_LINEAR);
		}
	} else {
		if (p_tex->flags & TEXTURE_FLAG_FILTER) {
			p_tex->GLSetFilter(p_tex->target, RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR);
			//texture->glTexParam_MinFilter(texture->target, GL_LINEAR);
		} else {
			p_tex->GLSetFilter(p_tex->target, RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST);
			//texture->glTexParam_MinFilter(texture->target, GL_NEAREST);
		}
	}

	if (((p_tex->flags & TEXTURE_FLAG_REPEAT) || (p_tex->flags & TEXTURE_FLAG_MIRRORED_REPEAT)) && p_tex->target != GL_TEXTURE_CUBE_MAP) {
		if (p_tex->flags & TEXTURE_FLAG_MIRRORED_REPEAT) {
			p_tex->GLSetRepeat(p_tex->target, RS::CANVAS_ITEM_TEXTURE_REPEAT_MIRROR);
		} else {
			p_tex->GLSetRepeat(p_tex->target, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
		}
	} else {
		p_tex->GLSetRepeat(p_tex->target, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	}
}

void TextureStorage::_texture_allocate_internal(RID p_texture, int p_width, int p_height, int p_depth_3d, Image::Format p_format, RenderingDevice::TextureType p_type, uint32_t p_flags) {
	//	GLenum format;
	//	GLenum internal_format;
	//	GLenum type;

	//	bool compressed = false;

	// Config *config = Config::get_singleton();

	if (p_flags & TEXTURE_FLAG_USED_FOR_STREAMING) {
		p_flags &= ~TEXTURE_FLAG_MIPMAPS; // no mipies for video
	}

	Texture *texture = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!texture);
	texture->width = p_width;
	texture->height = p_height;
	texture->format = p_format;
	texture->flags = p_flags;
	texture->stored_cube_sides = 0;
	texture->type = p_type;

	switch (p_type) {
		case RenderingDevice::TEXTURE_TYPE_2D: {
			texture->target = GL_TEXTURE_2D;
			texture->images.resize(1);
		} break;
			//		case RenderingDevice::TEXTURE_TYPE_EXTERNAL: {
			//#ifdef ANDROID_ENABLED
			//			texture->target = _GL_TEXTURE_EXTERNAL_OES;
			//#else
			//			texture->target = GL_TEXTURE_2D;
			//#endif
			//			texture->images.resize(0);
			//		} break;
		case RenderingDevice::TEXTURE_TYPE_CUBE: {
			texture->target = GL_TEXTURE_CUBE_MAP;
			texture->images.resize(6);
		} break;
		case RenderingDevice::TEXTURE_TYPE_2D_ARRAY:
		case RenderingDevice::TEXTURE_TYPE_3D: {
			texture->target = GL_TEXTURE_3D;
			ERR_PRINT("3D textures and Texture Arrays are not supported in OpenGL. Please switch to the Vulkan backend.");
			return;
		} break;
		default: {
			ERR_PRINT("Unknown texture type!");
			return;
		}
	}

#if 0
	//		if (p_type != RS::TEXTURE_TYPE_EXTERNAL) {
	if (p_type == RenderingDevice::TEXTURE_TYPE_2D) {
		texture->alloc_width = texture->width;
		texture->alloc_height = texture->height;
		texture->resize_to_po2 = false;
		if (!config->support_npot_repeat_mipmap) {
			int po2_width = next_power_of_2(p_width);
			int po2_height = next_power_of_2(p_height);

			bool is_po2 = p_width == po2_width && p_height == po2_height;

			if (!is_po2 && (p_flags & TEXTURE_FLAG_REPEAT || p_flags & TEXTURE_FLAG_MIPMAPS)) {
				if (p_flags & TEXTURE_FLAG_USED_FOR_STREAMING) {
					//not supported
					ERR_PRINT("Streaming texture for non power of 2 or has mipmaps on this hardware: " + texture->path + "'. Mipmaps and repeat disabled.");
					texture->flags &= ~(TEXTURE_FLAG_REPEAT | TEXTURE_FLAG_MIPMAPS);
				} else {
					texture->alloc_height = po2_height;
					texture->alloc_width = po2_width;
					texture->resize_to_po2 = true;
				}
			}
		}

		GLenum format;
		GLenum internal_format;
		GLenum type;
		bool compressed = false;

		Image::Format real_format;
		_get_gl_image_and_format(Ref<Image>(),
				texture->format,
				texture->flags,
				real_format,
				format,
				internal_format,
				type,
				compressed,
				texture->resize_to_po2);

		texture->gl_format_cache = format;
		texture->gl_type_cache = type;
		texture->gl_internal_format_cache = internal_format;
		texture->data_size = 0;
		texture->mipmaps = 1;

		texture->compressed = compressed;
	}
#endif

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(texture->target, texture->tex_id);

	//	if (p_type == RS::TEXTURE_TYPE_EXTERNAL) {
	//		glTexParameteri(texture->target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//		glTexParameteri(texture->target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//		glTexParameteri(texture->target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	//		glTexParameteri(texture->target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	//	} else if (p_flags & TEXTURE_FLAG_USED_FOR_STREAMING) {
	//		//prealloc if video
	//		glTexImage2D(texture->target, 0, internal_format, texture->alloc_width, texture->alloc_height, 0, format, type, NULL);
	//	}

	texture->active = true;
}

RID TextureStorage::texture_create() {
	ERR_FAIL_COND_V(!_is_main_thread(), RID());

	Texture *texture = memnew(Texture);
	ERR_FAIL_COND_V(!texture, RID());
	glGenTextures(1, &texture->tex_id);
	texture->active = false;
	texture->total_data_size = 0;

	return texture_owner.make_rid(texture);
}

RID TextureStorage::texture_allocate() {
	RID id = texture_create();
	ERR_FAIL_COND_V(id == RID(), id);
	return id;
}

void TextureStorage::texture_free(RID p_rid) {
	Texture *t = texture_owner.get_or_null(p_rid);

	// can't free a render target texture
	ERR_FAIL_COND(t->render_target);
	if (t->canvas_texture) {
		memdelete(t->canvas_texture);
	}

	// info.texture_mem -= t->total_data_size; // TODO make this work again!!
	texture_owner.free(p_rid);
	memdelete(t);
}

void TextureStorage::texture_2d_initialize(RID p_texture, const Ref<Image> &p_image) {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex);

	int w = p_image->get_width();
	int h = p_image->get_height();

	_texture_allocate_internal(p_texture, w, h, 1, p_image->get_format(), RenderingDevice::TEXTURE_TYPE_2D, 0);
	texture_set_data(p_texture, p_image);
}

void TextureStorage::texture_2d_layered_initialize(RID p_texture, const Vector<Ref<Image>> &p_layers, RS::TextureLayeredType p_layered_type) {
}

void TextureStorage::texture_3d_initialize(RID p_texture, Image::Format, int p_width, int p_height, int p_depth, bool p_mipmaps, const Vector<Ref<Image>> &p_data) {
}

void TextureStorage::texture_proxy_initialize(RID p_texture, RID p_base) {
	texture_set_proxy(p_texture, p_base);
}

//RID TextureStorage::texture_2d_create(const Ref<Image> &p_image) {
//	RID id = texture_create();
//	ERR_FAIL_COND_V(id == RID(), id);

//	int w = p_image->get_width();
//	int h = p_image->get_height();

//	texture_allocate(id, w, h, 1, p_image->get_format(), RenderingDevice::TEXTURE_TYPE_2D, 0);

//	texture_set_data(id, p_image);

//	return id;
//}

//RID TextureStorage::texture_2d_layered_create(const Vector<Ref<Image>> &p_layers, RS::TextureLayeredType p_layered_type) {
//	return RID();
//}

//void TextureStorage::texture_2d_update_immediate(RID p_texture, const Ref<Image> &p_image, int p_layer) {
//	// only 1 layer so far
//	texture_set_data(p_texture, p_image);
//}

void TextureStorage::texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer) {
	// only 1 layer so far
	texture_set_data(p_texture, p_image);
}

void TextureStorage::texture_2d_placeholder_initialize(RID p_texture) {
}

void TextureStorage::texture_2d_layered_placeholder_initialize(RID p_texture, RenderingServer::TextureLayeredType p_layered_type) {
}

void TextureStorage::texture_3d_placeholder_initialize(RID p_texture) {
}

Ref<Image> TextureStorage::texture_2d_get(RID p_texture) const {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND_V(!tex, Ref<Image>());

	/*
#ifdef TOOLS_ENABLED
	if (tex->image_cache_2d.is_valid()) {
		return tex->image_cache_2d;
	}
#endif
	Vector<uint8_t> data = RD::get_singleton()->texture_get_data(tex->rd_texture, 0);
	ERR_FAIL_COND_V(data.size() == 0, Ref<Image>());
	Ref<Image> image;
	image.instance();
	image->create(tex->width, tex->height, tex->mipmaps > 1, tex->validated_format, data);
	ERR_FAIL_COND_V(image->empty(), Ref<Image>());
	if (tex->format != tex->validated_format) {
		image->convert(tex->format);
	}

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		tex->image_cache_2d = image;
	}
#endif
*/
	ERR_FAIL_COND_V(!tex->images.size(), Ref<Image>());

	return tex->images[0];

	//	return image;

	//	return Ref<Image>();
}

void TextureStorage::texture_replace(RID p_texture, RID p_by_texture) {
	Texture *tex_to = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex_to);
	Texture *tex_from = texture_owner.get_or_null(p_by_texture);
	ERR_FAIL_COND(!tex_from);

	tex_to->destroy();
	tex_to->copy_from(*tex_from);

	// copy image data and upload to GL
	tex_to->images.resize(tex_from->images.size());

	for (int n = 0; n < tex_from->images.size(); n++) {
		texture_set_data(p_texture, tex_from->images[n], n);
	}

	texture_free(p_by_texture);
}

void TextureStorage::texture_set_size_override(RID p_texture, int p_width, int p_height) {
	Texture *texture = texture_owner.get_or_null(p_texture);

	ERR_FAIL_COND(!texture);
	ERR_FAIL_COND(texture->render_target);

	ERR_FAIL_COND(p_width <= 0 || p_width > 16384);
	ERR_FAIL_COND(p_height <= 0 || p_height > 16384);
	//real texture size is in alloc width and height
	texture->width = p_width;
	texture->height = p_height;
}

void TextureStorage::texture_set_path(RID p_texture, const String &p_path) {
	Texture *texture = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!texture);

	texture->path = p_path;
}

String TextureStorage::texture_get_path(RID p_texture) const {
	Texture *texture = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND_V(!texture, "");

	return texture->path;
}

void TextureStorage::texture_set_detect_3d_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) {
	Texture *texture = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!texture);

	texture->detect_3d = p_callback;
	texture->detect_3d_ud = p_userdata;
}

void TextureStorage::texture_set_detect_srgb_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) {
	Texture *texture = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!texture);

	texture->detect_srgb = p_callback;
	texture->detect_srgb_ud = p_userdata;
}

void TextureStorage::texture_set_detect_normal_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) {
	Texture *texture = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!texture);

	texture->detect_normal = p_callback;
	texture->detect_normal_ud = p_userdata;
}

void TextureStorage::texture_debug_usage(List<RS::TextureInfo> *r_info) {
	List<RID> textures;
	texture_owner.get_owned_list(&textures);

	for (List<RID>::Element *E = textures.front(); E; E = E->next()) {
		Texture *t = texture_owner.get_or_null(E->get());
		if (!t) {
			continue;
		}
		RS::TextureInfo tinfo;
		tinfo.path = t->path;
		tinfo.format = t->format;
		tinfo.width = t->alloc_width;
		tinfo.height = t->alloc_height;
		tinfo.depth = 0;
		tinfo.bytes = t->total_data_size;
		r_info->push_back(tinfo);
	}
}

void TextureStorage::texture_set_force_redraw_if_visible(RID p_texture, bool p_enable) {
	Texture *texture = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!texture);

	texture->redraw_if_visible = p_enable;
}

Size2 TextureStorage::texture_size_with_proxy(RID p_texture) {
	const Texture *texture = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND_V(!texture, Size2());
	if (texture->proxy) {
		return Size2(texture->proxy->width, texture->proxy->height);
	} else {
		return Size2(texture->width, texture->height);
	}
}

// example use in 3.2
// VS::get_singleton()->texture_set_proxy(default_texture->proxy, texture_rid);

// p_proxy is the source (pre-existing) texture?
// and p_texture is the one that is being made into a proxy?
//This naming is confusing. Comments!!!

// The naming of the parameters seemed to be reversed?
// The p_proxy is the source texture
// and p_texture is actually the proxy????

void TextureStorage::texture_set_proxy(RID p_texture, RID p_proxy) {
	Texture *texture = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!texture);

	if (texture->proxy) {
		texture->proxy->proxy_owners.erase(texture);
		texture->proxy = nullptr;
	}

	if (p_proxy.is_valid()) {
		Texture *proxy = texture_owner.get_or_null(p_proxy);
		ERR_FAIL_COND(!proxy);
		ERR_FAIL_COND(proxy == texture);
		proxy->proxy_owners.insert(texture);
		texture->proxy = proxy;
	}
}

void TextureStorage::texture_set_data(RID p_texture, const Ref<Image> &p_image, int p_layer) {
	Config *config = Config::get_singleton();
	Texture *texture = texture_owner.get_or_null(p_texture);

	ERR_FAIL_COND(!_is_main_thread());

	ERR_FAIL_COND(!texture);
	if (texture->target == GL_TEXTURE_3D) {
		// Target is set to a 3D texture or array texture, exit early to avoid spamming errors
		return;
	}
	ERR_FAIL_COND(!texture->active);
	ERR_FAIL_COND(texture->render_target);
	ERR_FAIL_COND(p_image.is_null());
	ERR_FAIL_COND(texture->format != p_image->get_format());

	ERR_FAIL_COND(!p_image->get_width());
	ERR_FAIL_COND(!p_image->get_height());

	//	ERR_FAIL_COND(texture->type == RS::TEXTURE_TYPE_EXTERNAL);

	GLenum type;
	GLenum format;
	GLenum internal_format;
	bool compressed = false;

	if (config->keep_original_textures && !(texture->flags & TEXTURE_FLAG_USED_FOR_STREAMING)) {
		texture->images.write[p_layer] = p_image;
	}

	// print_line("texture_set_data width " + itos (p_image->get_width()) + " height " + itos(p_image->get_height()));

	Image::Format real_format;
	Ref<Image> img = _get_gl_image_and_format(p_image, p_image->get_format(), texture->flags, real_format, format, internal_format, type, compressed, texture->resize_to_po2);

	if (texture->resize_to_po2) {
		if (p_image->is_compressed()) {
			ERR_PRINT("Texture '" + texture->path + "' is required to be a power of 2 because it uses either mipmaps or repeat, so it was decompressed. This will hurt performance and memory usage.");
		}

		if (img == p_image) {
			img = img->duplicate();
		}
		img->resize_to_po2(false);
	}

	if (config->shrink_textures_x2 && (p_image->has_mipmaps() || !p_image->is_compressed()) && !(texture->flags & TEXTURE_FLAG_USED_FOR_STREAMING)) {
		texture->alloc_height = MAX(1, texture->alloc_height / 2);
		texture->alloc_width = MAX(1, texture->alloc_width / 2);

		if (texture->alloc_width == img->get_width() / 2 && texture->alloc_height == img->get_height() / 2) {
			img->shrink_x2();
		} else if (img->get_format() <= Image::FORMAT_RGBA8) {
			img->resize(texture->alloc_width, texture->alloc_height, Image::INTERPOLATE_BILINEAR);
		}
	}

	GLenum blit_target = (texture->target == GL_TEXTURE_CUBE_MAP) ? _cube_side_enum[p_layer] : GL_TEXTURE_2D;

	texture->data_size = img->get_data().size();
	Vector<uint8_t> read = img->get_data();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(texture->target, texture->tex_id);

	texture->ignore_mipmaps = compressed && !img->has_mipmaps();

	// set filtering and repeat state
	_texture_set_state_from_flags(texture);

	//set swizle for older format compatibility
#ifdef GLES_OVER_GL
	switch (texture->format) {
		case Image::FORMAT_L8: {
			glTexParameteri(texture->target, GL_TEXTURE_SWIZZLE_R, GL_RED);
			glTexParameteri(texture->target, GL_TEXTURE_SWIZZLE_G, GL_RED);
			glTexParameteri(texture->target, GL_TEXTURE_SWIZZLE_B, GL_RED);
			glTexParameteri(texture->target, GL_TEXTURE_SWIZZLE_A, GL_ONE);

		} break;
		case Image::FORMAT_LA8: {
			glTexParameteri(texture->target, GL_TEXTURE_SWIZZLE_R, GL_RED);
			glTexParameteri(texture->target, GL_TEXTURE_SWIZZLE_G, GL_RED);
			glTexParameteri(texture->target, GL_TEXTURE_SWIZZLE_B, GL_RED);
			glTexParameteri(texture->target, GL_TEXTURE_SWIZZLE_A, GL_GREEN);
		} break;
		default: {
			glTexParameteri(texture->target, GL_TEXTURE_SWIZZLE_R, GL_RED);
			glTexParameteri(texture->target, GL_TEXTURE_SWIZZLE_G, GL_GREEN);
			glTexParameteri(texture->target, GL_TEXTURE_SWIZZLE_B, GL_BLUE);
			glTexParameteri(texture->target, GL_TEXTURE_SWIZZLE_A, GL_ALPHA);

		} break;
	}
#endif

	int mipmaps = ((texture->flags & TEXTURE_FLAG_MIPMAPS) && img->has_mipmaps()) ? img->get_mipmap_count() + 1 : 1;

	int w = img->get_width();
	int h = img->get_height();

	int tsize = 0;

	for (int i = 0; i < mipmaps; i++) {
		int size, ofs;
		img->get_mipmap_offset_and_size(i, ofs, size);

		if (compressed) {
			glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

			int bw = w;
			int bh = h;

			glCompressedTexImage2D(blit_target, i, internal_format, bw, bh, 0, size, &read[ofs]);
		} else {
			glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
			if (texture->flags & TEXTURE_FLAG_USED_FOR_STREAMING) {
				glTexSubImage2D(blit_target, i, 0, 0, w, h, format, type, &read[ofs]);
			} else {
				glTexImage2D(blit_target, i, internal_format, w, h, 0, format, type, &read[ofs]);
			}
		}

		tsize += size;

		w = MAX(1, w >> 1);
		h = MAX(1, h >> 1);
	}

	// info.texture_mem -= texture->total_data_size; // TODO make this work again!!
	texture->total_data_size = tsize;
	// info.texture_mem += texture->total_data_size; // TODO make this work again!!

	// printf("texture: %i x %i - size: %i - total: %i\n", texture->width, texture->height, tsize, info.texture_mem);

	texture->stored_cube_sides |= (1 << p_layer);

	if ((texture->flags & TEXTURE_FLAG_MIPMAPS) && mipmaps == 1 && !texture->ignore_mipmaps && (texture->type != RenderingDevice::TEXTURE_TYPE_CUBE || texture->stored_cube_sides == (1 << 6) - 1)) {
		//generate mipmaps if they were requested and the image does not contain them
		glGenerateMipmap(texture->target);
	}

	texture->mipmaps = mipmaps;
}

void TextureStorage::texture_set_data_partial(RID p_texture, const Ref<Image> &p_image, int src_x, int src_y, int src_w, int src_h, int dst_x, int dst_y, int p_dst_mip, int p_layer) {
	// TODO
	ERR_PRINT("Not implemented (ask Karroffel to do it :p)");
}

/*
Ref<Image> TextureStorage::texture_get_data(RID p_texture, int p_layer) const {
	Texture *texture = texture_owner.get_or_null(p_texture);

	ERR_FAIL_COND_V(!texture, Ref<Image>());
	ERR_FAIL_COND_V(!texture->active, Ref<Image>());
	ERR_FAIL_COND_V(texture->data_size == 0 && !texture->render_target, Ref<Image>());

	if (texture->type == RS::TEXTURE_TYPE_CUBEMAP && p_layer < 6 && p_layer >= 0 && !texture->images[p_layer].is_null()) {
		return texture->images[p_layer];
	}

#ifdef GLES_OVER_GL

	Image::Format real_format;
	GLenum gl_format;
	GLenum gl_internal_format;
	GLenum gl_type;
	bool compressed;
	_get_gl_image_and_format(Ref<Image>(), texture->format, texture->flags, real_format, gl_format, gl_internal_format, gl_type, compressed, false);

	PoolVector<uint8_t> data;

	int data_size = Image::get_image_data_size(texture->alloc_width, texture->alloc_height, real_format, texture->mipmaps > 1);

	data.resize(data_size * 2); //add some memory at the end, just in case for buggy drivers
	PoolVector<uint8_t>::Write wb = data.write();

	glActiveTexture(GL_TEXTURE0);

	glBindTexture(texture->target, texture->tex_id);

	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

	for (int i = 0; i < texture->mipmaps; i++) {
		int ofs = Image::get_image_mipmap_offset(texture->alloc_width, texture->alloc_height, real_format, i);

		if (texture->compressed) {
			glPixelStorei(GL_PACK_ALIGNMENT, 4);
			glGetCompressedTexImage(texture->target, i, &wb[ofs]);
		} else {
			glPixelStorei(GL_PACK_ALIGNMENT, 1);
			glGetTexImage(texture->target, i, texture->gl_format_cache, texture->gl_type_cache, &wb[ofs]);
		}
	}

	wb.release();

	data.resize(data_size);

	Image *img = memnew(Image(texture->alloc_width, texture->alloc_height, texture->mipmaps > 1, real_format, data));

	return Ref<Image>(img);
#else

	Image::Format real_format;
	GLenum gl_format;
	GLenum gl_internal_format;
	GLenum gl_type;
	bool compressed;
	_get_gl_image_and_format(Ref<Image>(), texture->format, texture->flags, real_format, gl_format, gl_internal_format, gl_type, compressed, texture->resize_to_po2);

	PoolVector<uint8_t> data;

	int data_size = Image::get_image_data_size(texture->alloc_width, texture->alloc_height, Image::FORMAT_RGBA8, false);

	data.resize(data_size * 2); //add some memory at the end, just in case for buggy drivers
	PoolVector<uint8_t>::Write wb = data.write();

	GLuint temp_framebuffer;
	glGenFramebuffers(1, &temp_framebuffer);

	GLuint temp_color_texture;
	glGenTextures(1, &temp_color_texture);

	glBindFramebuffer(GL_FRAMEBUFFER, temp_framebuffer);

	glBindTexture(GL_TEXTURE_2D, temp_color_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture->alloc_width, texture->alloc_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, temp_color_texture, 0);

	glDepthMask(GL_FALSE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDisable(GL_BLEND);
	glDepthFunc(GL_LEQUAL);
	glColorMask(1, 1, 1, 1);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture->tex_id);

	glViewport(0, 0, texture->alloc_width, texture->alloc_height);

	shaders.copy.bind();

	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	bind_quad_array();
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glReadPixels(0, 0, texture->alloc_width, texture->alloc_height, GL_RGBA, GL_UNSIGNED_BYTE, &wb[0]);

	glDeleteTextures(1, &temp_color_texture);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glDeleteFramebuffers(1, &temp_framebuffer);

	wb.release();

	data.resize(data_size);

	Image *img = memnew(Image(texture->alloc_width, texture->alloc_height, false, Image::FORMAT_RGBA8, data));
	if (!texture->compressed) {
		img->convert(real_format);
	}

	return Ref<Image>(img);

#endif
}
*/

void TextureStorage::texture_set_flags(RID p_texture, uint32_t p_flags) {
	Texture *texture = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!texture);

	bool had_mipmaps = texture->flags & TEXTURE_FLAG_MIPMAPS;

	texture->flags = p_flags;

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(texture->target, texture->tex_id);

	// set filtering and repeat state
	_texture_set_state_from_flags(texture);

	if ((texture->flags & TEXTURE_FLAG_MIPMAPS) && !texture->ignore_mipmaps) {
		if (!had_mipmaps && texture->mipmaps == 1) {
			glGenerateMipmap(texture->target);
		}
	}
}

uint32_t TextureStorage::texture_get_flags(RID p_texture) const {
	Texture *texture = texture_owner.get_or_null(p_texture);

	ERR_FAIL_COND_V(!texture, 0);

	return texture->flags;
}

Image::Format TextureStorage::texture_get_format(RID p_texture) const {
	Texture *texture = texture_owner.get_or_null(p_texture);

	ERR_FAIL_COND_V(!texture, Image::FORMAT_L8);

	return texture->format;
}

RenderingDevice::TextureType TextureStorage::texture_get_type(RID p_texture) const {
	Texture *texture = texture_owner.get_or_null(p_texture);

	ERR_FAIL_COND_V(!texture, RenderingDevice::TEXTURE_TYPE_2D);

	return texture->type;
}

uint32_t TextureStorage::texture_get_texid(RID p_texture) const {
	Texture *texture = texture_owner.get_or_null(p_texture);

	ERR_FAIL_COND_V(!texture, 0);

	return texture->tex_id;
}

uint32_t TextureStorage::texture_get_width(RID p_texture) const {
	Texture *texture = texture_owner.get_or_null(p_texture);

	ERR_FAIL_COND_V(!texture, 0);

	return texture->width;
}

uint32_t TextureStorage::texture_get_height(RID p_texture) const {
	Texture *texture = texture_owner.get_or_null(p_texture);

	ERR_FAIL_COND_V(!texture, 0);

	return texture->height;
}

uint32_t TextureStorage::texture_get_depth(RID p_texture) const {
	Texture *texture = texture_owner.get_or_null(p_texture);

	ERR_FAIL_COND_V(!texture, 0);

	return texture->depth;
}

void TextureStorage::texture_bind(RID p_texture, uint32_t p_texture_no) {
	Texture *texture = texture_owner.get_or_null(p_texture);

	ERR_FAIL_COND(!texture);

	glActiveTexture(GL_TEXTURE0 + p_texture_no);
	glBindTexture(texture->target, texture->tex_id);
}

void TextureStorage::texture_set_shrink_all_x2_on_set_data(bool p_enable) {
	Config::get_singleton()->shrink_textures_x2 = p_enable;
}

RID TextureStorage::texture_create_radiance_cubemap(RID p_source, int p_resolution) const {
	return RID();
}

void TextureStorage::textures_keep_original(bool p_enable) {
	Config::get_singleton()->keep_original_textures = p_enable;
}

#endif // GLES3_ENABLED
