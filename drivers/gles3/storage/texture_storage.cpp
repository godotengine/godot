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

static const GLenum _cube_side_enum[6] = {
	GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
	GL_TEXTURE_CUBE_MAP_POSITIVE_X,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
};

TextureStorage::TextureStorage() {
	singleton = this;

	system_fbo = 0;

	frame.current_rt = nullptr;

	{ //create default textures
		{ // White Textures

			Ref<Image> image;
			image.instantiate();
			image->create(4, 4, true, Image::FORMAT_RGBA8);
			image->fill(Color(1, 1, 1, 1));
			image->generate_mipmaps();

			default_gl_textures[DEFAULT_GL_TEXTURE_WHITE] = texture_allocate();
			texture_2d_initialize(default_gl_textures[DEFAULT_GL_TEXTURE_WHITE], image);

			Vector<Ref<Image>> images;
			images.push_back(image);

			default_gl_textures[DEFAULT_GL_TEXTURE_2D_ARRAY_WHITE] = texture_allocate();
			texture_2d_layered_initialize(default_gl_textures[DEFAULT_GL_TEXTURE_2D_ARRAY_WHITE], images, RS::TEXTURE_LAYERED_2D_ARRAY);

			for (int i = 0; i < 3; i++) {
				images.push_back(image);
			}

			default_gl_textures[DEFAULT_GL_TEXTURE_3D_WHITE] = texture_allocate();
			texture_3d_initialize(default_gl_textures[DEFAULT_GL_TEXTURE_3D_WHITE], image->get_format(), 4, 4, 4, false, images);

			for (int i = 0; i < 2; i++) {
				images.push_back(image);
			}

			default_gl_textures[DEFAULT_GL_TEXTURE_CUBEMAP_WHITE] = texture_allocate();
			texture_2d_layered_initialize(default_gl_textures[DEFAULT_GL_TEXTURE_CUBEMAP_WHITE], images, RS::TEXTURE_LAYERED_CUBEMAP);
		}

		{ // black
			Ref<Image> image;
			image.instantiate();
			image->create(4, 4, true, Image::FORMAT_RGBA8);
			image->fill(Color(0, 0, 0, 1));
			image->generate_mipmaps();

			default_gl_textures[DEFAULT_GL_TEXTURE_BLACK] = texture_allocate();
			texture_2d_initialize(default_gl_textures[DEFAULT_GL_TEXTURE_BLACK], image);

			Vector<Ref<Image>> images;

			for (int i = 0; i < 4; i++) {
				images.push_back(image);
			}

			default_gl_textures[DEFAULT_GL_TEXTURE_3D_BLACK] = texture_allocate();
			texture_3d_initialize(default_gl_textures[DEFAULT_GL_TEXTURE_3D_BLACK], image->get_format(), 4, 4, 4, false, images);

			for (int i = 0; i < 2; i++) {
				images.push_back(image);
			}
			default_gl_textures[DEFAULT_GL_TEXTURE_CUBEMAP_BLACK] = texture_allocate();
			texture_2d_layered_initialize(default_gl_textures[DEFAULT_GL_TEXTURE_CUBEMAP_BLACK], images, RS::TEXTURE_LAYERED_CUBEMAP);
		}

		{
			Ref<Image> image;
			image.instantiate();
			image->create(4, 4, true, Image::FORMAT_RGBA8);
			image->fill(Color(0.5, 0.5, 1, 1));
			image->generate_mipmaps();

			default_gl_textures[DEFAULT_GL_TEXTURE_NORMAL] = texture_allocate();
			texture_2d_initialize(default_gl_textures[DEFAULT_GL_TEXTURE_NORMAL], image);
		}

		{
			Ref<Image> image;
			image.instantiate();
			image->create(4, 4, true, Image::FORMAT_RGBA8);
			image->fill(Color(1.0, 0.5, 1, 1));
			image->generate_mipmaps();

			default_gl_textures[DEFAULT_GL_TEXTURE_ANISO] = texture_allocate();
			texture_2d_initialize(default_gl_textures[DEFAULT_GL_TEXTURE_ANISO], image);
		}

		{
			unsigned char pixel_data[4 * 4 * 4];
			for (int i = 0; i < 16; i++) {
				pixel_data[i * 4 + 0] = 0;
				pixel_data[i * 4 + 1] = 0;
				pixel_data[i * 4 + 2] = 0;
				pixel_data[i * 4 + 3] = 0;
			}

			default_gl_textures[DEFAULT_GL_TEXTURE_2D_UINT] = texture_allocate();
			Texture texture;
			texture.width = 4;
			texture.height = 4;
			texture.format = Image::FORMAT_RGBA8;
			texture.type = Texture::TYPE_2D;
			texture.target = GL_TEXTURE_2D;
			texture.active = true;
			glGenTextures(1, &texture.tex_id);
			texture_owner.initialize_rid(default_gl_textures[DEFAULT_GL_TEXTURE_2D_UINT], texture);

			glBindTexture(GL_TEXTURE_2D, texture.tex_id);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, 4, 4, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, pixel_data);
			texture.gl_set_filter(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST);
		}
		{
			uint16_t pixel_data[4 * 4];
			for (int i = 0; i < 16; i++) {
				pixel_data[i] = Math::make_half_float(1.0f);
			}

			default_gl_textures[DEFAULT_GL_TEXTURE_DEPTH] = texture_allocate();
			Texture texture;
			texture.width = 4;
			texture.height = 4;
			texture.format = Image::FORMAT_RGBA8;
			texture.type = Texture::TYPE_2D;
			texture.target = GL_TEXTURE_2D;
			texture.active = true;
			glGenTextures(1, &texture.tex_id);
			texture_owner.initialize_rid(default_gl_textures[DEFAULT_GL_TEXTURE_DEPTH], texture);

			glBindTexture(GL_TEXTURE_2D, texture.tex_id);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, 4, 4, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, pixel_data);
			texture.gl_set_filter(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST);
		}
	}
}

TextureStorage::~TextureStorage() {
	singleton = nullptr;
	for (int i = 0; i < DEFAULT_GL_TEXTURE_MAX; i++) {
		texture_free(default_gl_textures[i]);
	}
}

//TODO, move back to storage
bool TextureStorage::can_create_resources_async() const {
	return false;
}

/* Canvas Texture API */

RID TextureStorage::canvas_texture_allocate() {
	return canvas_texture_owner.allocate_rid();
}

void TextureStorage::canvas_texture_initialize(RID p_rid) {
	canvas_texture_owner.initialize_rid(p_rid);
}

void TextureStorage::canvas_texture_free(RID p_rid) {
	canvas_texture_owner.free(p_rid);
}

void TextureStorage::canvas_texture_set_channel(RID p_canvas_texture, RS::CanvasTextureChannel p_channel, RID p_texture) {
	CanvasTexture *ct = canvas_texture_owner.get_or_null(p_canvas_texture);
	switch (p_channel) {
		case RS::CANVAS_TEXTURE_CHANNEL_DIFFUSE: {
			ct->diffuse = p_texture;
		} break;
		case RS::CANVAS_TEXTURE_CHANNEL_NORMAL: {
			ct->normal_map = p_texture;
		} break;
		case RS::CANVAS_TEXTURE_CHANNEL_SPECULAR: {
			ct->specular = p_texture;
		} break;
	}
}

void TextureStorage::canvas_texture_set_shading_parameters(RID p_canvas_texture, const Color &p_specular_color, float p_shininess) {
	CanvasTexture *ct = canvas_texture_owner.get_or_null(p_canvas_texture);
	ct->specular_color.r = p_specular_color.r;
	ct->specular_color.g = p_specular_color.g;
	ct->specular_color.b = p_specular_color.b;
	ct->specular_color.a = p_shininess;
}

void TextureStorage::canvas_texture_set_texture_filter(RID p_canvas_texture, RS::CanvasItemTextureFilter p_filter) {
	CanvasTexture *ct = canvas_texture_owner.get_or_null(p_canvas_texture);
	ct->texture_filter = p_filter;
}

void TextureStorage::canvas_texture_set_texture_repeat(RID p_canvas_texture, RS::CanvasItemTextureRepeat p_repeat) {
	CanvasTexture *ct = canvas_texture_owner.get_or_null(p_canvas_texture);
	ct->texture_repeat = p_repeat;
}

/* Texture API */

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
			ERR_FAIL_V_MSG(Ref<Image>(), "Image Format: " + itos(p_format) + " is not supported by the OpenGL3 Renderer");
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

RID TextureStorage::texture_allocate() {
	return texture_owner.allocate_rid();
}

void TextureStorage::texture_free(RID p_texture) {
	Texture *t = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!t);
	ERR_FAIL_COND(t->is_render_target);

	if (t->canvas_texture) {
		memdelete(t->canvas_texture);
	}

	if (t->tex_id != 0) {
		glDeleteTextures(1, &t->tex_id);
		t->tex_id = 0;
	}

	if (t->is_proxy && t->proxy_to.is_valid()) {
		Texture *proxy_to = texture_owner.get_or_null(t->proxy_to);
		if (proxy_to) {
			proxy_to->proxies.erase(p_texture);
		}
	}

	//decal_atlas_remove_texture(p_texture);

	for (int i = 0; i < t->proxies.size(); i++) {
		Texture *p = texture_owner.get_or_null(t->proxies[i]);
		ERR_CONTINUE(!p);
		p->proxy_to = RID();
		p->tex_id = 0;
	}

	texture_owner.free(p_texture);
}

void TextureStorage::texture_2d_initialize(RID p_texture, const Ref<Image> &p_image) {
	Texture texture;
	texture.width = p_image->get_width();
	texture.height = p_image->get_height();
	texture.alloc_width = texture.width;
	texture.alloc_height = texture.height;
	texture.mipmaps = p_image->get_mipmap_count();
	texture.format = p_image->get_format();
	texture.type = Texture::TYPE_2D;
	texture.target = GL_TEXTURE_2D;
	_get_gl_image_and_format(Ref<Image>(), texture.format, 0, texture.real_format, texture.gl_format_cache, texture.gl_internal_format_cache, texture.gl_type_cache, texture.compressed, false);
	//texture.total_data_size = p_image->get_image_data_size(); // verify that this returns size in bytes
	texture.active = true;
	glGenTextures(1, &texture.tex_id);
	texture_owner.initialize_rid(p_texture, texture);
	texture_set_data(p_texture, p_image);
}

void TextureStorage::texture_2d_layered_initialize(RID p_texture, const Vector<Ref<Image>> &p_layers, RS::TextureLayeredType p_layered_type) {
	texture_owner.initialize_rid(p_texture, Texture());
}

void TextureStorage::texture_3d_initialize(RID p_texture, Image::Format, int p_width, int p_height, int p_depth, bool p_mipmaps, const Vector<Ref<Image>> &p_data) {
	texture_owner.initialize_rid(p_texture, Texture());
}

// Called internally when texture_proxy_create(p_base) is called.
// Note: p_base is the root and p_texture is the proxy.
void TextureStorage::texture_proxy_initialize(RID p_texture, RID p_base) {
	Texture *texture = texture_owner.get_or_null(p_base);
	ERR_FAIL_COND(!texture);
	Texture proxy_tex;
	proxy_tex.copy_from(*texture);
	proxy_tex.proxy_to = p_base;
	proxy_tex.is_render_target = false;
	proxy_tex.is_proxy = true;
	proxy_tex.proxies.clear();
	texture->proxies.push_back(p_texture);
	texture_owner.initialize_rid(p_texture, proxy_tex);
}

void TextureStorage::texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer) {
	// only 1 layer so far
	texture_set_data(p_texture, p_image);
#ifdef TOOLS_ENABLED
	Texture *tex = texture_owner.get_or_null(p_texture);

	tex->image_cache_2d.unref();
#endif
}

void TextureStorage::texture_proxy_update(RID p_texture, RID p_proxy_to) {
}

void TextureStorage::texture_2d_placeholder_initialize(RID p_texture) {
	//this could be better optimized to reuse an existing image , done this way
	//for now to get it working
	Ref<Image> image;
	image.instantiate();
	image->create(4, 4, false, Image::FORMAT_RGBA8);
	image->fill(Color(1, 0, 1, 1));

	texture_2d_initialize(p_texture, image);
}

void TextureStorage::texture_2d_layered_placeholder_initialize(RID p_texture, RenderingServer::TextureLayeredType p_layered_type) {
	//this could be better optimized to reuse an existing image , done this way
	//for now to get it working
	Ref<Image> image;
	image.instantiate();
	image->create(4, 4, false, Image::FORMAT_RGBA8);
	image->fill(Color(1, 0, 1, 1));

	Vector<Ref<Image>> images;
	if (p_layered_type == RS::TEXTURE_LAYERED_2D_ARRAY) {
		images.push_back(image);
	} else {
		//cube
		for (int i = 0; i < 6; i++) {
			images.push_back(image);
		}
	}

	texture_2d_layered_initialize(p_texture, images, p_layered_type);
}

void TextureStorage::texture_3d_placeholder_initialize(RID p_texture) {
	//this could be better optimized to reuse an existing image , done this way
	//for now to get it working
	Ref<Image> image;
	image.instantiate();
	image->create(4, 4, false, Image::FORMAT_RGBA8);
	image->fill(Color(1, 0, 1, 1));

	Vector<Ref<Image>> images;
	//cube
	for (int i = 0; i < 4; i++) {
		images.push_back(image);
	}

	texture_3d_initialize(p_texture, Image::FORMAT_RGBA8, 4, 4, 4, false, images);
}

Ref<Image> TextureStorage::texture_2d_get(RID p_texture) const {
	Texture *texture = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND_V(!texture, Ref<Image>());

#ifdef TOOLS_ENABLED
	if (texture->image_cache_2d.is_valid() && !texture->is_render_target) {
		return texture->image_cache_2d;
	}
#endif

#ifdef GLES_OVER_GL
	// OpenGL 3.3 supports glGetTexImage which is faster and simpler than glReadPixels.
	Vector<uint8_t> data;

	int data_size = Image::get_image_data_size(texture->alloc_width, texture->alloc_height, texture->real_format, texture->mipmaps > 1);

	data.resize(data_size * 2); //add some memory at the end, just in case for buggy drivers
	uint8_t *w = data.ptrw();

	glActiveTexture(GL_TEXTURE0);

	glBindTexture(texture->target, texture->tex_id);

	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

	for (int i = 0; i < texture->mipmaps; i++) {
		int ofs = Image::get_image_mipmap_offset(texture->alloc_width, texture->alloc_height, texture->real_format, i);

		if (texture->compressed) {
			glPixelStorei(GL_PACK_ALIGNMENT, 4);
			glGetCompressedTexImage(texture->target, i, &w[ofs]);

		} else {
			glPixelStorei(GL_PACK_ALIGNMENT, 1);

			glGetTexImage(texture->target, i, texture->gl_format_cache, texture->gl_type_cache, &w[ofs]);
		}
	}

	data.resize(data_size);

	ERR_FAIL_COND_V(data.size() == 0, Ref<Image>());
	Ref<Image> image;
	image.instantiate();
	image->create(texture->width, texture->height, texture->mipmaps > 1, texture->real_format, data);
	ERR_FAIL_COND_V(image->is_empty(), Ref<Image>());
	if (texture->format != texture->real_format) {
		image->convert(texture->format);
	}
#else
	// Support for Web and Mobile will come later.
	Ref<Image> image;
#endif

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint() && !texture->is_render_target) {
		texture->image_cache_2d = image;
	}
#endif

	return image;
}

void TextureStorage::texture_replace(RID p_texture, RID p_by_texture) {
	Texture *tex_to = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex_to);
	ERR_FAIL_COND(tex_to->is_proxy); //can't replace proxy
	Texture *tex_from = texture_owner.get_or_null(p_by_texture);
	ERR_FAIL_COND(!tex_from);
	ERR_FAIL_COND(tex_from->is_proxy); //can't replace proxy

	if (tex_to == tex_from) {
		return;
	}

	if (tex_to->canvas_texture) {
		memdelete(tex_to->canvas_texture);
		tex_to->canvas_texture = nullptr;
	}

	if (tex_to->tex_id) {
		glDeleteTextures(1, &tex_to->tex_id);
		tex_to->tex_id = 0;
	}

	Vector<RID> proxies_to_update = tex_to->proxies;
	Vector<RID> proxies_to_redirect = tex_from->proxies;

	tex_to->copy_from(*tex_from);

	tex_to->proxies = proxies_to_update; //restore proxies, so they can be updated

	if (tex_to->canvas_texture) {
		tex_to->canvas_texture->diffuse = p_texture; //update
	}

	for (int i = 0; i < proxies_to_update.size(); i++) {
		texture_proxy_update(proxies_to_update[i], p_texture);
	}
	for (int i = 0; i < proxies_to_redirect.size(); i++) {
		texture_proxy_update(proxies_to_redirect[i], p_texture);
	}
	//delete last, so proxies can be updated
	texture_owner.free(p_by_texture);

	//decal_atlas_mark_dirty_on_texture(p_texture);
}

void TextureStorage::texture_set_size_override(RID p_texture, int p_width, int p_height) {
	Texture *texture = texture_owner.get_or_null(p_texture);

	ERR_FAIL_COND(!texture);
	ERR_FAIL_COND(texture->is_render_target);

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

	texture->detect_3d_callback = p_callback;
	texture->detect_3d_callback_ud = p_userdata;
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

	texture->detect_normal_callback = p_callback;
	texture->detect_normal_callback_ud = p_userdata;
}

void TextureStorage::texture_set_detect_roughness_callback(RID p_texture, RS::TextureDetectRoughnessCallback p_callback, void *p_userdata) {
	Texture *texture = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!texture);

	texture->detect_roughness_callback = p_callback;
	texture->detect_roughness_callback_ud = p_userdata;
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
	if (texture->is_proxy) {
		const Texture *proxy = texture_owner.get_or_null(texture->proxy_to);
		return Size2(proxy->width, proxy->height);
	} else {
		return Size2(texture->width, texture->height);
	}
}

void TextureStorage::texture_set_data(RID p_texture, const Ref<Image> &p_image, int p_layer) {
	Texture *texture = texture_owner.get_or_null(p_texture);

	ERR_FAIL_COND(!texture);
	if (texture->target == GL_TEXTURE_3D) {
		// Target is set to a 3D texture or array texture, exit early to avoid spamming errors
		return;
	}
	ERR_FAIL_COND(!texture->active);
	ERR_FAIL_COND(texture->is_render_target);
	ERR_FAIL_COND(p_image.is_null());
	ERR_FAIL_COND(texture->format != p_image->get_format());

	ERR_FAIL_COND(!p_image->get_width());
	ERR_FAIL_COND(!p_image->get_height());

	//	ERR_FAIL_COND(texture->type == RS::TEXTURE_TYPE_EXTERNAL);

	GLenum type;
	GLenum format;
	GLenum internal_format;
	bool compressed = false;

	// print_line("texture_set_data width " + itos (p_image->get_width()) + " height " + itos(p_image->get_height()));

	Image::Format real_format;
	Ref<Image> img = _get_gl_image_and_format(p_image, p_image->get_format(), 0, real_format, format, internal_format, type, compressed, texture->resize_to_po2);
	ERR_FAIL_COND(img.is_null());
	if (texture->resize_to_po2) {
		if (p_image->is_compressed()) {
			ERR_PRINT("Texture '" + texture->path + "' is required to be a power of 2 because it uses either mipmaps or repeat, so it was decompressed. This will hurt performance and memory usage.");
		}

		if (img == p_image) {
			img = img->duplicate();
		}
		img->resize_to_po2(false);
	}

	GLenum blit_target = (texture->target == GL_TEXTURE_CUBE_MAP) ? _cube_side_enum[p_layer] : GL_TEXTURE_2D;

	Vector<uint8_t> read = img->get_data();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(texture->target, texture->tex_id);

	// set filtering and repeat state to default
	texture->gl_set_filter(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST);
	texture->gl_set_repeat(RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);

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

	int mipmaps = img->has_mipmaps() ? img->get_mipmap_count() + 1 : 1;

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
			glTexImage2D(blit_target, i, internal_format, w, h, 0, format, type, &read[ofs]);
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

	//if ((texture->flags & TEXTURE_FLAG_MIPMAPS) && mipmaps == 1 && !texture->ignore_mipmaps && (texture->type != RenderingDevice::TEXTURE_TYPE_CUBE || texture->stored_cube_sides == (1 << 6) - 1)) {
	//generate mipmaps if they were requested and the image does not contain them
	//	glGenerateMipmap(texture->target);
	//}

	texture->mipmaps = mipmaps;
}

void TextureStorage::texture_set_data_partial(RID p_texture, const Ref<Image> &p_image, int src_x, int src_y, int src_w, int src_h, int dst_x, int dst_y, int p_dst_mip, int p_layer) {
	ERR_PRINT("Not implemented yet, sorry :(");
}

/*
Ref<Image> TextureStorage::texture_get_data(RID p_texture, int p_layer) const {
	Texture *texture = texture_owner.get_or_null(p_texture);

	ERR_FAIL_COND_V(!texture, Ref<Image>());
	ERR_FAIL_COND_V(!texture->active, Ref<Image>());
	ERR_FAIL_COND_V(texture->data_size == 0 && !texture->render_target, Ref<Image>());


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

Image::Format TextureStorage::texture_get_format(RID p_texture) const {
	Texture *texture = texture_owner.get_or_null(p_texture);

	ERR_FAIL_COND_V(!texture, Image::FORMAT_L8);

	return texture->format;
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

RID TextureStorage::texture_create_radiance_cubemap(RID p_source, int p_resolution) const {
	return RID();
}

/* DECAL API */

RID TextureStorage::decal_allocate() {
	return RID();
}

void TextureStorage::decal_initialize(RID p_rid) {
}

void TextureStorage::decal_set_extents(RID p_decal, const Vector3 &p_extents) {
}

void TextureStorage::decal_set_texture(RID p_decal, RS::DecalTexture p_type, RID p_texture) {
}

void TextureStorage::decal_set_emission_energy(RID p_decal, float p_energy) {
}

void TextureStorage::decal_set_albedo_mix(RID p_decal, float p_mix) {
}

void TextureStorage::decal_set_modulate(RID p_decal, const Color &p_modulate) {
}

void TextureStorage::decal_set_cull_mask(RID p_decal, uint32_t p_layers) {
}

void TextureStorage::decal_set_distance_fade(RID p_decal, bool p_enabled, float p_begin, float p_length) {
}

void TextureStorage::decal_set_fade(RID p_decal, float p_above, float p_below) {
}

void TextureStorage::decal_set_normal_fade(RID p_decal, float p_fade) {
}

AABB TextureStorage::decal_get_aabb(RID p_decal) const {
	return AABB();
}

/* RENDER TARGET API */

GLuint TextureStorage::system_fbo = 0;

void TextureStorage::_set_current_render_target(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);

	if (rt) {
		if (rt->allocate_is_dirty) {
			rt->allocate_is_dirty = false;
			//_clear_render_target(rt);
			//_update_render_target(rt);
		}

		frame.current_rt = rt;
		ERR_FAIL_COND(!rt);

		glViewport(rt->position.x, rt->position.y, rt->size.x, rt->size.y);

		_dims.rt_width = rt->size.x;
		_dims.rt_height = rt->size.y;
		_dims.win_width = rt->size.x;
		_dims.win_height = rt->size.y;

	} else {
		frame.current_rt = nullptr;
		glBindFramebuffer(GL_FRAMEBUFFER, GLES3::TextureStorage::system_fbo);
	}
}

void TextureStorage::_update_render_target(RenderTarget *rt) {
	// do not allocate a render target with no size
	if (rt->size.x <= 0 || rt->size.y <= 0) {
		return;
	}

	// do not allocate a render target that is attached to the screen
	if (rt->flags[RENDER_TARGET_DIRECT_TO_SCREEN]) {
		rt->fbo = system_fbo;
		return;
	}

	rt->color_internal_format = rt->flags[RENDER_TARGET_TRANSPARENT] ? GL_RGBA8 : GL_RGB10_A2;
	rt->color_format = GL_RGBA;
	rt->color_type = rt->flags[RENDER_TARGET_TRANSPARENT] ? GL_BYTE : GL_UNSIGNED_INT_2_10_10_10_REV;
	rt->image_format = Image::FORMAT_RGBA8;

	glDisable(GL_SCISSOR_TEST);
	glColorMask(1, 1, 1, 1);
	glDepthMask(GL_FALSE);

	{
		/* Front FBO */

		Texture *texture = get_texture(rt->texture);
		ERR_FAIL_COND(!texture);

		// framebuffer
		glGenFramebuffers(1, &rt->fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, rt->fbo);

		// color
		glGenTextures(1, &rt->color);
		glBindTexture(GL_TEXTURE_2D, rt->color);

		glTexImage2D(GL_TEXTURE_2D, 0, rt->color_internal_format, rt->size.x, rt->size.y, 0, rt->color_format, rt->color_type, nullptr);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt->color, 0);

		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

		if (status != GL_FRAMEBUFFER_COMPLETE) {
			glDeleteFramebuffers(1, &rt->fbo);
			glDeleteTextures(1, &rt->color);
			rt->fbo = 0;
			rt->size.x = 0;
			rt->size.y = 0;
			rt->color = 0;
			texture->tex_id = 0;
			texture->active = false;
			WARN_PRINT("Could not create render target, status: " + get_framebuffer_error(status));
			return;
		}

		texture->format = rt->image_format;
		texture->real_format = rt->image_format;
		texture->type = Texture::TYPE_2D;
		texture->target = GL_TEXTURE_2D;
		texture->gl_format_cache = rt->color_format;
		texture->gl_type_cache = GL_UNSIGNED_BYTE;
		texture->gl_internal_format_cache = rt->color_internal_format;
		texture->tex_id = rt->color;
		texture->width = rt->size.x;
		texture->alloc_width = rt->size.x;
		texture->height = rt->size.y;
		texture->alloc_height = rt->size.y;
		texture->active = true;
	}

	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);

	// Allocate mipmap chains for full screen blur
	if (rt->size.x >= 2 && rt->size.y >= 2) {
		for (int i = 0; i < 2; i++) {
			ERR_FAIL_COND(rt->mip_maps[i].sizes.size());
			int w = rt->size.x;
			int h = rt->size.y;

			if (i > 0) {
				w >>= 1;
				h >>= 1;
			}

			int level = 0;
			GLsizei width = w;
			GLsizei height = h;

			while (true) {
				RenderTarget::MipMaps::Size mm;
				mm.width = w;
				mm.height = h;
				rt->mip_maps[i].sizes.push_back(mm);

				w >>= 1;
				h >>= 1;

				if (w < 2 || h < 2) {
					break;
				}

				level++;
			}

			glGenTextures(1, &rt->mip_maps[i].color);
			glBindTexture(GL_TEXTURE_2D, rt->mip_maps[i].color);

			for (int l = 0; l < level + 1; l++) {
				glTexImage2D(GL_TEXTURE_2D, l, rt->color_internal_format, width, height, 0, rt->color_format, rt->color_type, nullptr);
				width = MAX(1, (width / 2));
				height = MAX(1, (height / 2));
			}
#ifdef GLES_OVER_GL
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, level);
#endif

			for (int j = 0; j < rt->mip_maps[i].sizes.size(); j++) {
				RenderTarget::MipMaps::Size &mm = rt->mip_maps[i].sizes.write[j];

				glGenFramebuffers(1, &mm.fbo);
				bind_framebuffer(mm.fbo);

				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt->mip_maps[i].color, j);

				GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
				if (status != GL_FRAMEBUFFER_COMPLETE) {
					WARN_PRINT_ONCE("Cannot allocate mipmaps for canvas screen blur. Status: " + get_framebuffer_error(status));
					bind_framebuffer_system();
					return;
				}

				glClearColor(1.0, 0.0, 1.0, 0.0);
				glClear(GL_COLOR_BUFFER_BIT);
			}

			rt->mip_maps[i].levels = level;

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		}
		rt->mip_maps_allocated = true;
	}

	bind_framebuffer_system();
}

void TextureStorage::_clear_render_target(RenderTarget *rt) {
	// there is nothing to clear when DIRECT_TO_SCREEN is used
	if (rt->flags[RENDER_TARGET_DIRECT_TO_SCREEN]) {
		return;
	}

	if (rt->fbo) {
		glDeleteFramebuffers(1, &rt->fbo);
		glDeleteTextures(1, &rt->color);
		rt->fbo = 0;
		rt->color = 0;
	}
	/*
	if (rt->external.fbo != 0) {
		// free this
		glDeleteFramebuffers(1, &rt->external.fbo);

		// clean up our texture
		Texture *t = get_texture(rt->external.texture);
		t->alloc_height = 0;
		t->alloc_width = 0;
		t->width = 0;
		t->height = 0;
		t->active = false;
		texture_free(rt->external.texture);
		memdelete(t);

		rt->external.fbo = 0;
	}
	*/

	Texture *tex = get_texture(rt->texture);
	tex->alloc_height = 0;
	tex->alloc_width = 0;
	tex->width = 0;
	tex->height = 0;
	tex->active = false;

	for (int i = 0; i < 2; i++) {
		if (rt->mip_maps[i].sizes.size()) {
			for (int j = 0; j < rt->mip_maps[i].sizes.size(); j++) {
				glDeleteFramebuffers(1, &rt->mip_maps[i].sizes[j].fbo);
			}

			glDeleteTextures(1, &rt->mip_maps[i].color);
			rt->mip_maps[i].sizes.clear();
			rt->mip_maps[i].levels = 0;
			rt->mip_maps[i].color = 0;
		}
	}
}

RID TextureStorage::render_target_create() {
	RenderTarget render_target;
	//render_target.was_used = false;
	render_target.clear_requested = false;

	for (int i = 0; i < RENDER_TARGET_FLAG_MAX; i++) {
		render_target.flags[i] = false;
	}
	Texture t;
	t.active = true;
	t.render_target = &render_target;
	t.is_render_target = true;

	render_target.texture = texture_owner.make_rid(t);
	_update_render_target(&render_target);
	return render_target_owner.make_rid(render_target);
}

void TextureStorage::render_target_free(RID p_rid) {
	RenderTarget *rt = render_target_owner.get_or_null(p_rid);
	_clear_render_target(rt);

	Texture *t = get_texture(rt->texture);
	if (t) {
		t->is_render_target = false;
		texture_free(rt->texture);
		//memdelete(t);
	}
	render_target_owner.free(p_rid);
}

void TextureStorage::render_target_set_position(RID p_render_target, int p_x, int p_y) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	rt->position = Point2i(p_x, p_y);
}

void TextureStorage::render_target_set_size(RID p_render_target, int p_width, int p_height, uint32_t p_view_count) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	if (p_width == rt->size.x && p_height == rt->size.y) {
		return;
	}

	_clear_render_target(rt);

	rt->size = Size2i(p_width, p_height);

	// print_line("render_target_set_size " + itos(p_render_target.get_id()) + ", w " + itos(p_width) + " h " + itos(p_height));

	rt->allocate_is_dirty = true;
	_update_render_target(rt);
}

// TODO: convert to Size2i internally
Size2i TextureStorage::render_target_get_size(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, Size2());

	return rt->size;
}

RID TextureStorage::render_target_get_texture(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	if (rt->external.fbo == 0) {
		return rt->texture;
	} else {
		return rt->external.texture;
	}
}

void TextureStorage::render_target_set_external_texture(RID p_render_target, unsigned int p_texture_id) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	if (p_texture_id == 0) {
		if (rt->external.fbo != 0) {
			// free this
			glDeleteFramebuffers(1, &rt->external.fbo);

			// and this
			if (rt->external.depth != 0) {
				glDeleteRenderbuffers(1, &rt->external.depth);
			}

			// clean up our texture
			Texture *t = get_texture(rt->external.texture);
			t->alloc_height = 0;
			t->alloc_width = 0;
			t->width = 0;
			t->height = 0;
			t->active = false;
			texture_free(rt->external.texture);
			//memdelete(t);

			rt->external.fbo = 0;
			rt->external.color = 0;
			rt->external.depth = 0;
		}
	} else {
		Texture *t;

		if (rt->external.fbo == 0) {
			// create our fbo
			glGenFramebuffers(1, &rt->external.fbo);
			glBindFramebuffer(GL_FRAMEBUFFER, rt->external.fbo);

			// allocate a texture
			t = memnew(Texture);

			t->type = Texture::TYPE_2D;
			t->width = 0;
			t->height = 0;
			t->alloc_height = 0;
			t->alloc_width = 0;
			t->format = Image::FORMAT_RGBA8;
			t->target = GL_TEXTURE_2D;
			t->gl_format_cache = 0;
			t->gl_internal_format_cache = 0;
			t->gl_type_cache = 0;
			t->srgb = false;
			t->total_data_size = 0;
			t->mipmaps = 1;
			t->active = true;
			t->tex_id = 0;
			t->render_target = rt;
			t->is_render_target = true;

			//rt->external.texture = make_rid(t);

		} else {
			// bind our frame buffer
			glBindFramebuffer(GL_FRAMEBUFFER, rt->external.fbo);

			// find our texture
			t = get_texture(rt->external.texture);
		}

		// set our texture
		t->tex_id = p_texture_id;
		rt->external.color = p_texture_id;

		// size shouldn't be different
		t->width = rt->size.x;
		t->height = rt->size.y;
		t->alloc_height = rt->size.x;
		t->alloc_width = rt->size.y;

		// Switch our texture on our frame buffer
		{
			// set our texture as the destination for our framebuffer
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, p_texture_id, 0);
		}

		// check status and unbind
		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		glBindFramebuffer(GL_FRAMEBUFFER, GLES3::TextureStorage::system_fbo);

		if (status != GL_FRAMEBUFFER_COMPLETE) {
			WARN_PRINT("framebuffer fail, status: " + get_framebuffer_error(status));
		}

		ERR_FAIL_COND(status != GL_FRAMEBUFFER_COMPLETE);
	}
}

void TextureStorage::render_target_set_flag(RID p_render_target, RenderTargetFlags p_flag, bool p_value) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	// When setting DIRECT_TO_SCREEN, you need to clear before the value is set, but allocate after as
	// those functions change how they operate depending on the value of DIRECT_TO_SCREEN
	if (p_flag == RENDER_TARGET_DIRECT_TO_SCREEN && p_value != rt->flags[RENDER_TARGET_DIRECT_TO_SCREEN]) {
		_clear_render_target(rt);
		rt->flags[p_flag] = p_value;
		_update_render_target(rt);
	}

	rt->flags[p_flag] = p_value;

	switch (p_flag) {
		case RENDER_TARGET_TRANSPARENT: {
			//must reset for these formats
			_clear_render_target(rt);
			_update_render_target(rt);
		} break;
		default: {
		}
	}
}

bool TextureStorage::render_target_was_used(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, false);

	return rt->used_in_frame;
}

void TextureStorage::render_target_clear_used(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	rt->used_in_frame = false;
}

void TextureStorage::render_target_request_clear(RID p_render_target, const Color &p_clear_color) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->clear_requested = true;
	rt->clear_color = p_clear_color;
}

bool TextureStorage::render_target_is_clear_requested(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, false);
	return rt->clear_requested;
}
Color TextureStorage::render_target_get_clear_request_color(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, Color());
	return rt->clear_color;
}

void TextureStorage::render_target_disable_clear_request(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->clear_requested = false;
}

void TextureStorage::render_target_do_clear_request(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	if (!rt->clear_requested) {
		return;
	}

	glClearBufferfv(GL_COLOR, 0, rt->clear_color.components);
	rt->clear_requested = false;
}

void TextureStorage::render_target_set_sdf_size_and_scale(RID p_render_target, RS::ViewportSDFOversize p_size, RS::ViewportSDFScale p_scale) {
}

Rect2i TextureStorage::render_target_get_sdf_rect(RID p_render_target) const {
	return Rect2i();
}

void TextureStorage::render_target_mark_sdf_enabled(RID p_render_target, bool p_enabled) {
}

#endif // GLES3_ENABLED
