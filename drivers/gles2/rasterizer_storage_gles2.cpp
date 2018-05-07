/*************************************************************************/
/*  rasterizer_storage_gles2.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "rasterizer_storage_gles2.h"
#include "project_settings.h"
#include "rasterizer_canvas_gles2.h"
#include "rasterizer_scene_gles2.h"

GLuint RasterizerStorageGLES2::system_fbo = 0;

/* TEXTURE API */

Ref<Image> RasterizerStorageGLES2::_get_gl_image_and_format(const Ref<Image> &p_image, Image::Format p_format, uint32_t p_flags, GLenum &r_gl_format, GLenum &r_gl_internal_format, GLenum &r_gl_type) {

	r_gl_format = 0;
	Ref<Image> image = p_image;

	bool need_decompress = false;

	switch (p_format) {

		case Image::FORMAT_L8: {

			r_gl_internal_format = GL_LUMINANCE;
			r_gl_format = GL_LUMINANCE;
			r_gl_type = GL_UNSIGNED_BYTE;
		} break;
		case Image::FORMAT_LA8: {
			r_gl_internal_format = GL_LUMINANCE_ALPHA;
			r_gl_format = GL_LUMINANCE_ALPHA;
			r_gl_type = GL_UNSIGNED_BYTE;
		} break;
		case Image::FORMAT_R8: {

			r_gl_internal_format = GL_ALPHA;
			r_gl_format = GL_ALPHA;
			r_gl_type = GL_UNSIGNED_BYTE;

		} break;
		case Image::FORMAT_RG8: {

			ERR_EXPLAIN("RG texture not supported");
			ERR_FAIL_V(image);

		} break;
		case Image::FORMAT_RGB8: {

			r_gl_internal_format = GL_RGB;
			r_gl_format = GL_RGB;
			r_gl_type = GL_UNSIGNED_BYTE;

		} break;
		case Image::FORMAT_RGBA8: {

			r_gl_format = GL_RGBA;
			r_gl_internal_format = GL_RGBA;
			r_gl_type = GL_UNSIGNED_BYTE;

		} break;
		case Image::FORMAT_RGBA4444: {

			r_gl_internal_format = GL_RGBA;
			r_gl_format = GL_RGBA;
			r_gl_type = GL_UNSIGNED_SHORT_4_4_4_4;

		} break;
		case Image::FORMAT_RGBA5551: {

			r_gl_internal_format = GL_RGB5_A1;
			r_gl_format = GL_RGBA;
			r_gl_type = GL_UNSIGNED_SHORT_5_5_5_1;

		} break;
		case Image::FORMAT_RF: {
			ERR_EXPLAIN("R float texture not supported");
			ERR_FAIL_V(image);

		} break;
		case Image::FORMAT_RGF: {
			ERR_EXPLAIN("RG float texture not supported");
			ERR_FAIL_V(image);

		} break;
		case Image::FORMAT_RGBF: {

			ERR_EXPLAIN("RGB float texture not supported");
			ERR_FAIL_V(image);

		} break;
		case Image::FORMAT_RGBAF: {

			ERR_EXPLAIN("RGBA float texture not supported");
			ERR_FAIL_V(image);

		} break;
		case Image::FORMAT_RH: {
			ERR_EXPLAIN("R half float texture not supported");
			ERR_FAIL_V(image);
		} break;
		case Image::FORMAT_RGH: {
			ERR_EXPLAIN("RG half float texture not supported");
			ERR_FAIL_V(image);

		} break;
		case Image::FORMAT_RGBH: {
			ERR_EXPLAIN("RGB half float texture not supported");
			ERR_FAIL_V(image);

		} break;
		case Image::FORMAT_RGBAH: {
			ERR_EXPLAIN("RGBA half float texture not supported");
			ERR_FAIL_V(image);

		} break;
		case Image::FORMAT_RGBE9995: {
			ERR_EXPLAIN("RGBA float texture not supported");
			ERR_FAIL_V(image);

		} break;
		case Image::FORMAT_DXT1: {

			need_decompress = true;

		} break;
		case Image::FORMAT_DXT3: {

			need_decompress = true;

		} break;
		case Image::FORMAT_DXT5: {

			need_decompress = true;

		} break;
		case Image::FORMAT_RGTC_R: {

			need_decompress = true;

		} break;
		case Image::FORMAT_RGTC_RG: {

			need_decompress = true;

		} break;
		case Image::FORMAT_BPTC_RGBA: {

			need_decompress = true;
		} break;
		case Image::FORMAT_BPTC_RGBF: {

			need_decompress = true;
		} break;
		case Image::FORMAT_BPTC_RGBFU: {

			need_decompress = true;
		} break;
		case Image::FORMAT_PVRTC2: {

			need_decompress = true;
		} break;
		case Image::FORMAT_PVRTC2A: {

			need_decompress = true;
		} break;
		case Image::FORMAT_PVRTC4: {

			need_decompress = true;
		} break;
		case Image::FORMAT_PVRTC4A: {

			need_decompress = true;
		} break;
		case Image::FORMAT_ETC: {

			need_decompress = true;
		} break;
		case Image::FORMAT_ETC2_R11: {

			need_decompress = true;
		} break;
		case Image::FORMAT_ETC2_R11S: {

			need_decompress = true;
		} break;
		case Image::FORMAT_ETC2_RG11: {

			need_decompress = true;
		} break;
		case Image::FORMAT_ETC2_RG11S: {

			need_decompress = true;
		} break;
		case Image::FORMAT_ETC2_RGB8: {

			need_decompress = true;
		} break;
		case Image::FORMAT_ETC2_RGBA8: {

			need_decompress = true;
		} break;
		case Image::FORMAT_ETC2_RGB8A1: {

			need_decompress = true;
		} break;
		default: {

			ERR_FAIL_V(Ref<Image>());
		}
	}

	if (need_decompress) {

		if (!image.is_null()) {
			image = image->duplicate();
			image->decompress();
			ERR_FAIL_COND_V(image->is_compressed(), image);
			image->convert(Image::FORMAT_RGBA8);
		}

		r_gl_format = GL_RGBA;
		r_gl_internal_format = GL_RGBA;
		r_gl_type = GL_UNSIGNED_BYTE;

		return image;
	}

	return p_image;
}

static const GLenum _cube_side_enum[6] = {

	GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
	GL_TEXTURE_CUBE_MAP_POSITIVE_X,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Z,

};

RID RasterizerStorageGLES2::texture_create() {

	Texture *texture = memnew(Texture);
	ERR_FAIL_COND_V(!texture, RID());
	glGenTextures(1, &texture->tex_id);
	texture->active = false;
	texture->total_data_size = 0;

	return texture_owner.make_rid(texture);
}

void RasterizerStorageGLES2::texture_allocate(RID p_texture, int p_width, int p_height, Image::Format p_format, uint32_t p_flags) {
	GLenum format;
	GLenum internal_format;
	GLenum type;

	if (p_flags & VS::TEXTURE_FLAG_USED_FOR_STREAMING) {
		p_flags &= ~VS::TEXTURE_FLAG_MIPMAPS; // no mipies for video
	}

	Texture *texture = texture_owner.getornull(p_texture);
	ERR_FAIL_COND(!texture);
	texture->width = p_width;
	texture->height = p_height;
	texture->format = p_format;
	texture->flags = p_flags;
	texture->stored_cube_sides = 0;
	texture->target = (p_flags & VS::TEXTURE_FLAG_CUBEMAP) ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D;

	_get_gl_image_and_format(Ref<Image>(), texture->format, texture->flags, format, internal_format, type);

	texture->alloc_width = texture->width;
	texture->alloc_height = texture->height;

	texture->gl_format_cache = format;
	texture->gl_type_cache = type;
	texture->gl_internal_format_cache = internal_format;
	texture->data_size = 0;
	texture->mipmaps = 1;

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(texture->target, texture->tex_id);

	if (p_flags & VS::TEXTURE_FLAG_USED_FOR_STREAMING) {
		//prealloc if video
		glTexImage2D(texture->target, 0, internal_format, p_width, p_height, 0, format, type, NULL);
	}

	texture->active = true;
}

void RasterizerStorageGLES2::texture_set_data(RID p_texture, const Ref<Image> &p_image, VS::CubeMapSide p_cube_side) {
	Texture *texture = texture_owner.getornull(p_texture);

	ERR_FAIL_COND(!texture);
	ERR_FAIL_COND(!texture->active);
	ERR_FAIL_COND(texture->render_target);
	ERR_FAIL_COND(texture->format != p_image->get_format());
	ERR_FAIL_COND(p_image.is_null());

	GLenum type;
	GLenum format;
	GLenum internal_format;
	bool compressed = false;
	bool srgb;

	if (config.keep_original_textures && !(texture->flags & VS::TEXTURE_FLAG_USED_FOR_STREAMING)) {
		texture->images[p_cube_side] = p_image;
	}

	Ref<Image> img = _get_gl_image_and_format(p_image, p_image->get_format(), texture->flags, format, internal_format, type);

	if (config.shrink_textures_x2 && (p_image->has_mipmaps() || !p_image->is_compressed()) && !(texture->flags & VS::TEXTURE_FLAG_USED_FOR_STREAMING)) {

		texture->alloc_height = MAX(1, texture->alloc_height / 2);
		texture->alloc_width = MAX(1, texture->alloc_width / 2);

		if (texture->alloc_width == img->get_width() / 2 && texture->alloc_height == img->get_height() / 2) {

			img->shrink_x2();
		} else if (img->get_format() <= Image::FORMAT_RGBA8) {

			img->resize(texture->alloc_width, texture->alloc_height, Image::INTERPOLATE_BILINEAR);
		}
	};

	GLenum blit_target = (texture->target == GL_TEXTURE_CUBE_MAP) ? _cube_side_enum[p_cube_side] : GL_TEXTURE_2D;

	texture->data_size = img->get_data().size();
	PoolVector<uint8_t>::Read read = img->get_data().read();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(texture->target, texture->tex_id);

	texture->ignore_mipmaps = compressed && !img->has_mipmaps();

	if ((texture->flags & VS::TEXTURE_FLAG_MIPMAPS) && !texture->ignore_mipmaps)
		glTexParameteri(texture->target, GL_TEXTURE_MIN_FILTER, config.use_fast_texture_filter ? GL_LINEAR_MIPMAP_NEAREST : GL_LINEAR_MIPMAP_LINEAR);
	else {
		if (texture->flags & VS::TEXTURE_FLAG_FILTER) {
			glTexParameteri(texture->target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		} else {
			glTexParameteri(texture->target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		}
	}

	if (texture->flags & VS::TEXTURE_FLAG_FILTER) {

		glTexParameteri(texture->target, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // Linear Filtering

	} else {

		glTexParameteri(texture->target, GL_TEXTURE_MAG_FILTER, GL_NEAREST); // raw Filtering
	}

	if (((texture->flags & VS::TEXTURE_FLAG_REPEAT) || (texture->flags & VS::TEXTURE_FLAG_MIRRORED_REPEAT)) && texture->target != GL_TEXTURE_CUBE_MAP) {

		if (texture->flags & VS::TEXTURE_FLAG_MIRRORED_REPEAT) {
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
		} else {
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		}
	} else {

		//glTexParameterf( texture->target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
		glTexParameterf(texture->target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(texture->target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

//set swizle for older format compatibility
#ifdef GLES_OVER_GL
	switch (texture->format) {

		case Image::FORMAT_L8: {

		} break;
		case Image::FORMAT_LA8: {

		} break;
		default: {

		} break;
	}
#endif

	int mipmaps = ((texture->flags & VS::TEXTURE_FLAG_MIPMAPS) && img->has_mipmaps()) ? img->get_mipmap_count() + 1 : 1;

	int w = img->get_width();
	int h = img->get_height();

	int tsize = 0;

	for (int i = 0; i < mipmaps; i++) {

		int size, ofs;
		img->get_mipmap_offset_and_size(i, ofs, size);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		if (texture->flags & VS::TEXTURE_FLAG_USED_FOR_STREAMING) {
			glTexSubImage2D(blit_target, i, 0, 0, w, h, format, type, &read[ofs]);
		} else {
			glTexImage2D(blit_target, i, internal_format, w, h, 0, format, type, &read[ofs]);
		}

		tsize += size;

		w = MAX(1, w >> 1);
		h = MAX(1, h >> 1);
	}

	info.texture_mem -= texture->total_data_size;
	texture->total_data_size = tsize;
	info.texture_mem += texture->total_data_size;

	// printf("texture: %i x %i - size: %i - total: %i\n", texture->width, texture->height, tsize, info.texture_mem);

	texture->stored_cube_sides |= (1 << p_cube_side);

	if ((texture->flags & VS::TEXTURE_FLAG_MIPMAPS) && mipmaps == 1 && !texture->ignore_mipmaps && (!(texture->flags & VS::TEXTURE_FLAG_CUBEMAP) || texture->stored_cube_sides == (1 << 6) - 1)) {
		//generate mipmaps if they were requested and the image does not contain them
		glGenerateMipmap(texture->target);
	}

	texture->mipmaps = mipmaps;
}

void RasterizerStorageGLES2::texture_set_data_partial(RID p_texture, const Ref<Image> &p_image, int src_x, int src_y, int src_w, int src_h, int dst_x, int dst_y, int p_dst_mip, VS::CubeMapSide p_cube_side) {
	// TODO
	ERR_PRINT("Not implemented (ask Karroffel to do it :p)");
}

Ref<Image> RasterizerStorageGLES2::texture_get_data(RID p_texture, VS::CubeMapSide p_cube_side) const {

	Texture *texture = texture_owner.getornull(p_texture);

	ERR_FAIL_COND_V(!texture, Ref<Image>());
	ERR_FAIL_COND_V(!texture->active, Ref<Image>());
	ERR_FAIL_COND_V(texture->data_size == 0 && !texture->render_target, Ref<Image>());

	if (!texture->images[p_cube_side].is_null()) {
		return texture->images[p_cube_side];
	}
#ifdef GLES_OVER_GL

	PoolVector<uint8_t> data;

	int data_size = Image::get_image_data_size(texture->alloc_width, texture->alloc_height, texture->format, texture->mipmaps > 1 ? -1 : 0);

	data.resize(data_size * 2); //add some memory at the end, just in case for buggy drivers
	PoolVector<uint8_t>::Write wb = data.write();

	glActiveTexture(GL_TEXTURE0);

	glBindTexture(texture->target, texture->tex_id);

	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

	//print_line("GET FORMAT: " + Image::get_format_name(texture->format) + " mipmaps: " + itos(texture->mipmaps));

	for (int i = 0; i < texture->mipmaps; i++) {

		int ofs = 0;
		if (i > 0) {
			ofs = Image::get_image_data_size(texture->alloc_width, texture->alloc_height, texture->format, i - 1);
		}

		glPixelStorei(GL_PACK_ALIGNMENT, 1);

		glGetTexImage(texture->target, i, texture->gl_format_cache, texture->gl_type_cache, &wb[ofs]);
	}

	wb = PoolVector<uint8_t>::Write();

	data.resize(data_size);

	Image *img = memnew(Image(texture->alloc_width, texture->alloc_height, texture->mipmaps > 1 ? true : false, texture->format, data));

	return Ref<Image>(img);
#else

	ERR_EXPLAIN("Sorry, It's not posible to obtain images back in OpenGL ES");
	return Ref<Image>();
#endif
}

void RasterizerStorageGLES2::texture_set_flags(RID p_texture, uint32_t p_flags) {

	Texture *texture = texture_owner.getornull(p_texture);
	ERR_FAIL_COND(!texture);

	bool had_mipmaps = texture->flags & VS::TEXTURE_FLAG_MIPMAPS;

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(texture->target, texture->tex_id);
	uint32_t cube = texture->flags & VS::TEXTURE_FLAG_CUBEMAP;
	texture->flags = p_flags | cube; // can't remove a cube from being a cube

	if (((texture->flags & VS::TEXTURE_FLAG_REPEAT) || (texture->flags & VS::TEXTURE_FLAG_MIRRORED_REPEAT)) && texture->target != GL_TEXTURE_CUBE_MAP) {

		if (texture->flags & VS::TEXTURE_FLAG_MIRRORED_REPEAT) {
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
		} else {
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		}
	} else {
		//glTexParameterf( texture->target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
		glTexParameterf(texture->target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(texture->target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

	if ((texture->flags & VS::TEXTURE_FLAG_MIPMAPS) && !texture->ignore_mipmaps) {
		if (!had_mipmaps && texture->mipmaps == 1) {
			glGenerateMipmap(texture->target);
		}
		glTexParameteri(texture->target, GL_TEXTURE_MIN_FILTER, config.use_fast_texture_filter ? GL_LINEAR_MIPMAP_NEAREST : GL_LINEAR_MIPMAP_LINEAR);

	} else {
		if (texture->flags & VS::TEXTURE_FLAG_FILTER) {
			glTexParameteri(texture->target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		} else {
			glTexParameteri(texture->target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		}
	}

	if (texture->flags & VS::TEXTURE_FLAG_FILTER) {

		glTexParameteri(texture->target, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // Linear Filtering

	} else {

		glTexParameteri(texture->target, GL_TEXTURE_MAG_FILTER, GL_NEAREST); // raw Filtering
	}
}

uint32_t RasterizerStorageGLES2::texture_get_flags(RID p_texture) const {
	Texture *texture = texture_owner.getornull(p_texture);

	ERR_FAIL_COND_V(!texture, 0);

	return texture->flags;
}

Image::Format RasterizerStorageGLES2::texture_get_format(RID p_texture) const {
	Texture *texture = texture_owner.getornull(p_texture);

	ERR_FAIL_COND_V(!texture, Image::FORMAT_L8);

	return texture->format;
}

uint32_t RasterizerStorageGLES2::texture_get_texid(RID p_texture) const {
	Texture *texture = texture_owner.getornull(p_texture);

	ERR_FAIL_COND_V(!texture, 0);

	return texture->tex_id;
}

uint32_t RasterizerStorageGLES2::texture_get_width(RID p_texture) const {
	Texture *texture = texture_owner.getornull(p_texture);

	ERR_FAIL_COND_V(!texture, 0);

	return texture->width;
}

uint32_t RasterizerStorageGLES2::texture_get_height(RID p_texture) const {
	Texture *texture = texture_owner.getornull(p_texture);

	ERR_FAIL_COND_V(!texture, 0);

	return texture->height;
}

void RasterizerStorageGLES2::texture_set_size_override(RID p_texture, int p_width, int p_height) {
	Texture *texture = texture_owner.getornull(p_texture);

	ERR_FAIL_COND(!texture);
	ERR_FAIL_COND(texture->render_target);

	ERR_FAIL_COND(p_width <= 0 || p_width > 16384);
	ERR_FAIL_COND(p_height <= 0 || p_height > 16384);
	//real texture size is in alloc width and height
	texture->width = p_width;
	texture->height = p_height;
}

void RasterizerStorageGLES2::texture_set_path(RID p_texture, const String &p_path) {
	Texture *texture = texture_owner.getornull(p_texture);
	ERR_FAIL_COND(!texture);

	texture->path = p_path;
}

String RasterizerStorageGLES2::texture_get_path(RID p_texture) const {
	Texture *texture = texture_owner.getornull(p_texture);
	ERR_FAIL_COND_V(!texture, "");

	return texture->path;
}

void RasterizerStorageGLES2::texture_debug_usage(List<VS::TextureInfo> *r_info) {
	List<RID> textures;
	texture_owner.get_owned_list(&textures);

	for (List<RID>::Element *E = textures.front(); E; E = E->next()) {

		Texture *t = texture_owner.getornull(E->get());
		if (!t)
			continue;
		VS::TextureInfo tinfo;
		tinfo.path = t->path;
		tinfo.format = t->format;
		tinfo.size.x = t->alloc_width;
		tinfo.size.y = t->alloc_height;
		tinfo.bytes = t->total_data_size;
		r_info->push_back(tinfo);
	}
}

void RasterizerStorageGLES2::texture_set_shrink_all_x2_on_set_data(bool p_enable) {
	config.shrink_textures_x2 = p_enable;
}

void RasterizerStorageGLES2::textures_keep_original(bool p_enable) {
	config.keep_original_textures = p_enable;
}

void RasterizerStorageGLES2::texture_set_proxy(RID p_texture, RID p_proxy) {
	Texture *texture = texture_owner.getornull(p_texture);
	ERR_FAIL_COND(!texture);

	if (texture->proxy) {
		texture->proxy->proxy_owners.erase(texture);
		texture->proxy = NULL;
	}

	if (p_proxy.is_valid()) {
		Texture *proxy = texture_owner.get(p_proxy);
		ERR_FAIL_COND(!proxy);
		ERR_FAIL_COND(proxy == texture);
		proxy->proxy_owners.insert(texture);
		texture->proxy = proxy;
	}
}

void RasterizerStorageGLES2::texture_set_detect_3d_callback(RID p_texture, VisualServer::TextureDetectCallback p_callback, void *p_userdata) {
	// TODO
}

void RasterizerStorageGLES2::texture_set_detect_srgb_callback(RID p_texture, VisualServer::TextureDetectCallback p_callback, void *p_userdata) {
	// TODO
}

void RasterizerStorageGLES2::texture_set_detect_normal_callback(RID p_texture, VisualServer::TextureDetectCallback p_callback, void *p_userdata) {
	// TODO
}

RID RasterizerStorageGLES2::texture_create_radiance_cubemap(RID p_source, int p_resolution) const {
	// TODO
	return RID();
}

RID RasterizerStorageGLES2::sky_create() {
	return RID();
}

void RasterizerStorageGLES2::sky_set_texture(RID p_sky, RID p_panorama, int p_radiance_size) {
}

/* SHADER API */

RID RasterizerStorageGLES2::shader_create() {

	Shader *shader = memnew(Shader);
	shader->mode = VS::SHADER_SPATIAL;
	shader->shader = &scene->state.scene_shader;
	RID rid = shader_owner.make_rid(shader);
	_shader_make_dirty(shader);
	shader->self = rid;

	return rid;
}

void RasterizerStorageGLES2::_shader_make_dirty(Shader *p_shader) {
	if (p_shader->dirty_list.in_list())
		return;

	_shader_dirty_list.add(&p_shader->dirty_list);
}

void RasterizerStorageGLES2::shader_set_code(RID p_shader, const String &p_code) {

	Shader *shader = shader_owner.getornull(p_shader);
	ERR_FAIL_COND(!shader);

	shader->code = p_code;

	String mode_string = ShaderLanguage::get_shader_type(p_code);
	VS::ShaderMode mode;

	if (mode_string == "canvas_item")
		mode = VS::SHADER_CANVAS_ITEM;
	else if (mode_string == "particles")
		mode = VS::SHADER_PARTICLES;
	else
		mode = VS::SHADER_SPATIAL;

	if (shader->custom_code_id && mode != shader->mode) {
		shader->shader->free_custom_shader(shader->custom_code_id);
		shader->custom_code_id = 0;
	}

	shader->mode = mode;

	// TODO handle all shader types
	if (mode == VS::SHADER_CANVAS_ITEM) {
		shader->shader = &canvas->state.canvas_shader;

	} else {
		return;
	}

	if (shader->custom_code_id == 0) {
		shader->custom_code_id = shader->shader->create_custom_shader();
	}

	_shader_make_dirty(shader);
}

String RasterizerStorageGLES2::shader_get_code(RID p_shader) const {

	const Shader *shader = shader_owner.get(p_shader);
	ERR_FAIL_COND_V(!shader, "");

	return shader->code;
}

void RasterizerStorageGLES2::_update_shader(Shader *p_shader) const {

	_shader_dirty_list.remove(&p_shader->dirty_list);

	p_shader->valid = false;

	p_shader->uniforms.clear();

	ShaderCompilerGLES2::GeneratedCode gen_code;
	ShaderCompilerGLES2::IdentifierActions *actions = NULL;

	switch (p_shader->mode) {

			// TODO

		case VS::SHADER_CANVAS_ITEM: {

			p_shader->canvas_item.blend_mode = Shader::CanvasItem::BLEND_MODE_MIX;

			p_shader->canvas_item.uses_screen_texture = false;
			p_shader->canvas_item.uses_screen_uv = false;
			p_shader->canvas_item.uses_time = false;

			shaders.actions_canvas.render_mode_values["blend_add"] = Pair<int *, int>(&p_shader->canvas_item.blend_mode, Shader::CanvasItem::BLEND_MODE_ADD);
			shaders.actions_canvas.render_mode_values["blend_mix"] = Pair<int *, int>(&p_shader->canvas_item.blend_mode, Shader::CanvasItem::BLEND_MODE_MIX);
			shaders.actions_canvas.render_mode_values["blend_sub"] = Pair<int *, int>(&p_shader->canvas_item.blend_mode, Shader::CanvasItem::BLEND_MODE_SUB);
			shaders.actions_canvas.render_mode_values["blend_mul"] = Pair<int *, int>(&p_shader->canvas_item.blend_mode, Shader::CanvasItem::BLEND_MODE_MUL);
			shaders.actions_canvas.render_mode_values["blend_premul_alpha"] = Pair<int *, int>(&p_shader->canvas_item.blend_mode, Shader::CanvasItem::BLEND_MODE_PMALPHA);

			// shaders.actions_canvas.render_mode_values["unshaded"] = Pair<int *, int>(&p_shader->canvas_item.light_mode, Shader::CanvasItem::LIGHT_MODE_UNSHADED);
			// shaders.actions_canvas.render_mode_values["light_only"] = Pair<int *, int>(&p_shader->canvas_item.light_mode, Shader::CanvasItem::LIGHT_MODE_LIGHT_ONLY);

			shaders.actions_canvas.usage_flag_pointers["SCREEN_UV"] = &p_shader->canvas_item.uses_screen_uv;
			shaders.actions_canvas.usage_flag_pointers["SCREEN_PIXEL_SIZE"] = &p_shader->canvas_item.uses_screen_uv;
			shaders.actions_canvas.usage_flag_pointers["SCREEN_TEXTURE"] = &p_shader->canvas_item.uses_screen_texture;
			shaders.actions_canvas.usage_flag_pointers["TIME"] = &p_shader->canvas_item.uses_time;

			actions = &shaders.actions_canvas;
			actions->uniforms = &p_shader->uniforms;
		} break;

		default: {
			return;
		} break;
	}

	Error err = shaders.compiler.compile(p_shader->mode, p_shader->code, actions, p_shader->path, gen_code);

	ERR_FAIL_COND(err != OK);

	p_shader->shader->set_custom_shader_code(p_shader->custom_code_id, gen_code.vertex, gen_code.vertex_global, gen_code.fragment, gen_code.light, gen_code.fragment_global, gen_code.uniforms, gen_code.texture_uniforms, gen_code.custom_defines);

	p_shader->texture_count = gen_code.texture_uniforms.size();
	p_shader->texture_hints = gen_code.texture_hints;

	p_shader->uses_vertex_time = gen_code.uses_vertex_time;
	p_shader->uses_fragment_time = gen_code.uses_fragment_time;

	for (SelfList<Material> *E = p_shader->materials.first(); E; E = E->next()) {
		_material_make_dirty(E->self());
	}

	p_shader->valid = true;
	p_shader->version++;
}

void RasterizerStorageGLES2::update_dirty_shaders() {
	while (_shader_dirty_list.first()) {
		_update_shader(_shader_dirty_list.first()->self());
	}
}

void RasterizerStorageGLES2::shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const {

	Shader *shader = shader_owner.get(p_shader);
	ERR_FAIL_COND(!shader);

	if (shader->dirty_list.in_list()) {
		_update_shader(shader);
	}

	Map<int, StringName> order;

	for (Map<StringName, ShaderLanguage::ShaderNode::Uniform>::Element *E = shader->uniforms.front(); E; E = E->next()) {

		if (E->get().texture_order >= 0) {
			order[E->get().texture_order + 100000] = E->key();
		} else {
			order[E->get().order] = E->key();
		}
	}

	for (Map<int, StringName>::Element *E = order.front(); E; E = E->next()) {

		PropertyInfo pi;
		ShaderLanguage::ShaderNode::Uniform &u = shader->uniforms[E->get()];

		pi.name = E->get();

		switch (u.type) {
			case ShaderLanguage::TYPE_VOID: {
				pi.type = Variant::NIL;
			} break;

			case ShaderLanguage::TYPE_BOOL: {
				pi.type = Variant::BOOL;
			} break;

			// bool vectors
			case ShaderLanguage::TYPE_BVEC2: {
				pi.type = Variant::INT;
				pi.hint = PROPERTY_HINT_FLAGS;
				pi.hint_string = "x,y";
			} break;
			case ShaderLanguage::TYPE_BVEC3: {
				pi.type = Variant::INT;
				pi.hint = PROPERTY_HINT_FLAGS;
				pi.hint_string = "x,y,z";
			} break;
			case ShaderLanguage::TYPE_BVEC4: {
				pi.type = Variant::INT;
				pi.hint = PROPERTY_HINT_FLAGS;
				pi.hint_string = "x,y,z,w";
			} break;

				// int stuff
			case ShaderLanguage::TYPE_UINT:
			case ShaderLanguage::TYPE_INT: {
				pi.type = Variant::INT;

				if (u.hint == ShaderLanguage::ShaderNode::Uniform::HINT_RANGE) {
					pi.hint = PROPERTY_HINT_RANGE;
					pi.hint_string = rtos(u.hint_range[0]) + "," + rtos(u.hint_range[1]);
				}
			} break;

			case ShaderLanguage::TYPE_IVEC2:
			case ShaderLanguage::TYPE_UVEC2:
			case ShaderLanguage::TYPE_IVEC3:
			case ShaderLanguage::TYPE_UVEC3:
			case ShaderLanguage::TYPE_IVEC4:
			case ShaderLanguage::TYPE_UVEC4: {
				pi.type = Variant::POOL_INT_ARRAY;
			} break;

			case ShaderLanguage::TYPE_VEC2: {
				pi.type = Variant::VECTOR2;
			} break;
			case ShaderLanguage::TYPE_VEC3: {
				pi.type = Variant::VECTOR3;
			} break;

			case ShaderLanguage::TYPE_VEC4: {
				if (u.hint == ShaderLanguage::ShaderNode::Uniform::HINT_COLOR) {
					pi.type = Variant::COLOR;
				} else {
					pi.type = Variant::PLANE;
				}
			} break;

			case ShaderLanguage::TYPE_MAT2: {
				pi.type = Variant::TRANSFORM2D;
			} break;

			case ShaderLanguage::TYPE_MAT3: {
				pi.type = Variant::BASIS;
			} break;

			case ShaderLanguage::TYPE_MAT4: {
				pi.type = Variant::TRANSFORM;
			} break;

			case ShaderLanguage::TYPE_SAMPLER2D:
			case ShaderLanguage::TYPE_ISAMPLER2D:
			case ShaderLanguage::TYPE_USAMPLER2D: {
				pi.type = Variant::OBJECT;
				pi.hint = PROPERTY_HINT_RESOURCE_TYPE;
				pi.hint_string = "Texture";
			} break;

			case ShaderLanguage::TYPE_SAMPLERCUBE: {
				pi.type = Variant::OBJECT;
				pi.hint = PROPERTY_HINT_RESOURCE_TYPE;
				pi.hint_string = "CubeMap";
			} break;
		}

		p_param_list->push_back(pi);
	}
}

void RasterizerStorageGLES2::shader_set_default_texture_param(RID p_shader, const StringName &p_name, RID p_texture) {

	Shader *shader = shader_owner.get(p_shader);
	ERR_FAIL_COND(!shader);
	ERR_FAIL_COND(p_texture.is_valid() && !texture_owner.owns(p_texture));

	if (p_texture.is_valid()) {
		shader->default_textures[p_name] = p_texture;
	} else {
		shader->default_textures.erase(p_name);
	}

	_shader_make_dirty(shader);
}

RID RasterizerStorageGLES2::shader_get_default_texture_param(RID p_shader, const StringName &p_name) const {

	const Shader *shader = shader_owner.get(p_shader);
	ERR_FAIL_COND_V(!shader, RID());

	const Map<StringName, RID>::Element *E = shader->default_textures.find(p_name);

	if (!E) {
		return RID();
	}

	return E->get();
}

/* COMMON MATERIAL API */

void RasterizerStorageGLES2::_material_make_dirty(Material *p_material) const {

	if (p_material->dirty_list.in_list())
		return;

	_material_dirty_list.add(&p_material->dirty_list);
}

RID RasterizerStorageGLES2::material_create() {

	Material *material = memnew(Material);

	return material_owner.make_rid(material);
}

void RasterizerStorageGLES2::material_set_shader(RID p_material, RID p_shader) {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND(!material);

	Shader *shader = shader_owner.getornull(p_shader);

	if (material->shader) {
		// if a shader is present, remove the old shader
		material->shader->materials.remove(&material->list);
	}

	material->shader = shader;

	if (shader) {
		shader->materials.add(&material->list);
	}

	_material_make_dirty(material);
}

RID RasterizerStorageGLES2::material_get_shader(RID p_material) const {

	const Material *material = material_owner.get(p_material);
	ERR_FAIL_COND_V(!material, RID());

	if (material->shader) {
		return material->shader->self;
	}

	return RID();
}

void RasterizerStorageGLES2::material_set_param(RID p_material, const StringName &p_param, const Variant &p_value) {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND(!material);

	if (p_value.get_type() == Variant::NIL) {
		material->params.erase(p_param);
	} else {
		material->params[p_param] = p_value;
	}

	_material_make_dirty(material);
}

Variant RasterizerStorageGLES2::material_get_param(RID p_material, const StringName &p_param) const {

	const Material *material = material_owner.get(p_material);
	ERR_FAIL_COND_V(!material, RID());

	if (material->params.has(p_param)) {
		return material->params[p_param];
	}

	return Variant();
}

void RasterizerStorageGLES2::material_set_line_width(RID p_material, float p_width) {
}

void RasterizerStorageGLES2::material_set_next_pass(RID p_material, RID p_next_material) {
}

bool RasterizerStorageGLES2::material_is_animated(RID p_material) {
	return false;
}

bool RasterizerStorageGLES2::material_casts_shadows(RID p_material) {
	return false;
}

void RasterizerStorageGLES2::material_add_instance_owner(RID p_material, RasterizerScene::InstanceBase *p_instance) {
}

void RasterizerStorageGLES2::material_remove_instance_owner(RID p_material, RasterizerScene::InstanceBase *p_instance) {
}

void RasterizerStorageGLES2::material_set_render_priority(RID p_material, int priority) {
}

void RasterizerStorageGLES2::update_dirty_materials() {
}

/* MESH API */

RID RasterizerStorageGLES2::mesh_create() {
	return RID();
}

void RasterizerStorageGLES2::mesh_add_surface(RID p_mesh, uint32_t p_format, VS::PrimitiveType p_primitive, const PoolVector<uint8_t> &p_array, int p_vertex_count, const PoolVector<uint8_t> &p_index_array, int p_index_count, const AABB &p_aabb, const Vector<PoolVector<uint8_t> > &p_blend_shapes, const Vector<AABB> &p_bone_aabbs) {
}

void RasterizerStorageGLES2::mesh_set_blend_shape_count(RID p_mesh, int p_amount) {
}

int RasterizerStorageGLES2::mesh_get_blend_shape_count(RID p_mesh) const {
	return 0;
}

void RasterizerStorageGLES2::mesh_set_blend_shape_mode(RID p_mesh, VS::BlendShapeMode p_mode) {
}

VS::BlendShapeMode RasterizerStorageGLES2::mesh_get_blend_shape_mode(RID p_mesh) const {
	return VS::BLEND_SHAPE_MODE_NORMALIZED;
}

void RasterizerStorageGLES2::mesh_surface_update_region(RID p_mesh, int p_surface, int p_offset, const PoolVector<uint8_t> &p_data) {
}

void RasterizerStorageGLES2::mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material) {
}

RID RasterizerStorageGLES2::mesh_surface_get_material(RID p_mesh, int p_surface) const {
	return RID();
}

int RasterizerStorageGLES2::mesh_surface_get_array_len(RID p_mesh, int p_surface) const {
	return 0;
}

int RasterizerStorageGLES2::mesh_surface_get_array_index_len(RID p_mesh, int p_surface) const {
	return 0;
}

PoolVector<uint8_t> RasterizerStorageGLES2::mesh_surface_get_array(RID p_mesh, int p_surface) const {
	return PoolVector<uint8_t>();
}

PoolVector<uint8_t> RasterizerStorageGLES2::mesh_surface_get_index_array(RID p_mesh, int p_surface) const {
	return PoolVector<uint8_t>();
}

uint32_t RasterizerStorageGLES2::mesh_surface_get_format(RID p_mesh, int p_surface) const {
	return 0;
}

VS::PrimitiveType RasterizerStorageGLES2::mesh_surface_get_primitive_type(RID p_mesh, int p_surface) const {
	return VS::PRIMITIVE_TRIANGLES;
}

AABB RasterizerStorageGLES2::mesh_surface_get_aabb(RID p_mesh, int p_surface) const {
	return AABB();
}

Vector<PoolVector<uint8_t> > RasterizerStorageGLES2::mesh_surface_get_blend_shapes(RID p_mesh, int p_surface) const {
	return Vector<PoolVector<uint8_t> >();
}
Vector<AABB> RasterizerStorageGLES2::mesh_surface_get_skeleton_aabb(RID p_mesh, int p_surface) const {
	return Vector<AABB>();
}

void RasterizerStorageGLES2::mesh_remove_surface(RID p_mesh, int p_surface) {
}

int RasterizerStorageGLES2::mesh_get_surface_count(RID p_mesh) const {
	return 0;
}

void RasterizerStorageGLES2::mesh_set_custom_aabb(RID p_mesh, const AABB &p_aabb) {
}

AABB RasterizerStorageGLES2::mesh_get_custom_aabb(RID p_mesh) const {
	return AABB();
}

AABB RasterizerStorageGLES2::mesh_get_aabb(RID p_mesh, RID p_skeleton) const {
	return AABB();
}
void RasterizerStorageGLES2::mesh_clear(RID p_mesh) {
}

/* MULTIMESH API */

RID RasterizerStorageGLES2::multimesh_create() {
	return RID();
}

void RasterizerStorageGLES2::multimesh_allocate(RID p_multimesh, int p_instances, VS::MultimeshTransformFormat p_transform_format, VS::MultimeshColorFormat p_color_format) {
}

int RasterizerStorageGLES2::multimesh_get_instance_count(RID p_multimesh) const {
	return 0;
}

void RasterizerStorageGLES2::multimesh_set_mesh(RID p_multimesh, RID p_mesh) {
}

void RasterizerStorageGLES2::multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform &p_transform) {
}

void RasterizerStorageGLES2::multimesh_instance_set_transform_2d(RID p_multimesh, int p_index, const Transform2D &p_transform) {
}

void RasterizerStorageGLES2::multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color) {
}

RID RasterizerStorageGLES2::multimesh_get_mesh(RID p_multimesh) const {
	return RID();
}

Transform RasterizerStorageGLES2::multimesh_instance_get_transform(RID p_multimesh, int p_index) const {
	return Transform();
}

Transform2D RasterizerStorageGLES2::multimesh_instance_get_transform_2d(RID p_multimesh, int p_index) const {
	return Transform2D();
}

Color RasterizerStorageGLES2::multimesh_instance_get_color(RID p_multimesh, int p_index) const {
	return Color();
}

void RasterizerStorageGLES2::multimesh_set_visible_instances(RID p_multimesh, int p_visible) {
}

int RasterizerStorageGLES2::multimesh_get_visible_instances(RID p_multimesh) const {
	return 0;
}

AABB RasterizerStorageGLES2::multimesh_get_aabb(RID p_multimesh) const {

	return AABB();
}

void RasterizerStorageGLES2::update_dirty_multimeshes() {
}

/* IMMEDIATE API */

RID RasterizerStorageGLES2::immediate_create() {
	return RID();
}

void RasterizerStorageGLES2::immediate_begin(RID p_immediate, VS::PrimitiveType p_rimitive, RID p_texture) {
}

void RasterizerStorageGLES2::immediate_vertex(RID p_immediate, const Vector3 &p_vertex) {
}

void RasterizerStorageGLES2::immediate_normal(RID p_immediate, const Vector3 &p_normal) {
}

void RasterizerStorageGLES2::immediate_tangent(RID p_immediate, const Plane &p_tangent) {
}

void RasterizerStorageGLES2::immediate_color(RID p_immediate, const Color &p_color) {
}

void RasterizerStorageGLES2::immediate_uv(RID p_immediate, const Vector2 &tex_uv) {
}

void RasterizerStorageGLES2::immediate_uv2(RID p_immediate, const Vector2 &tex_uv) {
}

void RasterizerStorageGLES2::immediate_end(RID p_immediate) {
}

void RasterizerStorageGLES2::immediate_clear(RID p_immediate) {
}

AABB RasterizerStorageGLES2::immediate_get_aabb(RID p_immediate) const {
	return AABB();
}

void RasterizerStorageGLES2::immediate_set_material(RID p_immediate, RID p_material) {
}

RID RasterizerStorageGLES2::immediate_get_material(RID p_immediate) const {
	return RID();
}

/* SKELETON API */

RID RasterizerStorageGLES2::skeleton_create() {
	return RID();
}

void RasterizerStorageGLES2::skeleton_allocate(RID p_skeleton, int p_bones, bool p_2d_skeleton) {
}

int RasterizerStorageGLES2::skeleton_get_bone_count(RID p_skeleton) const {
	return 0;
}

void RasterizerStorageGLES2::skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform &p_transform) {
}

Transform RasterizerStorageGLES2::skeleton_bone_get_transform(RID p_skeleton, int p_bone) const {
	return Transform();
}
void RasterizerStorageGLES2::skeleton_bone_set_transform_2d(RID p_skeleton, int p_bone, const Transform2D &p_transform) {
}

Transform2D RasterizerStorageGLES2::skeleton_bone_get_transform_2d(RID p_skeleton, int p_bone) const {
	return Transform2D();
}

void RasterizerStorageGLES2::skeleton_set_base_transform_2d(RID p_skeleton, const Transform2D &p_base_transform) {

}

void RasterizerStorageGLES2::update_dirty_skeletons() {
}

/* Light API */

RID RasterizerStorageGLES2::light_create(VS::LightType p_type) {
	return RID();
}

void RasterizerStorageGLES2::light_set_color(RID p_light, const Color &p_color) {
}

void RasterizerStorageGLES2::light_set_param(RID p_light, VS::LightParam p_param, float p_value) {
}

void RasterizerStorageGLES2::light_set_shadow(RID p_light, bool p_enabled) {
}

void RasterizerStorageGLES2::light_set_shadow_color(RID p_light, const Color &p_color) {
}

void RasterizerStorageGLES2::light_set_projector(RID p_light, RID p_texture) {
}

void RasterizerStorageGLES2::light_set_negative(RID p_light, bool p_enable) {
}

void RasterizerStorageGLES2::light_set_cull_mask(RID p_light, uint32_t p_mask) {
}

void RasterizerStorageGLES2::light_set_reverse_cull_face_mode(RID p_light, bool p_enabled) {
}

void RasterizerStorageGLES2::light_omni_set_shadow_mode(RID p_light, VS::LightOmniShadowMode p_mode) {
}

VS::LightOmniShadowMode RasterizerStorageGLES2::light_omni_get_shadow_mode(RID p_light) {
	return VS::LIGHT_OMNI_SHADOW_DUAL_PARABOLOID;
}

void RasterizerStorageGLES2::light_omni_set_shadow_detail(RID p_light, VS::LightOmniShadowDetail p_detail) {
}

void RasterizerStorageGLES2::light_directional_set_shadow_mode(RID p_light, VS::LightDirectionalShadowMode p_mode) {
}

void RasterizerStorageGLES2::light_directional_set_blend_splits(RID p_light, bool p_enable) {
}

bool RasterizerStorageGLES2::light_directional_get_blend_splits(RID p_light) const {
	return false;
}

VS::LightDirectionalShadowMode RasterizerStorageGLES2::light_directional_get_shadow_mode(RID p_light) {
	return VS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL;
}

void RasterizerStorageGLES2::light_directional_set_shadow_depth_range_mode(RID p_light, VS::LightDirectionalShadowDepthRangeMode p_range_mode) {
}

VS::LightDirectionalShadowDepthRangeMode RasterizerStorageGLES2::light_directional_get_shadow_depth_range_mode(RID p_light) const {
	return VS::LIGHT_DIRECTIONAL_SHADOW_DEPTH_RANGE_STABLE;
}

VS::LightType RasterizerStorageGLES2::light_get_type(RID p_light) const {
	return VS::LIGHT_DIRECTIONAL;
}

float RasterizerStorageGLES2::light_get_param(RID p_light, VS::LightParam p_param) {

	return VS::LIGHT_DIRECTIONAL;
}

Color RasterizerStorageGLES2::light_get_color(RID p_light) {
	return Color();
}

bool RasterizerStorageGLES2::light_has_shadow(RID p_light) const {

	return VS::LIGHT_DIRECTIONAL;
}

uint64_t RasterizerStorageGLES2::light_get_version(RID p_light) const {
	return 0;
}

AABB RasterizerStorageGLES2::light_get_aabb(RID p_light) const {
	return AABB();
}

/* PROBE API */

RID RasterizerStorageGLES2::reflection_probe_create() {
	return RID();
}

void RasterizerStorageGLES2::reflection_probe_set_update_mode(RID p_probe, VS::ReflectionProbeUpdateMode p_mode) {
}

void RasterizerStorageGLES2::reflection_probe_set_intensity(RID p_probe, float p_intensity) {
}

void RasterizerStorageGLES2::reflection_probe_set_interior_ambient(RID p_probe, const Color &p_ambient) {
}

void RasterizerStorageGLES2::reflection_probe_set_interior_ambient_energy(RID p_probe, float p_energy) {
}

void RasterizerStorageGLES2::reflection_probe_set_interior_ambient_probe_contribution(RID p_probe, float p_contrib) {
}

void RasterizerStorageGLES2::reflection_probe_set_max_distance(RID p_probe, float p_distance) {
}

void RasterizerStorageGLES2::reflection_probe_set_extents(RID p_probe, const Vector3 &p_extents) {
}

void RasterizerStorageGLES2::reflection_probe_set_origin_offset(RID p_probe, const Vector3 &p_offset) {
}

void RasterizerStorageGLES2::reflection_probe_set_as_interior(RID p_probe, bool p_enable) {
}

void RasterizerStorageGLES2::reflection_probe_set_enable_box_projection(RID p_probe, bool p_enable) {
}

void RasterizerStorageGLES2::reflection_probe_set_enable_shadows(RID p_probe, bool p_enable) {
}

void RasterizerStorageGLES2::reflection_probe_set_cull_mask(RID p_probe, uint32_t p_layers) {
}

AABB RasterizerStorageGLES2::reflection_probe_get_aabb(RID p_probe) const {
	return AABB();
}
VS::ReflectionProbeUpdateMode RasterizerStorageGLES2::reflection_probe_get_update_mode(RID p_probe) const {
	return VS::REFLECTION_PROBE_UPDATE_ALWAYS;
}

uint32_t RasterizerStorageGLES2::reflection_probe_get_cull_mask(RID p_probe) const {
	return 0;
}

Vector3 RasterizerStorageGLES2::reflection_probe_get_extents(RID p_probe) const {
	return Vector3();
}
Vector3 RasterizerStorageGLES2::reflection_probe_get_origin_offset(RID p_probe) const {
	return Vector3();
}

bool RasterizerStorageGLES2::reflection_probe_renders_shadows(RID p_probe) const {
	return false;
}

float RasterizerStorageGLES2::reflection_probe_get_origin_max_distance(RID p_probe) const {
	return 0;
}

RID RasterizerStorageGLES2::gi_probe_create() {
	return RID();
}

void RasterizerStorageGLES2::gi_probe_set_bounds(RID p_probe, const AABB &p_bounds) {
}

AABB RasterizerStorageGLES2::gi_probe_get_bounds(RID p_probe) const {
	return AABB();
}

void RasterizerStorageGLES2::gi_probe_set_cell_size(RID p_probe, float p_size) {
}

float RasterizerStorageGLES2::gi_probe_get_cell_size(RID p_probe) const {
	return 0.0;
}

void RasterizerStorageGLES2::gi_probe_set_to_cell_xform(RID p_probe, const Transform &p_xform) {
}

Transform RasterizerStorageGLES2::gi_probe_get_to_cell_xform(RID p_probe) const {
	return Transform();
}

void RasterizerStorageGLES2::gi_probe_set_dynamic_data(RID p_probe, const PoolVector<int> &p_data) {
}

PoolVector<int> RasterizerStorageGLES2::gi_probe_get_dynamic_data(RID p_probe) const {
	return PoolVector<int>();
}

void RasterizerStorageGLES2::gi_probe_set_dynamic_range(RID p_probe, int p_range) {
}

int RasterizerStorageGLES2::gi_probe_get_dynamic_range(RID p_probe) const {
	return 0;
}

void RasterizerStorageGLES2::gi_probe_set_energy(RID p_probe, float p_range) {
}

void RasterizerStorageGLES2::gi_probe_set_bias(RID p_probe, float p_range) {
}

void RasterizerStorageGLES2::gi_probe_set_normal_bias(RID p_probe, float p_range) {
}

void RasterizerStorageGLES2::gi_probe_set_propagation(RID p_probe, float p_range) {
}

void RasterizerStorageGLES2::gi_probe_set_interior(RID p_probe, bool p_enable) {
}

bool RasterizerStorageGLES2::gi_probe_is_interior(RID p_probe) const {
	return false;
}

void RasterizerStorageGLES2::gi_probe_set_compress(RID p_probe, bool p_enable) {
}

bool RasterizerStorageGLES2::gi_probe_is_compressed(RID p_probe) const {
	return false;
}
float RasterizerStorageGLES2::gi_probe_get_energy(RID p_probe) const {
	return 0;
}

float RasterizerStorageGLES2::gi_probe_get_bias(RID p_probe) const {
	return 0;
}

float RasterizerStorageGLES2::gi_probe_get_normal_bias(RID p_probe) const {
	return 0;
}

float RasterizerStorageGLES2::gi_probe_get_propagation(RID p_probe) const {
	return 0;
}

uint32_t RasterizerStorageGLES2::gi_probe_get_version(RID p_probe) {
	return 0;
}

RasterizerStorage::GIProbeCompression RasterizerStorageGLES2::gi_probe_get_dynamic_data_get_preferred_compression() const {
	return GI_PROBE_UNCOMPRESSED;
}

RID RasterizerStorageGLES2::gi_probe_dynamic_data_create(int p_width, int p_height, int p_depth, GIProbeCompression p_compression) {
	return RID();
}

void RasterizerStorageGLES2::gi_probe_dynamic_data_update(RID p_gi_probe_data, int p_depth_slice, int p_slice_count, int p_mipmap, const void *p_data) {
}

///////

RID RasterizerStorageGLES2::lightmap_capture_create() {
	return RID();
}

void RasterizerStorageGLES2::lightmap_capture_set_bounds(RID p_capture, const AABB &p_bounds) {
}

AABB RasterizerStorageGLES2::lightmap_capture_get_bounds(RID p_capture) const {
	return AABB();
}

void RasterizerStorageGLES2::lightmap_capture_set_octree(RID p_capture, const PoolVector<uint8_t> &p_octree) {
}

PoolVector<uint8_t> RasterizerStorageGLES2::lightmap_capture_get_octree(RID p_capture) const {
	return PoolVector<uint8_t>();
}

void RasterizerStorageGLES2::lightmap_capture_set_octree_cell_transform(RID p_capture, const Transform &p_xform) {
}

Transform RasterizerStorageGLES2::lightmap_capture_get_octree_cell_transform(RID p_capture) const {
	return Transform();
}

void RasterizerStorageGLES2::lightmap_capture_set_octree_cell_subdiv(RID p_capture, int p_subdiv) {
}

int RasterizerStorageGLES2::lightmap_capture_get_octree_cell_subdiv(RID p_capture) const {
	return 0;
}

void RasterizerStorageGLES2::lightmap_capture_set_energy(RID p_capture, float p_energy) {
}

float RasterizerStorageGLES2::lightmap_capture_get_energy(RID p_capture) const {
	return 0.0;
}

const PoolVector<RasterizerStorage::LightmapCaptureOctree> *RasterizerStorageGLES2::lightmap_capture_get_octree_ptr(RID p_capture) const {
	return NULL;
}

///////

RID RasterizerStorageGLES2::particles_create() {
	return RID();
}

void RasterizerStorageGLES2::particles_set_emitting(RID p_particles, bool p_emitting) {
}

bool RasterizerStorageGLES2::particles_get_emitting(RID p_particles) {
	return false;
}

void RasterizerStorageGLES2::particles_set_amount(RID p_particles, int p_amount) {
}

void RasterizerStorageGLES2::particles_set_lifetime(RID p_particles, float p_lifetime) {
}

void RasterizerStorageGLES2::particles_set_one_shot(RID p_particles, bool p_one_shot) {
}

void RasterizerStorageGLES2::particles_set_pre_process_time(RID p_particles, float p_time) {
}

void RasterizerStorageGLES2::particles_set_explosiveness_ratio(RID p_particles, float p_ratio) {
}

void RasterizerStorageGLES2::particles_set_randomness_ratio(RID p_particles, float p_ratio) {
}

void RasterizerStorageGLES2::particles_set_custom_aabb(RID p_particles, const AABB &p_aabb) {
}

void RasterizerStorageGLES2::particles_set_speed_scale(RID p_particles, float p_scale) {
}

void RasterizerStorageGLES2::particles_set_use_local_coordinates(RID p_particles, bool p_enable) {
}

void RasterizerStorageGLES2::particles_set_fixed_fps(RID p_particles, int p_fps) {
}

void RasterizerStorageGLES2::particles_set_fractional_delta(RID p_particles, bool p_enable) {
}

void RasterizerStorageGLES2::particles_set_process_material(RID p_particles, RID p_material) {
}

void RasterizerStorageGLES2::particles_set_draw_order(RID p_particles, VS::ParticlesDrawOrder p_order) {
}

void RasterizerStorageGLES2::particles_set_draw_passes(RID p_particles, int p_passes) {
}

void RasterizerStorageGLES2::particles_set_draw_pass_mesh(RID p_particles, int p_pass, RID p_mesh) {
}

void RasterizerStorageGLES2::particles_restart(RID p_particles) {
}

void RasterizerStorageGLES2::particles_request_process(RID p_particles) {
}

AABB RasterizerStorageGLES2::particles_get_current_aabb(RID p_particles) {
	return AABB();
}

AABB RasterizerStorageGLES2::particles_get_aabb(RID p_particles) const {
	return AABB();
}

void RasterizerStorageGLES2::particles_set_emission_transform(RID p_particles, const Transform &p_transform) {
}

int RasterizerStorageGLES2::particles_get_draw_passes(RID p_particles) const {
	return 0;
}

RID RasterizerStorageGLES2::particles_get_draw_pass_mesh(RID p_particles, int p_pass) const {
	return RID();
}

void RasterizerStorageGLES2::update_particles() {
}

////////

void RasterizerStorageGLES2::instance_add_skeleton(RID p_skeleton, RasterizerScene::InstanceBase *p_instance) {
}

void RasterizerStorageGLES2::instance_remove_skeleton(RID p_skeleton, RasterizerScene::InstanceBase *p_instance) {
}

void RasterizerStorageGLES2::instance_add_dependency(RID p_base, RasterizerScene::InstanceBase *p_instance) {
}

void RasterizerStorageGLES2::instance_remove_dependency(RID p_base, RasterizerScene::InstanceBase *p_instance) {
}

/* RENDER TARGET */

void RasterizerStorageGLES2::_render_target_allocate(RenderTarget *rt) {

	if (rt->width <= 0 || rt->height <= 0)
		return;

	Texture *texture = texture_owner.getornull(rt->texture);
	ERR_FAIL_COND(!texture);

	// create fbo

	glGenFramebuffers(1, &rt->fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, rt->fbo);

	// color

	glGenTextures(1, &rt->color);
	glBindTexture(GL_TEXTURE_2D, rt->color);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rt->width, rt->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	if (texture->flags & VS::TEXTURE_FLAG_FILTER) {

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	} else {

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	}
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt->color, 0);

	// depth

	glGenRenderbuffers(1, &rt->depth);
	glBindRenderbuffer(GL_RENDERBUFFER, rt->depth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, rt->width, rt->height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rt->depth);

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

	if (status != GL_FRAMEBUFFER_COMPLETE) {

		glDeleteRenderbuffers(1, &rt->fbo);
		glDeleteTextures(1, &rt->depth);
		glDeleteTextures(1, &rt->color);
		rt->fbo = 0;
		rt->width = 0;
		rt->height = 0;
		rt->color = 0;
		rt->depth = 0;
		texture->tex_id = 0;
		texture->active = false;
		WARN_PRINT("Could not create framebuffer!!");
		return;
	}

	texture->format = Image::FORMAT_RGBA8;
	texture->gl_format_cache = GL_RGBA;
	texture->gl_type_cache = GL_UNSIGNED_BYTE;
	texture->gl_internal_format_cache = GL_RGBA;
	texture->tex_id = rt->color;
	texture->width = rt->width;
	texture->alloc_width = rt->width;
	texture->height = rt->height;
	texture->alloc_height = rt->height;
	texture->active = true;

	texture_set_flags(rt->texture, texture->flags);

	// copy texscreen buffers
	{
		int w = rt->width;
		int h = rt->height;

		glGenTextures(1, &rt->copy_screen_effect.color);
		glBindTexture(GL_TEXTURE_2D, rt->copy_screen_effect.color);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		glGenFramebuffers(1, &rt->copy_screen_effect.fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, rt->copy_screen_effect.fbo);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt->color, 0);

		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			_render_target_clear(rt);
			ERR_FAIL_COND(status != GL_FRAMEBUFFER_COMPLETE);
		}
	}

	glBindFramebuffer(GL_FRAMEBUFFER, RasterizerStorageGLES2::system_fbo);
}

void RasterizerStorageGLES2::_render_target_clear(RenderTarget *rt) {

	if (rt->fbo) {
		glDeleteFramebuffers(1, &rt->fbo);
		glDeleteTextures(1, &rt->color);
		rt->fbo = 0;
	}

	if (rt->depth) {
		glDeleteRenderbuffers(1, &rt->depth);
		rt->depth = 0;
	}

	Texture *tex = texture_owner.get(rt->texture);
	tex->alloc_height = 0;
	tex->alloc_width = 0;
	tex->width = 0;
	tex->height = 0;
	tex->active = false;

	// TODO hardcoded texscreen copy effect
	if (rt->copy_screen_effect.color) {
		glDeleteFramebuffers(1, &rt->copy_screen_effect.fbo);
		rt->copy_screen_effect.fbo = 0;

		glDeleteTextures(1, &rt->copy_screen_effect.color);
		rt->copy_screen_effect.color = 0;
	}
}

RID RasterizerStorageGLES2::render_target_create() {

	RenderTarget *rt = memnew(RenderTarget);

	Texture *t = memnew(Texture);

	t->flags = 0;
	t->width = 0;
	t->height = 0;
	t->alloc_height = 0;
	t->alloc_width = 0;
	t->format = Image::FORMAT_R8;
	t->target = GL_TEXTURE_2D;
	t->gl_format_cache = 0;
	t->gl_internal_format_cache = 0;
	t->gl_type_cache = 0;
	t->data_size = 0;
	t->total_data_size = 0;
	t->ignore_mipmaps = false;
	t->mipmaps = 1;
	t->active = true;
	t->tex_id = 0;
	t->render_target = rt;

	rt->texture = texture_owner.make_rid(t);

	return render_target_owner.make_rid(rt);
}

void RasterizerStorageGLES2::render_target_set_size(RID p_render_target, int p_width, int p_height) {

	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);

	if (p_width == rt->width && p_height == rt->height)
		return;

	_render_target_clear(rt);

	rt->width = p_width;
	rt->height = p_height;

	_render_target_allocate(rt);
}

RID RasterizerStorageGLES2::render_target_get_texture(RID p_render_target) const {

	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	return rt->texture;
}

void RasterizerStorageGLES2::render_target_set_flag(RID p_render_target, RenderTargetFlags p_flag, bool p_value) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);

	rt->flags[p_flag] = p_value;

	switch (p_flag) {
		case RENDER_TARGET_HDR:
		case RENDER_TARGET_NO_3D:
		case RENDER_TARGET_NO_SAMPLING:
		case RENDER_TARGET_NO_3D_EFFECTS: {
			//must reset for these formats
			_render_target_clear(rt);
			_render_target_allocate(rt);

		} break;
		default: {}
	}
}

bool RasterizerStorageGLES2::render_target_was_used(RID p_render_target) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND_V(!rt, false);

	return rt->used_in_frame;
}

void RasterizerStorageGLES2::render_target_clear_used(RID p_render_target) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);

	rt->used_in_frame = false;
}

void RasterizerStorageGLES2::render_target_set_msaa(RID p_render_target, VS::ViewportMSAA p_msaa) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);

	if (rt->msaa == p_msaa)
		return;

	_render_target_clear(rt);
	rt->msaa = p_msaa;
	_render_target_allocate(rt);
}

/* CANVAS SHADOW */

RID RasterizerStorageGLES2::canvas_light_shadow_buffer_create(int p_width) {
	return RID();
}

/* LIGHT SHADOW MAPPING */

RID RasterizerStorageGLES2::canvas_light_occluder_create() {
	return RID();
}

void RasterizerStorageGLES2::canvas_light_occluder_set_polylines(RID p_occluder, const PoolVector<Vector2> &p_lines) {
}

VS::InstanceType RasterizerStorageGLES2::get_base_type(RID p_rid) const {
	return VS::INSTANCE_NONE;
}

bool RasterizerStorageGLES2::free(RID p_rid) {
	return false;
}

bool RasterizerStorageGLES2::has_os_feature(const String &p_feature) const {
	return false;
}

////////////////////////////////////////////

void RasterizerStorageGLES2::set_debug_generate_wireframes(bool p_generate) {
}

void RasterizerStorageGLES2::render_info_begin_capture() {
}

void RasterizerStorageGLES2::render_info_end_capture() {
}

int RasterizerStorageGLES2::get_captured_render_info(VS::RenderInfo p_info) {

	return get_render_info(p_info);
}

int RasterizerStorageGLES2::get_render_info(VS::RenderInfo p_info) {
	return 0;
}

void RasterizerStorageGLES2::initialize() {
	RasterizerStorageGLES2::system_fbo = 0;

	{
		const char *gl_extensions = (const char *)glGetString(GL_EXTENSIONS);
		Vector<String> strings = String(gl_extensions).split(" ", false);
		for (int i = 0; i < strings.size(); i++) {
			config.extensions.insert(strings[i]);
		}
	}

	frame.count = 0;
	frame.prev_tick = 0;
	frame.delta = 0;
	frame.current_rt = NULL;
	frame.clear_request = false;
	// config.keep_original_textures = false;

	glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &config.max_texture_image_units);
	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &config.max_texture_size);

	shaders.copy.init();

	{
		//default textures

		glGenTextures(1, &resources.white_tex);
		unsigned char whitetexdata[8 * 8 * 3];
		for (int i = 0; i < 8 * 8 * 3; i++) {
			whitetexdata[i] = 255;
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, resources.white_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 8, 8, 0, GL_RGB, GL_UNSIGNED_BYTE, whitetexdata);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);

		glGenTextures(1, &resources.black_tex);
		unsigned char blacktexdata[8 * 8 * 3];
		for (int i = 0; i < 8 * 8 * 3; i++) {
			blacktexdata[i] = 0;
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, resources.black_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 8, 8, 0, GL_RGB, GL_UNSIGNED_BYTE, blacktexdata);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);

		glGenTextures(1, &resources.normal_tex);
		unsigned char normaltexdata[8 * 8 * 3];
		for (int i = 0; i < 8 * 8 * 3; i += 3) {
			normaltexdata[i + 0] = 128;
			normaltexdata[i + 1] = 128;
			normaltexdata[i + 2] = 255;
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, resources.normal_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 8, 8, 0, GL_RGB, GL_UNSIGNED_BYTE, normaltexdata);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);

		glGenTextures(1, &resources.aniso_tex);
		unsigned char anisotexdata[8 * 8 * 3];
		for (int i = 0; i < 8 * 8 * 3; i += 3) {
			anisotexdata[i + 0] = 255;
			anisotexdata[i + 1] = 128;
			anisotexdata[i + 2] = 0;
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, resources.aniso_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 8, 8, 0, GL_RGB, GL_UNSIGNED_BYTE, anisotexdata);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
}

void RasterizerStorageGLES2::finalize() {
}

void RasterizerStorageGLES2::update_dirty_resources() {
	update_dirty_shaders();
	update_dirty_materials();
}

RasterizerStorageGLES2::RasterizerStorageGLES2() {
	RasterizerStorageGLES2::system_fbo = 0;
}
