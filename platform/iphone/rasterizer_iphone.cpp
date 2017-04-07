/*************************************************************************/
/*  rasterizer_iphone.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifdef IPHONE_ENABLED

#include "rasterizer_iphone.h"
#include "global_config.h"
#include "os/os.h"
#include <stdio.h>

_FORCE_INLINE_ static void _gl_load_transform(const Transform &tr) {

	GLfloat matrix[16] = { /* build a 16x16 matrix */
		tr.basis.elements[0][0],
		tr.basis.elements[1][0],
		tr.basis.elements[2][0],
		0,
		tr.basis.elements[0][1],
		tr.basis.elements[1][1],
		tr.basis.elements[2][1],
		0,
		tr.basis.elements[0][2],
		tr.basis.elements[1][2],
		tr.basis.elements[2][2],
		0,
		tr.origin.x,
		tr.origin.y,
		tr.origin.z,
		1
	};

	glLoadMatrixf(matrix);
};

_FORCE_INLINE_ static void _gl_mult_transform(const Transform &tr) {

	GLfloat matrix[16] = { /* build a 16x16 matrix */
		tr.basis.elements[0][0],
		tr.basis.elements[1][0],
		tr.basis.elements[2][0],
		0,
		tr.basis.elements[0][1],
		tr.basis.elements[1][1],
		tr.basis.elements[2][1],
		0,
		tr.basis.elements[0][2],
		tr.basis.elements[1][2],
		tr.basis.elements[2][2],
		0,
		tr.origin.x,
		tr.origin.y,
		tr.origin.z,
		1
	};

	glMultMatrixf(matrix);
};

static const GLenum prim_type[] = { GL_POINTS, GL_LINES, GL_TRIANGLES, GL_TRIANGLE_FAN };

static void _draw_primitive(int p_points, const float *p_vertices, const float *p_normals, const float *p_colors, const float *p_uvs, const Plane *p_tangents = NULL, int p_instanced = 1) {

	ERR_FAIL_COND(!p_vertices);
	ERR_FAIL_COND(p_points < 1 || p_points > 4);

	GLenum type = prim_type[p_points - 1];

	if (!p_colors) {
		glColor4f(1, 1, 1, 1);
	};

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, (GLvoid *)p_vertices);

	if (p_normals) {

		glEnableClientState(GL_NORMAL_ARRAY);
		glNormalPointer(GL_FLOAT, 0, (GLvoid *)p_normals);
	};

	if (p_colors) {
		glEnableClientState(GL_COLOR_ARRAY);
		glColorPointer(4, GL_FLOAT, 0, p_colors);
	};

	if (p_uvs) {

		glClientActiveTexture(GL_TEXTURE0);
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		glTexCoordPointer(2, GL_FLOAT, 0, p_uvs);
	};

	glDrawArrays(type, 0, p_points);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
};

/* TEXTURE API */

static Image _get_gl_image_and_format(const Image &p_image, Image::Format p_format, uint32_t p_flags, GLenum &r_gl_format, int &r_gl_components, bool &r_has_alpha_cache) {

	r_has_alpha_cache = false;
	Image image = p_image;

	switch (p_format) {

		case Image::FORMAT_L8: {
			r_gl_components = 1;
			r_gl_format = GL_LUMINANCE;

		} break;
		case Image::FORMAT_INTENSITY: {

			image.convert(Image::FORMAT_RGBA8);
			r_gl_components = 4;
			r_gl_format = GL_RGBA;
			r_has_alpha_cache = true;
		} break;
		case Image::FORMAT_LA8: {

			image.convert(Image::FORMAT_RGBA8);
			r_gl_components = 4;
			r_gl_format = GL_RGBA;
			r_has_alpha_cache = true;
		} break;

		case Image::FORMAT_INDEXED: {

			image.convert(Image::FORMAT_RGB8);
			r_gl_components = 3;
			r_gl_format = GL_RGB;

		} break;

		case Image::FORMAT_INDEXED_ALPHA: {

			image.convert(Image::FORMAT_RGBA8);
			r_gl_components = 4;
			r_gl_format = GL_RGB;
			r_has_alpha_cache = true;

		} break;
		case Image::FORMAT_RGB8: {

			r_gl_components = 3;
			r_gl_format = GL_RGB;
		} break;
		case Image::FORMAT_RGBA8: {

			r_gl_components = 4;
			r_gl_format = GL_RGBA;
			r_has_alpha_cache = true;
		} break;
		default: {

			ERR_FAIL_V(Image());
		}
	}

	return image;
}

RID RasterizerIPhone::texture_create() {

	Texture *texture = memnew(Texture);
	ERR_FAIL_COND_V(!texture, RID());
	glGenTextures(1, &texture->tex_id);
	texture->active = false;

	return texture_owner.make_rid(texture);
}

void RasterizerIPhone::texture_allocate(RID p_texture, int p_width, int p_height, Image::Format p_format, uint32_t p_flags) {

	bool has_alpha_cache;
	int components;
	GLenum format;

	Texture *texture = texture_owner.get(p_texture);
	ERR_FAIL_COND(!texture);
	texture->width = p_width;
	texture->height = p_height;
	texture->format = p_format;
	texture->flags = p_flags;
	//texture->target = (p_flags & VS::TEXTURE_FLAG_CUBEMAP) ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D;
	texture->target = GL_TEXTURE_2D;

	_get_gl_image_and_format(Image(), texture->format, texture->flags, format, components, has_alpha_cache);

	texture->gl_components_cache = components;
	texture->gl_format_cache = format;
	texture->format_has_alpha = has_alpha_cache;
	texture->has_alpha = false; //by default it doesn't have alpha unless something with alpha is blitteds

	glBindTexture(texture->target, texture->tex_id);

	if (texture->flags & VS::TEXTURE_FLAG_MIPMAPS) {
		glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
	}

	if (texture->target == GL_TEXTURE_2D) {
		glTexImage2D(texture->target, 0, format, texture->width, texture->height, 0, format, GL_UNSIGNED_BYTE, NULL);
	}

	/*
	else {
		//cubemappor
		for (int i=0;i<6;i++)
			glTexImage2D(_cube_side_enum[i], 0, format, texture->width, texture->height, 0, format, GL_UNSIGNED_BYTE,NULL);
	}
	*/

	glTexParameteri(texture->target, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // Linear Filtering

	if (texture->flags & VS::TEXTURE_FLAG_FILTER) {

		glTexParameteri(texture->target, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // Linear Filtering
		if (texture->flags & VS::TEXTURE_FLAG_MIPMAPS) {
			//glTexParameteri(texture->target,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
		};
	}

	if (texture->flags & VS::TEXTURE_FLAG_REPEAT /* && texture->target != GL_TEXTURE_CUBE_MAP*/) {

		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	} else {

		//glTexParameterf( texture->target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
		glTexParameterf(texture->target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(texture->target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

	texture->active = true;
}

void RasterizerIPhone::texture_blit_rect(RID p_texture, int p_x, int p_y, const Image &p_image, VS::CubeMapSide p_cube_side) {

	Texture *texture = texture_owner.get(p_texture);

	ERR_FAIL_COND(!texture);
	ERR_FAIL_COND(!texture->active);
	ERR_FAIL_COND(texture->format != p_image.get_format());

	int components;
	GLenum format;
	bool alpha;

	Image img = _get_gl_image_and_format(p_image, p_image.get_format(), texture->flags, format, components, alpha);

	if (img.detect_alpha())
		texture->has_alpha = true;

	GLenum blit_target = GL_TEXTURE_2D; //(texture->target == GL_TEXTURE_CUBE_MAP)?_cube_side_enum[p_cube_side]:GL_TEXTURE_2D;

	PoolVector<uint8_t>::Read read = img.get_data().read();

	glBindTexture(texture->target, texture->tex_id);
	glTexSubImage2D(blit_target, 0, p_x, p_y, img.get_width(), img.get_height(), format, GL_UNSIGNED_BYTE, read.ptr());

	//glGenerateMipmap( texture->target );
}

Image RasterizerIPhone::texture_get_rect(RID p_texture, int p_x, int p_y, int p_width, int p_height, VS::CubeMapSide p_cube_side) const {

	return Image();
}
void RasterizerIPhone::texture_set_flags(RID p_texture, uint32_t p_flags) {

	Texture *texture = texture_owner.get(p_texture);
	ERR_FAIL_COND(!texture);

	glBindTexture(texture->target, texture->tex_id);
	uint32_t cube = texture->flags & VS::TEXTURE_FLAG_CUBEMAP;
	texture->flags = p_flags | cube; // can't remove a cube from being a cube

	if (texture->flags & VS::TEXTURE_FLAG_REPEAT /*&& texture->target != GL_TEXTURE_CUBE_MAP*/) {

		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	} else {
		//glTexParameterf( texture->target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
		glTexParameterf(texture->target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(texture->target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

	if (texture->flags & VS::TEXTURE_FLAG_FILTER) {

		glTexParameteri(texture->target, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // Linear Filtering
		if (texture->flags & VS::TEXTURE_FLAG_MIPMAPS)
			glTexParameteri(texture->target, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

	} else {

		glTexParameteri(texture->target, GL_TEXTURE_MAG_FILTER, GL_NEAREST); // nearest
	}
}
uint32_t RasterizerIPhone::texture_get_flags(RID p_texture) const {

	Texture *texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture, 0);

	return texture->flags;
}
Image::Format RasterizerIPhone::texture_get_format(RID p_texture) const {

	Texture *texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture, Image::FORMAT_L8);

	return texture->format;
}
uint32_t RasterizerIPhone::texture_get_width(RID p_texture) const {

	Texture *texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture, 0);

	return texture->width;
}
uint32_t RasterizerIPhone::texture_get_height(RID p_texture) const {

	Texture *texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture, 0);

	return texture->height;
}

bool RasterizerIPhone::texture_has_alpha(RID p_texture) const {

	Texture *texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture, 0);

	return texture->has_alpha;
}

/* SHADER API */

RID RasterizerIPhone::shader_create() {

	return RID();
}

void RasterizerIPhone::shader_node_add(RID p_shader, VS::ShaderNodeType p_type, int p_id) {
}
void RasterizerIPhone::shader_node_remove(RID p_shader, int p_id) {
}
void RasterizerIPhone::shader_node_change_type(RID p_shader, int p_id, VS::ShaderNodeType p_type) {
}
void RasterizerIPhone::shader_node_set_param(RID p_shader, int p_id, const Variant &p_value) {
}

void RasterizerIPhone::shader_get_node_list(RID p_shader, List<int> *p_node_list) const {
}
VS::ShaderNodeType RasterizerIPhone::shader_node_get_type(RID p_shader, int p_id) const {

	return VS::NODE_ADD;
}
Variant RasterizerIPhone::shader_node_get_param(RID p_shader, int p_id) const {

	return Variant();
}

void RasterizerIPhone::shader_connect(RID p_shader, int p_src_id, int p_src_slot, int p_dst_id, int p_dst_slot) {
}
bool RasterizerIPhone::shader_is_connected(RID p_shader, int p_src_id, int p_src_slot, int p_dst_id, int p_dst_slot) const {

	return false;
}

void RasterizerIPhone::shader_disconnect(RID p_shader, int p_src_id, int p_src_slot, int p_dst_id, int p_dst_slot) {
}

void RasterizerIPhone::shader_get_connections(RID p_shader, List<VS::ShaderConnection> *p_connections) const {
}

void RasterizerIPhone::shader_clear(RID p_shader) {
}

/* COMMON MATERIAL API */

void RasterizerIPhone::material_set_param(RID p_material, const StringName &p_param, const Variant &p_value) {
}
Variant RasterizerIPhone::material_get_param(RID p_material, const StringName &p_param) const {

	return Variant();
}
void RasterizerIPhone::material_get_param_list(RID p_material, List<String> *p_param_list) const {
}

void RasterizerIPhone::material_set_flag(RID p_material, VS::MaterialFlag p_flag, bool p_enabled) {
}
bool RasterizerIPhone::material_get_flag(RID p_material, VS::MaterialFlag p_flag) const {

	return false;
}

void RasterizerIPhone::material_set_blend_mode(RID p_material, VS::MaterialBlendMode p_mode) {
}
VS::MaterialBlendMode RasterizerIPhone::material_get_blend_mode(RID p_material) const {

	return VS::MATERIAL_BLEND_MODE_ADD;
}

void RasterizerIPhone::material_set_line_width(RID p_material, float p_line_width) {
}
float RasterizerIPhone::material_get_line_width(RID p_material) const {

	return 0;
}

/* FIXED MATERIAL */

RID RasterizerIPhone::material_create() {

	return material_owner.make_rid(memnew(Material));
}

void RasterizerIPhone::fixed_material_set_parameter(RID p_material, VS::SpatialMaterialParam p_parameter, const Variant &p_value) {

	Material *m = material_owner.get(p_material);
	ERR_FAIL_COND(!m);
	ERR_FAIL_INDEX(p_parameter, VisualServer::FIXED_MATERIAL_PARAM_MAX);

	m->parameters[p_parameter] = p_value;
}
Variant RasterizerIPhone::fixed_material_get_parameter(RID p_material, VS::SpatialMaterialParam p_parameter) const {

	Material *m = material_owner.get(p_material);
	ERR_FAIL_COND_V(!m, Variant());
	ERR_FAIL_INDEX_V(p_parameter, VisualServer::FIXED_MATERIAL_PARAM_MAX, Variant());

	return m->parameters[p_parameter];
}

void RasterizerIPhone::fixed_material_set_texture(RID p_material, VS::SpatialMaterialParam p_parameter, RID p_texture) {

	Material *m = material_owner.get(p_material);
	ERR_FAIL_COND(!m);
	ERR_FAIL_INDEX(p_parameter, VisualServer::FIXED_MATERIAL_PARAM_MAX);

	m->textures[p_parameter] = p_texture;
}
RID RasterizerIPhone::fixed_material_get_texture(RID p_material, VS::SpatialMaterialParam p_parameter) const {

	Material *m = material_owner.get(p_material);
	ERR_FAIL_COND_V(!m, RID());
	ERR_FAIL_INDEX_V(p_parameter, VisualServer::FIXED_MATERIAL_PARAM_MAX, Variant());

	return m->textures[p_parameter];
}

void RasterizerIPhone::fixed_material_set_detail_blend_mode(RID p_material, VS::MaterialBlendMode p_mode) {

	Material *m = material_owner.get(p_material);
	ERR_FAIL_COND(!m);

	m->detail_blend_mode = p_mode;
}
VS::MaterialBlendMode RasterizerIPhone::fixed_material_get_detail_blend_mode(RID p_material) const {

	Material *m = material_owner.get(p_material);
	ERR_FAIL_COND_V(!m, VS::MATERIAL_BLEND_MODE_MIX);

	return m->detail_blend_mode;
}

void RasterizerIPhone::fixed_material_set_texcoord_mode(RID p_material, VS::SpatialMaterialParam p_parameter, VS::SpatialMaterialTexCoordMode p_mode) {

	Material *m = material_owner.get(p_material);
	ERR_FAIL_COND(!m);
	ERR_FAIL_INDEX(p_parameter, VisualServer::FIXED_MATERIAL_PARAM_MAX);

	m->texcoord_mode[p_parameter] = p_mode;
}
VS::SpatialMaterialTexCoordMode RasterizerIPhone::fixed_material_get_texcoord_mode(RID p_material, VS::SpatialMaterialParam p_parameter) const {

	Material *m = material_owner.get(p_material);
	ERR_FAIL_COND_V(!m, VS::FIXED_MATERIAL_TEXCOORD_TEXGEN);
	ERR_FAIL_INDEX_V(p_parameter, VisualServer::FIXED_MATERIAL_PARAM_MAX, VS::FIXED_MATERIAL_TEXCOORD_UV);

	return m->texcoord_mode[p_parameter]; // for now
}

void RasterizerIPhone::fixed_material_set_texgen_mode(RID p_material, VS::SpatialMaterialTexGenMode p_mode) {

	Material *m = material_owner.get(p_material);
	ERR_FAIL_COND(!m);

	m->texgen_mode = p_mode;
};

VS::SpatialMaterialTexGenMode RasterizerIPhone::fixed_material_get_texgen_mode(RID p_material) const {

	Material *m = material_owner.get(p_material);
	ERR_FAIL_COND_V(!m, VS::FIXED_MATERIAL_TEXGEN_SPHERE);

	return m->texgen_mode;
};

void RasterizerIPhone::fixed_material_set_uv_transform(RID p_material, const Transform &p_transform) {

	Material *m = material_owner.get(p_material);
	ERR_FAIL_COND(!m);

	m->uv_transform = p_transform;
}
Transform RasterizerIPhone::fixed_material_get_uv_transform(RID p_material) const {

	Material *m = material_owner.get(p_material);
	ERR_FAIL_COND_V(!m, Transform());

	return m->uv_transform;
}

/* SHADER MATERIAL */

RID RasterizerIPhone::shader_material_create() const {

	return RID();
}

void RasterizerIPhone::shader_material_set_vertex_shader(RID p_material, RID p_shader, bool p_owned) {
}
RID RasterizerIPhone::shader_material_get_vertex_shader(RID p_material) const {

	return RID();
}

void RasterizerIPhone::shader_material_set_fragment_shader(RID p_material, RID p_shader, bool p_owned) {
}
RID RasterizerIPhone::shader_material_get_fragment_shader(RID p_material) const {

	return RID();
}

/* MESH API */

RID RasterizerIPhone::mesh_create() {

	return mesh_owner.make_rid(memnew(Mesh));
}

void RasterizerIPhone::mesh_add_surface(RID p_mesh, VS::PrimitiveType p_primitive, uint32_t p_format, int p_array_len, int p_index_array_len) {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND(!mesh);

	ERR_FAIL_COND((p_format & VS::ARRAY_FORMAT_VERTEX) == 0); // mandatory
	ERR_FAIL_COND(p_array_len <= 0);
	ERR_FAIL_COND(p_index_array_len == 0);
	ERR_FAIL_INDEX(p_primitive, VS::PRIMITIVE_MAX);

	Surface *surface = memnew(Surface);
	ERR_FAIL_COND(!surface);

	int total_elem_size = 0;

	bool use_VBO = true; //glGenBuffersARB!=NULL; // TODO detect if it's in there
	if (p_format & VS::ARRAY_FORMAT_WEIGHTS) {

		use_VBO = false;
	}

	for (int i = 0; i < VS::ARRAY_MAX; i++) {

		Surface::ArrayData &ad = surface->array[i];
		ad.size = 0;
		ad.configured = false;
		ad.ofs = 0;
		int elem_size = 0;
		int elem_count = 0;

		if (!(p_format & (1 << i))) // no array
			continue;

		switch (i) {

			case VS::ARRAY_VERTEX:
			case VS::ARRAY_NORMAL: {

				elem_size = 3 * sizeof(GLfloat); // vertex
				elem_count = 3;
			} break;
			case VS::ARRAY_TANGENT: {
				elem_size = 4 * sizeof(GLfloat); // vertex
				elem_count = 4;

			} break;
			case VS::ARRAY_COLOR: {

				elem_size = 4; /* RGBA */
				elem_count = 4;
			} break;
			case VS::ARRAY_TEX_UV: {
				elem_size = 2 * sizeof(GLfloat);
				elem_count = 2;

			} break;
			case VS::ARRAY_WEIGHTS:
			case VS::ARRAY_BONES: {

				elem_size = VS::ARRAY_WEIGHTS_SIZE * sizeof(GLfloat);
				elem_count = VS::ARRAY_WEIGHTS_SIZE;

			} break;
			case VS::ARRAY_INDEX: {

				if (p_index_array_len <= 0) {
					ERR_PRINT("p_index_array_len==NO_INDEX_ARRAY");
					break;
				}
				/* determine wether using 8 or 16 bits indices */
				if (p_index_array_len > (1 << 8)) {

					elem_size = 2;
				} else {
					elem_size = 1;
				}

				if (use_VBO) {

					glGenBuffers(1, &surface->index_id);
					ERR_FAIL_COND(surface->index_id == 0);
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, surface->index_id);
					glBufferData(GL_ELEMENT_ARRAY_BUFFER, p_index_array_len * elem_size, NULL, GL_STATIC_DRAW);
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); //unbind
				} else {
					surface->index_array_local = (uint8_t *)memalloc(p_index_array_len * elem_size);
				};

				surface->index_array_len = p_index_array_len; // only way it can exist
				ad.ofs = 0;
				ad.size = elem_size;
				ad.configured = false;
				ad.components = 1;

				continue;
			} break;
			default: {
				ERR_FAIL();
			}
		}

		ad.ofs = total_elem_size;
		ad.size = elem_size;
		ad.components = elem_count;
		total_elem_size += elem_size;
		ad.configured = false;
	}

	surface->stride = total_elem_size;
	surface->array_len = p_array_len;
	surface->format = p_format;
	surface->primitive = p_primitive;

	/* bind the bigass buffers */
	if (use_VBO) {

		glGenBuffers(1, &surface->vertex_id);
		ERR_FAIL_COND(surface->vertex_id == 0);
		glBindBuffer(GL_ARRAY_BUFFER, surface->vertex_id);
		glBufferData(GL_ARRAY_BUFFER, surface->array_len * surface->stride, NULL, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind
	} else {
		surface->array_local = (uint8_t *)memalloc(surface->array_len * surface->stride);
	};

	mesh->surfaces.push_back(surface);
}

Error RasterizerIPhone::mesh_surface_set_array(RID p_mesh, int p_surface, VS::ArrayType p_type, const Variant &p_array) {

	ERR_FAIL_INDEX_V(p_type, VS::ARRAY_MAX, ERR_INVALID_PARAMETER);

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, ERR_INVALID_PARAMETER);
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), ERR_INVALID_PARAMETER);
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V(!surface, ERR_INVALID_PARAMETER);

	ERR_FAIL_COND_V(surface->array[p_type].size == 0, ERR_INVALID_PARAMETER);

	Surface::ArrayData &a = surface->array[p_type];

	switch (p_type) {

		case VS::ARRAY_INDEX: {
			ERR_FAIL_COND_V(surface->index_array_len <= 0, ERR_INVALID_DATA);
			ERR_FAIL_COND_V(p_array.get_type() != Variant::INT_ARRAY, ERR_INVALID_PARAMETER);

			PoolVector<int> indices = p_array;
			ERR_FAIL_COND_V(indices.size() == 0, ERR_INVALID_PARAMETER);
			ERR_FAIL_COND_V(indices.size() != surface->index_array_len, ERR_INVALID_PARAMETER);

			/* determine wether using 16 or 32 bits indices */

			if (surface->index_array_local == 0) {
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, surface->index_id);
			};

			PoolVector<int>::Read read = indices.read();
			const int *src = read.ptr();

			for (int i = 0; i < surface->index_array_len; i++) {

				if (surface->index_array_local) {

					if (a.size <= (1 << 8)) {
						uint8_t v = src[i];

						copymem(&surface->array_local[i * a.size], &v, a.size);
					} else {
						uint16_t v = src[i];

						copymem(&surface->array_local[i * a.size], &v, a.size);
					}

				} else {
					if (a.size <= (1 << 8)) {
						uint8_t v = src[i];

						glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, i * a.size, a.size, &v);
					} else {
						uint16_t v = src[i];

						glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, i * a.size, a.size, &v);
					}
				};
			}
			if (surface->index_array_local == 0) {
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			};
			a.configured = true;
			return OK;
		} break;
		case VS::ARRAY_VERTEX:
		case VS::ARRAY_NORMAL: {

			ERR_FAIL_COND_V(p_array.get_type() != Variant::VECTOR3_ARRAY, ERR_INVALID_PARAMETER);

			PoolVector<Vector3> array = p_array;
			ERR_FAIL_COND_V(array.size() != surface->array_len, ERR_INVALID_PARAMETER);

			if (surface->array_local == 0) {
				glBindBuffer(GL_ARRAY_BUFFER, surface->vertex_id);
			};

			PoolVector<Vector3>::Read read = array.read();
			const Vector3 *src = read.ptr();

			// setting vertices means regenerating the AABB
			if (p_type == VS::ARRAY_VERTEX)
				surface->aabb = AABB();

			for (int i = 0; i < surface->array_len; i++) {

				GLfloat vector[3] = { src[i].x, src[i].y, src[i].z };

				if (surface->array_local == 0) {
					glBufferSubData(GL_ARRAY_BUFFER, a.ofs + i * surface->stride, a.size, vector);
				} else {
					copymem(&surface->array_local[a.ofs + i * surface->stride], vector, a.size);
				}

				if (p_type == VS::ARRAY_VERTEX) {

					if (i == 0) {

						surface->aabb = AABB(src[i], Vector3());
					} else {

						surface->aabb.expand_to(src[i]);
					}
				}
			}

			if (surface->array_local == 0) {
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			};

		} break;
		case VS::ARRAY_TANGENT: {

			ERR_FAIL_COND_V(p_array.get_type() != Variant::REAL_ARRAY, ERR_INVALID_PARAMETER);

			PoolVector<real_t> array = p_array;

			ERR_FAIL_COND_V(array.size() != surface->array_len * 4, ERR_INVALID_PARAMETER);

			if (surface->array_local == 0) {
				glBindBuffer(GL_ARRAY_BUFFER, surface->vertex_id);
			};

			PoolVector<real_t>::Read read = array.read();
			const real_t *src = read.ptr();

			for (int i = 0; i < surface->array_len; i++) {

				GLfloat xyzw[4] = {
					src[i * 4 + 0],
					src[i * 4 + 1],
					src[i * 4 + 2],
					src[i * 4 + 3]
				};

				if (surface->array_local == 0) {

					glBufferSubData(GL_ARRAY_BUFFER, a.ofs + i * surface->stride, a.size, xyzw);
				} else {

					copymem(&surface->array_local[a.ofs + i * surface->stride], xyzw, a.size);
				};
			}

			if (surface->array_local == 0) {
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			};
		} break;
		case VS::ARRAY_COLOR: {

			ERR_FAIL_COND_V(p_array.get_type() != Variant::COLOR_ARRAY, ERR_INVALID_PARAMETER);

			PoolVector<Color> array = p_array;

			ERR_FAIL_COND_V(array.size() != surface->array_len, ERR_INVALID_PARAMETER);

			if (surface->array_local == 0)
				glBindBuffer(GL_ARRAY_BUFFER, surface->vertex_id);

			PoolVector<Color>::Read read = array.read();
			const Color *src = read.ptr();
			surface->has_alpha_cache = false;

			for (int i = 0; i < surface->array_len; i++) {

				if (src[i].a < 0.98) // tolerate alpha a bit, for crappy exporters
					surface->has_alpha_cache = true;
				uint8_t colors[4] = { src[i].r * 255.0, src[i].g * 255.0, src[i].b * 255.0, src[i].a * 255.0 };
				// I'm not sure if this is correct, endianness-wise, i should re-check the GL spec

				if (surface->array_local == 0)
					glBufferSubData(GL_ARRAY_BUFFER, a.ofs + i * surface->stride, a.size, colors);
				else
					copymem(&surface->array_local[a.ofs + i * surface->stride], colors, a.size);
			}

			if (surface->array_local == 0)
				glBindBuffer(GL_ARRAY_BUFFER, 0);

		} break;
		case VS::ARRAY_TEX_UV: {

			ERR_FAIL_COND_V(p_array.get_type() != Variant::VECTOR3_ARRAY, ERR_INVALID_PARAMETER);

			PoolVector<Vector3> array = p_array;

			ERR_FAIL_COND_V(array.size() != surface->array_len, ERR_INVALID_PARAMETER);

			if (surface->array_local == 0)
				glBindBuffer(GL_ARRAY_BUFFER, surface->vertex_id);

			PoolVector<Vector3>::Read read = array.read();

			const Vector3 *src = read.ptr();

			for (int i = 0; i < surface->array_len; i++) {

				GLfloat uv[2] = { src[i].x, src[i].y };

				if (surface->array_local == 0)
					glBufferSubData(GL_ARRAY_BUFFER, a.ofs + i * surface->stride, a.size, uv);
				else
					copymem(&surface->array_local[a.ofs + i * surface->stride], uv, a.size);
			}

			if (surface->array_local == 0)
				glBindBuffer(GL_ARRAY_BUFFER, 0);

		} break;
		case VS::ARRAY_BONES:
		case VS::ARRAY_WEIGHTS: {

			ERR_FAIL_COND_V(p_array.get_type() != Variant::REAL_ARRAY, ERR_INVALID_PARAMETER);

			PoolVector<real_t> array = p_array;

			ERR_FAIL_COND_V(array.size() != surface->array_len * VS::ARRAY_WEIGHTS_SIZE, ERR_INVALID_PARAMETER);

			if (surface->array_local == 0)
				glBindBuffer(GL_ARRAY_BUFFER, surface->vertex_id);

			PoolVector<real_t>::Read read = array.read();

			const real_t *src = read.ptr();

			for (int i = 0; i < surface->array_len; i++) {

				GLfloat data[VS::ARRAY_WEIGHTS_SIZE];
				for (int j = 0; j < VS::ARRAY_WEIGHTS_SIZE; j++)
					data[j] = src[i * VS::ARRAY_WEIGHTS_SIZE + j];

				if (surface->array_local == 0)
					glBufferSubData(GL_ARRAY_BUFFER, a.ofs + i * surface->stride, a.size, data);
				else
					copymem(&surface->array_local[a.ofs + i * surface->stride], data, a.size);
			}

			if (surface->array_local == 0)
				glBindBuffer(GL_ARRAY_BUFFER, 0);
		} break;
		default: { ERR_FAIL_V(ERR_INVALID_PARAMETER); }
	}

	a.configured = true;

	return OK;
}
Variant RasterizerIPhone::mesh_surface_get_array(RID p_mesh, int p_surface, VS::ArrayType p_type) const {

	return Variant();
}

void RasterizerIPhone::mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material, bool p_owned) {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND(!mesh);
	ERR_FAIL_INDEX(p_surface, mesh->surfaces.size());
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND(!surface);

	if (surface->material_owned && surface->material.is_valid())
		free(surface->material);

	surface->material_owned = p_owned;

	surface->material = p_material;
}

RID RasterizerIPhone::mesh_surface_get_material(RID p_mesh, int p_surface) const {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, RID());
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), RID());
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V(!surface, RID());

	return surface->material;
}

int RasterizerIPhone::mesh_surface_get_array_len(RID p_mesh, int p_surface) const {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, -1);
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), -1);
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V(!surface, -1);

	return surface->array_len;
}
int RasterizerIPhone::mesh_surface_get_array_index_len(RID p_mesh, int p_surface) const {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, -1);
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), -1);
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V(!surface, -1);

	return surface->index_array_len;
}
uint32_t RasterizerIPhone::mesh_surface_get_format(RID p_mesh, int p_surface) const {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, 0);
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), 0);
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V(!surface, 0);

	return surface->format;
}
VS::PrimitiveType RasterizerIPhone::mesh_surface_get_primitive_type(RID p_mesh, int p_surface) const {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, VS::PRIMITIVE_POINTS);
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), VS::PRIMITIVE_POINTS);
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V(!surface, VS::PRIMITIVE_POINTS);

	return surface->primitive;
}

void RasterizerIPhone::mesh_erase_surface(RID p_mesh, int p_index) {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND(!mesh);
	ERR_FAIL_INDEX(p_index, mesh->surfaces.size());
	Surface *surface = mesh->surfaces[p_index];
	ERR_FAIL_COND(!surface);

	memdelete(mesh->surfaces[p_index]);
	mesh->surfaces.remove(p_index);
}
int RasterizerIPhone::mesh_get_surface_count(RID p_mesh) const {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, -1);

	return mesh->surfaces.size();
}

AABB RasterizerIPhone::mesh_get_aabb(RID p_mesh) const {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, AABB());

	AABB aabb;

	for (int i = 0; i < mesh->surfaces.size(); i++) {

		if (i == 0)
			aabb = mesh->surfaces[i]->aabb;
		else
			aabb.merge_with(mesh->surfaces[i]->aabb);
	}

	return aabb;
}

/* MULTIMESH API */

RID RasterizerIPhone::multimesh_create() {

	return RID();
}

void RasterizerIPhone::multimesh_set_instance_count(RID p_multimesh, int p_count) {
}
int RasterizerIPhone::multimesh_get_instance_count(RID p_multimesh) const {

	return 0;
}

void RasterizerIPhone::multimesh_set_mesh(RID p_multimesh, RID p_mesh) {
}
void RasterizerIPhone::multimesh_set_aabb(RID p_multimesh, const AABB &p_aabb) {
}
void RasterizerIPhone::multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform &p_transform) {
}
void RasterizerIPhone::multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color) {
}

RID RasterizerIPhone::multimesh_get_mesh(RID p_multimesh) const {

	return RID();
}
AABB RasterizerIPhone::multimesh_get_aabb(RID p_multimesh) const {

	return AABB();
}

Transform RasterizerIPhone::multimesh_instance_get_transform(RID p_multimesh, int p_index) const {

	return Transform();
}
Color RasterizerIPhone::multimesh_instance_get_color(RID p_multimesh, int p_index) const {

	return Color();
}

/* POLY API */

RID RasterizerIPhone::poly_create() {

	return RID();
}
void RasterizerIPhone::poly_set_material(RID p_poly, RID p_material, bool p_owned) {
}
void RasterizerIPhone::poly_add_primitive(RID p_poly, const Vector<Vector3> &p_points, const Vector<Vector3> &p_normals, const Vector<Color> &p_colors, const Vector<Vector3> &p_uvs) {
}
void RasterizerIPhone::poly_clear(RID p_poly) {
}

AABB RasterizerIPhone::poly_get_aabb(RID p_poly) const {

	return AABB();
}

/* PARTICLES API */

RID RasterizerIPhone::particles_create() {

	return RID();
}

void RasterizerIPhone::particles_set_amount(RID p_particles, int p_amount) {
}
int RasterizerIPhone::particles_get_amount(RID p_particles) const {

	return 0;
}

void RasterizerIPhone::particles_set_emitting(RID p_particles, bool p_emitting) {
}

bool RasterizerIPhone::particles_is_emitting(RID p_particles) const {

	return false;
}

void RasterizerIPhone::particles_set_visibility_aabb(RID p_particles, const AABB &p_visibility) {
}
AABB RasterizerIPhone::particles_get_visibility_aabb(RID p_particles) const {

	return AABB();
}

void RasterizerIPhone::particles_set_emission_half_extents(RID p_particles, const Vector3 &p_half_extents) {
}
Vector3 RasterizerIPhone::particles_get_emission_half_extents(RID p_particles) const {

	return Vector3();
}

void RasterizerIPhone::particles_set_gravity_normal(RID p_particles, const Vector3 &p_normal) {
}
Vector3 RasterizerIPhone::particles_get_gravity_normal(RID p_particles) const {

	return Vector3();
}

void RasterizerIPhone::particles_set_variable(RID p_particles, VS::ParticleVariable p_variable, float p_value) {
}
float RasterizerIPhone::particles_get_variable(RID p_particles, VS::ParticleVariable p_variable) const {

	return 0;
}

void RasterizerIPhone::particles_set_randomness(RID p_particles, VS::ParticleVariable p_variable, float p_randomness) {
}
float RasterizerIPhone::particles_get_randomness(RID p_particles, VS::ParticleVariable p_variable) const {

	return 0;
}

void RasterizerIPhone::particles_set_color_phase_pos(RID p_particles, int p_phase, float p_pos) {
}
float RasterizerIPhone::particles_get_color_phase_pos(RID p_particles, int p_phase) const {

	return 0;
}

void RasterizerIPhone::particles_set_color_phases(RID p_particles, int p_phases) {
}
int RasterizerIPhone::particles_get_color_phases(RID p_particles) const {

	return 0;
}

void RasterizerIPhone::particles_set_color_phase_color(RID p_particles, int p_phase, const Color &p_color) {
}
Color RasterizerIPhone::particles_get_color_phase_color(RID p_particles, int p_phase) const {

	return Color();
}

void RasterizerIPhone::particles_set_attractors(RID p_particles, int p_attractors) {
}
int RasterizerIPhone::particles_get_attractors(RID p_particles) const {

	return 0;
}

void RasterizerIPhone::particles_set_attractor_pos(RID p_particles, int p_attractor, const Vector3 &p_pos) {
}
Vector3 RasterizerIPhone::particles_get_attractor_pos(RID p_particles, int p_attractor) const {

	return Vector3();
}

void RasterizerIPhone::particles_set_attractor_strength(RID p_particles, int p_attractor, float p_force) {
}
float RasterizerIPhone::particles_get_attractor_strength(RID p_particles, int p_attractor) const {

	return 0;
}

void RasterizerIPhone::particles_set_material(RID p_particles, RID p_material, bool p_owned) {
}

RID RasterizerIPhone::particles_get_material(RID p_particles) const {

	return RID();
}

AABB RasterizerIPhone::particles_get_aabb(RID p_particles) const {

	return AABB();
}
/* BEAM API */

RID RasterizerIPhone::beam_create() {

	return RID();
}

void RasterizerIPhone::beam_set_point_count(RID p_beam, int p_count) {
}
int RasterizerIPhone::beam_get_point_count(RID p_beam) const {

	return 0;
}
void RasterizerIPhone::beam_clear(RID p_beam) {
}

void RasterizerIPhone::beam_set_point(RID p_beam, int p_point, Vector3 &p_pos) {
}
Vector3 RasterizerIPhone::beam_get_point(RID p_beam, int p_point) const {

	return Vector3();
}

void RasterizerIPhone::beam_set_primitive(RID p_beam, VS::BeamPrimitive p_primitive) {
}

VS::BeamPrimitive RasterizerIPhone::beam_get_primitive(RID p_beam) const {

	return VS::BEAM_CUBIC;
}

void RasterizerIPhone::beam_set_material(RID p_beam, RID p_material) {
}
RID RasterizerIPhone::beam_get_material(RID p_beam) const {

	return RID();
}

AABB RasterizerIPhone::beam_get_aabb(RID p_particles) const {

	return AABB();
}
/* SKELETON API */

RID RasterizerIPhone::skeleton_create() {

	Skeleton *skeleton = memnew(Skeleton);
	ERR_FAIL_COND_V(!skeleton, RID());
	return skeleton_owner.make_rid(skeleton);
}
void RasterizerIPhone::skeleton_resize(RID p_skeleton, int p_bones) {

	Skeleton *skeleton = skeleton_owner.get(p_skeleton);
	ERR_FAIL_COND(!skeleton);
	if (p_bones == skeleton->bones.size()) {
		return;
	};
	ERR_FAIL_COND(p_bones < 0 || p_bones > 256);

	skeleton->bones.resize(p_bones);
}
int RasterizerIPhone::skeleton_get_bone_count(RID p_skeleton) const {

	Skeleton *skeleton = skeleton_owner.get(p_skeleton);
	ERR_FAIL_COND_V(!skeleton, -1);
	return skeleton->bones.size();
}
void RasterizerIPhone::skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform &p_transform) {

	Skeleton *skeleton = skeleton_owner.get(p_skeleton);
	ERR_FAIL_COND(!skeleton);
	ERR_FAIL_INDEX(p_bone, skeleton->bones.size());

	skeleton->bones[p_bone] = p_transform;
}
Transform RasterizerIPhone::skeleton_bone_get_transform(RID p_skeleton, int p_bone) {

	Skeleton *skeleton = skeleton_owner.get(p_skeleton);
	ERR_FAIL_COND_V(!skeleton, Transform());
	ERR_FAIL_INDEX_V(p_bone, skeleton->bones.size(), Transform());

	// something
	return skeleton->bones[p_bone];
}

/* LIGHT API */

RID RasterizerIPhone::light_create(VS::LightType p_type) {

	Light *light = memnew(Light);
	light->type = p_type;
	return light_owner.make_rid(light);
}

VS::LightType RasterizerIPhone::light_get_type(RID p_light) const {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light, VS::LIGHT_OMNI);
	return light->type;
}

void RasterizerIPhone::light_set_color(RID p_light, VS::LightColor p_type, const Color &p_color) {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND(!light);
	ERR_FAIL_INDEX(p_type, 3);
	light->colors[p_type] = p_color;
}
Color RasterizerIPhone::light_get_color(RID p_light, VS::LightColor p_type) const {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light, Color());
	ERR_FAIL_INDEX_V(p_type, 3, Color());
	return light->colors[p_type];
}

void RasterizerIPhone::light_set_shadow(RID p_light, bool p_enabled) {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND(!light);
	light->shadow_enabled = p_enabled;
}

bool RasterizerIPhone::light_has_shadow(RID p_light) const {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light, false);
	return light->shadow_enabled;
}

void RasterizerIPhone::light_set_volumetric(RID p_light, bool p_enabled) {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND(!light);
	light->volumetric_enabled = p_enabled;
}
bool RasterizerIPhone::light_is_volumetric(RID p_light) const {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light, false);
	return light->volumetric_enabled;
}

void RasterizerIPhone::light_set_projector(RID p_light, RID p_texture) {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND(!light);
	light->projector = p_texture;
}
RID RasterizerIPhone::light_get_projector(RID p_light) const {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light, RID());
	return light->projector;
}

void RasterizerIPhone::light_set_var(RID p_light, VS::LightParam p_var, float p_value) {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND(!light);
	ERR_FAIL_INDEX(p_var, VS::LIGHT_PARAM_MAX);

	light->vars[p_var] = p_value;
}
float RasterizerIPhone::light_get_var(RID p_light, VS::LightParam p_var) const {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light, 0);

	ERR_FAIL_INDEX_V(p_var, VS::LIGHT_PARAM_MAX, 0);

	return light->vars[p_var];
}

AABB RasterizerIPhone::light_get_aabb(RID p_light) const {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light, AABB());

	switch (light->type) {

		case VS::LIGHT_SPOT: {

			float len = light->vars[VS::LIGHT_PARAM_RADIUS];
			float size = Math::tan(Math::deg2rad(light->vars[VS::LIGHT_PARAM_SPOT_ANGLE])) * len;
			return AABB(Vector3(-size, -size, -len), Vector3(size * 2, size * 2, len));
		} break;
		case VS::LIGHT_OMNI: {

			float r = light->vars[VS::LIGHT_PARAM_RADIUS];
			return AABB(-Vector3(r, r, r), Vector3(r, r, r) * 2);
		} break;
		case VS::LIGHT_DIRECTIONAL: {

			return AABB();
		} break;
		default: {}
	}

	ERR_FAIL_V(AABB());
}

RID RasterizerIPhone::light_instance_create(RID p_light) {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light, RID());

	LightInstance *light_instance = memnew(LightInstance);

	light_instance->light = p_light;
	light_instance->base = light;
	light_instance->last_pass = 0;

	return light_instance_owner.make_rid(light_instance);
}
void RasterizerIPhone::light_instance_set_transform(RID p_light_instance, const Transform &p_transform) {

	LightInstance *lighti = light_instance_owner.get(p_light_instance);
	ERR_FAIL_COND(!lighti);
	lighti->transform = p_transform;
}

void RasterizerIPhone::light_instance_set_active_hint(RID p_light_instance) {

	LightInstance *lighti = light_instance_owner.get(p_light_instance);
	ERR_FAIL_COND(!lighti);
	lighti->last_pass = frame;
}
bool RasterizerIPhone::light_instance_has_shadow(RID p_light_instance) const {

	return false;
}
bool RasterizerIPhone::light_instance_assign_shadow(RID p_light_instance) {

	return false;
}
Rasterizer::ShadowType RasterizerIPhone::light_instance_get_shadow_type(RID p_light_instance) const {

	return Rasterizer::SHADOW_CUBE;
}
int RasterizerIPhone::light_instance_get_shadow_passes(RID p_light_instance) const {

	return 0;
}
void RasterizerIPhone::light_instance_set_pssm_split_info(RID p_light_instance, int p_split, float p_near, float p_far, const CameraMatrix &p_camera, const Transform &p_transform) {
}

/* PARTICLES INSTANCE */

RID RasterizerIPhone::particles_instance_create(RID p_particles) {

	return RID();
}
void RasterizerIPhone::particles_instance_set_transform(RID p_particles_instance, const Transform &p_transform) {
}

/* RENDER API */
/* all calls (inside begin/end shadow) are always warranted to be in the following order: */

static GLfloat rtri; // Angle For The Triangle ( NEW )
static GLfloat rquad; // Angle For The Quad ( NEW )

void RasterizerIPhone::begin_frame() {

	window_size = Size2(OS::get_singleton()->get_video_mode().width, OS::get_singleton()->get_video_mode().height);

	double time = (OS::get_singleton()->get_ticks_usec() / 1000); // get msec
	time /= 1000.0; // make secs
	time_delta = time - last_time;
	last_time = time;
	frame++;
	glClearColor(0, 0, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT);

/* nehe ?*/

#if 0
	glViewport(0,0,window_size.width,window_size.height);						// Reset The Current Viewport

	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();									// Reset The Projection Matrix

	// Calculate The Aspect Ratio Of The Window
	gluPerspective(45.0f,(GLfloat)window_size.width/(GLfloat)window_size.height,0.1f,100.0f);

	glMatrixMode(GL_MODELVIEW);							// Select The Modelview Matrix
	glLoadIdentity();									// Reset The Modelview Matrix



	glShadeModel(GL_SMOOTH);							// Enable Smooth Shading
	glClearColor(0.0f, 0.0f, 0.0f, 0.5f);				// Black Background
	glClearDepth(1.0f);									// Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);							// Enables Depth Testing
	glDepthFunc(GL_LEQUAL);								// The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);	// Really Nice Perspective Calculations

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clear Screen And Depth Buffer
	glLoadIdentity();									// Reset The Current Modelview Matrix
	glTranslatef(-1.5f,0.0f,-6.0f);						// Move Left 1.5 Units And Into The Screen 6.0
	glRotatef(rtri,0.0f,1.0f,0.0f);						// Rotate The Triangle On The Y axis ( NEW )
	glBegin(GL_TRIANGLES);								// Start Drawing A Triangle
		glColor3f(1.0f,0.0f,0.0f);						// Red
		glVertex3f( 0.0f, 1.0f, 0.0f);					// Top Of Triangle (Front)
		glColor3f(0.0f,1.0f,0.0f);						// Green
		glVertex3f(-1.0f,-1.0f, 1.0f);					// Left Of Triangle (Front)
		glColor3f(0.0f,0.0f,1.0f);						// Blue
		glVertex3f( 1.0f,-1.0f, 1.0f);					// Right Of Triangle (Front)
		glColor3f(1.0f,0.0f,0.0f);						// Red
		glVertex3f( 0.0f, 1.0f, 0.0f);					// Top Of Triangle (Right)
		glColor3f(0.0f,0.0f,1.0f);						// Blue
		glVertex3f( 1.0f,-1.0f, 1.0f);					// Left Of Triangle (Right)
		glColor3f(0.0f,1.0f,0.0f);						// Green
		glVertex3f( 1.0f,-1.0f, -1.0f);					// Right Of Triangle (Right)
		glColor3f(1.0f,0.0f,0.0f);						// Red
		glVertex3f( 0.0f, 1.0f, 0.0f);					// Top Of Triangle (Back)
		glColor3f(0.0f,1.0f,0.0f);						// Green
		glVertex3f( 1.0f,-1.0f, -1.0f);					// Left Of Triangle (Back)
		glColor3f(0.0f,0.0f,1.0f);						// Blue
		glVertex3f(-1.0f,-1.0f, -1.0f);					// Right Of Triangle (Back)
		glColor3f(1.0f,0.0f,0.0f);						// Red
		glVertex3f( 0.0f, 1.0f, 0.0f);					// Top Of Triangle (Left)
		glColor3f(0.0f,0.0f,1.0f);						// Blue
		glVertex3f(-1.0f,-1.0f,-1.0f);					// Left Of Triangle (Left)
		glColor3f(0.0f,1.0f,0.0f);						// Green
		glVertex3f(-1.0f,-1.0f, 1.0f);					// Right Of Triangle (Left)
	glEnd();											// Done Drawing The Pyramid

	glLoadIdentity();									// Reset The Current Modelview Matrix
	glTranslatef(1.5f,0.0f,-7.0f);						// Move Right 1.5 Units And Into The Screen 7.0
	glRotatef(rquad,1.0f,1.0f,1.0f);					// Rotate The Quad On The X axis ( NEW )
	glBegin(GL_QUADS);									// Draw A Quad
		glColor3f(0.0f,1.0f,0.0f);						// Set The Color To Green
		glVertex3f( 1.0f, 1.0f,-1.0f);					// Top Right Of The Quad (Top)
		glVertex3f(-1.0f, 1.0f,-1.0f);					// Top Left Of The Quad (Top)
		glVertex3f(-1.0f, 1.0f, 1.0f);					// Bottom Left Of The Quad (Top)
		glVertex3f( 1.0f, 1.0f, 1.0f);					// Bottom Right Of The Quad (Top)
		glColor3f(1.0f,0.5f,0.0f);						// Set The Color To Orange
		glVertex3f( 1.0f,-1.0f, 1.0f);					// Top Right Of The Quad (Bottom)
		glVertex3f(-1.0f,-1.0f, 1.0f);					// Top Left Of The Quad (Bottom)
		glVertex3f(-1.0f,-1.0f,-1.0f);					// Bottom Left Of The Quad (Bottom)
		glVertex3f( 1.0f,-1.0f,-1.0f);					// Bottom Right Of The Quad (Bottom)
		glColor3f(1.0f,0.0f,0.0f);						// Set The Color To Red
		glVertex3f( 1.0f, 1.0f, 1.0f);					// Top Right Of The Quad (Front)
		glVertex3f(-1.0f, 1.0f, 1.0f);					// Top Left Of The Quad (Front)
		glVertex3f(-1.0f,-1.0f, 1.0f);					// Bottom Left Of The Quad (Front)
		glVertex3f( 1.0f,-1.0f, 1.0f);					// Bottom Right Of The Quad (Front)
		glColor3f(1.0f,1.0f,0.0f);						// Set The Color To Yellow
		glVertex3f( 1.0f,-1.0f,-1.0f);					// Top Right Of The Quad (Back)
		glVertex3f(-1.0f,-1.0f,-1.0f);					// Top Left Of The Quad (Back)
		glVertex3f(-1.0f, 1.0f,-1.0f);					// Bottom Left Of The Quad (Back)
		glVertex3f( 1.0f, 1.0f,-1.0f);					// Bottom Right Of The Quad (Back)
		glColor3f(0.0f,0.0f,1.0f);						// Set The Color To Blue
		glVertex3f(-1.0f, 1.0f, 1.0f);					// Top Right Of The Quad (Left)
		glVertex3f(-1.0f, 1.0f,-1.0f);					// Top Left Of The Quad (Left)
		glVertex3f(-1.0f,-1.0f,-1.0f);					// Bottom Left Of The Quad (Left)
		glVertex3f(-1.0f,-1.0f, 1.0f);					// Bottom Right Of The Quad (Left)
		glColor3f(1.0f,0.0f,1.0f);						// Set The Color To Violet
		glVertex3f( 1.0f, 1.0f,-1.0f);					// Top Right Of The Quad (Right)
		glVertex3f( 1.0f, 1.0f, 1.0f);					// Top Left Of The Quad (Right)
		glVertex3f( 1.0f,-1.0f, 1.0f);					// Bottom Left Of The Quad (Right)
		glVertex3f( 1.0f,-1.0f,-1.0f);					// Bottom Right Of The Quad (Right)
	glEnd();											// Done Drawing The Quad

	rtri+=0.2f;											// Increase The Rotation Variable For The Triangle ( NEW )
	rquad-=0.15f;										// Decrease The Rotation Variable For The Quad ( NEW )

#endif
}

void RasterizerIPhone::set_viewport(const VS::ViewportRect &p_viewport) {

	viewport = p_viewport;
	canvas_transform = Transform();
	canvas_transform.translate(-(viewport.width / 2.0f), -(viewport.height / 2.0f), 0.0f);
	canvas_transform.scale(Vector3(2.0f / viewport.width, -2.0f / viewport.height, 1.0f));

	glViewport(viewport.x, window_size.height - (viewport.height + viewport.y), viewport.width, viewport.height);
}

void RasterizerIPhone::begin_scene(RID p_fx, VS::ScenarioDebugMode p_debug) {

	opaque_render_list.clear();
	alpha_render_list.clear();
	light_instance_count = 0;
	scene_fx = p_fx.is_valid() ? fx_owner.get(p_fx) : NULL;
};

void RasterizerIPhone::begin_shadow_map(RID p_light_instance, int p_shadow_pass) {
}

void RasterizerIPhone::set_camera(const Transform &p_world, const CameraMatrix &p_projection) {

	camera_transform = p_world;
	camera_transform_inverse = camera_transform.inverse();
	camera_projection = p_projection;
	camera_plane = Plane(camera_transform.origin, camera_transform.basis.get_axis(2));
	camera_z_near = camera_projection.get_z_near();
	camera_z_far = camera_projection.get_z_far();
	camera_projection.get_viewport_size(camera_vp_size.x, camera_vp_size.y);
}

void RasterizerIPhone::add_light(RID p_light_instance) {

#define LIGHT_FADE_TRESHOLD 0.05

	ERR_FAIL_COND(light_instance_count >= MAX_LIGHTS);

	LightInstance *li = light_instance_owner.get(p_light_instance);
	ERR_FAIL_COND(!li);

	/* make light hash */

	// actually, not really a hash, but helps to sort the lights
	// and avoid recompiling redudant shader versions

	li->hash_aux = li->base->type;

	if (li->base->shadow_enabled)
		li->hash_aux |= (1 << 3);

	if (li->base->projector.is_valid())
		li->hash_aux |= (1 << 4);

	if (li->base->shadow_enabled && li->base->volumetric_enabled)
		li->hash_aux |= (1 << 5);

	switch (li->base->type) {

		case VisualServer::LIGHT_DIRECTIONAL: {

			Vector3 dir = li->transform.basis.get_axis(2);
			li->light_vector.x = dir.x;
			li->light_vector.y = dir.y;
			li->light_vector.z = dir.z;

		} break;
		case VisualServer::LIGHT_OMNI: {

			float radius = li->base->vars[VisualServer::LIGHT_PARAM_RADIUS];
			if (radius == 0)
				radius = 0.0001;
			li->linear_att = (1 / LIGHT_FADE_TRESHOLD) / radius;
			li->light_vector.x = li->transform.origin.x;
			li->light_vector.y = li->transform.origin.y;
			li->light_vector.z = li->transform.origin.z;

		} break;
		case VisualServer::LIGHT_SPOT: {

			float radius = li->base->vars[VisualServer::LIGHT_PARAM_RADIUS];
			if (radius == 0)
				radius = 0.0001;
			li->linear_att = (1 / LIGHT_FADE_TRESHOLD) / radius;
			li->light_vector.x = li->transform.origin.x;
			li->light_vector.y = li->transform.origin.y;
			li->light_vector.z = li->transform.origin.z;
			Vector3 dir = -li->transform.basis.get_axis(2);
			li->spot_vector.x = dir.x;
			li->spot_vector.y = dir.y;
			li->spot_vector.z = dir.z;

		} break;
	}

	light_instances[light_instance_count++] = li;
}

void RasterizerIPhone::_add_geometry(const Geometry *p_geometry, const Transform &p_world, uint32_t p_vertex_format, const RID *p_light_instances, int p_light_count, const ParamOverrideMap *p_material_overrides, const Skeleton *p_skeleton, GeometryOwner *p_owner) {

	Material *m = NULL;

	if (p_geometry->material.is_valid())
		m = material_owner.get(p_geometry->material);

	if (!m) {
		m = material_owner.get(default_material);
	}

	ERR_FAIL_COND(!m);

	LightInstance *lights[RenderList::MAX_LIGHTS];
	int light_count = 0;

	RenderList *render_list = &opaque_render_list;
	if (p_geometry->has_alpha || m->detail_blend_mode != VS::MATERIAL_BLEND_MODE_MIX) {
		render_list = &alpha_render_list;
	};

	if (!m->flags[VS::MATERIAL_FLAG_UNSHADED]) {

		light_count = p_light_count;
		for (int i = 0; i < light_count; i++) {
			lights[i] = light_instance_owner.get(p_light_instances[i]);
		}
	}

	render_list->add_element(p_geometry, m, p_world, lights, light_count, p_material_overrides, p_skeleton, camera_plane.distance(p_world.origin), p_owner);
}

void RasterizerIPhone::add_mesh(RID p_mesh, const Transform *p_world, const RID *p_light_instances, int p_light_count, const ParamOverrideMap *p_material_overrides, RID p_skeleton) {

	Mesh *mesh = mesh_owner.get(p_mesh);

	int ssize = mesh->surfaces.size();

	for (int i = 0; i < ssize; i++) {

		Surface *s = mesh->surfaces[i];
		Skeleton *sk = p_skeleton.is_valid() ? skeleton_owner.get(p_skeleton) : NULL;

		_add_geometry(s, *p_world, s->format, p_light_instances, p_light_count, p_material_overrides, sk, NULL);
	}

	mesh->last_pass = frame;
}

void RasterizerIPhone::add_multimesh(RID p_multimesh, const Transform *p_world, const RID *p_light_instances, int p_light_count, const ParamOverrideMap *p_material_overrides) {
}

void RasterizerIPhone::add_poly(RID p_poly, const Transform *p_world, const RID *p_light_instances, int p_light_count, const ParamOverrideMap *p_material_overrides) {

	Poly *p = poly_owner.get(p_poly);
	if (!p->primitives.empty()) {
		const Poly::Primitive *pp = &p->primitives[0];

		uint32_t format = VisualServer::ARRAY_FORMAT_VERTEX;

		if (!pp->normals.empty())
			format |= VisualServer::ARRAY_FORMAT_NORMAL;
		if (!pp->colors.empty())
			format |= VisualServer::ARRAY_FORMAT_COLOR;
		if (!pp->uvs.empty())
			format |= VisualServer::ARRAY_TEX_UV;

		_add_geometry(p, *p_world, format, p_light_instances, p_light_count, p_material_overrides, NULL, NULL);
	}
}

void RasterizerIPhone::add_beam(RID p_beam, const Transform *p_world, const RID *p_light_instances, int p_light_count, const ParamOverrideMap *p_material_overrides) {
}

void RasterizerIPhone::add_particles(RID p_particle_instance, const RID *p_light_instances, int p_light_count, const ParamOverrideMap *p_material_overrides) {
}

void RasterizerIPhone::_setup_material(const Geometry *p_geometry, const Material *p_material) {

	if (p_material->flags[VS::MATERIAL_FLAG_DOUBLE_SIDED])
		glDisable(GL_CULL_FACE);
	else {
		glEnable(GL_CULL_FACE);
		glCullFace((p_material->flags[VS::MATERIAL_FLAG_INVERT_FACES]) ? GL_FRONT : GL_BACK);
	}

	glEnable(GL_COLOR_MATERIAL); /* unused, unless color array */
	//glColorMaterial( GL_FRONT_AND_BACK, GL_DIFFUSE );
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

	///ambient @TODO offer global ambient group option
	float ambient_rgba[4] = {
		1,
		1,
		1,
		1.0
	};
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient_rgba);

	///diffuse
	const Color &diffuse_color = p_material->parameters[VS::FIXED_MATERIAL_PARAM_DIFFUSE];
	float diffuse_rgba[4] = {
		(float)diffuse_color.r,
		(float)diffuse_color.g,
		(float)diffuse_color.b,
		(float)diffuse_color.a
	};

	glColor4f(diffuse_rgba[0], diffuse_rgba[1], diffuse_rgba[2], diffuse_rgba[3]);

	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse_rgba);

	//specular

	const Color &specular_color = p_material->parameters[VS::FIXED_MATERIAL_PARAM_SPECULAR];
	float specular_rgba[4] = {
		(float)specular_color.r,
		(float)specular_color.g,
		(float)specular_color.b,
		1.0
	};

	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular_rgba);

	const Color &emission_color = p_material->parameters[VS::FIXED_MATERIAL_PARAM_EMISSION];
	float emission_rgba[4] = {
		(float)emission_color.r,
		(float)emission_color.g,
		(float)emission_color.b,
		1.0
	};

	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, emission_rgba);

	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, p_material->parameters[VS::FIXED_MATERIAL_PARAM_SPECULAR_EXP]);

	if (p_material->flags[VS::MATERIAL_FLAG_UNSHADED]) {
		glDisable(GL_LIGHTING);
	} else {
		glEnable(GL_LIGHTING);
		glDisable(GL_LIGHTING);
	}

	//depth test?
	/*
	if (p_material->flags[VS::MATERIAL_FLAG_WIREFRAME])
		glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	*/
	if (p_material->textures[VS::FIXED_MATERIAL_PARAM_DIFFUSE]) {

		Texture *texture = texture_owner.get(p_material->textures[VS::FIXED_MATERIAL_PARAM_DIFFUSE]);
		ERR_FAIL_COND(!texture);
		glActiveTexture(GL_TEXTURE0);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, texture->tex_id);
	};
};

void RasterizerIPhone::_setup_light(LightInstance *p_instance, int p_idx) {

	Light *ld = p_instance->base;

	int glid = GL_LIGHT0 + p_idx;
	glLightfv(glid, GL_AMBIENT, ld->colors[VS::LIGHT_COLOR_AMBIENT].components);
	glLightfv(glid, GL_DIFFUSE, ld->colors[VS::LIGHT_COLOR_DIFFUSE].components);
	glLightfv(glid, GL_SPECULAR, ld->colors[VS::LIGHT_COLOR_SPECULAR].components);

	switch (ld->type) {

		case VS::LIGHT_DIRECTIONAL: {
			/* This doesn't have attenuation */

			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glLoadIdentity();
			Vector3 v(0.0, 0.0, -1.0); // directional lights point up by default
			v = p_instance->transform.get_basis().xform(v);
			v = camera_transform_inverse.get_basis().xform(v);
			v.normalize(); // this sucks, so it will be optimized at some point
			v = -v;
			float lightpos[4] = { v.x, v.y, v.z, 0.0 };

			glLightfv(glid, GL_POSITION, lightpos); //at modelview

			glPopMatrix();

		} break;
		case VS::LIGHT_OMNI: {

			glLightf(glid, GL_SPOT_CUTOFF, 180.0);
			glLightf(glid, GL_CONSTANT_ATTENUATION, ld->vars[VS::LIGHT_PARAM_ATTENUATION]);
			glLightf(glid, GL_LINEAR_ATTENUATION, ld->vars[VS::LIGHT_PARAM_RADIUS]);
			glLightf(glid, GL_QUADRATIC_ATTENUATION, ld->vars[VS::LIGHT_PARAM_ENERGY]); // wut?

			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glLoadIdentity();
			Vector3 pos = p_instance->transform.get_origin();
			pos = camera_transform_inverse.xform(pos);
			float lightpos[4] = { pos.x, pos.y, pos.z, 1.0 };
			glLightfv(glid, GL_POSITION, lightpos); //at modelview

			glPopMatrix();

		} break;
		case VS::LIGHT_SPOT: {

			glLightf(glid, GL_SPOT_CUTOFF, ld->vars[VS::LIGHT_PARAM_SPOT_ANGLE]);
			glLightf(glid, GL_SPOT_EXPONENT, ld->vars[VS::LIGHT_PARAM_SPOT_ATTENUATION]);
			glLightf(glid, GL_CONSTANT_ATTENUATION, ld->vars[VS::LIGHT_PARAM_ATTENUATION]);
			glLightf(glid, GL_LINEAR_ATTENUATION, ld->vars[VS::LIGHT_PARAM_RADIUS]);
			glLightf(glid, GL_QUADRATIC_ATTENUATION, ld->vars[VS::LIGHT_PARAM_ENERGY]); // wut?

			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glLoadIdentity();

			Vector3 v(0.0, 0.0, -1.0); // directional lights point up by default
			v = p_instance->transform.get_basis().xform(v);
			v = camera_transform_inverse.get_basis().xform(v);
			v.normalize(); // this sucks, so it will be optimized at some point
			float lightdir[4] = { v.x, v.y, v.z, 1.0 };
			glLightfv(glid, GL_SPOT_DIRECTION, lightdir); //at modelview

			v = p_instance->transform.get_origin();
			v = camera_transform_inverse.xform(v);
			float lightpos[4] = { v.x, v.y, v.z, 1.0 };
			glLightfv(glid, GL_POSITION, lightpos); //at modelview

			glPopMatrix();

		} break;
		default: break;
	}
};

void RasterizerIPhone::_setup_lights(LightInstance **p_lights, int p_light_count) {

	for (int i = 0; i < MAX_LIGHTS; i++) {

		if (i < p_light_count) {
			glEnable(GL_LIGHT0 + i);
			_setup_light(p_lights[i], i);
		} else {
			glDisable(GL_LIGHT0 + i);
		}
	}
}

static const int gl_client_states[] = {

	GL_VERTEX_ARRAY,
	GL_NORMAL_ARRAY,
	-1, // ARRAY_TANGENT
	GL_COLOR_ARRAY,
	GL_TEXTURE_COORD_ARRAY, // ARRAY_TEX_UV
	GL_TEXTURE_COORD_ARRAY, // ARRAY_TEX_UV2
	-1, // ARRAY_BONES
	-1, // ARRAY_WEIGHTS
	-1, // ARRAY_INDEX
};

void RasterizerIPhone::_setup_geometry(const Geometry *p_geometry, const Material *p_material) {

	switch (p_geometry->type) {

		case Geometry::GEOMETRY_SURFACE: {

			Surface *surf = (Surface *)p_geometry;
			uint8_t *base = 0;
			bool use_VBO = (surf->array_local == 0);

			if (!use_VBO) {

				base = surf->array_local;
				glBindBuffer(GL_ARRAY_BUFFER, 0);

			} else {

				glBindBuffer(GL_ARRAY_BUFFER, surf->vertex_id);
			};

			const Surface::ArrayData *a = surf->array;
			for (int i = 0; i < VS::ARRAY_MAX; i++) {

				const Surface::ArrayData &ad = surf->array[i];
				if (ad.size == 0) {
					if (gl_client_states[i] != -1) {
						glDisableClientState(gl_client_states[i]);
					};
					continue; // this one is disabled.
				}
				ERR_CONTINUE(!ad.configured);

				if (gl_client_states[i] != -1) {
					glEnableClientState(gl_client_states[i]);
				};

				switch (i) {

					case VS::ARRAY_VERTEX:
						if (!use_VBO)
							glVertexPointer(3, GL_FLOAT, surf->stride, (GLvoid *)&base[a->ofs]);
						else if (surf->array[VS::ARRAY_BONES].size)
							glVertexPointer(3, GL_FLOAT, 0, skinned_buffer);
						else
							glVertexPointer(3, GL_FLOAT, surf->stride, (GLvoid *)a->ofs);
						break;

					case VS::ARRAY_NORMAL:
						if (use_VBO)
							glNormalPointer(GL_FLOAT, surf->stride, (GLvoid *)a->ofs);
						else
							glNormalPointer(GL_FLOAT, surf->stride, (GLvoid *)&base[a->ofs]);
						break;
					case VS::ARRAY_TANGENT:
						break;
					case VS::ARRAY_COLOR:
						if (use_VBO)
							glColorPointer(4, GL_UNSIGNED_BYTE, surf->stride, (GLvoid *)a->ofs);
						else
							glColorPointer(4, GL_UNSIGNED_BYTE, surf->stride, (GLvoid *)&base[a->ofs]);
						break;
					case VS::ARRAY_TEX_UV:
					case VS::ARRAY_TEX_UV2:
						if (use_VBO)
							glTexCoordPointer(2, GL_FLOAT, surf->stride, (GLvoid *)a->ofs);
						else
							glTexCoordPointer(2, GL_FLOAT, surf->stride, &base[a->ofs]);
						break;
					case VS::ARRAY_BONES:
					case VS::ARRAY_WEIGHTS:
					case VS::ARRAY_INDEX:
						break;
				};
			}

			// process skeleton here

		} break;

		default: break;
	};
};

static const GLenum gl_primitive[] = {
	GL_POINTS,
	GL_LINES,
	GL_LINE_STRIP,
	GL_LINE_LOOP,
	GL_TRIANGLES,
	GL_TRIANGLE_STRIP,
	GL_TRIANGLE_FAN
};

void RasterizerIPhone::_render(const Geometry *p_geometry, const Material *p_material, const Skeleton *p_skeleton) {

	switch (p_geometry->type) {

		case Geometry::GEOMETRY_SURFACE: {

			Surface *s = (Surface *)p_geometry;

			if (s->index_array_len > 0) {

				if (s->index_array_local) {

					glDrawElements(gl_primitive[s->primitive], s->index_array_len, (s->index_array_len > (1 << 8)) ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE, s->index_array_local);

				} else {

					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->index_id);
					glDrawElements(gl_primitive[s->primitive], s->index_array_len, (s->index_array_len > (1 << 8)) ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE, 0);
				}

			} else {

				glDrawArrays(gl_primitive[s->primitive], 0, s->array_len);
			};
		} break;

		default: break;
	};
};

void RasterizerIPhone::_render_list_forward(RenderList *p_render_list) {

	const Material *prev_material = NULL;
	uint64_t prev_light_hash = 0;
	const Skeleton *prev_skeleton = NULL;
	const Geometry *prev_geometry = NULL;
	const ParamOverrideMap *prev_overrides = NULL; // make it different than NULL

	Geometry::Type prev_geometry_type = Geometry::GEOMETRY_INVALID;

	glMatrixMode(GL_PROJECTION);
	glLoadMatrixf(&camera_projection.matrix[0][0]);

	for (int i = 0; i < p_render_list->element_count; i++) {

		RenderList::Element *e = p_render_list->elements[i];
		const Material *material = e->material;
		uint64_t light_hash = e->light_hash;
		const Skeleton *skeleton = e->skeleton;
		const Geometry *geometry = e->geometry;
		const ParamOverrideMap *material_overrides = e->material_overrides;

		if (material != prev_material || geometry->type != prev_geometry_type) {
			_setup_material(e->geometry, material);
			//_setup_material_overrides(e->material,NULL,material_overrides);
			//_setup_material_skeleton(material,skeleton);
		} else {

			if (material_overrides != prev_overrides) {

				//_setup_material_overrides(e->material,prev_overrides,material_overrides);
			}

			if (prev_skeleton != skeleton) {
				//_setup_material_skeleton(material,skeleton);
			};
		}

		if (geometry != prev_geometry || geometry->type != prev_geometry_type) {

			_setup_geometry(geometry, material);
		};

		if (i == 0 || light_hash != prev_light_hash)
			_setup_lights(e->lights, e->light_count);

		glMatrixMode(GL_MODELVIEW);
		_gl_load_transform(camera_transform_inverse);
		_gl_mult_transform(e->transform);

		_render(geometry, material, skeleton);

		prev_material = material;
		prev_skeleton = skeleton;
		prev_geometry = geometry;
		prev_light_hash = e->light_hash;
		prev_geometry_type = geometry->type;
		prev_overrides = material_overrides;
	}
};

void RasterizerIPhone::end_scene() {

	glEnable(GL_BLEND);
	glDepthMask(GL_FALSE);

	opaque_render_list.sort_mat_light();
	_render_list_forward(&opaque_render_list);

	glDisable(GL_BLEND);
	glDepthMask(GL_TRUE);

	alpha_render_list.sort_z();
	_render_list_forward(&alpha_render_list);
}
void RasterizerIPhone::end_shadow_map() {
}

void RasterizerIPhone::end_frame() {

	//ContextGL::get_singleton()->swap_buffers();
}

/* CANVAS API */

void RasterizerIPhone::canvas_begin() {

	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glLineWidth(1.0);
	glDisable(GL_LIGHTING);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
}
void RasterizerIPhone::canvas_set_transparency(float p_transparency) {
}

void RasterizerIPhone::canvas_set_rect(const Rect2 &p_rect, bool p_clip) {

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glScalef(2.0 / window_size.x, -2.0 / window_size.y, 0);
	glTranslatef((-(window_size.x / 2.0)) + p_rect.pos.x, (-(window_size.y / 2.0)) + p_rect.pos.y, 0);

	if (p_clip) {

		glEnable(GL_SCISSOR_TEST);
		glScissor(viewport.x + p_rect.pos.x, viewport.y + (viewport.height - (p_rect.pos.y + p_rect.size.height)),
				p_rect.size.width, p_rect.size.height);
	} else {

		glDisable(GL_SCISSOR_TEST);
	}
}
void RasterizerIPhone::canvas_draw_line(const Point2 &p_from, const Point2 &p_to, const Color &p_color, float p_width) {

	glColor4f(1, 1, 1, 1);

	float verts[6] = {
		p_from.x, p_from.y, 0,
		p_to.x, p_to.y, 0
	};

	float colors[] = {
		p_color.r, p_color.g, p_color.b, p_color.a,
		p_color.r, p_color.g, p_color.b, p_color.a,
	};
	glLineWidth(p_width);
	_draw_primitive(2, verts, 0, colors, 0);
}

static void _draw_textured_quad(const Rect2 &p_rect, const Rect2 &p_src_region, const Size2 &p_tex_size) {

	float texcoords[] = {
		p_src_region.pos.x / p_tex_size.width,
		p_src_region.pos.y / p_tex_size.height,

		(p_src_region.pos.x + p_src_region.size.width) / p_tex_size.width,
		p_src_region.pos.y / p_tex_size.height,

		(p_src_region.pos.x + p_src_region.size.width) / p_tex_size.width,
		(p_src_region.pos.y + p_src_region.size.height) / p_tex_size.height,

		p_src_region.pos.x / p_tex_size.width,
		(p_src_region.pos.y + p_src_region.size.height) / p_tex_size.height,
	};

	float coords[] = {
		p_rect.pos.x, p_rect.pos.y, 0,
		p_rect.pos.x + p_rect.size.width, p_rect.pos.y, 0,
		p_rect.pos.x + p_rect.size.width, p_rect.pos.y + p_rect.size.height, 0,
		p_rect.pos.x, p_rect.pos.y + p_rect.size.height, 0
	};

	_draw_primitive(4, coords, 0, 0, texcoords);
}

static void _draw_quad(const Rect2 &p_rect) {

	float coords[] = {
		p_rect.pos.x, p_rect.pos.y, 0,
		p_rect.pos.x + p_rect.size.width, p_rect.pos.y, 0,
		p_rect.pos.x + p_rect.size.width, p_rect.pos.y + p_rect.size.height, 0,
		p_rect.pos.x, p_rect.pos.y + p_rect.size.height, 0
	};

	_draw_primitive(4, coords, 0, 0, 0);
}

void RasterizerIPhone::canvas_draw_rect(const Rect2 &p_rect, bool p_region, const Rect2 &p_source, bool p_tile, RID p_texture, const Color &p_modulate) {

	glColor4f(p_modulate.r, p_modulate.g, p_modulate.b, p_modulate.a);

	if (p_texture.is_valid()) {

		glEnable(GL_TEXTURE_2D);
		Texture *texture = texture_owner.get(p_texture);
		ERR_FAIL_COND(!texture);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture->tex_id);

		if (!p_region) {

			Rect2 region = Rect2(0, 0, texture->width, texture->height);
			_draw_textured_quad(p_rect, region, region.size);

		} else {

			_draw_textured_quad(p_rect, p_source, Size2(texture->width, texture->height));
		}
	} else {

		_draw_quad(p_rect);
	}
}
void RasterizerIPhone::canvas_draw_style_box(const Rect2 &p_rect, const Rect2 &p_src_region, RID p_texture, const float *p_margin, bool p_draw_center) {

	glColor4f(1, 1, 1, 1);

	Texture *texture = texture_owner.get(p_texture);
	ERR_FAIL_COND(!texture);

	glEnable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture->tex_id);

	Rect2 region = p_src_region;
	if (region.size.width <= 0)
		region.size.width = texture->width;
	if (region.size.height <= 0)
		region.size.height = texture->height;
	/* CORNERS */
	_draw_textured_quad( // top left
			Rect2(p_rect.pos, Size2(p_margin[MARGIN_LEFT], p_margin[MARGIN_TOP])),
			Rect2(region.pos, Size2(p_margin[MARGIN_LEFT], p_margin[MARGIN_TOP])),
			Size2(texture->width, texture->height));

	_draw_textured_quad( // top right
			Rect2(Point2(p_rect.pos.x + p_rect.size.width - p_margin[MARGIN_RIGHT], p_rect.pos.y), Size2(p_margin[MARGIN_RIGHT], p_margin[MARGIN_TOP])),
			Rect2(Point2(region.pos.x + region.size.width - p_margin[MARGIN_RIGHT], region.pos.y), Size2(p_margin[MARGIN_RIGHT], p_margin[MARGIN_TOP])),
			Size2(texture->width, texture->height));

	_draw_textured_quad( // bottom left
			Rect2(Point2(p_rect.pos.x, p_rect.pos.y + p_rect.size.height - p_margin[MARGIN_BOTTOM]), Size2(p_margin[MARGIN_LEFT], p_margin[MARGIN_BOTTOM])),
			Rect2(Point2(region.pos.x, region.pos.y + region.size.height - p_margin[MARGIN_BOTTOM]), Size2(p_margin[MARGIN_LEFT], p_margin[MARGIN_BOTTOM])),
			Size2(texture->width, texture->height));

	_draw_textured_quad( // bottom right
			Rect2(Point2(p_rect.pos.x + p_rect.size.width - p_margin[MARGIN_RIGHT], p_rect.pos.y + p_rect.size.height - p_margin[MARGIN_BOTTOM]), Size2(p_margin[MARGIN_RIGHT], p_margin[MARGIN_BOTTOM])),
			Rect2(Point2(region.pos.x + region.size.width - p_margin[MARGIN_RIGHT], region.pos.y + region.size.height - p_margin[MARGIN_BOTTOM]), Size2(p_margin[MARGIN_RIGHT], p_margin[MARGIN_BOTTOM])),
			Size2(texture->width, texture->height));

	Rect2 rect_center(p_rect.pos + Point2(p_margin[MARGIN_LEFT], p_margin[MARGIN_TOP]), Size2(p_rect.size.width - p_margin[MARGIN_LEFT] - p_margin[MARGIN_RIGHT], p_rect.size.height - p_margin[MARGIN_TOP] - p_margin[MARGIN_BOTTOM]));

	Rect2 src_center(Point2(region.pos.x + p_margin[MARGIN_LEFT], region.pos.y + p_margin[MARGIN_TOP]), Size2(region.size.width - p_margin[MARGIN_LEFT] - p_margin[MARGIN_RIGHT], region.size.height - p_margin[MARGIN_TOP] - p_margin[MARGIN_BOTTOM]));

	_draw_textured_quad( // top
			Rect2(Point2(rect_center.pos.x, p_rect.pos.y), Size2(rect_center.size.width, p_margin[MARGIN_TOP])),
			Rect2(Point2(src_center.pos.x, region.pos.y), Size2(src_center.size.width, p_margin[MARGIN_TOP])),
			Size2(texture->width, texture->height));

	_draw_textured_quad( // bottom
			Rect2(Point2(rect_center.pos.x, rect_center.pos.y + rect_center.size.height), Size2(rect_center.size.width, p_margin[MARGIN_BOTTOM])),
			Rect2(Point2(src_center.pos.x, src_center.pos.y + src_center.size.height), Size2(src_center.size.width, p_margin[MARGIN_BOTTOM])),
			Size2(texture->width, texture->height));

	_draw_textured_quad( // left
			Rect2(Point2(p_rect.pos.x, rect_center.pos.y), Size2(p_margin[MARGIN_LEFT], rect_center.size.height)),
			Rect2(Point2(region.pos.x, region.pos.y + p_margin[MARGIN_TOP]), Size2(p_margin[MARGIN_LEFT], src_center.size.height)),
			Size2(texture->width, texture->height));

	_draw_textured_quad( // right
			Rect2(Point2(rect_center.pos.x + rect_center.size.width, rect_center.pos.y), Size2(p_margin[MARGIN_RIGHT], rect_center.size.height)),
			Rect2(Point2(src_center.pos.x + src_center.size.width, region.pos.y + p_margin[MARGIN_TOP]), Size2(p_margin[MARGIN_RIGHT], src_center.size.height)),
			Size2(texture->width, texture->height));

	if (p_draw_center) {

		_draw_textured_quad(
				rect_center,
				src_center,
				Size2(texture->width, texture->height));
	}
}
void RasterizerIPhone::canvas_draw_primitive(const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, RID p_texture) {

	ERR_FAIL_COND(p_points.size() < 1);
	float verts[12];
	float uvs[8];
	float colors[16];

	glColor4f(1, 1, 1, 1);

	int idx = 0;
	for (int i = 0; i < p_points.size(); i++) {

		verts[idx++] = p_points[i].x;
		verts[idx++] = p_points[i].y;
		verts[idx++] = 0;
	}

	idx = 0;
	for (int i = 0; i < p_uvs.size(); i++) {

		uvs[idx++] = p_uvs[i].x;
		uvs[idx++] = p_uvs[i].y;
	}

	idx = 0;
	for (int i = 0; i < p_colors.size(); i++) {

		colors[idx++] = p_colors[i].r;
		colors[idx++] = p_colors[i].g;
		colors[idx++] = p_colors[i].b;
		colors[idx++] = p_colors[i].a;
	};

	if (p_texture.is_valid()) {
		glEnable(GL_TEXTURE_2D);
		Texture *texture = texture_owner.get(p_texture);
		if (texture) {
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, texture->tex_id);
		}
	}

	_draw_primitive(p_points.size(), &verts[0], NULL, p_colors.size() ? &colors[0] : NULL, p_uvs.size() ? uvs : NULL);
}

/* FX */

RID RasterizerIPhone::fx_create() {

	return RID();
}
void RasterizerIPhone::fx_get_effects(RID p_fx, List<String> *p_effects) const {
}
void RasterizerIPhone::fx_set_active(RID p_fx, const String &p_effect, bool p_active) {
}
bool RasterizerIPhone::fx_is_active(RID p_fx, const String &p_effect) const {

	return false;
}
void RasterizerIPhone::fx_get_effect_params(RID p_fx, const String &p_effect, List<PropertyInfo> *p_params) const {
}
Variant RasterizerIPhone::fx_get_effect_param(RID p_fx, const String &p_effect, const String &p_param) const {

	return Variant();
}
void RasterizerIPhone::fx_set_effect_param(RID p_fx, const String &p_effect, const String &p_param, const Variant &p_pvalue) {
}

/*MISC*/

bool RasterizerIPhone::is_texture(const RID &p_rid) const {

	return texture_owner.owns(p_rid);
}
bool RasterizerIPhone::is_material(const RID &p_rid) const {

	return material_owner.owns(p_rid);
}
bool RasterizerIPhone::is_mesh(const RID &p_rid) const {

	return mesh_owner.owns(p_rid);
}
bool RasterizerIPhone::is_multimesh(const RID &p_rid) const {

	return false;
}
bool RasterizerIPhone::is_poly(const RID &p_rid) const {

	return poly_owner.owns(p_rid);
}
bool RasterizerIPhone::is_particles(const RID &p_beam) const {

	return false;
}

bool RasterizerIPhone::is_beam(const RID &p_beam) const {

	return false;
}

bool RasterizerIPhone::is_light(const RID &p_rid) const {

	return light_owner.owns(p_rid);
}
bool RasterizerIPhone::is_light_instance(const RID &p_rid) const {

	return light_instance_owner.owns(p_rid);
}
bool RasterizerIPhone::is_particles_instance(const RID &p_rid) const {

	return false;
}
bool RasterizerIPhone::is_skeleton(const RID &p_rid) const {

	return skeleton_owner.owns(p_rid);
}
bool RasterizerIPhone::is_fx(const RID &p_rid) const {

	return fx_owner.owns(p_rid);
}
bool RasterizerIPhone::is_shader(const RID &p_rid) const {

	return false;
}

void RasterizerIPhone::free(const RID &p_rid) const {

	if (texture_owner.owns(p_rid)) {

		// delete the texture
		Texture *texture = texture_owner.get(p_rid);

		glDeleteTextures(1, &texture->tex_id);

		texture_owner.free(p_rid);
		memdelete(texture);

	} else if (material_owner.owns(p_rid)) {

		Material *material = material_owner.get(p_rid);
		ERR_FAIL_COND(!material);

		material_owner.free(p_rid);
		memdelete(material);

	} else if (mesh_owner.owns(p_rid)) {

		Mesh *mesh = mesh_owner.get(p_rid);
		ERR_FAIL_COND(!mesh);
		for (int i = 0; i < mesh->surfaces.size(); i++) {

			Surface *surface = mesh->surfaces[i];
			if (surface->array_local != 0) {
				memfree(surface->array_local);
			};
			if (surface->index_array_local != 0) {
				memfree(surface->index_array_local);
			};

			if (surface->vertex_id)
				glDeleteBuffers(1, &surface->vertex_id);
			if (surface->index_id)
				glDeleteBuffers(1, &surface->index_id);

			memdelete(surface);
		};

		mesh->surfaces.clear();

		mesh_owner.free(p_rid);
		memdelete(mesh);

	} else if (skeleton_owner.owns(p_rid)) {

		Skeleton *skeleton = skeleton_owner.get(p_rid);
		ERR_FAIL_COND(!skeleton)

		skeleton_owner.free(p_rid);
		memdelete(skeleton);

	} else if (light_owner.owns(p_rid)) {

		Light *light = light_owner.get(p_rid);
		ERR_FAIL_COND(!light)

		light_owner.free(p_rid);
		memdelete(light);

	} else if (light_instance_owner.owns(p_rid)) {

		LightInstance *light_instance = light_instance_owner.get(p_rid);
		ERR_FAIL_COND(!light_instance);

		light_instance_owner.free(p_rid);
		memdelete(light_instance);

	} else if (fx_owner.owns(p_rid)) {

		FX *fx = fx_owner.get(p_rid);
		ERR_FAIL_COND(!fx);

		fx_owner.free(p_rid);
		memdelete(fx);
	};
}

void RasterizerIPhone::init() {

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glFrontFace(GL_CW);

	glEnable(GL_TEXTURE_2D);
}

void RasterizerIPhone::finish() {
}

int RasterizerIPhone::get_render_info(VS::RenderInfo p_info) {

	return false;
}

RasterizerIPhone::RasterizerIPhone() {

	frame = 0;
};

RasterizerIPhone::~RasterizerIPhone(){

};

#endif
