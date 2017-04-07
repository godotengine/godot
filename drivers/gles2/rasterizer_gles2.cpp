/*************************************************************************/
/*  rasterizer_gles2.cpp                                                 */
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
#ifdef GLES2_ENABLED

#include "rasterizer_gles2.h"
#include "gl_context/context_gl.h"
#include "global_config.h"
#include "os/os.h"
#include "servers/visual/particle_system_sw.h"
#include "servers/visual/shader_language.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef GLEW_ENABLED
#define _GL_HALF_FLOAT_OES 0x140B
#else
#define _GL_HALF_FLOAT_OES 0x8D61
#endif

#define _GL_RGBA16F_EXT 0x881A
#define _GL_RGB16F_EXT 0x881B
#define _GL_RG16F_EXT 0x822F
#define _GL_R16F_EXT 0x822D
#define _GL_R32F_EXT 0x822E

#define _GL_RED_EXT 0x1903
#define _GL_RG_EXT 0x8227
#define _GL_R8_EXT 0x8229
#define _GL_RG8_EXT 0x822B

#define _DEPTH_COMPONENT24_OES 0x81A6

#ifdef GLEW_ENABLED
#define _glClearDepth glClearDepth
#else
#define _glClearDepth glClearDepthf
#endif

#define _GL_SRGB_EXT 0x8C40
#define _GL_SRGB_ALPHA_EXT 0x8C42

#define _GL_TEXTURE_MAX_ANISOTROPY_EXT 0x84FE
#define _GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT 0x84FF

//#define DEBUG_OPENGL

#ifdef DEBUG_OPENGL

#define DEBUG_TEST_ERROR(m_section)                                         \
	{                                                                       \
		print_line("AT: " + String(m_section));                             \
		glFlush();                                                          \
		uint32_t err = glGetError();                                        \
		if (err) {                                                          \
			print_line("OpenGL Error #" + itos(err) + " at: " + m_section); \
		}                                                                   \
	}

#else

#define DEBUG_TEST_ERROR(m_section)

#endif

static RasterizerGLES2 *_singleton = NULL;

#ifdef GLES_NO_CLIENT_ARRAYS
static float GlobalVertexBuffer[MAX_POLYGON_VERTICES * 8] = { 0 };
#endif

static const GLenum prim_type[] = { GL_POINTS, GL_LINES, GL_TRIANGLES, GL_TRIANGLE_FAN };

_FORCE_INLINE_ static void _set_color_attrib(const Color &p_color) {

	GLfloat c[4] = { p_color.r, p_color.g, p_color.b, p_color.a };
	glVertexAttrib4fv(VS::ARRAY_COLOR, c);
}

static _FORCE_INLINE_ uint16_t make_half_float(float f) {

	union {
		float fv;
		uint32_t ui;
	} ci;
	ci.fv = f;

	unsigned int x = ci.ui;
	unsigned int sign = (unsigned short)(x >> 31);
	unsigned int mantissa;
	unsigned int exp;
	uint16_t hf;

	// get mantissa
	mantissa = x & ((1 << 23) - 1);
	// get exponent bits
	exp = x & (0xFF << 23);
	if (exp >= 0x47800000) {
		// check if the original single precision float number is a NaN
		if (mantissa && (exp == (0xFF << 23))) {
			// we have a single precision NaN
			mantissa = (1 << 23) - 1;
		} else {
			// 16-bit half-float representation stores number as Inf
			mantissa = 0;
		}
		hf = (((uint16_t)sign) << 15) | (uint16_t)((0x1F << 10)) |
			 (uint16_t)(mantissa >> 13);
	}
	// check if exponent is <= -15
	else if (exp <= 0x38000000) {

		/*// store a denorm half-float value or zero
	exp = (0x38000000 - exp) >> 23;
	mantissa >>= (14 + exp);

	hf = (((uint16_t)sign) << 15) | (uint16_t)(mantissa);
	*/
		hf = 0; //denormals do not work for 3D, convert to zero
	} else {
		hf = (((uint16_t)sign) << 15) |
			 (uint16_t)((exp - 0x38000000) >> 13) |
			 (uint16_t)(mantissa >> 13);
	}

	return hf;
}

void RasterizerGLES2::_draw_primitive(int p_points, const Vector3 *p_vertices, const Vector3 *p_normals, const Color *p_colors, const Vector3 *p_uvs, const Plane *p_tangents, int p_instanced) {

	ERR_FAIL_COND(!p_vertices);
	ERR_FAIL_COND(p_points < 1 || p_points > 4);

	bool quad = false;

	GLenum type;
	switch (p_points) {

		case 1: type = GL_POINTS; break;
		case 2: type = GL_LINES; break;
		case 4: quad = true; p_points = 3;
		case 3: type = GL_TRIANGLES; break;
	};

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	GLfloat vert_array[18];
	GLfloat normal_array[18];
	GLfloat color_array[24];
	GLfloat tangent_array[24];
	GLfloat uv_array[18];

	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	glVertexAttribPointer(VS::ARRAY_VERTEX, 3, GL_FLOAT, false, 0, vert_array);

	for (int i = 0; i < p_points; i++) {

		vert_array[i * 3 + 0] = p_vertices[i].x;
		vert_array[i * 3 + 1] = p_vertices[i].y;
		vert_array[i * 3 + 2] = p_vertices[i].z;
		if (quad) {
			int idx = 2 + i;
			if (idx == 4)
				idx = 0;
			vert_array[9 + i * 3 + 0] = p_vertices[idx].x;
			vert_array[9 + i * 3 + 1] = p_vertices[idx].y;
			vert_array[9 + i * 3 + 2] = p_vertices[idx].z;
		}
	}

	if (p_normals) {
		glEnableVertexAttribArray(VS::ARRAY_NORMAL);
		glVertexAttribPointer(VS::ARRAY_NORMAL, 3, GL_FLOAT, false, 0, normal_array);
		for (int i = 0; i < p_points; i++) {

			normal_array[i * 3 + 0] = p_normals[i].x;
			normal_array[i * 3 + 1] = p_normals[i].y;
			normal_array[i * 3 + 2] = p_normals[i].z;
			if (quad) {
				int idx = 2 + i;
				if (idx == 4)
					idx = 0;
				normal_array[9 + i * 3 + 0] = p_normals[idx].x;
				normal_array[9 + i * 3 + 1] = p_normals[idx].y;
				normal_array[9 + i * 3 + 2] = p_normals[idx].z;
			}
		}
	} else {
		glDisableVertexAttribArray(VS::ARRAY_NORMAL);
	}

	if (p_colors) {
		glEnableVertexAttribArray(VS::ARRAY_COLOR);
		glVertexAttribPointer(VS::ARRAY_COLOR, 4, GL_FLOAT, false, 0, color_array);
		for (int i = 0; i < p_points; i++) {

			color_array[i * 4 + 0] = p_colors[i].r;
			color_array[i * 4 + 1] = p_colors[i].g;
			color_array[i * 4 + 2] = p_colors[i].b;
			color_array[i * 4 + 3] = p_colors[i].a;
			if (quad) {
				int idx = 2 + i;
				if (idx == 4)
					idx = 0;
				color_array[12 + i * 4 + 0] = p_colors[idx].r;
				color_array[12 + i * 4 + 1] = p_colors[idx].g;
				color_array[12 + i * 4 + 2] = p_colors[idx].b;
				color_array[12 + i * 4 + 3] = p_colors[idx].a;
			}
		}
	} else {
		glDisableVertexAttribArray(VS::ARRAY_COLOR);
	}

	if (p_tangents) {
		glEnableVertexAttribArray(VS::ARRAY_TANGENT);
		glVertexAttribPointer(VS::ARRAY_TANGENT, 4, GL_FLOAT, false, 0, tangent_array);
		for (int i = 0; i < p_points; i++) {

			tangent_array[i * 4 + 0] = p_tangents[i].normal.x;
			tangent_array[i * 4 + 1] = p_tangents[i].normal.y;
			tangent_array[i * 4 + 2] = p_tangents[i].normal.z;
			tangent_array[i * 4 + 3] = p_tangents[i].d;
			if (quad) {
				int idx = 2 + i;
				if (idx == 4)
					idx = 0;
				tangent_array[12 + i * 4 + 0] = p_tangents[idx].normal.x;
				tangent_array[12 + i * 4 + 1] = p_tangents[idx].normal.y;
				tangent_array[12 + i * 4 + 2] = p_tangents[idx].normal.z;
				tangent_array[12 + i * 4 + 3] = p_tangents[idx].d;
			}
		}
	} else {
		glDisableVertexAttribArray(VS::ARRAY_TANGENT);
	}

	if (p_uvs) {

		glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
		glVertexAttribPointer(VS::ARRAY_TEX_UV, 3, GL_FLOAT, false, 0, uv_array);
		for (int i = 0; i < p_points; i++) {

			uv_array[i * 3 + 0] = p_uvs[i].x;
			uv_array[i * 3 + 1] = p_uvs[i].y;
			uv_array[i * 3 + 2] = p_uvs[i].z;
			if (quad) {
				int idx = 2 + i;
				if (idx == 4)
					idx = 0;
				uv_array[9 + i * 3 + 0] = p_uvs[idx].x;
				uv_array[9 + i * 3 + 1] = p_uvs[idx].y;
				uv_array[9 + i * 3 + 2] = p_uvs[idx].z;
			}
		}

	} else {
		glDisableVertexAttribArray(VS::ARRAY_TEX_UV);
	}

	/*
	if (p_instanced>1)
		glDrawArraysInstanced(type,0,p_points,p_instanced);
	else
	*/

	glDrawArrays(type, 0, quad ? 6 : p_points);
};

/* TEXTURE API */
#define _EXT_COMPRESSED_RGB_PVRTC_4BPPV1_IMG 0x8C00
#define _EXT_COMPRESSED_RGB_PVRTC_2BPPV1_IMG 0x8C01
#define _EXT_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG 0x8C02
#define _EXT_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG 0x8C03

#define _EXT_COMPRESSED_SRGB_PVRTC_2BPPV1_EXT 0x8A54
#define _EXT_COMPRESSED_SRGB_PVRTC_4BPPV1_EXT 0x8A55
#define _EXT_COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV1_EXT 0x8A56
#define _EXT_COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV1_EXT 0x8A57

#define _EXT_COMPRESSED_RGBA_S3TC_DXT1_EXT 0x83F1
#define _EXT_COMPRESSED_RGBA_S3TC_DXT3_EXT 0x83F2
#define _EXT_COMPRESSED_RGBA_S3TC_DXT5_EXT 0x83F3

#define _EXT_COMPRESSED_LUMINANCE_LATC1_EXT 0x8C70
#define _EXT_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT 0x8C71
#define _EXT_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT 0x8C72
#define _EXT_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT 0x8C73

#define _EXT_COMPRESSED_RED_RGTC1_EXT 0x8DBB
#define _EXT_COMPRESSED_RED_RGTC1 0x8DBB
#define _EXT_COMPRESSED_SIGNED_RED_RGTC1 0x8DBC
#define _EXT_COMPRESSED_RG_RGTC2 0x8DBD
#define _EXT_COMPRESSED_SIGNED_RG_RGTC2 0x8DBE
#define _EXT_COMPRESSED_SIGNED_RED_RGTC1_EXT 0x8DBC
#define _EXT_COMPRESSED_RED_GREEN_RGTC2_EXT 0x8DBD
#define _EXT_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT 0x8DBE
#define _EXT_ETC1_RGB8_OES 0x8D64

#define _EXT_SLUMINANCE_NV 0x8C46
#define _EXT_SLUMINANCE_ALPHA_NV 0x8C44
#define _EXT_SRGB8_NV 0x8C41
#define _EXT_SLUMINANCE8_NV 0x8C47
#define _EXT_SLUMINANCE8_ALPHA8_NV 0x8C45

#define _EXT_COMPRESSED_SRGB_S3TC_DXT1_NV 0x8C4C
#define _EXT_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_NV 0x8C4D
#define _EXT_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_NV 0x8C4E
#define _EXT_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_NV 0x8C4F

#define _EXT_ATC_RGB_AMD 0x8C92
#define _EXT_ATC_RGBA_EXPLICIT_ALPHA_AMD 0x8C93
#define _EXT_ATC_RGBA_INTERPOLATED_ALPHA_AMD 0x87EE

/* TEXTURE API */

Image RasterizerGLES2::_get_gl_image_and_format(const Image &p_image, Image::Format p_format, uint32_t p_flags, GLenum &r_gl_format, GLenum &r_gl_internal_format, int &r_gl_components, bool &r_has_alpha_cache, bool &r_compressed) {

	r_has_alpha_cache = false;
	r_compressed = false;
	r_gl_format = 0;
	Image image = p_image;

	switch (p_format) {

		case Image::FORMAT_L8: {
			r_gl_components = 1;
			r_gl_format = GL_LUMINANCE;
			r_gl_internal_format = (srgb_supported && p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) ? _EXT_SLUMINANCE_NV : GL_LUMINANCE;

		} break;
		case Image::FORMAT_INTENSITY: {

			if (!image.empty())
				image.convert(Image::FORMAT_RGBA8);
			r_gl_components = 4;
			r_gl_format = GL_RGBA;
			r_gl_internal_format = (srgb_supported && p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) ? _GL_SRGB_ALPHA_EXT : GL_RGBA;
			r_has_alpha_cache = true;
		} break;
		case Image::FORMAT_LA8: {

			//image.convert(Image::FORMAT_RGBA8);
			r_gl_components = 2;
			r_gl_format = GL_LUMINANCE_ALPHA;
			r_gl_internal_format = (srgb_supported && p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) ? _EXT_SLUMINANCE_ALPHA_NV : GL_LUMINANCE_ALPHA;
			r_has_alpha_cache = true;
		} break;

		case Image::FORMAT_INDEXED: {

			if (!image.empty())
				image.convert(Image::FORMAT_RGB8);
			r_gl_components = 3;
			r_gl_format = GL_RGB;
			r_gl_internal_format = (srgb_supported && p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) ? _GL_SRGB_EXT : GL_RGB;

		} break;

		case Image::FORMAT_INDEXED_ALPHA: {

			if (!image.empty())
				image.convert(Image::FORMAT_RGBA8);
			r_gl_components = 4;

			if (p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) {

				if (srgb_supported) {
					r_gl_format = GL_RGBA;
					r_gl_internal_format = _GL_SRGB_ALPHA_EXT;
				} else {
					r_gl_internal_format = GL_RGBA;
					if (!image.empty())
						image.srgb_to_linear();
				}
			} else {
				r_gl_internal_format = GL_RGBA;
			}
			r_has_alpha_cache = true;

		} break;
		case Image::FORMAT_RGB8: {

			r_gl_components = 3;

			if (p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) {

				if (srgb_supported) {
					r_gl_internal_format = _GL_SRGB_EXT;
					r_gl_format = GL_RGB;
				} else {
					r_gl_internal_format = GL_RGB;
					if (!image.empty())
						image.srgb_to_linear();
				}
			} else {
				r_gl_internal_format = GL_RGB;
			}
		} break;
		case Image::FORMAT_RGBA8: {

			r_gl_components = 4;
			if (p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) {

				if (srgb_supported) {
					r_gl_internal_format = _GL_SRGB_ALPHA_EXT;
					r_gl_format = GL_RGBA;
					//r_gl_internal_format=GL_RGBA;
				} else {
					r_gl_internal_format = GL_RGBA;
					if (!image.empty())
						image.srgb_to_linear();
				}
			} else {
				r_gl_internal_format = GL_RGBA;
			}

			r_has_alpha_cache = true;
		} break;
		case Image::FORMAT_DXT1: {

			if (!s3tc_supported || (!s3tc_srgb_supported && p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)) {

				if (!image.empty()) {
					image.decompress();
				}
				r_gl_components = 4;
				if (p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) {

					if (srgb_supported) {
						r_gl_format = GL_RGBA;
						r_gl_internal_format = _GL_SRGB_ALPHA_EXT;
					} else {
						r_gl_internal_format = GL_RGBA;
						if (!image.empty())
							image.srgb_to_linear();
					}
				} else {
					r_gl_internal_format = GL_RGBA;
				}
				r_has_alpha_cache = true;

			} else {

				r_gl_components = 1; //doesn't matter much
				r_gl_internal_format = (srgb_supported && p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) ? _EXT_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_NV : _EXT_COMPRESSED_RGBA_S3TC_DXT1_EXT;
				r_compressed = true;
			};

		} break;
		case Image::FORMAT_DXT3: {

			if (!s3tc_supported || (!s3tc_srgb_supported && p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)) {

				if (!image.empty()) {
					image.decompress();
				}
				r_gl_components = 4;
				if (p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) {

					if (srgb_supported) {
						r_gl_format = GL_RGBA;
						r_gl_internal_format = _GL_SRGB_ALPHA_EXT;
					} else {
						r_gl_internal_format = GL_RGBA;
						if (!image.empty())
							image.srgb_to_linear();
					}
				} else {
					r_gl_internal_format = GL_RGBA;
				}
				r_has_alpha_cache = true;

			} else {
				r_gl_components = 1; //doesn't matter much
				r_gl_internal_format = (srgb_supported && p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) ? _EXT_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_NV : _EXT_COMPRESSED_RGBA_S3TC_DXT3_EXT;

				r_has_alpha_cache = true;
				r_compressed = true;
			};

		} break;
		case Image::FORMAT_DXT5: {

			if (!s3tc_supported || (!s3tc_srgb_supported && p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)) {

				if (!image.empty()) {
					image.decompress();
				}
				r_gl_components = 4;
				if (p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) {

					if (srgb_supported) {
						r_gl_format = GL_RGBA;
						r_gl_internal_format = _GL_SRGB_ALPHA_EXT;
					} else {
						r_gl_internal_format = GL_RGBA;
						if (!image.empty())
							image.srgb_to_linear();
					}
				} else {
					r_gl_internal_format = GL_RGBA;
				}
				r_has_alpha_cache = true;

			} else {
				r_gl_components = 1; //doesn't matter much
				r_gl_internal_format = (srgb_supported && p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) ? _EXT_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_NV : _EXT_COMPRESSED_RGBA_S3TC_DXT5_EXT;
				r_has_alpha_cache = true;
				r_compressed = true;
			};

		} break;
		case Image::FORMAT_ATI1: {

			if (!latc_supported) {

				if (!image.empty()) {
					image.decompress();
				}
				r_gl_components = 4;
				if (p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) {

					if (srgb_supported) {
						r_gl_format = GL_RGBA;
						r_gl_internal_format = _GL_SRGB_ALPHA_EXT;
					} else {
						r_gl_internal_format = GL_RGBA;
						if (!image.empty())
							image.srgb_to_linear();
					}
				} else {
					r_gl_internal_format = GL_RGBA;
				}
				r_has_alpha_cache = true;

			} else {

				r_gl_internal_format = _EXT_COMPRESSED_LUMINANCE_LATC1_EXT;
				r_gl_components = 1; //doesn't matter much
				r_compressed = true;
			};

		} break;
		case Image::FORMAT_ATI2: {

			if (!latc_supported) {

				if (!image.empty()) {
					image.decompress();
				}
				r_gl_components = 4;
				if (p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) {

					if (srgb_supported) {
						r_gl_format = GL_RGBA;
						r_gl_internal_format = _GL_SRGB_ALPHA_EXT;
					} else {
						r_gl_internal_format = GL_RGBA;
						if (!image.empty())
							image.srgb_to_linear();
					}
				} else {
					r_gl_internal_format = GL_RGBA;
				}
				r_has_alpha_cache = true;

			} else {
				r_gl_internal_format = _EXT_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT;
				r_gl_components = 1; //doesn't matter much
				r_compressed = true;
			};
		} break;
		case Image::FORMAT_PVRTC2: {

			if (!pvr_supported || (!pvr_srgb_supported && p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)) {

				if (!image.empty()) {
					image.decompress();
				}
				r_gl_components = 4;
				if (p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) {

					if (srgb_supported) {
						r_gl_format = GL_RGBA;
						r_gl_internal_format = _GL_SRGB_ALPHA_EXT;
					} else {
						r_gl_internal_format = GL_RGBA;
						if (!image.empty())
							image.srgb_to_linear();
					}
				} else {
					r_gl_internal_format = GL_RGBA;
				}
				r_has_alpha_cache = true;

			} else {

				r_gl_internal_format = (srgb_supported && p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) ? _EXT_COMPRESSED_SRGB_PVRTC_2BPPV1_EXT : _EXT_COMPRESSED_RGB_PVRTC_2BPPV1_IMG;
				r_gl_components = 1; //doesn't matter much
				r_compressed = true;
			}

		} break;
		case Image::FORMAT_PVRTC2A: {

			if (!pvr_supported || (!pvr_srgb_supported && p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)) {

				if (!image.empty())
					image.decompress();
				r_gl_components = 4;
				if (p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) {

					if (srgb_supported) {
						r_gl_format = GL_RGBA;
						r_gl_internal_format = _GL_SRGB_ALPHA_EXT;
					} else {
						r_gl_internal_format = GL_RGBA;
						if (!image.empty())
							image.srgb_to_linear();
					}
				} else {
					r_gl_internal_format = GL_RGBA;
				}
				r_has_alpha_cache = true;

			} else {

				r_gl_internal_format = _EXT_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG;
				r_gl_internal_format = (srgb_supported && p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) ? _EXT_COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV1_EXT : _EXT_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG;
				r_gl_components = 1; //doesn't matter much
				r_compressed = true;
			}

		} break;
		case Image::FORMAT_PVRTC4: {

			if (!pvr_supported || (!pvr_srgb_supported && p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)) {

				if (!image.empty())
					image.decompress();
				r_gl_components = 4;
				if (p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) {

					if (srgb_supported) {
						r_gl_format = GL_RGBA;
						r_gl_internal_format = _GL_SRGB_ALPHA_EXT;
					} else {
						r_gl_internal_format = GL_RGBA;
						if (!image.empty())
							image.srgb_to_linear();
					}
				} else {
					r_gl_internal_format = GL_RGBA;
				}
				r_has_alpha_cache = true;
			} else {

				r_gl_internal_format = (srgb_supported && p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) ? _EXT_COMPRESSED_SRGB_PVRTC_4BPPV1_EXT : _EXT_COMPRESSED_RGB_PVRTC_4BPPV1_IMG;
				r_gl_components = 1; //doesn't matter much
				r_compressed = true;
			}

		} break;
		case Image::FORMAT_PVRTC4A: {

			if (!pvr_supported || (!pvr_srgb_supported && p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)) {

				if (!image.empty())
					image.decompress();
				r_gl_components = 4;
				if (p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) {

					if (srgb_supported) {
						r_gl_format = GL_RGBA;
						r_gl_internal_format = _GL_SRGB_ALPHA_EXT;
					} else {
						r_gl_internal_format = GL_RGBA;
						if (!image.empty())
							image.srgb_to_linear();
					}
				} else {
					r_gl_internal_format = GL_RGBA;
				}
				r_has_alpha_cache = true;

			} else {
				r_gl_internal_format = (srgb_supported && p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) ? _EXT_COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV1_EXT : _EXT_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;
				r_gl_components = 1; //doesn't matter much
				r_compressed = true;
			}

		} break;
		case Image::FORMAT_ETC: {

			if (!etc_supported || p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) {

				if (!image.empty()) {
					image.decompress();
				}
				r_gl_components = 3;
				if (p_flags & VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) {

					if (srgb_supported) {
						r_gl_format = GL_RGB;
						r_gl_internal_format = _GL_SRGB_EXT;
					} else {
						r_gl_internal_format = GL_RGB;
						if (!image.empty())
							image.srgb_to_linear();
					}
				} else {
					r_gl_internal_format = GL_RGB;
				}
				r_gl_internal_format = GL_RGB;

			} else {

				r_gl_internal_format = _EXT_ETC1_RGB8_OES;
				r_gl_components = 1; //doesn't matter much
				r_compressed = true;
			}

		} break;
		case Image::FORMAT_ATC: {

			if (!atitc_supported) {

				if (!image.empty()) {
					image.decompress();
				}
				r_gl_components = 3;
				r_gl_internal_format = GL_RGB;

			} else {

				r_gl_internal_format = _EXT_ATC_RGB_AMD;
				r_gl_components = 1; //doesn't matter much
				r_compressed = true;
			}

		} break;
		case Image::FORMAT_ATC_ALPHA_EXPLICIT: {

			if (!atitc_supported) {

				if (!image.empty()) {
					image.decompress();
				}
				r_gl_components = 4;
				r_gl_internal_format = GL_RGBA;

			} else {

				r_gl_internal_format = _EXT_ATC_RGBA_EXPLICIT_ALPHA_AMD;
				r_gl_components = 1; //doesn't matter much
				r_compressed = true;
			}

		} break;
		case Image::FORMAT_ATC_ALPHA_INTERPOLATED: {

			if (!atitc_supported) {

				if (!image.empty()) {
					image.decompress();
				}
				r_gl_components = 4;
				r_gl_internal_format = GL_RGBA;

			} else {

				r_gl_internal_format = _EXT_ATC_RGBA_INTERPOLATED_ALPHA_AMD;
				r_gl_components = 1; //doesn't matter much
				r_compressed = true;
			}

		} break;
		case Image::FORMAT_YUV_422:
		case Image::FORMAT_YUV_444: {

			if (!image.empty())
				image.convert(Image::FORMAT_RGB8);
			r_gl_internal_format = GL_RGB;
			r_gl_components = 3;

		} break;

		default: {

			ERR_FAIL_V(Image());
		}
	}

	if (r_gl_format == 0) {
		r_gl_format = r_gl_internal_format;
	}

	return image;
}

static const GLenum _cube_side_enum[6] = {

	GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
	GL_TEXTURE_CUBE_MAP_POSITIVE_X,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Z,

};

RID RasterizerGLES2::texture_create() {

	Texture *texture = memnew(Texture);
	ERR_FAIL_COND_V(!texture, RID());
	glGenTextures(1, &texture->tex_id);
	texture->active = false;
	texture->total_data_size = 0;

	return texture_owner.make_rid(texture);
}

void RasterizerGLES2::texture_allocate(RID p_texture, int p_width, int p_height, Image::Format p_format, uint32_t p_flags) {

	bool has_alpha_cache;
	int components;
	GLenum format;
	GLenum internal_format;
	bool compressed;

	int po2_width = nearest_power_of_2(p_width);
	int po2_height = nearest_power_of_2(p_height);

	if (p_flags & VS::TEXTURE_FLAG_VIDEO_SURFACE) {
		p_flags &= ~VS::TEXTURE_FLAG_MIPMAPS; // no mipies for video
	}

	Texture *texture = texture_owner.get(p_texture);
	ERR_FAIL_COND(!texture);
	texture->width = p_width;
	texture->height = p_height;
	texture->format = p_format;
	texture->flags = p_flags;
	texture->target = (p_flags & VS::TEXTURE_FLAG_CUBEMAP) ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D;

	_get_gl_image_and_format(Image(), texture->format, texture->flags, format, internal_format, components, has_alpha_cache, compressed);

	bool scale_textures = !compressed && !(p_flags & VS::TEXTURE_FLAG_VIDEO_SURFACE) && (!npo2_textures_available || p_flags & VS::TEXTURE_FLAG_MIPMAPS);

	if (scale_textures) {
		texture->alloc_width = po2_width;
		texture->alloc_height = po2_height;
		//print_line("scale because npo2: "+itos(npo2_textures_available)+" mm: "+itos(p_format&VS::TEXTURE_FLAG_MIPMAPS)+" "+itos(p_mipmap_count) );
	} else {

		texture->alloc_width = texture->width;
		texture->alloc_height = texture->height;
	};

	if (!(p_flags & VS::TEXTURE_FLAG_VIDEO_SURFACE) && shrink_textures_x2) {
		texture->alloc_height = MAX(1, texture->alloc_height / 2);
		texture->alloc_width = MAX(1, texture->alloc_width / 2);
	}

	texture->gl_components_cache = components;
	texture->gl_format_cache = format;
	texture->gl_internal_format_cache = internal_format;
	texture->format_has_alpha = has_alpha_cache;
	texture->compressed = compressed;
	texture->has_alpha = false; //by default it doesn't have alpha unless something with alpha is blitteds
	texture->data_size = 0;
	texture->mipmaps = 0;

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(texture->target, texture->tex_id);

	if (p_flags & VS::TEXTURE_FLAG_VIDEO_SURFACE) {
		//prealloc if video
		glTexImage2D(texture->target, 0, internal_format, p_width, p_height, 0, format, GL_UNSIGNED_BYTE, NULL);
	}

	texture->active = true;
}

void RasterizerGLES2::texture_set_data(RID p_texture, const Image &p_image, VS::CubeMapSide p_cube_side) {

	Texture *texture = texture_owner.get(p_texture);

	ERR_FAIL_COND(!texture);
	ERR_FAIL_COND(!texture->active);
	ERR_FAIL_COND(texture->render_target);
	ERR_FAIL_COND(texture->format != p_image.get_format());
	ERR_FAIL_COND(p_image.empty());

	int components;
	GLenum format;
	GLenum internal_format;
	bool alpha;
	bool compressed;

	if (keep_copies && !(texture->flags & VS::TEXTURE_FLAG_VIDEO_SURFACE) && !(use_reload_hooks && texture->reloader)) {
		texture->image[p_cube_side] = p_image;
	}

	Image img = _get_gl_image_and_format(p_image, p_image.get_format(), texture->flags, format, internal_format, components, alpha, compressed);

	if (texture->alloc_width != img.get_width() || texture->alloc_height != img.get_height()) {

		if (texture->alloc_width == img.get_width() / 2 && texture->alloc_height == img.get_height() / 2) {

			img.shrink_x2();
		} else if (img.get_format() <= Image::FORMAT_INDEXED_ALPHA) {

			img.resize(texture->alloc_width, texture->alloc_height, Image::INTERPOLATE_BILINEAR);
		}
	};

	if (!(texture->flags & VS::TEXTURE_FLAG_VIDEO_SURFACE) && img.detect_alpha() == Image::ALPHA_BLEND) {
		texture->has_alpha = true;
	}

	GLenum blit_target = (texture->target == GL_TEXTURE_CUBE_MAP) ? _cube_side_enum[p_cube_side] : GL_TEXTURE_2D;

	texture->data_size = img.get_data().size();
	PoolVector<uint8_t>::Read read = img.get_data().read();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(texture->target, texture->tex_id);

	texture->ignore_mipmaps = compressed && img.get_mipmaps() == 0;

	if (texture->flags & VS::TEXTURE_FLAG_MIPMAPS && !texture->ignore_mipmaps)
		glTexParameteri(texture->target, GL_TEXTURE_MIN_FILTER, use_fast_texture_filter ? GL_LINEAR_MIPMAP_NEAREST : GL_LINEAR_MIPMAP_LINEAR);
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

	bool force_clamp_to_edge = !(texture->flags & VS::TEXTURE_FLAG_MIPMAPS && !texture->ignore_mipmaps) && (nearest_power_of_2(texture->alloc_height) != texture->alloc_height || nearest_power_of_2(texture->alloc_width) != texture->alloc_width);

	if (!force_clamp_to_edge && (texture->flags & VS::TEXTURE_FLAG_REPEAT || texture->flags & VS::TEXTURE_FLAG_MIRRORED_REPEAT) && texture->target != GL_TEXTURE_CUBE_MAP) {

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

	if (use_anisotropic_filter) {

		if (texture->flags & VS::TEXTURE_FLAG_ANISOTROPIC_FILTER) {

			glTexParameterf(texture->target, _GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropic_level);
		} else {
			glTexParameterf(texture->target, _GL_TEXTURE_MAX_ANISOTROPY_EXT, 1);
		}
	}

	int mipmaps = (texture->flags & VS::TEXTURE_FLAG_MIPMAPS && img.get_mipmaps() > 0) ? img.get_mipmaps() + 1 : 1;

	int w = img.get_width();
	int h = img.get_height();

	int tsize = 0;
	for (int i = 0; i < mipmaps; i++) {

		int size, ofs;
		img.get_mipmap_offset_and_size(i, ofs, size);

		//print_line("mipmap: "+itos(i)+" size: "+itos(size)+" w: "+itos(mm_w)+", h: "+itos(mm_h));

		if (texture->compressed) {
			glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
			glCompressedTexImage2D(blit_target, i, format, w, h, 0, size, &read[ofs]);

		} else {
			glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
			if (texture->flags & VS::TEXTURE_FLAG_VIDEO_SURFACE) {
				glTexSubImage2D(blit_target, i, 0, 0, w, h, format, GL_UNSIGNED_BYTE, &read[ofs]);
			} else {
				glTexImage2D(blit_target, i, internal_format, w, h, 0, format, GL_UNSIGNED_BYTE, &read[ofs]);
			}
		}
		tsize += size;

		w = MAX(1, w >> 1);
		h = MAX(1, h >> 1);
	}

	_rinfo.texture_mem -= texture->total_data_size;
	texture->total_data_size = tsize;
	_rinfo.texture_mem += texture->total_data_size;

	//printf("texture: %i x %i - size: %i - total: %i\n",texture->width,texture->height,tsize,_rinfo.texture_mem);

	if (texture->flags & VS::TEXTURE_FLAG_MIPMAPS && mipmaps == 1 && !texture->ignore_mipmaps) {
		//generate mipmaps if they were requested and the image does not contain them
		glGenerateMipmap(texture->target);
	}

	texture->mipmaps = mipmaps;

	if (mipmaps > 1) {

		//glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, mipmaps-1 ); - assumed to have all, always
	}

	//texture_set_flags(p_texture,texture->flags);
}

Image RasterizerGLES2::texture_get_data(RID p_texture, VS::CubeMapSide p_cube_side) const {

	Texture *texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture, Image());
	ERR_FAIL_COND_V(!texture->active, Image());
	ERR_FAIL_COND_V(texture->data_size == 0, Image());
	ERR_FAIL_COND_V(texture->render_target, Image());

	return texture->image[p_cube_side];

#if 0

	Texture * texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture,Image());
	ERR_FAIL_COND_V(!texture->active,Image());
	ERR_FAIL_COND_V(texture->data_size==0,Image());

	PoolVector<uint8_t> data;
	GLenum format,type=GL_UNSIGNED_BYTE;
	Image::Format fmt;
	int pixelsize=0;
	int pixelshift=0;
	int minw=1,minh=1;
	bool compressed=false;

	fmt=texture->format;

	switch(texture->format) {

		case Image::FORMAT_L8: {

			format=GL_LUMINANCE;
			type=GL_UNSIGNED_BYTE;
			data.resize(texture->alloc_width*texture->alloc_height);
			pixelsize=1;

		} break;
		case Image::FORMAT_INTENSITY: {
			return Image();
		} break;
		case Image::FORMAT_LA8: {

			format=GL_LUMINANCE_ALPHA;
			type=GL_UNSIGNED_BYTE;
			pixelsize=2;

		} break;
		case Image::FORMAT_RGB8: {
			format=GL_RGB;
			type=GL_UNSIGNED_BYTE;
			pixelsize=3;
		} break;
		case Image::FORMAT_RGBA8: {

			format=GL_RGBA;
			type=GL_UNSIGNED_BYTE;
			pixelsize=4;
		} break;
		case Image::FORMAT_INDEXED: {

			format=GL_RGB;
			type=GL_UNSIGNED_BYTE;
			fmt=Image::FORMAT_RGB8;
			pixelsize=3;
		} break;
		case Image::FORMAT_INDEXED_ALPHA: {

			format=GL_RGBA;
			type=GL_UNSIGNED_BYTE;
			fmt=Image::FORMAT_RGBA8;
			pixelsize=4;

		} break;
		case Image::FORMAT_DXT1: {

			pixelsize=1; //doesn't matter much
			format=GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
			compressed=true;
			pixelshift=1;
			minw=minh=4;

		} break;
		case Image::FORMAT_DXT3: {
			pixelsize=1; //doesn't matter much
			format=GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
			compressed=true;
			minw=minh=4;

		} break;
		case Image::FORMAT_DXT5: {

			pixelsize=1; //doesn't matter much
			format=GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
			compressed=true;
			minw=minh=4;

		} break;
		case Image::FORMAT_ATI1: {

			format=GL_COMPRESSED_RED_RGTC1;
			pixelsize=1; //doesn't matter much
			compressed=true;
			pixelshift=1;
			minw=minh=4;

		} break;
		case Image::FORMAT_ATI2: {

			format=GL_COMPRESSED_RG_RGTC2;
			pixelsize=1; //doesn't matter much
			compressed=true;
			minw=minh=4;

		} break;

		default:{}
	}

	data.resize(texture->data_size);
	PoolVector<uint8_t>::Write wb = data.write();

	glActiveTexture(GL_TEXTURE0);
	int ofs=0;
	glBindTexture(texture->target,texture->tex_id);

	int w=texture->alloc_width;
	int h=texture->alloc_height;
	for(int i=0;i<texture->mipmaps+1;i++) {

		if (compressed) {

			glPixelStorei(GL_PACK_ALIGNMENT, 4);
			glGetCompressedTexImage(texture->target,i,&wb[ofs]);

		} else {
			glPixelStorei(GL_PACK_ALIGNMENT, 1);
			glGetTexImage(texture->target,i,format,type,&wb[ofs]);
		}

		int size = (w*h*pixelsize)>>pixelshift;
		ofs+=size;

		w=MAX(minw,w>>1);
		h=MAX(minh,h>>1);

	}


	wb=PoolVector<uint8_t>::Write();

	Image img(texture->alloc_width,texture->alloc_height,texture->mipmaps,fmt,data);

	if (texture->format<Image::FORMAT_INDEXED && (texture->alloc_width!=texture->width || texture->alloc_height!=texture->height))
		img.resize(texture->width,texture->height);

	return img;
#endif
}

void RasterizerGLES2::texture_set_flags(RID p_texture, uint32_t p_flags) {

	Texture *texture = texture_owner.get(p_texture);
	ERR_FAIL_COND(!texture);
	if (texture->render_target) {

		p_flags &= VS::TEXTURE_FLAG_FILTER; //can change only filter
	}

	bool had_mipmaps = texture->flags & VS::TEXTURE_FLAG_MIPMAPS;

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(texture->target, texture->tex_id);
	uint32_t cube = texture->flags & VS::TEXTURE_FLAG_CUBEMAP;
	texture->flags = p_flags | cube; // can't remove a cube from being a cube

	bool force_clamp_to_edge = !(p_flags & VS::TEXTURE_FLAG_MIPMAPS && !texture->ignore_mipmaps) && (nearest_power_of_2(texture->alloc_height) != texture->alloc_height || nearest_power_of_2(texture->alloc_width) != texture->alloc_width);

	if (!force_clamp_to_edge && (texture->flags & VS::TEXTURE_FLAG_REPEAT || texture->flags & VS::TEXTURE_FLAG_MIRRORED_REPEAT) && texture->target != GL_TEXTURE_CUBE_MAP) {

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

	if (use_anisotropic_filter) {

		if (texture->flags & VS::TEXTURE_FLAG_ANISOTROPIC_FILTER) {

			glTexParameterf(texture->target, _GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropic_level);
		} else {
			glTexParameterf(texture->target, _GL_TEXTURE_MAX_ANISOTROPY_EXT, 1);
		}
	}

	if (texture->flags & VS::TEXTURE_FLAG_MIPMAPS && !texture->ignore_mipmaps) {
		if (!had_mipmaps && texture->mipmaps == 1) {
			glGenerateMipmap(texture->target);
		}
		glTexParameteri(texture->target, GL_TEXTURE_MIN_FILTER, use_fast_texture_filter ? GL_LINEAR_MIPMAP_NEAREST : GL_LINEAR_MIPMAP_LINEAR);

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
uint32_t RasterizerGLES2::texture_get_flags(RID p_texture) const {

	Texture *texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture, 0);

	return texture->flags;
}
Image::Format RasterizerGLES2::texture_get_format(RID p_texture) const {

	Texture *texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture, Image::FORMAT_L8);

	return texture->format;
}
uint32_t RasterizerGLES2::texture_get_width(RID p_texture) const {

	Texture *texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture, 0);

	return texture->width;
}
uint32_t RasterizerGLES2::texture_get_height(RID p_texture) const {

	Texture *texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture, 0);

	return texture->height;
}

bool RasterizerGLES2::texture_has_alpha(RID p_texture) const {

	Texture *texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture, 0);

	return texture->has_alpha;
}

void RasterizerGLES2::texture_set_size_override(RID p_texture, int p_width, int p_height) {

	Texture *texture = texture_owner.get(p_texture);

	ERR_FAIL_COND(!texture);
	ERR_FAIL_COND(texture->render_target);

	ERR_FAIL_COND(p_width <= 0 || p_width > 16384);
	ERR_FAIL_COND(p_height <= 0 || p_height > 16384);
	//real texture size is in alloc width and height
	texture->width = p_width;
	texture->height = p_height;
}

void RasterizerGLES2::texture_set_reload_hook(RID p_texture, ObjectID p_owner, const StringName &p_function) const {

	Texture *texture = texture_owner.get(p_texture);

	ERR_FAIL_COND(!texture);
	ERR_FAIL_COND(texture->render_target);

	texture->reloader = p_owner;
	texture->reloader_func = p_function;
	if (use_reload_hooks && p_owner && keep_copies) {

		for (int i = 0; i < 6; i++)
			texture->image[i] = Image();
	}
}

GLuint RasterizerGLES2::_texture_get_name(RID p_tex) {

	Texture *texture = texture_owner.get(p_tex);
	ERR_FAIL_COND_V(!texture, 0);

	return texture->tex_id;
};

void RasterizerGLES2::texture_set_path(RID p_texture, const String &p_path) {
	Texture *texture = texture_owner.get(p_texture);
	ERR_FAIL_COND(!texture);

	texture->path = p_path;
}

String RasterizerGLES2::texture_get_path(RID p_texture) const {

	Texture *texture = texture_owner.get(p_texture);
	ERR_FAIL_COND_V(!texture, String());
	return texture->path;
}
void RasterizerGLES2::texture_debug_usage(List<VS::TextureInfo> *r_info) {

	List<RID> textures;
	texture_owner.get_owned_list(&textures);

	for (List<RID>::Element *E = textures.front(); E; E = E->next()) {

		Texture *t = texture_owner.get(E->get());
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

void RasterizerGLES2::texture_set_shrink_all_x2_on_set_data(bool p_enable) {

	shrink_textures_x2 = p_enable;
}

/* SHADER API */

RID RasterizerGLES2::shader_create(VS::ShaderMode p_mode) {

	Shader *shader = memnew(Shader);
	shader->mode = p_mode;
	RID rid = shader_owner.make_rid(shader);
	shader_set_mode(rid, p_mode);
	_shader_make_dirty(shader);

	return rid;
}

void RasterizerGLES2::shader_set_mode(RID p_shader, VS::ShaderMode p_mode) {

	ERR_FAIL_INDEX(p_mode, 3);
	Shader *shader = shader_owner.get(p_shader);
	ERR_FAIL_COND(!shader);
	if (shader->custom_code_id && p_mode == shader->mode)
		return;

	if (shader->custom_code_id) {

		switch (shader->mode) {
			case VS::SHADER_MATERIAL: {
				material_shader.free_custom_shader(shader->custom_code_id);
			} break;
			case VS::SHADER_CANVAS_ITEM: {
				canvas_shader.free_custom_shader(shader->custom_code_id);
			} break;
		}

		shader->custom_code_id = 0;
	}

	shader->mode = p_mode;

	switch (shader->mode) {
		case VS::SHADER_MATERIAL: {
			shader->custom_code_id = material_shader.create_custom_shader();
		} break;
		case VS::SHADER_CANVAS_ITEM: {
			shader->custom_code_id = canvas_shader.create_custom_shader();
		} break;
	}
	_shader_make_dirty(shader);
}
VS::ShaderMode RasterizerGLES2::shader_get_mode(RID p_shader) const {

	Shader *shader = shader_owner.get(p_shader);
	ERR_FAIL_COND_V(!shader, VS::SHADER_MATERIAL);
	return shader->mode;
}

void RasterizerGLES2::shader_set_code(RID p_shader, const String &p_vertex, const String &p_fragment, const String &p_light, int p_vertex_ofs, int p_fragment_ofs, int p_light_ofs) {

	Shader *shader = shader_owner.get(p_shader);
	ERR_FAIL_COND(!shader);

#ifdef DEBUG_ENABLED
	if (shader->vertex_code == p_vertex && shader->fragment_code == p_fragment && shader->light_code == p_light)
		return;
#endif
	shader->fragment_code = p_fragment;
	shader->vertex_code = p_vertex;
	shader->light_code = p_light;
	shader->fragment_line = p_fragment_ofs;
	shader->vertex_line = p_vertex_ofs;
	shader->light_line = p_light_ofs;
	_shader_make_dirty(shader);
}

String RasterizerGLES2::shader_get_vertex_code(RID p_shader) const {

	Shader *shader = shader_owner.get(p_shader);
	ERR_FAIL_COND_V(!shader, String());
	return shader->vertex_code;
}

String RasterizerGLES2::shader_get_fragment_code(RID p_shader) const {

	Shader *shader = shader_owner.get(p_shader);
	ERR_FAIL_COND_V(!shader, String());
	return shader->fragment_code;
}

String RasterizerGLES2::shader_get_light_code(RID p_shader) const {

	Shader *shader = shader_owner.get(p_shader);
	ERR_FAIL_COND_V(!shader, String());
	return shader->light_code;
}

void RasterizerGLES2::_shader_make_dirty(Shader *p_shader) {

	if (p_shader->dirty_list.in_list())
		return;

	_shader_dirty_list.add(&p_shader->dirty_list);
}

void RasterizerGLES2::shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const {

	Shader *shader = shader_owner.get(p_shader);
	ERR_FAIL_COND(!shader);

	if (shader->dirty_list.in_list())
		_update_shader(shader); // ok should be not anymore dirty

	Map<int, StringName> order;

	for (Map<StringName, ShaderLanguage::Uniform>::Element *E = shader->uniforms.front(); E; E = E->next()) {

		order[E->get().order] = E->key();
	}

	for (Map<int, StringName>::Element *E = order.front(); E; E = E->next()) {

		PropertyInfo pi;
		ShaderLanguage::Uniform &u = shader->uniforms[E->get()];
		pi.name = E->get();
		switch (u.type) {

			case ShaderLanguage::TYPE_VOID:
			case ShaderLanguage::TYPE_BOOL:
			case ShaderLanguage::TYPE_FLOAT:
			case ShaderLanguage::TYPE_VEC2:
			case ShaderLanguage::TYPE_VEC3:
			case ShaderLanguage::TYPE_MAT3:
			case ShaderLanguage::TYPE_MAT4:
			case ShaderLanguage::TYPE_VEC4:
				pi.type = u.default_value.get_type();
				break;
			case ShaderLanguage::TYPE_TEXTURE:
				pi.type = Variant::_RID;
				pi.hint = PROPERTY_HINT_RESOURCE_TYPE;
				pi.hint_string = "Texture";
				break;
			case ShaderLanguage::TYPE_CUBEMAP:
				pi.type = Variant::_RID;
				pi.hint = PROPERTY_HINT_RESOURCE_TYPE;
				pi.hint_string = "CubeMap";
				break;
		};

		p_param_list->push_back(pi);
	}
}

void RasterizerGLES2::shader_set_default_texture_param(RID p_shader, const StringName &p_name, RID p_texture) {

	Shader *shader = shader_owner.get(p_shader);
	ERR_FAIL_COND(!shader);
	ERR_FAIL_COND(p_texture.is_valid() && !texture_owner.owns(p_texture));

	if (p_texture.is_valid())
		shader->default_textures[p_name] = p_texture;
	else
		shader->default_textures.erase(p_name);

	_shader_make_dirty(shader);
}

RID RasterizerGLES2::shader_get_default_texture_param(RID p_shader, const StringName &p_name) const {
	const Shader *shader = shader_owner.get(p_shader);
	ERR_FAIL_COND_V(!shader, RID());

	const Map<StringName, RID>::Element *E = shader->default_textures.find(p_name);
	if (!E)
		return RID();
	return E->get();
}

Variant RasterizerGLES2::shader_get_default_param(RID p_shader, const StringName &p_name) {

	Shader *shader = shader_owner.get(p_shader);
	ERR_FAIL_COND_V(!shader, Variant());

	//update shader params if necessary
	//make sure the shader is compiled and everything
	//so the actual parameters can be properly retrieved!
	if (shader->dirty_list.in_list()) {
		_update_shader(shader);
	}
	if (shader->valid && shader->uniforms.has(p_name))
		return shader->uniforms[p_name].default_value;

	return Variant();
}

/* COMMON MATERIAL API */

RID RasterizerGLES2::material_create() {

	RID material = material_owner.make_rid(memnew(Material));
	return material;
}

void RasterizerGLES2::material_set_shader(RID p_material, RID p_shader) {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND(!material);
	if (material->shader == p_shader)
		return;
	material->shader = p_shader;
	material->shader_version = 0;
}

RID RasterizerGLES2::material_get_shader(RID p_material) const {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND_V(!material, RID());
	return material->shader;
}

void RasterizerGLES2::material_set_param(RID p_material, const StringName &p_param, const Variant &p_value) {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND(!material);

	Map<StringName, Material::UniformData>::Element *E = material->shader_params.find(p_param);
	if (E) {

		if (p_value.get_type() == Variant::NIL) {

			material->shader_params.erase(E);
			material->shader_version = 0; //get default!
		} else {
			E->get().value = p_value;
			E->get().inuse = true;
		}
	} else {

		if (p_value.get_type() == Variant::NIL)
			return;

		Material::UniformData ud;
		ud.index = -1;
		ud.value = p_value;
		ud.istexture = p_value.get_type() == Variant::_RID; /// cache it being texture
		ud.inuse = true;
		material->shader_params[p_param] = ud; //may be got at some point, or erased
	}
}
Variant RasterizerGLES2::material_get_param(RID p_material, const StringName &p_param) const {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND_V(!material, Variant());

	if (material->shader.is_valid()) {
		//update shader params if necessary
		//make sure the shader is compiled and everything
		//so the actual parameters can be properly retrieved!
		material->shader_cache = shader_owner.get(material->shader);
		if (!material->shader_cache) {
			//invalidate
			material->shader = RID();
			material->shader_cache = NULL;
		} else {

			if (material->shader_cache->dirty_list.in_list())
				_update_shader(material->shader_cache);
			if (material->shader_cache->valid && material->shader_cache->version != material->shader_version) {
				//validate
				_update_material_shader_params(material);
			}
		}
	}

	if (material->shader_params.has(p_param) && material->shader_params[p_param].inuse)
		return material->shader_params[p_param].value;
	else
		return Variant();
}

void RasterizerGLES2::material_set_flag(RID p_material, VS::MaterialFlag p_flag, bool p_enabled) {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND(!material);
	ERR_FAIL_INDEX(p_flag, VS::MATERIAL_FLAG_MAX);

	material->flags[p_flag] = p_enabled;
}
bool RasterizerGLES2::material_get_flag(RID p_material, VS::MaterialFlag p_flag) const {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND_V(!material, false);
	ERR_FAIL_INDEX_V(p_flag, VS::MATERIAL_FLAG_MAX, false);
	return material->flags[p_flag];
}

void RasterizerGLES2::material_set_depth_draw_mode(RID p_material, VS::MaterialDepthDrawMode p_mode) {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND(!material);
	material->depth_draw_mode = p_mode;
}

VS::MaterialDepthDrawMode RasterizerGLES2::material_get_depth_draw_mode(RID p_material) const {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND_V(!material, VS::MATERIAL_DEPTH_DRAW_ALWAYS);
	return material->depth_draw_mode;
}

void RasterizerGLES2::material_set_blend_mode(RID p_material, VS::MaterialBlendMode p_mode) {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND(!material);
	material->blend_mode = p_mode;
}
VS::MaterialBlendMode RasterizerGLES2::material_get_blend_mode(RID p_material) const {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND_V(!material, VS::MATERIAL_BLEND_MODE_ADD);
	return material->blend_mode;
}

void RasterizerGLES2::material_set_line_width(RID p_material, float p_line_width) {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND(!material);
	material->line_width = p_line_width;
}
float RasterizerGLES2::material_get_line_width(RID p_material) const {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND_V(!material, 0);

	return material->line_width;
}

/* MESH API */

RID RasterizerGLES2::mesh_create() {

	return mesh_owner.make_rid(memnew(Mesh));
}

void RasterizerGLES2::mesh_add_surface(RID p_mesh, VS::PrimitiveType p_primitive, const Array &p_arrays, const Array &p_blend_shapes, bool p_alpha_sort) {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND(!mesh);

	ERR_FAIL_INDEX(p_primitive, VS::PRIMITIVE_MAX);
	ERR_FAIL_COND(p_arrays.size() != VS::ARRAY_MAX);

	uint32_t format = 0;

	// validation
	int index_array_len = 0;
	int array_len = 0;

	for (int i = 0; i < p_arrays.size(); i++) {

		if (p_arrays[i].get_type() == Variant::NIL)
			continue;

		format |= (1 << i);

		if (i == VS::ARRAY_VERTEX) {

			array_len = Vector3Array(p_arrays[i]).size();
			ERR_FAIL_COND(array_len == 0);
		} else if (i == VS::ARRAY_INDEX) {

			index_array_len = IntArray(p_arrays[i]).size();
		}
	}

	ERR_FAIL_COND((format & VS::ARRAY_FORMAT_VERTEX) == 0); // mandatory

	ERR_FAIL_COND(mesh->morph_target_count != p_blend_shapes.size());
	if (mesh->morph_target_count) {
		//validate format for morphs
		for (int i = 0; i < p_blend_shapes.size(); i++) {

			uint32_t bsformat = 0;
			Array arr = p_blend_shapes[i];
			for (int j = 0; j < arr.size(); j++) {

				if (arr[j].get_type() != Variant::NIL)
					bsformat |= (1 << j);
			}

			ERR_FAIL_COND((bsformat) != (format & (VS::ARRAY_FORMAT_BONES - 1)));
		}
	}

	Surface *surface = memnew(Surface);
	ERR_FAIL_COND(!surface);

	bool use_VBO = true; //glGenBuffersARB!=NULL; // TODO detect if it's in there
	if ((!use_hw_skeleton_xform && format & VS::ARRAY_FORMAT_WEIGHTS) || mesh->morph_target_count > 0) {

		use_VBO = false;
	}

	//surface->packed=pack_arrays && use_VBO;

	int total_elem_size = 0;

	for (int i = 0; i < VS::ARRAY_MAX; i++) {

		Surface::ArrayData &ad = surface->array[i];
		ad.size = 0;
		ad.ofs = 0;
		int elem_size = 0;
		int elem_count = 0;
		bool valid_local = true;
		GLenum datatype;
		bool normalize = false;
		bool bind = false;

		if (!(format & (1 << i))) // no array
			continue;

		switch (i) {

			case VS::ARRAY_VERTEX: {

				if (use_VBO && use_half_float) {
					elem_size = 3 * sizeof(int16_t); // vertex
					datatype = _GL_HALF_FLOAT_OES;
				} else {

					elem_size = 3 * sizeof(GLfloat); // vertex
					datatype = GL_FLOAT;
				}
				bind = true;
				elem_count = 3;

			} break;
			case VS::ARRAY_NORMAL: {

				if (use_VBO) {
					elem_size = 4 * sizeof(int8_t); // vertex
					datatype = GL_BYTE;
					normalize = true;
				} else {
					elem_size = 3 * sizeof(GLfloat); // vertex
					datatype = GL_FLOAT;
				}
				bind = true;
				elem_count = 3;
			} break;
			case VS::ARRAY_TANGENT: {
				if (use_VBO) {
					elem_size = 4 * sizeof(int8_t); // vertex
					datatype = GL_BYTE;
					normalize = true;
				} else {
					elem_size = 4 * sizeof(GLfloat); // vertex
					datatype = GL_FLOAT;
				}
				bind = true;
				elem_count = 4;

			} break;
			case VS::ARRAY_COLOR: {

				elem_size = 4 * sizeof(uint8_t); /* RGBA */
				datatype = GL_UNSIGNED_BYTE;
				elem_count = 4;
				bind = true;
				normalize = true;
			} break;
			case VS::ARRAY_TEX_UV:
			case VS::ARRAY_TEX_UV2: {
				if (use_VBO && use_half_float) {
					elem_size = 2 * sizeof(int16_t); // vertex
					datatype = _GL_HALF_FLOAT_OES;
				} else {
					elem_size = 2 * sizeof(GLfloat); // vertex
					datatype = GL_FLOAT;
				}
				bind = true;
				elem_count = 2;

			} break;
			case VS::ARRAY_WEIGHTS: {

				if (use_VBO) {

					elem_size = VS::ARRAY_WEIGHTS_SIZE * sizeof(GLushort);
					valid_local = false;
					bind = true;
					normalize = true;
					datatype = GL_UNSIGNED_SHORT;
					elem_count = 4;

				} else {
					elem_size = VS::ARRAY_WEIGHTS_SIZE * sizeof(GLfloat);
					valid_local = false;
					bind = false;
					datatype = GL_FLOAT;
					elem_count = 4;
				}

			} break;
			case VS::ARRAY_BONES: {

				if (use_VBO) {
					elem_size = VS::ARRAY_WEIGHTS_SIZE * sizeof(GLubyte);
					valid_local = false;
					bind = true;
					datatype = GL_UNSIGNED_BYTE;
					elem_count = 4;
				} else {

					elem_size = VS::ARRAY_WEIGHTS_SIZE * sizeof(GLushort);
					valid_local = false;
					bind = false;
					datatype = GL_UNSIGNED_SHORT;
					elem_count = 4;
				}

			} break;
			case VS::ARRAY_INDEX: {

				if (index_array_len <= 0) {
					ERR_PRINT("index_array_len==NO_INDEX_ARRAY");
					break;
				}
				/* determine wether using 16 or 32 bits indices */
				if (array_len > (1 << 16)) {

					elem_size = 4;
					datatype = GL_UNSIGNED_INT;
				} else {
					elem_size = 2;
					datatype = GL_UNSIGNED_SHORT;
				}

				/*
				if (use_VBO) {

					glGenBuffers(1,&surface->index_id);
					ERR_FAIL_COND(surface->index_id==0);
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,surface->index_id);
					glBufferData(GL_ELEMENT_ARRAY_BUFFER,index_array_len*elem_size,NULL,GL_STATIC_DRAW);
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0); //unbind
				} else {
					surface->index_array_local = (uint8_t*)memalloc(index_array_len*elem_size);
				};
*/
				surface->index_array_len = index_array_len; // only way it can exist
				ad.ofs = 0;
				ad.size = elem_size;

				continue;
			} break;
			default: {
				ERR_FAIL();
			}
		}

		ad.ofs = total_elem_size;
		ad.size = elem_size;
		ad.datatype = datatype;
		ad.normalize = normalize;
		ad.bind = bind;
		ad.count = elem_count;
		total_elem_size += elem_size;
		if (valid_local) {
			surface->local_stride += elem_size;
			surface->morph_format |= (1 << i);
		}
	}

	surface->stride = total_elem_size;
	surface->array_len = array_len;
	surface->format = format;
	surface->primitive = p_primitive;
	surface->morph_target_count = mesh->morph_target_count;
	surface->configured_format = 0;
	surface->mesh = mesh;
	if (keep_copies) {
		surface->data = p_arrays;
		surface->morph_data = p_blend_shapes;
	}

	uint8_t *array_ptr = NULL;
	uint8_t *index_array_ptr = NULL;
	PoolVector<uint8_t> array_pre_vbo;
	PoolVector<uint8_t>::Write vaw;
	PoolVector<uint8_t> index_array_pre_vbo;
	PoolVector<uint8_t>::Write iaw;

	/* create pointers */
	if (use_VBO) {

		array_pre_vbo.resize(surface->array_len * surface->stride);
		vaw = array_pre_vbo.write();
		array_ptr = vaw.ptr();

		if (surface->index_array_len) {

			index_array_pre_vbo.resize(surface->index_array_len * surface->array[VS::ARRAY_INDEX].size);
			iaw = index_array_pre_vbo.write();
			index_array_ptr = iaw.ptr();
		}

		_surface_set_arrays(surface, array_ptr, index_array_ptr, p_arrays, true);

	} else {

		surface->array_local = (uint8_t *)memalloc(surface->array_len * surface->stride);
		array_ptr = (uint8_t *)surface->array_local;
		if (surface->index_array_len) {
			surface->index_array_local = (uint8_t *)memalloc(index_array_len * surface->array[VS::ARRAY_INDEX].size);
			index_array_ptr = (uint8_t *)surface->index_array_local;
		}

		_surface_set_arrays(surface, array_ptr, index_array_ptr, p_arrays, true);

		if (mesh->morph_target_count) {

			surface->morph_targets_local = memnew_arr(Surface::MorphTarget, mesh->morph_target_count);
			for (int i = 0; i < mesh->morph_target_count; i++) {

				surface->morph_targets_local[i].array = memnew_arr(uint8_t, surface->local_stride * surface->array_len);
				surface->morph_targets_local[i].configured_format = surface->morph_format;
				_surface_set_arrays(surface, surface->morph_targets_local[i].array, NULL, p_blend_shapes[i], false);
			}
		}
	}

	/* create buffers!! */
	if (use_VBO) {
		glGenBuffers(1, &surface->vertex_id);
		ERR_FAIL_COND(surface->vertex_id == 0);
		glBindBuffer(GL_ARRAY_BUFFER, surface->vertex_id);
		glBufferData(GL_ARRAY_BUFFER, surface->array_len * surface->stride, array_ptr, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind
		if (surface->index_array_len) {

			glGenBuffers(1, &surface->index_id);
			ERR_FAIL_COND(surface->index_id == 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, surface->index_id);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_array_len * surface->array[VS::ARRAY_INDEX].size, index_array_ptr, GL_STATIC_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); //unbind
		}
	}

	mesh->surfaces.push_back(surface);
}

Error RasterizerGLES2::_surface_set_arrays(Surface *p_surface, uint8_t *p_mem, uint8_t *p_index_mem, const Array &p_arrays, bool p_main) {

	uint32_t stride = p_main ? p_surface->stride : p_surface->local_stride;

	for (int ai = 0; ai < VS::ARRAY_MAX; ai++) {
		if (ai >= p_arrays.size())
			break;
		if (p_arrays[ai].get_type() == Variant::NIL)
			continue;
		Surface::ArrayData &a = p_surface->array[ai];

		switch (ai) {

			case VS::ARRAY_VERTEX: {

				ERR_FAIL_COND_V(p_arrays[ai].get_type() != Variant::VECTOR3_ARRAY, ERR_INVALID_PARAMETER);

				PoolVector<Vector3> array = p_arrays[ai];
				ERR_FAIL_COND_V(array.size() != p_surface->array_len, ERR_INVALID_PARAMETER);

				PoolVector<Vector3>::Read read = array.read();
				const Vector3 *src = read.ptr();

				// setting vertices means regenerating the AABB
				AABB aabb;

				float scale = 1;

				if (p_surface->array[VS::ARRAY_VERTEX].datatype == _GL_HALF_FLOAT_OES) {

					for (int i = 0; i < p_surface->array_len; i++) {

						uint16_t vector[3] = { make_half_float(src[i].x), make_half_float(src[i].y), make_half_float(src[i].z) };

						copymem(&p_mem[a.ofs + i * stride], vector, a.size);

						if (i == 0) {

							aabb = AABB(src[i], Vector3());
						} else {

							aabb.expand_to(src[i]);
						}
					}

				} else {
					for (int i = 0; i < p_surface->array_len; i++) {

						GLfloat vector[3] = { src[i].x, src[i].y, src[i].z };

						copymem(&p_mem[a.ofs + i * stride], vector, a.size);

						if (i == 0) {

							aabb = AABB(src[i], Vector3());
						} else {

							aabb.expand_to(src[i]);
						}
					}
				}

				if (p_main) {
					p_surface->aabb = aabb;
					p_surface->vertex_scale = scale;
				}

			} break;
			case VS::ARRAY_NORMAL: {

				ERR_FAIL_COND_V(p_arrays[ai].get_type() != Variant::VECTOR3_ARRAY, ERR_INVALID_PARAMETER);

				PoolVector<Vector3> array = p_arrays[ai];
				ERR_FAIL_COND_V(array.size() != p_surface->array_len, ERR_INVALID_PARAMETER);

				PoolVector<Vector3>::Read read = array.read();
				const Vector3 *src = read.ptr();

				// setting vertices means regenerating the AABB

				if (p_surface->array[VS::ARRAY_NORMAL].datatype == GL_BYTE) {

					for (int i = 0; i < p_surface->array_len; i++) {

						GLbyte vector[4] = {
							CLAMP(src[i].x * 127, -128, 127),
							CLAMP(src[i].y * 127, -128, 127),
							CLAMP(src[i].z * 127, -128, 127),
							0,
						};

						copymem(&p_mem[a.ofs + i * stride], vector, a.size);
					}

				} else {
					for (int i = 0; i < p_surface->array_len; i++) {

						GLfloat vector[3] = { src[i].x, src[i].y, src[i].z };
						copymem(&p_mem[a.ofs + i * stride], vector, a.size);
					}
				}

			} break;
			case VS::ARRAY_TANGENT: {

				ERR_FAIL_COND_V(p_arrays[ai].get_type() != Variant::REAL_ARRAY, ERR_INVALID_PARAMETER);

				PoolVector<real_t> array = p_arrays[ai];

				ERR_FAIL_COND_V(array.size() != p_surface->array_len * 4, ERR_INVALID_PARAMETER);

				PoolVector<real_t>::Read read = array.read();
				const real_t *src = read.ptr();

				if (p_surface->array[VS::ARRAY_TANGENT].datatype == GL_BYTE) {

					for (int i = 0; i < p_surface->array_len; i++) {

						GLbyte xyzw[4] = {
							CLAMP(src[i * 4 + 0] * 127, -128, 127),
							CLAMP(src[i * 4 + 1] * 127, -128, 127),
							CLAMP(src[i * 4 + 2] * 127, -128, 127),
							CLAMP(src[i * 4 + 3] * 127, -128, 127)
						};

						copymem(&p_mem[a.ofs + i * stride], xyzw, a.size);
					}

				} else {
					for (int i = 0; i < p_surface->array_len; i++) {

						GLfloat xyzw[4] = {
							src[i * 4 + 0],
							src[i * 4 + 1],
							src[i * 4 + 2],
							src[i * 4 + 3]
						};

						copymem(&p_mem[a.ofs + i * stride], xyzw, a.size);
					}
				}

			} break;
			case VS::ARRAY_COLOR: {

				ERR_FAIL_COND_V(p_arrays[ai].get_type() != Variant::COLOR_ARRAY, ERR_INVALID_PARAMETER);

				PoolVector<Color> array = p_arrays[ai];

				ERR_FAIL_COND_V(array.size() != p_surface->array_len, ERR_INVALID_PARAMETER);

				PoolVector<Color>::Read read = array.read();
				const Color *src = read.ptr();
				bool alpha = false;

				for (int i = 0; i < p_surface->array_len; i++) {

					if (src[i].a < 0.98) // tolerate alpha a bit, for crappy exporters
						alpha = true;

					uint8_t colors[4];

					for (int j = 0; j < 4; j++) {

						colors[j] = CLAMP(int((src[i][j]) * 255.0), 0, 255);
					}

					copymem(&p_mem[a.ofs + i * stride], colors, a.size);
				}

				if (p_main)
					p_surface->has_alpha = alpha;

			} break;
			case VS::ARRAY_TEX_UV:
			case VS::ARRAY_TEX_UV2: {

				ERR_FAIL_COND_V(p_arrays[ai].get_type() != Variant::VECTOR3_ARRAY && p_arrays[ai].get_type() != Variant::VECTOR2_ARRAY, ERR_INVALID_PARAMETER);

				PoolVector<Vector2> array = p_arrays[ai];

				ERR_FAIL_COND_V(array.size() != p_surface->array_len, ERR_INVALID_PARAMETER);

				PoolVector<Vector2>::Read read = array.read();

				const Vector2 *src = read.ptr();
				float scale = 1.0;

				if (p_surface->array[ai].datatype == _GL_HALF_FLOAT_OES) {

					for (int i = 0; i < p_surface->array_len; i++) {

						uint16_t uv[2] = { make_half_float(src[i].x), make_half_float(src[i].y) };
						copymem(&p_mem[a.ofs + i * stride], uv, a.size);
					}

				} else {
					for (int i = 0; i < p_surface->array_len; i++) {

						GLfloat uv[2] = { src[i].x, src[i].y };

						copymem(&p_mem[a.ofs + i * stride], uv, a.size);
					}
				}

				if (p_main) {

					if (ai == VS::ARRAY_TEX_UV) {

						p_surface->uv_scale = scale;
					}
					if (ai == VS::ARRAY_TEX_UV2) {

						p_surface->uv2_scale = scale;
					}
				}

			} break;
			case VS::ARRAY_WEIGHTS: {

				ERR_FAIL_COND_V(p_arrays[ai].get_type() != Variant::REAL_ARRAY, ERR_INVALID_PARAMETER);

				PoolVector<real_t> array = p_arrays[ai];

				ERR_FAIL_COND_V(array.size() != p_surface->array_len * VS::ARRAY_WEIGHTS_SIZE, ERR_INVALID_PARAMETER);

				PoolVector<real_t>::Read read = array.read();

				const real_t *src = read.ptr();

				if (p_surface->array[VS::ARRAY_WEIGHTS].datatype == GL_UNSIGNED_SHORT) {

					for (int i = 0; i < p_surface->array_len; i++) {

						GLushort data[VS::ARRAY_WEIGHTS_SIZE];
						for (int j = 0; j < VS::ARRAY_WEIGHTS_SIZE; j++) {
							data[j] = CLAMP(src[i * VS::ARRAY_WEIGHTS_SIZE + j] * 65535, 0, 65535);
						}

						copymem(&p_mem[a.ofs + i * stride], data, a.size);
					}
				} else {

					for (int i = 0; i < p_surface->array_len; i++) {

						GLfloat data[VS::ARRAY_WEIGHTS_SIZE];
						for (int j = 0; j < VS::ARRAY_WEIGHTS_SIZE; j++) {
							data[j] = src[i * VS::ARRAY_WEIGHTS_SIZE + j];
						}

						copymem(&p_mem[a.ofs + i * stride], data, a.size);
					}
				}

			} break;
			case VS::ARRAY_BONES: {

				ERR_FAIL_COND_V(p_arrays[ai].get_type() != Variant::REAL_ARRAY, ERR_INVALID_PARAMETER);

				PoolVector<int> array = p_arrays[ai];

				ERR_FAIL_COND_V(array.size() != p_surface->array_len * VS::ARRAY_WEIGHTS_SIZE, ERR_INVALID_PARAMETER);

				PoolVector<int>::Read read = array.read();

				const int *src = read.ptr();

				p_surface->max_bone = 0;

				if (p_surface->array[VS::ARRAY_BONES].datatype == GL_UNSIGNED_BYTE) {

					for (int i = 0; i < p_surface->array_len; i++) {

						GLubyte data[VS::ARRAY_WEIGHTS_SIZE];
						for (int j = 0; j < VS::ARRAY_WEIGHTS_SIZE; j++) {
							data[j] = CLAMP(src[i * VS::ARRAY_WEIGHTS_SIZE + j], 0, 255);
							p_surface->max_bone = MAX(data[j], p_surface->max_bone);
						}

						copymem(&p_mem[a.ofs + i * stride], data, a.size);
					}

				} else {
					for (int i = 0; i < p_surface->array_len; i++) {

						GLushort data[VS::ARRAY_WEIGHTS_SIZE];
						for (int j = 0; j < VS::ARRAY_WEIGHTS_SIZE; j++) {
							data[j] = src[i * VS::ARRAY_WEIGHTS_SIZE + j];
							p_surface->max_bone = MAX(data[j], p_surface->max_bone);
						}

						copymem(&p_mem[a.ofs + i * stride], data, a.size);
					}
				}

			} break;
			case VS::ARRAY_INDEX: {

				ERR_FAIL_COND_V(p_surface->index_array_len <= 0, ERR_INVALID_DATA);
				ERR_FAIL_COND_V(p_arrays[ai].get_type() != Variant::INT_ARRAY, ERR_INVALID_PARAMETER);

				PoolVector<int> indices = p_arrays[ai];
				ERR_FAIL_COND_V(indices.size() == 0, ERR_INVALID_PARAMETER);
				ERR_FAIL_COND_V(indices.size() != p_surface->index_array_len, ERR_INVALID_PARAMETER);

				/* determine wether using 16 or 32 bits indices */

				PoolVector<int>::Read read = indices.read();
				const int *src = read.ptr();

				for (int i = 0; i < p_surface->index_array_len; i++) {

					if (a.size == 2) {
						uint16_t v = src[i];

						copymem(&p_index_mem[i * a.size], &v, a.size);
					} else {
						uint32_t v = src[i];

						copymem(&p_index_mem[i * a.size], &v, a.size);
					}
				}

			} break;

			default: { ERR_FAIL_V(ERR_INVALID_PARAMETER); }
		}

		p_surface->configured_format |= (1 << ai);
	}

	if (p_surface->format & VS::ARRAY_FORMAT_BONES) {
		//create AABBs for each detected bone
		int total_bones = p_surface->max_bone + 1;
		if (p_main) {
			p_surface->skeleton_bone_aabb.resize(total_bones);
			p_surface->skeleton_bone_used.resize(total_bones);
			for (int i = 0; i < total_bones; i++)
				p_surface->skeleton_bone_used[i] = false;
		}
		PoolVector<Vector3> vertices = p_arrays[VS::ARRAY_VERTEX];
		PoolVector<int> bones = p_arrays[VS::ARRAY_BONES];
		PoolVector<float> weights = p_arrays[VS::ARRAY_WEIGHTS];

		bool any_valid = false;

		if (vertices.size() && bones.size() == vertices.size() * 4 && weights.size() == bones.size()) {
			//print_line("MAKING SKELETHONG");
			int vs = vertices.size();
			PoolVector<Vector3>::Read rv = vertices.read();
			PoolVector<int>::Read rb = bones.read();
			PoolVector<float>::Read rw = weights.read();

			Vector<bool> first;
			first.resize(total_bones);
			for (int i = 0; i < total_bones; i++) {
				first[i] = p_main;
			}
			AABB *bptr = p_surface->skeleton_bone_aabb.ptr();
			bool *fptr = first.ptr();
			bool *usedptr = p_surface->skeleton_bone_used.ptr();

			for (int i = 0; i < vs; i++) {

				Vector3 v = rv[i];
				for (int j = 0; j < 4; j++) {

					int idx = rb[i * 4 + j];
					float w = rw[i * 4 + j];
					if (w == 0)
						continue; //break;
					ERR_FAIL_INDEX_V(idx, total_bones, ERR_INVALID_DATA);

					if (fptr[idx]) {
						bptr[idx].pos = v;
						fptr[idx] = false;
						any_valid = true;
					} else {
						bptr[idx].expand_to(v);
					}
					usedptr[idx] = true;
				}
			}
		}

		if (p_main && !any_valid) {

			p_surface->skeleton_bone_aabb.clear();
			p_surface->skeleton_bone_used.clear();
		}
	}

	return OK;
}

void RasterizerGLES2::mesh_add_custom_surface(RID p_mesh, const Variant &p_dat) {

	ERR_EXPLAIN("OpenGL Rasterizer does not support custom surfaces. Running on wrong platform?");
	ERR_FAIL();
}

Array RasterizerGLES2::mesh_get_surface_arrays(RID p_mesh, int p_surface) const {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, Array());
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), Array());
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V(!surface, Array());

	return surface->data;
}
Array RasterizerGLES2::mesh_get_surface_morph_arrays(RID p_mesh, int p_surface) const {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, Array());
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), Array());
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V(!surface, Array());

	return surface->morph_data;
}

void RasterizerGLES2::mesh_set_morph_target_count(RID p_mesh, int p_amount) {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND(!mesh);
	ERR_FAIL_COND(mesh->surfaces.size() != 0);

	mesh->morph_target_count = p_amount;
}

int RasterizerGLES2::mesh_get_morph_target_count(RID p_mesh) const {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, -1);

	return mesh->morph_target_count;
}

void RasterizerGLES2::mesh_set_morph_target_mode(RID p_mesh, VS::MorphTargetMode p_mode) {

	ERR_FAIL_INDEX(p_mode, 2);
	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND(!mesh);

	mesh->morph_target_mode = p_mode;
}

VS::MorphTargetMode RasterizerGLES2::mesh_get_morph_target_mode(RID p_mesh) const {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, VS::MORPH_MODE_NORMALIZED);

	return mesh->morph_target_mode;
}

void RasterizerGLES2::mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material, bool p_owned) {

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

RID RasterizerGLES2::mesh_surface_get_material(RID p_mesh, int p_surface) const {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, RID());
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), RID());
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V(!surface, RID());

	return surface->material;
}

int RasterizerGLES2::mesh_surface_get_array_len(RID p_mesh, int p_surface) const {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, -1);
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), -1);
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V(!surface, -1);

	return surface->array_len;
}
int RasterizerGLES2::mesh_surface_get_array_index_len(RID p_mesh, int p_surface) const {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, -1);
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), -1);
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V(!surface, -1);

	return surface->index_array_len;
}
uint32_t RasterizerGLES2::mesh_surface_get_format(RID p_mesh, int p_surface) const {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, 0);
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), 0);
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V(!surface, 0);

	return surface->format;
}
VS::PrimitiveType RasterizerGLES2::mesh_surface_get_primitive_type(RID p_mesh, int p_surface) const {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, VS::PRIMITIVE_POINTS);
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), VS::PRIMITIVE_POINTS);
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V(!surface, VS::PRIMITIVE_POINTS);

	return surface->primitive;
}

void RasterizerGLES2::mesh_remove_surface(RID p_mesh, int p_index) {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND(!mesh);
	ERR_FAIL_INDEX(p_index, mesh->surfaces.size());
	Surface *surface = mesh->surfaces[p_index];
	ERR_FAIL_COND(!surface);

	if (surface->vertex_id)
		glDeleteBuffers(1, &surface->vertex_id);
	if (surface->index_id)
		glDeleteBuffers(1, &surface->index_id);

	if (mesh->morph_target_count) {
		for (int i = 0; i < mesh->morph_target_count; i++)
			memfree(surface->morph_targets_local[i].array);
		memfree(surface->morph_targets_local);
	}

	memdelete(mesh->surfaces[p_index]);
	mesh->surfaces.remove(p_index);
}
int RasterizerGLES2::mesh_get_surface_count(RID p_mesh) const {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, -1);

	return mesh->surfaces.size();
}

AABB RasterizerGLES2::mesh_get_aabb(RID p_mesh, RID p_skeleton) const {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, AABB());

	if (mesh->custom_aabb != AABB())
		return mesh->custom_aabb;

	Skeleton *sk = NULL;
	if (p_skeleton.is_valid())
		sk = skeleton_owner.get(p_skeleton);

	AABB aabb;
	if (sk && sk->bones.size() != 0) {

		for (int i = 0; i < mesh->surfaces.size(); i++) {

			AABB laabb;
			if (mesh->surfaces[i]->format & VS::ARRAY_FORMAT_BONES && mesh->surfaces[i]->skeleton_bone_aabb.size()) {

				int bs = mesh->surfaces[i]->skeleton_bone_aabb.size();
				const AABB *skbones = mesh->surfaces[i]->skeleton_bone_aabb.ptr();
				const bool *skused = mesh->surfaces[i]->skeleton_bone_used.ptr();

				int sbs = sk->bones.size();
				ERR_CONTINUE(bs > sbs);
				Skeleton::Bone *skb = sk->bones.ptr();

				bool first = true;
				for (int j = 0; j < bs; j++) {

					if (!skused[j])
						continue;
					AABB baabb = skb[j].transform_aabb(skbones[j]);
					if (first) {
						laabb = baabb;
						first = false;
					} else {
						laabb.merge_with(baabb);
					}
				}

			} else {

				laabb = mesh->surfaces[i]->aabb;
			}

			if (i == 0)
				aabb = laabb;
			else
				aabb.merge_with(laabb);
		}
	} else {

		for (int i = 0; i < mesh->surfaces.size(); i++) {

			if (i == 0)
				aabb = mesh->surfaces[i]->aabb;
			else
				aabb.merge_with(mesh->surfaces[i]->aabb);
		}
	}

	return aabb;
}

void RasterizerGLES2::mesh_set_custom_aabb(RID p_mesh, const AABB &p_aabb) {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND(!mesh);

	mesh->custom_aabb = p_aabb;
}

AABB RasterizerGLES2::mesh_get_custom_aabb(RID p_mesh) const {

	const Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND_V(!mesh, AABB());

	return mesh->custom_aabb;
}
/* MULTIMESH API */

RID RasterizerGLES2::multimesh_create() {

	return multimesh_owner.make_rid(memnew(MultiMesh));
}

void RasterizerGLES2::multimesh_set_instance_count(RID p_multimesh, int p_count) {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND(!multimesh);

	//multimesh->elements.clear(); // make sure to delete everything, so it "fails" in all implementations

	if (use_texture_instancing) {

		if (nearest_power_of_2(p_count) != nearest_power_of_2(multimesh->elements.size())) {
			if (multimesh->tex_id) {
				glDeleteTextures(1, &multimesh->tex_id);
				multimesh->tex_id = 0;
			}

			if (p_count) {

				uint32_t po2 = nearest_power_of_2(p_count);
				if (po2 & 0xAAAAAAAA) {
					//half width

					multimesh->tw = Math::sqrt(po2 * 2);
					multimesh->th = multimesh->tw / 2;
				} else {

					multimesh->tw = Math::sqrt(po2);
					multimesh->th = multimesh->tw;
				}
				multimesh->tw *= 4;
				if (multimesh->th == 0)
					multimesh->th = 1;

				glGenTextures(1, &multimesh->tex_id);
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, multimesh->tex_id);

#ifdef GLEW_ENABLED
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, multimesh->tw, multimesh->th, 0, GL_RGBA, GL_FLOAT, NULL);
#else
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, multimesh->tw, multimesh->th, 0, GL_RGBA, GL_FLOAT, NULL);
#endif
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
				glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
				//multimesh->pixel_size=1.0/ps;

				glBindTexture(GL_TEXTURE_2D, 0);
			}
		}

		if (!multimesh->dirty_list.in_list()) {
			_multimesh_dirty_list.add(&multimesh->dirty_list);
		}
	}

	multimesh->elements.resize(p_count);
}
int RasterizerGLES2::multimesh_get_instance_count(RID p_multimesh) const {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, -1);

	return multimesh->elements.size();
}

void RasterizerGLES2::multimesh_set_mesh(RID p_multimesh, RID p_mesh) {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND(!multimesh);

	multimesh->mesh = p_mesh;
}
void RasterizerGLES2::multimesh_set_aabb(RID p_multimesh, const AABB &p_aabb) {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	multimesh->aabb = p_aabb;
}
void RasterizerGLES2::multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform &p_transform) {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	ERR_FAIL_INDEX(p_index, multimesh->elements.size());
	MultiMesh::Element &e = multimesh->elements[p_index];

	e.matrix[0] = p_transform.basis.elements[0][0];
	e.matrix[1] = p_transform.basis.elements[1][0];
	e.matrix[2] = p_transform.basis.elements[2][0];
	e.matrix[3] = 0;
	e.matrix[4] = p_transform.basis.elements[0][1];
	e.matrix[5] = p_transform.basis.elements[1][1];
	e.matrix[6] = p_transform.basis.elements[2][1];
	e.matrix[7] = 0;
	e.matrix[8] = p_transform.basis.elements[0][2];
	e.matrix[9] = p_transform.basis.elements[1][2];
	e.matrix[10] = p_transform.basis.elements[2][2];
	e.matrix[11] = 0;
	e.matrix[12] = p_transform.origin.x;
	e.matrix[13] = p_transform.origin.y;
	e.matrix[14] = p_transform.origin.z;
	e.matrix[15] = 1;

	if (!multimesh->dirty_list.in_list()) {
		_multimesh_dirty_list.add(&multimesh->dirty_list);
	}
}
void RasterizerGLES2::multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color) {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND(!multimesh)
	ERR_FAIL_INDEX(p_index, multimesh->elements.size());
	MultiMesh::Element &e = multimesh->elements[p_index];
	e.color[0] = CLAMP(p_color.r * 255, 0, 255);
	e.color[1] = CLAMP(p_color.g * 255, 0, 255);
	e.color[2] = CLAMP(p_color.b * 255, 0, 255);
	e.color[3] = CLAMP(p_color.a * 255, 0, 255);

	if (!multimesh->dirty_list.in_list()) {
		_multimesh_dirty_list.add(&multimesh->dirty_list);
	}
}

RID RasterizerGLES2::multimesh_get_mesh(RID p_multimesh) const {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, RID());

	return multimesh->mesh;
}
AABB RasterizerGLES2::multimesh_get_aabb(RID p_multimesh) const {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, AABB());

	return multimesh->aabb;
}

Transform RasterizerGLES2::multimesh_instance_get_transform(RID p_multimesh, int p_index) const {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, Transform());

	ERR_FAIL_INDEX_V(p_index, multimesh->elements.size(), Transform());
	MultiMesh::Element &e = multimesh->elements[p_index];

	Transform tr;

	tr.basis.elements[0][0] = e.matrix[0];
	tr.basis.elements[1][0] = e.matrix[1];
	tr.basis.elements[2][0] = e.matrix[2];
	tr.basis.elements[0][1] = e.matrix[4];
	tr.basis.elements[1][1] = e.matrix[5];
	tr.basis.elements[2][1] = e.matrix[6];
	tr.basis.elements[0][2] = e.matrix[8];
	tr.basis.elements[1][2] = e.matrix[9];
	tr.basis.elements[2][2] = e.matrix[10];
	tr.origin.x = e.matrix[12];
	tr.origin.y = e.matrix[13];
	tr.origin.z = e.matrix[14];

	return tr;
}
Color RasterizerGLES2::multimesh_instance_get_color(RID p_multimesh, int p_index) const {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, Color());
	ERR_FAIL_INDEX_V(p_index, multimesh->elements.size(), Color());
	MultiMesh::Element &e = multimesh->elements[p_index];
	Color c;
	c.r = e.color[0] / 255.0;
	c.g = e.color[1] / 255.0;
	c.b = e.color[2] / 255.0;
	c.a = e.color[3] / 255.0;

	return c;
}

void RasterizerGLES2::multimesh_set_visible_instances(RID p_multimesh, int p_visible) {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	multimesh->visible = p_visible;
}

int RasterizerGLES2::multimesh_get_visible_instances(RID p_multimesh) const {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, -1);
	return multimesh->visible;
}

/* IMMEDIATE API */

RID RasterizerGLES2::immediate_create() {

	Immediate *im = memnew(Immediate);
	return immediate_owner.make_rid(im);
}

void RasterizerGLES2::immediate_begin(RID p_immediate, VS::PrimitiveType p_rimitive, RID p_texture) {

	Immediate *im = immediate_owner.get(p_immediate);
	ERR_FAIL_COND(!im);
	ERR_FAIL_COND(im->building);

	Immediate::Chunk ic;
	ic.texture = p_texture;
	ic.primitive = p_rimitive;
	im->chunks.push_back(ic);
	im->mask = 0;
	im->building = true;
}
void RasterizerGLES2::immediate_vertex(RID p_immediate, const Vector3 &p_vertex) {

	Immediate *im = immediate_owner.get(p_immediate);
	ERR_FAIL_COND(!im);
	ERR_FAIL_COND(!im->building);

	Immediate::Chunk *c = &im->chunks.back()->get();

	if (c->vertices.empty() && im->chunks.size() == 1) {

		im->aabb.pos = p_vertex;
		im->aabb.size = Vector3();
	} else {
		im->aabb.expand_to(p_vertex);
	}

	if (im->mask & VS::ARRAY_FORMAT_NORMAL)
		c->normals.push_back(chunk_normal);
	if (im->mask & VS::ARRAY_FORMAT_TANGENT)
		c->tangents.push_back(chunk_tangent);
	if (im->mask & VS::ARRAY_FORMAT_COLOR)
		c->colors.push_back(chunk_color);
	if (im->mask & VS::ARRAY_FORMAT_TEX_UV)
		c->uvs.push_back(chunk_uv);
	if (im->mask & VS::ARRAY_FORMAT_TEX_UV2)
		c->uvs2.push_back(chunk_uv2);
	im->mask |= VS::ARRAY_FORMAT_VERTEX;
	c->vertices.push_back(p_vertex);
}

void RasterizerGLES2::immediate_normal(RID p_immediate, const Vector3 &p_normal) {

	Immediate *im = immediate_owner.get(p_immediate);
	ERR_FAIL_COND(!im);
	ERR_FAIL_COND(!im->building);

	im->mask |= VS::ARRAY_FORMAT_NORMAL;
	chunk_normal = p_normal;
}
void RasterizerGLES2::immediate_tangent(RID p_immediate, const Plane &p_tangent) {

	Immediate *im = immediate_owner.get(p_immediate);
	ERR_FAIL_COND(!im);
	ERR_FAIL_COND(!im->building);

	im->mask |= VS::ARRAY_FORMAT_TANGENT;
	chunk_tangent = p_tangent;
}
void RasterizerGLES2::immediate_color(RID p_immediate, const Color &p_color) {

	Immediate *im = immediate_owner.get(p_immediate);
	ERR_FAIL_COND(!im);
	ERR_FAIL_COND(!im->building);

	im->mask |= VS::ARRAY_FORMAT_COLOR;
	chunk_color = p_color;
}
void RasterizerGLES2::immediate_uv(RID p_immediate, const Vector2 &tex_uv) {

	Immediate *im = immediate_owner.get(p_immediate);
	ERR_FAIL_COND(!im);
	ERR_FAIL_COND(!im->building);

	im->mask |= VS::ARRAY_FORMAT_TEX_UV;
	chunk_uv = tex_uv;
}
void RasterizerGLES2::immediate_uv2(RID p_immediate, const Vector2 &tex_uv) {

	Immediate *im = immediate_owner.get(p_immediate);
	ERR_FAIL_COND(!im);
	ERR_FAIL_COND(!im->building);

	im->mask |= VS::ARRAY_FORMAT_TEX_UV2;
	chunk_uv2 = tex_uv;
}

void RasterizerGLES2::immediate_end(RID p_immediate) {

	Immediate *im = immediate_owner.get(p_immediate);
	ERR_FAIL_COND(!im);
	ERR_FAIL_COND(!im->building);

	im->building = false;
}
void RasterizerGLES2::immediate_clear(RID p_immediate) {

	Immediate *im = immediate_owner.get(p_immediate);
	ERR_FAIL_COND(!im);
	ERR_FAIL_COND(im->building);

	im->chunks.clear();
}

AABB RasterizerGLES2::immediate_get_aabb(RID p_immediate) const {

	Immediate *im = immediate_owner.get(p_immediate);
	ERR_FAIL_COND_V(!im, AABB());
	return im->aabb;
}

void RasterizerGLES2::immediate_set_material(RID p_immediate, RID p_material) {

	Immediate *im = immediate_owner.get(p_immediate);
	ERR_FAIL_COND(!im);
	im->material = p_material;
}

RID RasterizerGLES2::immediate_get_material(RID p_immediate) const {

	const Immediate *im = immediate_owner.get(p_immediate);
	ERR_FAIL_COND_V(!im, RID());
	return im->material;
}

/* PARTICLES API */

RID RasterizerGLES2::particles_create() {

	Particles *particles = memnew(Particles);
	ERR_FAIL_COND_V(!particles, RID());
	return particles_owner.make_rid(particles);
}

void RasterizerGLES2::particles_set_amount(RID p_particles, int p_amount) {

	ERR_FAIL_COND(p_amount < 1);
	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND(!particles);
	particles->data.amount = p_amount;
}

int RasterizerGLES2::particles_get_amount(RID p_particles) const {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND_V(!particles, -1);
	return particles->data.amount;
}

void RasterizerGLES2::particles_set_emitting(RID p_particles, bool p_emitting) {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND(!particles);
	particles->data.emitting = p_emitting;
}
bool RasterizerGLES2::particles_is_emitting(RID p_particles) const {

	const Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND_V(!particles, false);
	return particles->data.emitting;
}

void RasterizerGLES2::particles_set_visibility_aabb(RID p_particles, const AABB &p_visibility) {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND(!particles);
	particles->data.visibility_aabb = p_visibility;
}

void RasterizerGLES2::particles_set_emission_half_extents(RID p_particles, const Vector3 &p_half_extents) {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND(!particles);

	particles->data.emission_half_extents = p_half_extents;
}
Vector3 RasterizerGLES2::particles_get_emission_half_extents(RID p_particles) const {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND_V(!particles, Vector3());

	return particles->data.emission_half_extents;
}

void RasterizerGLES2::particles_set_emission_base_velocity(RID p_particles, const Vector3 &p_base_velocity) {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND(!particles);

	particles->data.emission_base_velocity = p_base_velocity;
}

Vector3 RasterizerGLES2::particles_get_emission_base_velocity(RID p_particles) const {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND_V(!particles, Vector3());

	return particles->data.emission_base_velocity;
}

void RasterizerGLES2::particles_set_emission_points(RID p_particles, const PoolVector<Vector3> &p_points) {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND(!particles);

	particles->data.emission_points = p_points;
}

PoolVector<Vector3> RasterizerGLES2::particles_get_emission_points(RID p_particles) const {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND_V(!particles, PoolVector<Vector3>());

	return particles->data.emission_points;
}

void RasterizerGLES2::particles_set_gravity_normal(RID p_particles, const Vector3 &p_normal) {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND(!particles);

	particles->data.gravity_normal = p_normal;
}
Vector3 RasterizerGLES2::particles_get_gravity_normal(RID p_particles) const {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND_V(!particles, Vector3());

	return particles->data.gravity_normal;
}

AABB RasterizerGLES2::particles_get_visibility_aabb(RID p_particles) const {

	const Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND_V(!particles, AABB());
	return particles->data.visibility_aabb;
}

void RasterizerGLES2::particles_set_variable(RID p_particles, VS::ParticleVariable p_variable, float p_value) {

	ERR_FAIL_INDEX(p_variable, VS::PARTICLE_VAR_MAX);

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND(!particles);
	particles->data.particle_vars[p_variable] = p_value;
}
float RasterizerGLES2::particles_get_variable(RID p_particles, VS::ParticleVariable p_variable) const {

	const Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND_V(!particles, -1);
	return particles->data.particle_vars[p_variable];
}

void RasterizerGLES2::particles_set_randomness(RID p_particles, VS::ParticleVariable p_variable, float p_randomness) {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND(!particles);
	particles->data.particle_randomness[p_variable] = p_randomness;
}
float RasterizerGLES2::particles_get_randomness(RID p_particles, VS::ParticleVariable p_variable) const {

	const Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND_V(!particles, -1);
	return particles->data.particle_randomness[p_variable];
}

void RasterizerGLES2::particles_set_color_phases(RID p_particles, int p_phases) {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND(!particles);
	ERR_FAIL_COND(p_phases < 0 || p_phases > VS::MAX_PARTICLE_COLOR_PHASES);
	particles->data.color_phase_count = p_phases;
}
int RasterizerGLES2::particles_get_color_phases(RID p_particles) const {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND_V(!particles, -1);
	return particles->data.color_phase_count;
}

void RasterizerGLES2::particles_set_color_phase_pos(RID p_particles, int p_phase, float p_pos) {

	ERR_FAIL_INDEX(p_phase, VS::MAX_PARTICLE_COLOR_PHASES);
	if (p_pos < 0.0)
		p_pos = 0.0;
	if (p_pos > 1.0)
		p_pos = 1.0;

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND(!particles);
	particles->data.color_phases[p_phase].pos = p_pos;
}
float RasterizerGLES2::particles_get_color_phase_pos(RID p_particles, int p_phase) const {

	ERR_FAIL_INDEX_V(p_phase, VS::MAX_PARTICLE_COLOR_PHASES, -1.0);

	const Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND_V(!particles, -1);
	return particles->data.color_phases[p_phase].pos;
}

void RasterizerGLES2::particles_set_color_phase_color(RID p_particles, int p_phase, const Color &p_color) {

	ERR_FAIL_INDEX(p_phase, VS::MAX_PARTICLE_COLOR_PHASES);
	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND(!particles);
	particles->data.color_phases[p_phase].color = p_color;

	//update alpha
	particles->has_alpha = false;
	for (int i = 0; i < VS::MAX_PARTICLE_COLOR_PHASES; i++) {
		if (particles->data.color_phases[i].color.a < 0.99)
			particles->has_alpha = true;
	}
}

Color RasterizerGLES2::particles_get_color_phase_color(RID p_particles, int p_phase) const {

	ERR_FAIL_INDEX_V(p_phase, VS::MAX_PARTICLE_COLOR_PHASES, Color());

	const Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND_V(!particles, Color());
	return particles->data.color_phases[p_phase].color;
}

void RasterizerGLES2::particles_set_attractors(RID p_particles, int p_attractors) {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND(!particles);
	ERR_FAIL_COND(p_attractors < 0 || p_attractors > VisualServer::MAX_PARTICLE_ATTRACTORS);
	particles->data.attractor_count = p_attractors;
}
int RasterizerGLES2::particles_get_attractors(RID p_particles) const {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND_V(!particles, -1);
	return particles->data.attractor_count;
}

void RasterizerGLES2::particles_set_attractor_pos(RID p_particles, int p_attractor, const Vector3 &p_pos) {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND(!particles);
	ERR_FAIL_INDEX(p_attractor, particles->data.attractor_count);
	particles->data.attractors[p_attractor].pos = p_pos;
}
Vector3 RasterizerGLES2::particles_get_attractor_pos(RID p_particles, int p_attractor) const {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND_V(!particles, Vector3());
	ERR_FAIL_INDEX_V(p_attractor, particles->data.attractor_count, Vector3());
	return particles->data.attractors[p_attractor].pos;
}

void RasterizerGLES2::particles_set_attractor_strength(RID p_particles, int p_attractor, float p_force) {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND(!particles);
	ERR_FAIL_INDEX(p_attractor, particles->data.attractor_count);
	particles->data.attractors[p_attractor].force = p_force;
}

float RasterizerGLES2::particles_get_attractor_strength(RID p_particles, int p_attractor) const {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND_V(!particles, 0);
	ERR_FAIL_INDEX_V(p_attractor, particles->data.attractor_count, 0);
	return particles->data.attractors[p_attractor].force;
}

void RasterizerGLES2::particles_set_material(RID p_particles, RID p_material, bool p_owned) {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND(!particles);
	if (particles->material_owned && particles->material.is_valid())
		free(particles->material);

	particles->material_owned = p_owned;

	particles->material = p_material;
}
RID RasterizerGLES2::particles_get_material(RID p_particles) const {

	const Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND_V(!particles, RID());
	return particles->material;
}

void RasterizerGLES2::particles_set_use_local_coordinates(RID p_particles, bool p_enable) {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND(!particles);
	particles->data.local_coordinates = p_enable;
}

bool RasterizerGLES2::particles_is_using_local_coordinates(RID p_particles) const {

	const Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND_V(!particles, false);
	return particles->data.local_coordinates;
}
bool RasterizerGLES2::particles_has_height_from_velocity(RID p_particles) const {

	const Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND_V(!particles, false);
	return particles->data.height_from_velocity;
}

void RasterizerGLES2::particles_set_height_from_velocity(RID p_particles, bool p_enable) {

	Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND(!particles);
	particles->data.height_from_velocity = p_enable;
}

AABB RasterizerGLES2::particles_get_aabb(RID p_particles) const {

	const Particles *particles = particles_owner.get(p_particles);
	ERR_FAIL_COND_V(!particles, AABB());
	return particles->data.visibility_aabb;
}

/* SKELETON API */

RID RasterizerGLES2::skeleton_create() {

	Skeleton *skeleton = memnew(Skeleton);
	ERR_FAIL_COND_V(!skeleton, RID());
	return skeleton_owner.make_rid(skeleton);
}
void RasterizerGLES2::skeleton_resize(RID p_skeleton, int p_bones) {

	Skeleton *skeleton = skeleton_owner.get(p_skeleton);
	ERR_FAIL_COND(!skeleton);
	if (p_bones == skeleton->bones.size()) {
		return;
	};
	if (use_hw_skeleton_xform) {

		if (nearest_power_of_2(p_bones) != nearest_power_of_2(skeleton->bones.size())) {
			if (skeleton->tex_id) {
				glDeleteTextures(1, &skeleton->tex_id);
				skeleton->tex_id = 0;
			}

			if (p_bones) {

				glGenTextures(1, &skeleton->tex_id);
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, skeleton->tex_id);
				int ps = nearest_power_of_2(p_bones * 3);
#ifdef GLEW_ENABLED
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, ps, 1, 0, GL_RGBA, GL_FLOAT, skel_default.ptr());
#else
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ps, 1, 0, GL_RGBA, GL_FLOAT, skel_default.ptr());
#endif
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
				glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
				skeleton->pixel_size = 1.0 / ps;

				glBindTexture(GL_TEXTURE_2D, 0);
			}
		}

		if (!skeleton->dirty_list.in_list()) {
			_skeleton_dirty_list.add(&skeleton->dirty_list);
		}
	}
	skeleton->bones.resize(p_bones);
}
int RasterizerGLES2::skeleton_get_bone_count(RID p_skeleton) const {

	Skeleton *skeleton = skeleton_owner.get(p_skeleton);
	ERR_FAIL_COND_V(!skeleton, -1);
	return skeleton->bones.size();
}
void RasterizerGLES2::skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform &p_transform) {

	Skeleton *skeleton = skeleton_owner.get(p_skeleton);
	ERR_FAIL_COND(!skeleton);
	ERR_FAIL_INDEX(p_bone, skeleton->bones.size());

	Skeleton::Bone &b = skeleton->bones[p_bone];

	b.mtx[0][0] = p_transform.basis[0][0];
	b.mtx[0][1] = p_transform.basis[1][0];
	b.mtx[0][2] = p_transform.basis[2][0];
	b.mtx[1][0] = p_transform.basis[0][1];
	b.mtx[1][1] = p_transform.basis[1][1];
	b.mtx[1][2] = p_transform.basis[2][1];
	b.mtx[2][0] = p_transform.basis[0][2];
	b.mtx[2][1] = p_transform.basis[1][2];
	b.mtx[2][2] = p_transform.basis[2][2];
	b.mtx[3][0] = p_transform.origin[0];
	b.mtx[3][1] = p_transform.origin[1];
	b.mtx[3][2] = p_transform.origin[2];

	if (skeleton->tex_id) {
		if (!skeleton->dirty_list.in_list()) {
			_skeleton_dirty_list.add(&skeleton->dirty_list);
		}
	}
}

Transform RasterizerGLES2::skeleton_bone_get_transform(RID p_skeleton, int p_bone) {

	Skeleton *skeleton = skeleton_owner.get(p_skeleton);
	ERR_FAIL_COND_V(!skeleton, Transform());
	ERR_FAIL_INDEX_V(p_bone, skeleton->bones.size(), Transform());

	const Skeleton::Bone &b = skeleton->bones[p_bone];

	Transform t;
	t.basis[0][0] = b.mtx[0][0];
	t.basis[1][0] = b.mtx[0][1];
	t.basis[2][0] = b.mtx[0][2];
	t.basis[0][1] = b.mtx[1][0];
	t.basis[1][1] = b.mtx[1][1];
	t.basis[2][1] = b.mtx[1][2];
	t.basis[0][2] = b.mtx[2][0];
	t.basis[1][2] = b.mtx[2][1];
	t.basis[2][2] = b.mtx[2][2];
	t.origin[0] = b.mtx[3][0];
	t.origin[1] = b.mtx[3][1];
	t.origin[2] = b.mtx[3][2];

	return t;
}

/* LIGHT API */

RID RasterizerGLES2::light_create(VS::LightType p_type) {

	Light *light = memnew(Light);
	light->type = p_type;
	return light_owner.make_rid(light);
}

VS::LightType RasterizerGLES2::light_get_type(RID p_light) const {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light, VS::LIGHT_OMNI);
	return light->type;
}

void RasterizerGLES2::light_set_color(RID p_light, VS::LightColor p_type, const Color &p_color) {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND(!light);
	ERR_FAIL_INDEX(p_type, 3);
	light->colors[p_type] = p_color;
}
Color RasterizerGLES2::light_get_color(RID p_light, VS::LightColor p_type) const {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light, Color());
	ERR_FAIL_INDEX_V(p_type, 3, Color());
	return light->colors[p_type];
}

void RasterizerGLES2::light_set_shadow(RID p_light, bool p_enabled) {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND(!light);
	light->shadow_enabled = p_enabled;
}

bool RasterizerGLES2::light_has_shadow(RID p_light) const {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light, false);
	return light->shadow_enabled;
}

void RasterizerGLES2::light_set_volumetric(RID p_light, bool p_enabled) {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND(!light);
	light->volumetric_enabled = p_enabled;
}
bool RasterizerGLES2::light_is_volumetric(RID p_light) const {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light, false);
	return light->volumetric_enabled;
}

void RasterizerGLES2::light_set_projector(RID p_light, RID p_texture) {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND(!light);
	light->projector = p_texture;
}
RID RasterizerGLES2::light_get_projector(RID p_light) const {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light, RID());
	return light->projector;
}

void RasterizerGLES2::light_set_var(RID p_light, VS::LightParam p_var, float p_value) {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND(!light);
	ERR_FAIL_INDEX(p_var, VS::LIGHT_PARAM_MAX);

	light->vars[p_var] = p_value;
}
float RasterizerGLES2::light_get_var(RID p_light, VS::LightParam p_var) const {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light, 0);

	ERR_FAIL_INDEX_V(p_var, VS::LIGHT_PARAM_MAX, 0);

	return light->vars[p_var];
}

void RasterizerGLES2::light_set_operator(RID p_light, VS::LightOp p_op){

};

VS::LightOp RasterizerGLES2::light_get_operator(RID p_light) const {

	return VS::LightOp();
};

void RasterizerGLES2::light_omni_set_shadow_mode(RID p_light, VS::LightOmniShadowMode p_mode) {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND(!light);

	light->omni_shadow_mode = p_mode;
}
VS::LightOmniShadowMode RasterizerGLES2::light_omni_get_shadow_mode(RID p_light) const {

	const Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light, VS::LIGHT_OMNI_SHADOW_DEFAULT);

	return light->omni_shadow_mode;
}

void RasterizerGLES2::light_directional_set_shadow_mode(RID p_light, VS::LightDirectionalShadowMode p_mode) {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND(!light);

	light->directional_shadow_mode = p_mode;
}

VS::LightDirectionalShadowMode RasterizerGLES2::light_directional_get_shadow_mode(RID p_light) const {

	const Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light, VS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL);

	return light->directional_shadow_mode;
}

void RasterizerGLES2::light_directional_set_shadow_param(RID p_light, VS::LightDirectionalShadowParam p_param, float p_value) {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND(!light);

	light->directional_shadow_param[p_param] = p_value;
}

float RasterizerGLES2::light_directional_get_shadow_param(RID p_light, VS::LightDirectionalShadowParam p_param) const {

	const Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light, 0);
	return light->directional_shadow_param[p_param];
}

AABB RasterizerGLES2::light_get_aabb(RID p_light) const {

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

RID RasterizerGLES2::light_instance_create(RID p_light) {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light, RID());

	LightInstance *light_instance = memnew(LightInstance);

	light_instance->light = p_light;
	light_instance->base = light;
	light_instance->last_pass = 0;

	return light_instance_owner.make_rid(light_instance);
}
void RasterizerGLES2::light_instance_set_transform(RID p_light_instance, const Transform &p_transform) {

	LightInstance *lighti = light_instance_owner.get(p_light_instance);
	ERR_FAIL_COND(!lighti);
	lighti->transform = p_transform;
}

Rasterizer::ShadowType RasterizerGLES2::light_instance_get_shadow_type(RID p_light_instance, bool p_far) const {

	LightInstance *lighti = light_instance_owner.get(p_light_instance);
	ERR_FAIL_COND_V(!lighti, Rasterizer::SHADOW_NONE);

	switch (lighti->base->type) {

		case VS::LIGHT_DIRECTIONAL: {
			switch (lighti->base->directional_shadow_mode) {
				case VS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL: {
					return SHADOW_ORTHOGONAL;
				} break;
				case VS::LIGHT_DIRECTIONAL_SHADOW_PERSPECTIVE: {
					return SHADOW_PSM;
				} break;
				case VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS:
				case VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS: {
					return SHADOW_PSSM;
				} break;
			}

		} break;
		case VS::LIGHT_OMNI: return SHADOW_DUAL_PARABOLOID; break;
		case VS::LIGHT_SPOT: return SHADOW_SIMPLE; break;
	}

	return Rasterizer::SHADOW_NONE;
}

int RasterizerGLES2::light_instance_get_shadow_passes(RID p_light_instance) const {

	LightInstance *lighti = light_instance_owner.get(p_light_instance);
	ERR_FAIL_COND_V(!lighti, 0);

	if (lighti->base->type == VS::LIGHT_DIRECTIONAL && lighti->base->directional_shadow_mode == VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS) {

		return 4; // dp4
	} else if (lighti->base->type == VS::LIGHT_OMNI || (lighti->base->type == VS::LIGHT_DIRECTIONAL && lighti->base->directional_shadow_mode == VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS)) {
		return 2; // dp
	} else
		return 1;
}

bool RasterizerGLES2::light_instance_get_pssm_shadow_overlap(RID p_light_instance) const {

	return shadow_filter >= SHADOW_FILTER_ESM;
}

void RasterizerGLES2::light_instance_set_shadow_transform(RID p_light_instance, int p_index, const CameraMatrix &p_camera, const Transform &p_transform, float p_split_near, float p_split_far) {

	LightInstance *lighti = light_instance_owner.get(p_light_instance);
	ERR_FAIL_COND(!lighti);

	ERR_FAIL_COND(lighti->base->type != VS::LIGHT_DIRECTIONAL);
	//ERR_FAIL_INDEX(p_index,1);

	lighti->custom_projection[p_index] = p_camera;
	lighti->custom_transform[p_index] = p_transform;
	lighti->shadow_split[p_index] = 1.0 / p_split_far;
#if 0
	if (p_index==0) {
		lighti->custom_projection=p_camera;
		lighti->custom_transform=p_transform;
		//Plane p(0,0,-p_split_far,1);
		//p=camera_projection.xform4(p);
		//lighti->shadow_split=p.normal.z/p.d;
		lighti->shadow_split=1.0/p_split_far;

		//lighti->shadow_split=-p_split_far;
	} else {

		lighti->custom_projection2=p_camera;
		lighti->custom_transform2=p_transform;
		lighti->shadow_split2=p_split_far;

	}
#endif
}

int RasterizerGLES2::light_instance_get_shadow_size(RID p_light_instance, int p_index) const {

	LightInstance *lighti = light_instance_owner.get(p_light_instance);
	ERR_FAIL_COND_V(!lighti, 1);
	ERR_FAIL_COND_V(!lighti->near_shadow_buffer, 256);
	return lighti->near_shadow_buffer->size / 2;
}

void RasterizerGLES2::shadow_clear_near() {

	for (int i = 0; i < near_shadow_buffers.size(); i++) {

		if (near_shadow_buffers[i].owner)
			near_shadow_buffers[i].owner->clear_near_shadow_buffers();
	}
}

bool RasterizerGLES2::shadow_allocate_near(RID p_light) {

	if (!use_shadow_mapping || !use_framebuffers)
		return false;

	LightInstance *li = light_instance_owner.get(p_light);
	ERR_FAIL_COND_V(!li, false);
	ERR_FAIL_COND_V(li->near_shadow_buffer, false);

	int skip = 0;
	if (framebuffer.active) {

		int sc = framebuffer.scale;
		while (sc > 1) {
			sc /= 2;
			skip++;
		}
	}

	for (int i = 0; i < near_shadow_buffers.size(); i++) {

		if (skip > 0) {
			skip--;
			continue;
		}

		if (near_shadow_buffers[i].owner != NULL)
			continue;

		near_shadow_buffers[i].owner = li;
		li->near_shadow_buffer = &near_shadow_buffers[i];
		return true;
	}

	return false;
}

bool RasterizerGLES2::shadow_allocate_far(RID p_light) {

	return false;
}

/* PARTICLES INSTANCE */

RID RasterizerGLES2::particles_instance_create(RID p_particles) {

	ERR_FAIL_COND_V(!particles_owner.owns(p_particles), RID());
	ParticlesInstance *particles_instance = memnew(ParticlesInstance);
	ERR_FAIL_COND_V(!particles_instance, RID());
	particles_instance->particles = p_particles;
	return particles_instance_owner.make_rid(particles_instance);
}

void RasterizerGLES2::particles_instance_set_transform(RID p_particles_instance, const Transform &p_transform) {

	ParticlesInstance *particles_instance = particles_instance_owner.get(p_particles_instance);
	ERR_FAIL_COND(!particles_instance);
	particles_instance->transform = p_transform;
}

RID RasterizerGLES2::viewport_data_create() {

	ViewportData *vd = memnew(ViewportData);

	glActiveTexture(GL_TEXTURE0);
	glGenFramebuffers(1, &vd->lum_fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, vd->lum_fbo);

	GLuint format_luminance = use_fp16_fb ? _GL_RG_EXT : GL_RGBA;
	GLuint format_luminance_type = use_fp16_fb ? (full_float_fb_supported ? GL_FLOAT : _GL_HALF_FLOAT_OES) : GL_UNSIGNED_BYTE;
	GLuint format_luminance_components = use_fp16_fb ? _GL_RG_EXT : GL_RGBA;

	glGenTextures(1, &vd->lum_color);
	glBindTexture(GL_TEXTURE_2D, vd->lum_color);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	/*
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0,
			GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	*/
	glTexImage2D(GL_TEXTURE_2D, 0, format_luminance, 1, 1, 0,
			format_luminance_components, format_luminance_type, NULL);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			GL_TEXTURE_2D, vd->lum_color, 0);

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

	glBindFramebuffer(GL_FRAMEBUFFER, base_framebuffer);
	DEBUG_TEST_ERROR("Viewport Data Init");
	if (status != GL_FRAMEBUFFER_COMPLETE) {
		WARN_PRINT("Can't create framebuffer for vd");
	}

	return viewport_data_owner.make_rid(vd);
}

RID RasterizerGLES2::render_target_create() {

	RenderTarget *rt = memnew(RenderTarget);
	rt->fbo = 0;
	rt->width = 0;
	rt->height = 0;
	rt->last_pass = 0;

	Texture *texture = memnew(Texture);
	texture->active = false;
	texture->total_data_size = 0;
	texture->render_target = rt;
	texture->ignore_mipmaps = true;
	rt->texture_ptr = texture;
	rt->texture = texture_owner.make_rid(texture);
	rt->texture_ptr->active = false;
	return render_target_owner.make_rid(rt);
}
void RasterizerGLES2::render_target_set_size(RID p_render_target, int p_width, int p_height) {

	RenderTarget *rt = render_target_owner.get(p_render_target);

	if (p_width == rt->width && p_height == rt->height)
		return;

	if (rt->width != 0 && rt->height != 0) {

		glDeleteFramebuffers(1, &rt->fbo);
		glDeleteRenderbuffers(1, &rt->depth);
		glDeleteTextures(1, &rt->color);

		rt->fbo = 0;
		rt->depth = 0;
		rt->color = 0;
		rt->width = 0;
		rt->height = 0;
		rt->texture_ptr->tex_id = 0;
		rt->texture_ptr->active = false;
	}

	if (p_width == 0 || p_height == 0)
		return;

	rt->width = p_width;
	rt->height = p_height;

	//fbo
	glGenFramebuffers(1, &rt->fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, rt->fbo);

	//depth
	if (!low_memory_2d) {
		glGenRenderbuffers(1, &rt->depth);
		glBindRenderbuffer(GL_RENDERBUFFER, rt->depth);

		glRenderbufferStorage(GL_RENDERBUFFER, use_depth24 ? _DEPTH_COMPONENT24_OES : GL_DEPTH_COMPONENT16, rt->width, rt->height);

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rt->depth);
	}

	//color
	glGenTextures(1, &rt->color);
	glBindTexture(GL_TEXTURE_2D, rt->color);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rt->width, rt->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	if (rt->texture_ptr->flags & VS::TEXTURE_FLAG_FILTER) {

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	} else {

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	}
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt->color, 0);

	rt->texture_ptr->tex_id = rt->color;
	rt->texture_ptr->active = true;
	rt->texture_ptr->width = p_width;
	rt->texture_ptr->height = p_height;

#
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
		rt->texture_ptr->tex_id = 0;
		rt->texture_ptr->active = false;
		WARN_PRINT("Could not create framebuffer!!");
	}

	glBindFramebuffer(GL_FRAMEBUFFER, base_framebuffer);
}

RID RasterizerGLES2::render_target_get_texture(RID p_render_target) const {

	const RenderTarget *rt = render_target_owner.get(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());
	return rt->texture;
}
bool RasterizerGLES2::render_target_renedered_in_frame(RID p_render_target) {

	RenderTarget *rt = render_target_owner.get(p_render_target);
	ERR_FAIL_COND_V(!rt, false);
	return rt->last_pass == frame;
}

/* RENDER API */
/* all calls (inside begin/end shadow) are always warranted to be in the following order: */

void RasterizerGLES2::begin_frame() {

	_update_framebuffer();

	glDepthFunc(GL_LEQUAL);
	glFrontFace(GL_CW);

//fragment_lighting=Globals::get_singleton()->get("rasterizer/use_fragment_lighting");
#ifdef TOOLS_ENABLED
	canvas_shader.set_conditional(CanvasShaderGLES2::USE_PIXEL_SNAP, GLOBAL_DEF("display/use_2d_pixel_snap", false));
	shadow_filter = ShadowFilterTechnique(int(GlobalConfig::get_singleton()->get("rasterizer/shadow_filter")));
#endif

	canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_PCF5, shadow_filter == SHADOW_FILTER_PCF5);
	canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_PCF13, shadow_filter == SHADOW_FILTER_PCF13);
	canvas_shader.set_conditional(CanvasShaderGLES2::SHADOW_ESM, shadow_filter == SHADOW_FILTER_ESM);

	window_size = Size2(OS::get_singleton()->get_video_mode().width, OS::get_singleton()->get_video_mode().height);

	double time = (OS::get_singleton()->get_ticks_usec() / 1000); // get msec
	time /= 1000.0; // make secs
	time_delta = time - last_time;
	last_time = time;
	frame++;

	_rinfo.vertex_count = 0;
	_rinfo.object_count = 0;
	_rinfo.mat_change_count = 0;
	_rinfo.shader_change_count = 0;
	_rinfo.ci_draw_commands = 0;
	_rinfo.surface_count = 0;
	_rinfo.draw_calls = 0;

	_update_fixed_materials();
	while (_shader_dirty_list.first()) {

		_update_shader(_shader_dirty_list.first()->self());
	}

	while (_skeleton_dirty_list.first()) {

		Skeleton *s = _skeleton_dirty_list.first()->self();

		float *sk_float = (float *)skinned_buffer;
		for (int i = 0; i < s->bones.size(); i++) {

			float *m = &sk_float[i * 12];
			const Skeleton::Bone &b = s->bones[i];
			m[0] = b.mtx[0][0];
			m[1] = b.mtx[1][0];
			m[2] = b.mtx[2][0];
			m[3] = b.mtx[3][0];

			m[4] = b.mtx[0][1];
			m[5] = b.mtx[1][1];
			m[6] = b.mtx[2][1];
			m[7] = b.mtx[3][1];

			m[8] = b.mtx[0][2];
			m[9] = b.mtx[1][2];
			m[10] = b.mtx[2][2];
			m[11] = b.mtx[3][2];
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, s->tex_id);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nearest_power_of_2(s->bones.size() * 3), 1, GL_RGBA, GL_FLOAT, sk_float);
		_skeleton_dirty_list.remove(_skeleton_dirty_list.first());
	}

	while (_multimesh_dirty_list.first()) {

		MultiMesh *s = _multimesh_dirty_list.first()->self();

		float *sk_float = (float *)skinned_buffer;
		for (int i = 0; i < s->elements.size(); i++) {

			float *m = &sk_float[i * 16];
			const float *im = s->elements[i].matrix;
			for (int j = 0; j < 16; j++) {
				m[j] = im[j];
			}
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, s->tex_id);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, s->tw, s->th, GL_RGBA, GL_FLOAT, sk_float);
		_multimesh_dirty_list.remove(_multimesh_dirty_list.first());
	}

	draw_next_frame = false;
	//material_shader.set_uniform_default(MaterialShaderGLES2::SCREENZ_SCALE, Math::fmod(time, 3600.0));
	/* nehe ?*/

	//glClearColor(0,0,1,1);
	//glClear(GL_COLOR_BUFFER_BIT); //should not clear if anything else cleared..
}

void RasterizerGLES2::capture_viewport(Image *r_capture) {
#if 0
	PoolVector<uint8_t> pixels;
	pixels.resize(viewport.width*viewport.height*3);
	PoolVector<uint8_t>::Write w = pixels.write();
#ifdef GLEW_ENABLED
	glReadBuffer(GL_COLOR_ATTACHMENT0);
#endif
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	if (current_rt)
		glReadPixels( 0, 0, viewport.width, viewport.height,GL_RGB,GL_UNSIGNED_BYTE,w.ptr() );
	else
		glReadPixels( viewport.x, window_size.height-(viewport.height+viewport.y), viewport.width,viewport.height,GL_RGB,GL_UNSIGNED_BYTE,w.ptr());

	glPixelStorei(GL_PACK_ALIGNMENT, 4);

	w=PoolVector<uint8_t>::Write();

	r_capture->create(viewport.width,viewport.height,0,Image::FORMAT_RGB8,pixels);
#else

	PoolVector<uint8_t> pixels;
	pixels.resize(viewport.width * viewport.height * 4);
	PoolVector<uint8_t>::Write w = pixels.write();
	glPixelStorei(GL_PACK_ALIGNMENT, 4);

	//uint64_t time = OS::get_singleton()->get_ticks_usec();

	if (current_rt) {
#ifdef GLEW_ENABLED
		glReadBuffer(GL_COLOR_ATTACHMENT0);
#endif
		glReadPixels(0, 0, viewport.width, viewport.height, GL_RGBA, GL_UNSIGNED_BYTE, w.ptr());
	} else {
		// back?
		glReadPixels(viewport.x, window_size.height - (viewport.height + viewport.y), viewport.width, viewport.height, GL_RGBA, GL_UNSIGNED_BYTE, w.ptr());
	}

	bool flip = current_rt == NULL;

	if (flip) {
		uint32_t *imgptr = (uint32_t *)w.ptr();
		for (int y = 0; y < (viewport.height / 2); y++) {

			uint32_t *ptr1 = &imgptr[y * viewport.width];
			uint32_t *ptr2 = &imgptr[(viewport.height - y - 1) * viewport.width];

			for (int x = 0; x < viewport.width; x++) {

				uint32_t tmp = ptr1[x];
				ptr1[x] = ptr2[x];
				ptr2[x] = tmp;
			}
		}
	}

	w = PoolVector<uint8_t>::Write();
	r_capture->create(viewport.width, viewport.height, 0, Image::FORMAT_RGBA8, pixels);
//r_capture->flip_y();

#endif
}

void RasterizerGLES2::clear_viewport(const Color &p_color) {

	if (current_rt || using_canvas_bg) {

		glScissor(0, 0, viewport.width, viewport.height);
	} else {
		glScissor(viewport.x, window_size.height - (viewport.height + viewport.y), viewport.width, viewport.height);
	}

	glEnable(GL_SCISSOR_TEST);
	glClearColor(p_color.r, p_color.g, p_color.b, p_color.a);
	glClear(GL_COLOR_BUFFER_BIT); //should not clear if anything else cleared..
	glDisable(GL_SCISSOR_TEST);
};

void RasterizerGLES2::set_render_target(RID p_render_target, bool p_transparent_bg, bool p_vflip) {

	if (!p_render_target.is_valid()) {
		glBindFramebuffer(GL_FRAMEBUFFER, base_framebuffer);
		current_rt = NULL;
		current_rt_vflip = false;

	} else {
		RenderTarget *rt = render_target_owner.get(p_render_target);
		ERR_FAIL_COND(!rt);
		ERR_FAIL_COND(rt->fbo == 0);
		glBindFramebuffer(GL_FRAMEBUFFER, rt->fbo);
		current_rt = rt;
		current_rt_transparent = p_transparent_bg;
		current_rt_vflip = !p_vflip;
	}
}

void RasterizerGLES2::set_viewport(const VS::ViewportRect &p_viewport) {

	viewport = p_viewport;
	//viewport.width/=2;
	//viewport.height/=2;
	//print_line("viewport: "+itos(p_viewport.x)+","+itos(p_viewport.y)+","+itos(p_viewport.width)+","+itos(p_viewport.height));

	if (current_rt) {

		glViewport(0, 0, viewport.width, viewport.height);
	} else {
		glViewport(viewport.x, window_size.height - (viewport.height + viewport.y), viewport.width, viewport.height);
	}
}

void RasterizerGLES2::begin_scene(RID p_viewport_data, RID p_env, VS::ScenarioDebugMode p_debug) {

	current_debug = p_debug;
	opaque_render_list.clear();
	alpha_render_list.clear();
	light_instance_count = 0;
	current_env = p_env.is_valid() ? environment_owner.get(p_env) : NULL;
	scene_pass++;
	last_light_id = 0;
	directional_light_count = 0;
	lights_use_shadow = false;
	texscreen_used = false;
	current_vd = viewport_data_owner.get(p_viewport_data);
	if (current_debug == VS::SCENARIO_DEBUG_WIREFRAME) {
#ifdef GLEW_ENABLED
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
#endif
	}

	//set state

	glCullFace(GL_FRONT);
	cull_front = true;
};

void RasterizerGLES2::begin_shadow_map(RID p_light_instance, int p_shadow_pass) {

	ERR_FAIL_COND(shadow);
	shadow = light_instance_owner.get(p_light_instance);
	shadow_pass = p_shadow_pass;
	ERR_FAIL_COND(!shadow);

	opaque_render_list.clear();
	alpha_render_list.clear();
	//pre_zpass_render_list.clear();
	light_instance_count = 0;

	glCullFace(GL_FRONT);
	cull_front = true;
}

void RasterizerGLES2::set_camera(const Transform &p_world, const CameraMatrix &p_projection, bool p_ortho_hint) {

	camera_transform = p_world;
	if (current_rt && current_rt_vflip) {
		camera_transform.basis.set_axis(1, -camera_transform.basis.get_axis(1));
	}
	camera_transform_inverse = camera_transform.inverse();
	camera_projection = p_projection;
	camera_plane = Plane(camera_transform.origin, -camera_transform.basis.get_axis(2));
	camera_z_near = camera_projection.get_z_near();
	camera_z_far = camera_projection.get_z_far();
	camera_projection.get_viewport_size(camera_vp_size.x, camera_vp_size.y);
	camera_ortho = p_ortho_hint;
}

void RasterizerGLES2::add_light(RID p_light_instance) {

#define LIGHT_FADE_TRESHOLD 0.05

	ERR_FAIL_COND(light_instance_count >= MAX_SCENE_LIGHTS);

	LightInstance *li = light_instance_owner.get(p_light_instance);
	ERR_FAIL_COND(!li);

	switch (li->base->type) {

		case VS::LIGHT_DIRECTIONAL: {

			ERR_FAIL_COND(directional_light_count >= RenderList::MAX_LIGHTS);
			directional_lights[directional_light_count++] = li;

			if (li->base->shadow_enabled) {
				CameraMatrix bias;
				bias.set_light_bias();

				int passes = light_instance_get_shadow_passes(p_light_instance);

				for (int i = 0; i < passes; i++) {
					Transform modelview = Transform(camera_transform_inverse * li->custom_transform[i]).inverse();
					li->shadow_projection[i] = bias * li->custom_projection[i] * modelview;
				}

				lights_use_shadow = true;
			}
		} break;
		case VS::LIGHT_OMNI: {

			if (li->base->shadow_enabled) {
				li->shadow_projection[0] = Transform(camera_transform_inverse * li->transform).inverse();
				lights_use_shadow = true;
			}
		} break;
		case VS::LIGHT_SPOT: {

			if (li->base->shadow_enabled) {
				CameraMatrix bias;
				bias.set_light_bias();
				Transform modelview = Transform(camera_transform_inverse * li->transform).inverse();
				li->shadow_projection[0] = bias * li->projection * modelview;
				lights_use_shadow = true;
			}
		} break;
	}

	/* make light hash */

	// actually, not really a hash, but helps to sort the lights
	// and avoid recompiling redudant shader versions

	li->last_pass = scene_pass;
	li->sort_key = light_instance_count;

	light_instances[light_instance_count++] = li;
}

void RasterizerGLES2::_update_shader(Shader *p_shader) const {

	_shader_dirty_list.remove(&p_shader->dirty_list);

	p_shader->valid = false;

	p_shader->uniforms.clear();
	Vector<StringName> uniform_names;

	String vertex_code;
	String vertex_globals;
	ShaderCompilerGLES2::Flags vertex_flags;
	ShaderCompilerGLES2::Flags fragment_flags;
	ShaderCompilerGLES2::Flags light_flags;

	if (p_shader->mode == VS::SHADER_MATERIAL) {
		Error err = shader_precompiler.compile(p_shader->vertex_code, ShaderLanguage::SHADER_MATERIAL_VERTEX, vertex_code, vertex_globals, vertex_flags, &p_shader->uniforms);
		if (err) {
			return; //invalid
		}
	} else if (p_shader->mode == VS::SHADER_CANVAS_ITEM) {

		Error err = shader_precompiler.compile(p_shader->vertex_code, ShaderLanguage::SHADER_CANVAS_ITEM_VERTEX, vertex_code, vertex_globals, vertex_flags, &p_shader->uniforms);
		if (err) {
			return; //invalid
		}
	}

	//print_line("compiled vertex: "+vertex_code);
	//print_line("compiled vertex globals: "+vertex_globals);

	//print_line("UCV: "+itos(p_shader->uniforms.size()));
	String fragment_code;
	String fragment_globals;

	if (p_shader->mode == VS::SHADER_MATERIAL) {
		Error err = shader_precompiler.compile(p_shader->fragment_code, ShaderLanguage::SHADER_MATERIAL_FRAGMENT, fragment_code, fragment_globals, fragment_flags, &p_shader->uniforms);
		if (err) {
			return; //invalid
		}
	} else if (p_shader->mode == VS::SHADER_CANVAS_ITEM) {
		Error err = shader_precompiler.compile(p_shader->fragment_code, ShaderLanguage::SHADER_CANVAS_ITEM_FRAGMENT, fragment_code, fragment_globals, fragment_flags, &p_shader->uniforms);
		if (err) {
			return; //invalid
		}
	}

	String light_code;
	String light_globals;

	if (p_shader->mode == VS::SHADER_MATERIAL) {

		Error err = shader_precompiler.compile(p_shader->light_code, (ShaderLanguage::SHADER_MATERIAL_LIGHT), light_code, light_globals, light_flags, &p_shader->uniforms);
		if (err) {
			return; //invalid
		}
	} else if (p_shader->mode == VS::SHADER_CANVAS_ITEM) {
		Error err = shader_precompiler.compile(p_shader->light_code, (ShaderLanguage::SHADER_CANVAS_ITEM_LIGHT), light_code, light_globals, light_flags, &p_shader->uniforms);
		if (err) {
			return; //invalid
		}
	}

	fragment_globals += light_globals; //both fragment anyway

	//print_line("compiled fragment: "+fragment_code);
	//("compiled fragment globals: "+fragment_globals);

	//print_line("UCF: "+itos(p_shader->uniforms.size()));

	int first_tex_index = 0xFFFFF;
	p_shader->first_texture = StringName();

	for (Map<StringName, ShaderLanguage::Uniform>::Element *E = p_shader->uniforms.front(); E; E = E->next()) {

		uniform_names.push_back("_" + String(E->key()));
		if (E->get().type == ShaderLanguage::TYPE_TEXTURE && E->get().order < first_tex_index) {
			p_shader->first_texture = E->key();
			first_tex_index = E->get().order;
		}
	}

	bool uses_time = false;

	if (p_shader->mode == VS::SHADER_MATERIAL) {
		//print_line("setting code to id.. "+itos(p_shader->custom_code_id));
		Vector<const char *> enablers;
		if (fragment_flags.use_color_interp || vertex_flags.use_color_interp)
			enablers.push_back("#define ENABLE_COLOR_INTERP\n");
		if (fragment_flags.use_uv_interp || vertex_flags.use_uv_interp)
			enablers.push_back("#define ENABLE_UV_INTERP\n");
		if (fragment_flags.use_uv2_interp || vertex_flags.use_uv2_interp)
			enablers.push_back("#define ENABLE_UV2_INTERP\n");
		if (fragment_flags.use_tangent_interp || vertex_flags.use_tangent_interp || fragment_flags.uses_normalmap)
			enablers.push_back("#define ENABLE_TANGENT_INTERP\n");
		if (fragment_flags.use_var1_interp || vertex_flags.use_var1_interp)
			enablers.push_back("#define ENABLE_VAR1_INTERP\n");
		if (fragment_flags.use_var2_interp || vertex_flags.use_var2_interp)
			enablers.push_back("#define ENABLE_VAR2_INTERP\n");
		if (fragment_flags.uses_texscreen) {
			enablers.push_back("#define ENABLE_TEXSCREEN\n");
		}
		if (fragment_flags.uses_screen_uv) {
			enablers.push_back("#define ENABLE_SCREEN_UV\n");
		}
		if (fragment_flags.uses_discard) {
			enablers.push_back("#define ENABLE_DISCARD\n");
		}
		if (fragment_flags.uses_normalmap) {
			enablers.push_back("#define ENABLE_NORMALMAP\n");
		}
		if (light_flags.uses_light) {
			enablers.push_back("#define USE_LIGHT_SHADER_CODE\n");
		}
		if (light_flags.uses_shadow_color) {
			enablers.push_back("#define USE_OUTPUT_SHADOW_COLOR\n");
		}
		if (light_flags.uses_time || fragment_flags.uses_time || vertex_flags.uses_time) {
			enablers.push_back("#define USE_TIME\n");
			uses_time = true;
		}
		if (vertex_flags.vertex_code_writes_position) {
			enablers.push_back("#define VERTEX_SHADER_WRITE_POSITION\n");
		}

		material_shader.set_custom_shader_code(p_shader->custom_code_id, vertex_code, vertex_globals, fragment_code, light_code, fragment_globals, uniform_names, enablers);
	} else if (p_shader->mode == VS::SHADER_CANVAS_ITEM) {

		Vector<const char *> enablers;

		if (light_flags.uses_time || fragment_flags.uses_time || vertex_flags.uses_time) {
			enablers.push_back("#define USE_TIME\n");
			uses_time = true;
		}
		if (fragment_flags.uses_normal) {
			enablers.push_back("#define NORMAL_USED\n");
		}
		if (fragment_flags.uses_normalmap) {
			enablers.push_back("#define USE_NORMALMAP\n");
		}

		if (light_flags.uses_light) {
			enablers.push_back("#define USE_LIGHT_SHADER_CODE\n");
		}
		if (fragment_flags.use_var1_interp || vertex_flags.use_var1_interp)
			enablers.push_back("#define ENABLE_VAR1_INTERP\n");
		if (fragment_flags.use_var2_interp || vertex_flags.use_var2_interp)
			enablers.push_back("#define ENABLE_VAR2_INTERP\n");
		if (fragment_flags.uses_texscreen) {
			enablers.push_back("#define ENABLE_TEXSCREEN\n");
		}
		if (fragment_flags.uses_screen_uv) {
			enablers.push_back("#define ENABLE_SCREEN_UV\n");
		}
		if (fragment_flags.uses_texpixel_size) {
			enablers.push_back("#define USE_TEXPIXEL_SIZE\n");
		}
		if (light_flags.uses_shadow_color) {
			enablers.push_back("#define USE_OUTPUT_SHADOW_COLOR\n");
		}

		if (vertex_flags.uses_worldvec) {
			enablers.push_back("#define USE_WORLD_VEC\n");
		}
		canvas_shader.set_custom_shader_code(p_shader->custom_code_id, vertex_code, vertex_globals, fragment_code, light_code, fragment_globals, uniform_names, enablers);

		//postprocess_shader.set_custom_shader_code(p_shader->custom_code_id,vertex_code, vertex_globals,fragment_code, fragment_globals,uniform_names);
	}

	p_shader->valid = true;
	p_shader->has_alpha = fragment_flags.uses_alpha || fragment_flags.uses_texscreen;
	p_shader->writes_vertex = vertex_flags.vertex_code_writes_vertex;
	p_shader->uses_discard = fragment_flags.uses_discard;
	p_shader->has_texscreen = fragment_flags.uses_texscreen;
	p_shader->has_screen_uv = fragment_flags.uses_screen_uv;
	p_shader->can_zpass = !fragment_flags.uses_discard && !vertex_flags.vertex_code_writes_vertex;
	p_shader->uses_normal = fragment_flags.uses_normal || light_flags.uses_normal;
	p_shader->uses_time = uses_time;
	p_shader->uses_texpixel_size = fragment_flags.uses_texpixel_size;
	p_shader->version++;
}

void RasterizerGLES2::_add_geometry(const Geometry *p_geometry, const InstanceData *p_instance, const Geometry *p_geometry_cmp, const GeometryOwner *p_owner, int p_material) {

	Material *m = NULL;
	RID m_src = p_instance->material_override.is_valid() ? p_instance->material_override : (p_material >= 0 ? p_instance->materials[p_material] : p_geometry->material);

#ifdef DEBUG_ENABLED
	if (current_debug == VS::SCENARIO_DEBUG_OVERDRAW) {
		m_src = overdraw_material;
	}

#endif

	if (m_src)
		m = material_owner.get(m_src);

	if (!m) {
		m = material_owner.get(default_material);
	}

	ERR_FAIL_COND(!m);

	if (m->last_pass != frame) {

		if (m->shader.is_valid()) {

			m->shader_cache = shader_owner.get(m->shader);
			if (m->shader_cache) {

				if (!m->shader_cache->valid) {
					m->shader_cache = NULL;
				} else {
					if (m->shader_cache->has_texscreen)
						texscreen_used = true;
				}
			} else {
				m->shader = RID();
			}

		} else {
			m->shader_cache = NULL;
		}

		m->last_pass = frame;
	}

	RenderList *render_list = NULL;

	bool has_base_alpha = (m->shader_cache && m->shader_cache->has_alpha);
	bool has_blend_alpha = m->blend_mode != VS::MATERIAL_BLEND_MODE_MIX || m->flags[VS::MATERIAL_FLAG_ONTOP];
	bool has_alpha = has_base_alpha || has_blend_alpha;

	if (shadow) {

		if (has_blend_alpha || (has_base_alpha && m->depth_draw_mode != VS::MATERIAL_DEPTH_DRAW_OPAQUE_PRE_PASS_ALPHA))
			return; //bye

		if (!m->shader_cache || (!m->shader_cache->writes_vertex && !m->shader_cache->uses_discard && m->depth_draw_mode != VS::MATERIAL_DEPTH_DRAW_OPAQUE_PRE_PASS_ALPHA)) {
			//shader does not use discard and does not write a vertex position, use generic material
			if (p_instance->cast_shadows == VS::SHADOW_CASTING_SETTING_DOUBLE_SIDED)
				m = shadow_mat_double_sided_ptr;
			else
				m = shadow_mat_ptr;
			if (m->last_pass != frame) {

				if (m->shader.is_valid()) {

					m->shader_cache = shader_owner.get(m->shader);
					if (m->shader_cache) {

						if (!m->shader_cache->valid)
							m->shader_cache = NULL;
					} else {
						m->shader = RID();
					}

				} else {
					m->shader_cache = NULL;
				}

				m->last_pass = frame;
			}
		}

		render_list = &opaque_render_list;
		/* notyet
		if (!m->shader_cache || m->shader_cache->can_zpass)
			render_list = &alpha_render_list;
		} else {
			render_list = &opaque_render_list;
		}*/

	} else {
		if (has_alpha) {
			render_list = &alpha_render_list;
		} else {
			render_list = &opaque_render_list;
		}
	}

	RenderList::Element *e = render_list->add_element();

	if (!e)
		return;

	e->geometry = p_geometry;
	e->geometry_cmp = p_geometry_cmp;
	e->material = m;
	e->instance = p_instance;
	if (camera_ortho) {
		e->depth = camera_plane.distance_to(p_instance->transform.origin);
	} else {
		e->depth = camera_transform.origin.distance_to(p_instance->transform.origin);
	}
	e->owner = p_owner;
	e->light_type = 0;
	e->additive = false;
	e->additive_ptr = &e->additive;
	e->sort_flags = 0;

	if (p_instance->skeleton.is_valid()) {
		e->skeleton = skeleton_owner.get(p_instance->skeleton);
		if (!e->skeleton)
			const_cast<InstanceData *>(p_instance)->skeleton = RID();
		else
			e->sort_flags |= RenderList::SORT_FLAG_SKELETON;
	} else {
		e->skeleton = NULL;
	}

	if (e->geometry->type == Geometry::GEOMETRY_MULTISURFACE)
		e->sort_flags |= RenderList::SORT_FLAG_INSTANCING;

	e->mirror = p_instance->mirror;
	if (m->flags[VS::MATERIAL_FLAG_INVERT_FACES])
		e->mirror = !e->mirror;

	//e->light_type=0xFF; // no lights!
	e->light_type = 3; //light type 3 is no light?
	e->light = 0xFFFF;

	if (!shadow && !has_blend_alpha && has_alpha && m->depth_draw_mode == VS::MATERIAL_DEPTH_DRAW_OPAQUE_PRE_PASS_ALPHA) {

		//if nothing exists, add this element as opaque too
		RenderList::Element *oe = opaque_render_list.add_element();

		if (!oe)
			return;

		memcpy(oe, e, sizeof(RenderList::Element));
		oe->additive_ptr = &oe->additive;
	}

	if (shadow || m->flags[VS::MATERIAL_FLAG_UNSHADED] || current_debug == VS::SCENARIO_DEBUG_SHADELESS) {

		e->light_type = 0x7F; //unshaded is zero
	} else {

		bool duplicate = false;

		for (int i = 0; i < directional_light_count; i++) {
			uint16_t sort_key = directional_lights[i]->sort_key;
			uint8_t light_type = VS::LIGHT_DIRECTIONAL;
			if (directional_lights[i]->base->shadow_enabled) {
				light_type |= 0x8;
				if (directional_lights[i]->base->directional_shadow_mode == VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS)
					light_type |= 0x10;
				else if (directional_lights[i]->base->directional_shadow_mode == VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS)
					light_type |= 0x30;
			}

			RenderList::Element *ec;
			if (duplicate) {

				ec = render_list->add_element();
				memcpy(ec, e, sizeof(RenderList::Element));
			} else {

				ec = e;
				duplicate = true;
			}

			ec->light_type = light_type;
			ec->light = sort_key;
			ec->additive_ptr = &e->additive;
		}

		const RID *liptr = p_instance->light_instances.ptr();
		int ilc = p_instance->light_instances.size();

		for (int i = 0; i < ilc; i++) {

			LightInstance *li = light_instance_owner.get(liptr[i]);
			if (!li || li->last_pass != scene_pass) //lit by light not in visible scene
				continue;
			uint8_t light_type = li->base->type | 0x40; //penalty to ensure directionals always go first
			if (li->base->shadow_enabled) {
				light_type |= 0x8;
			}
			uint16_t sort_key = li->sort_key;

			RenderList::Element *ec;
			if (duplicate) {

				ec = render_list->add_element();
				memcpy(ec, e, sizeof(RenderList::Element));
			} else {

				duplicate = true;
				ec = e;
			}

			ec->light_type = light_type;
			ec->light = sort_key;
			ec->additive_ptr = &e->additive;
		}
	}

	DEBUG_TEST_ERROR("Add Geometry");
}

void RasterizerGLES2::add_mesh(const RID &p_mesh, const InstanceData *p_data) {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND(!mesh);

	int ssize = mesh->surfaces.size();

	for (int i = 0; i < ssize; i++) {

		int mat_idx = p_data->materials[i].is_valid() ? i : -1;
		Surface *s = mesh->surfaces[i];
		_add_geometry(s, p_data, s, NULL, mat_idx);
	}

	mesh->last_pass = frame;
}

void RasterizerGLES2::add_multimesh(const RID &p_multimesh, const InstanceData *p_data) {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND(!multimesh);

	if (!multimesh->mesh.is_valid())
		return;
	if (multimesh->elements.empty())
		return;

	Mesh *mesh = mesh_owner.get(multimesh->mesh);
	ERR_FAIL_COND(!mesh);

	int surf_count = mesh->surfaces.size();
	if (multimesh->last_pass != scene_pass) {

		multimesh->cache_surfaces.resize(surf_count);
		for (int i = 0; i < surf_count; i++) {

			multimesh->cache_surfaces[i].material = mesh->surfaces[i]->material;
			multimesh->cache_surfaces[i].has_alpha = mesh->surfaces[i]->has_alpha;
			multimesh->cache_surfaces[i].surface = mesh->surfaces[i];
		}

		multimesh->last_pass = scene_pass;
	}

	for (int i = 0; i < surf_count; i++) {

		_add_geometry(&multimesh->cache_surfaces[i], p_data, multimesh->cache_surfaces[i].surface, multimesh);
	}
}

void RasterizerGLES2::add_immediate(const RID &p_immediate, const InstanceData *p_data) {

	Immediate *immediate = immediate_owner.get(p_immediate);
	ERR_FAIL_COND(!immediate);

	_add_geometry(immediate, p_data, immediate, NULL);
}

void RasterizerGLES2::add_particles(const RID &p_particle_instance, const InstanceData *p_data) {

	//print_line("adding particles");
	ParticlesInstance *particles_instance = particles_instance_owner.get(p_particle_instance);
	ERR_FAIL_COND(!particles_instance);
	Particles *p = particles_owner.get(particles_instance->particles);
	ERR_FAIL_COND(!p);

	_add_geometry(p, p_data, p, particles_instance);
	draw_next_frame = true;
}

Color RasterizerGLES2::_convert_color(const Color &p_color) {

	if (current_env && current_env->fx_enabled[VS::ENV_FX_SRGB])
		return p_color.to_linear();
	else
		return p_color;
}

void RasterizerGLES2::_set_cull(bool p_front, bool p_reverse_cull) {

	bool front = p_front;
	if (p_reverse_cull)
		front = !front;

	if (front != cull_front) {

		glCullFace(front ? GL_FRONT : GL_BACK);
		cull_front = front;
	}
}

_FORCE_INLINE_ void RasterizerGLES2::_update_material_shader_params(Material *p_material) const {

	Map<StringName, Material::UniformData> old_mparams = p_material->shader_params;
	Map<StringName, Material::UniformData> &mparams = p_material->shader_params;
	mparams.clear();
	int idx = 0;
	for (Map<StringName, ShaderLanguage::Uniform>::Element *E = p_material->shader_cache->uniforms.front(); E; E = E->next()) {

		Material::UniformData ud;

		bool keep = true; //keep material value

		Map<StringName, Material::UniformData>::Element *OLD = old_mparams.find(E->key());
		bool has_old = OLD;
		bool old_inuse = has_old && old_mparams[E->key()].inuse;

		ud.istexture = (E->get().type == ShaderLanguage::TYPE_TEXTURE || E->get().type == ShaderLanguage::TYPE_CUBEMAP);

		if (!has_old || !old_inuse) {
			keep = false;
		} else if (OLD->get().value.get_type() != E->value().default_value.get_type()) {

			if (OLD->get().value.get_type() == Variant::INT && E->get().type == ShaderLanguage::TYPE_FLOAT) {
				//handle common mistake using shaders (feeding ints instead of float)
				OLD->get().value = float(OLD->get().value);
				keep = true;
			} else if (!ud.istexture && E->value().default_value.get_type() != Variant::NIL) {

				keep = false;
			}
			//type changed between old and new
			/*	if (old_mparams[E->key()].value.get_type()==Variant::OBJECT) {
				if (E->value().default_value.get_type()!=Variant::_RID) //hackfor textures
					keep=false;
			} else if (!old_mparams[E->key()].value.is_num() || !E->value().default_value.get_type())
				keep=false;*/

			//value is invalid because type differs and default is not null
			;
		}

		if (keep) {
			ud.value = old_mparams[E->key()].value;

			//print_line("KEEP: "+String(E->key()));
		} else {
			if (ud.istexture && p_material->shader_cache->default_textures.has(E->key()))
				ud.value = p_material->shader_cache->default_textures[E->key()];
			else
				ud.value = E->value().default_value;
			old_inuse = false; //if reverted to default, obviously did not work

			/*
			print_line("NEW: "+String(E->key())+" because: hasold-"+itos(old_mparams.has(E->key())));
			if (old_mparams.has(E->key()))
				print_line(" told "+Variant::get_type_name(old_mparams[E->key()].value.get_type())+" tnew "+Variant::get_type_name(E->value().default_value.get_type()));
			*/
		}

		ud.index = idx++;
		ud.inuse = old_inuse;
		mparams[E->key()] = ud;
	}

	p_material->shader_version = p_material->shader_cache->version;
}

bool RasterizerGLES2::_setup_material(const Geometry *p_geometry, const Material *p_material, bool p_no_const_light, bool p_opaque_pass) {

	if (p_material->flags[VS::MATERIAL_FLAG_DOUBLE_SIDED]) {
		glDisable(GL_CULL_FACE);
	} else {
		glEnable(GL_CULL_FACE);
	}

	//glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);

	/*
	if (p_material->flags[VS::MATERIAL_FLAG_WIREFRAME])
		glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	*/

	if (p_material->line_width)
		glLineWidth(p_material->line_width);

	//all goes to false by default
	material_shader.set_conditional(MaterialShaderGLES2::USE_SHADOW_PASS, shadow != NULL);
	material_shader.set_conditional(MaterialShaderGLES2::USE_SHADOW_PCF, shadow_filter == SHADOW_FILTER_PCF5 || shadow_filter == SHADOW_FILTER_PCF13);
	material_shader.set_conditional(MaterialShaderGLES2::USE_SHADOW_PCF_HQ, shadow_filter == SHADOW_FILTER_PCF13);
	material_shader.set_conditional(MaterialShaderGLES2::USE_SHADOW_ESM, shadow_filter == SHADOW_FILTER_ESM);
	material_shader.set_conditional(MaterialShaderGLES2::USE_LIGHTMAP_ON_UV2, p_material->flags[VS::MATERIAL_FLAG_LIGHTMAP_ON_UV2]);
	material_shader.set_conditional(MaterialShaderGLES2::USE_COLOR_ATTRIB_SRGB_TO_LINEAR, p_material->flags[VS::MATERIAL_FLAG_COLOR_ARRAY_SRGB] && current_env && current_env->fx_enabled[VS::ENV_FX_SRGB]);

	if (p_opaque_pass && p_material->depth_draw_mode == VS::MATERIAL_DEPTH_DRAW_OPAQUE_PRE_PASS_ALPHA && p_material->shader_cache && p_material->shader_cache->has_alpha) {

		material_shader.set_conditional(MaterialShaderGLES2::ENABLE_CLIP_ALPHA, true);
	} else {
		material_shader.set_conditional(MaterialShaderGLES2::ENABLE_CLIP_ALPHA, false);
	}

	if (!shadow) {

		bool depth_test = !p_material->flags[VS::MATERIAL_FLAG_ONTOP];
		bool depth_write = p_material->depth_draw_mode != VS::MATERIAL_DEPTH_DRAW_NEVER && (p_opaque_pass || p_material->depth_draw_mode == VS::MATERIAL_DEPTH_DRAW_ALWAYS);
		//bool depth_write=!p_material->hints[VS::MATERIAL_HINT_NO_DEPTH_DRAW] && (p_opaque_pass || !p_material->hints[VS::MATERIAL_HINT_NO_DEPTH_DRAW_FOR_ALPHA]);

		if (current_depth_mask != depth_write) {
			current_depth_mask = depth_write;
			glDepthMask(depth_write);
		}

		if (current_depth_test != depth_test) {

			current_depth_test = depth_test;
			if (depth_test)
				glEnable(GL_DEPTH_TEST);
			else
				glDisable(GL_DEPTH_TEST);
		}

		material_shader.set_conditional(MaterialShaderGLES2::USE_FOG, current_env && current_env->fx_enabled[VS::ENV_FX_FOG]);
		//glDepthMask( true );
	}

	DEBUG_TEST_ERROR("Pre Shader Bind");

	bool rebind = false;

	if (p_material->shader_cache && p_material->shader_cache->valid) {

		/*
		// reduce amount of conditional compilations
		for(int i=0;i<_tex_version_count;i++)
			material_shader.set_conditional((MaterialShaderGLES2::Conditionals)_tex_version[i],false);
		*/

		//material_shader.set_custom_shader(p_material->shader_cache->custom_code_id);

		if (p_material->shader_version != p_material->shader_cache->version) {
			//shader changed somehow, must update uniforms

			_update_material_shader_params((Material *)p_material);
		}
		material_shader.set_custom_shader(p_material->shader_cache->custom_code_id);
		rebind = material_shader.bind();

		DEBUG_TEST_ERROR("Shader Bind");

		//set uniforms!
		int texcoord = 0;
		for (Map<StringName, Material::UniformData>::Element *E = p_material->shader_params.front(); E; E = E->next()) {

			if (E->get().index < 0)
				continue;
			//print_line(String(E->key())+": "+E->get().value);
			if (E->get().istexture) {
				//clearly a texture..
				RID rid = E->get().value;
				int loc = material_shader.get_custom_uniform_location(E->get().index); //should be automatic..

				Texture *t = NULL;
				if (rid.is_valid()) {

					t = texture_owner.get(rid);
					if (!t) {
						E->get().value = RID(); //nullify, invalid texture
						rid = RID();
					}
				}

				glActiveTexture(GL_TEXTURE0 + texcoord);
				glUniform1i(loc, texcoord); //TODO - this could happen automatically on compile...
				if (t) {
					if (t->render_target)
						t->render_target->last_pass = frame;
					if (E->key() == p_material->shader_cache->first_texture) {
						tc0_idx = texcoord;
						tc0_id_cache = t->tex_id;
					}
					glBindTexture(t->target, t->tex_id);
				} else
					glBindTexture(GL_TEXTURE_2D, white_tex); //no texture
				texcoord++;

			} else if (E->get().value.get_type() == Variant::COLOR) {
				Color c = E->get().value;
				material_shader.set_custom_uniform(E->get().index, _convert_color(c));
			} else {
				material_shader.set_custom_uniform(E->get().index, E->get().value);
			}
		}

		if (p_material->shader_cache->has_texscreen && framebuffer.active) {
			material_shader.set_uniform(MaterialShaderGLES2::TEXSCREEN_SCREEN_MULT, Vector2(float(viewport.width) / framebuffer.width, float(viewport.height) / framebuffer.height));
			material_shader.set_uniform(MaterialShaderGLES2::TEXSCREEN_SCREEN_CLAMP, Color(0, 0, float(viewport.width) / framebuffer.width, float(viewport.height) / framebuffer.height));
			material_shader.set_uniform(MaterialShaderGLES2::TEXSCREEN_TEX, texcoord);
			glActiveTexture(GL_TEXTURE0 + texcoord);
			glBindTexture(GL_TEXTURE_2D, framebuffer.sample_color);
		}
		if (p_material->shader_cache->has_screen_uv) {
			material_shader.set_uniform(MaterialShaderGLES2::SCREEN_UV_MULT, Vector2(1.0 / viewport.width, 1.0 / viewport.height));
		}
		DEBUG_TEST_ERROR("Material arameters");

		if (p_material->shader_cache->uses_time) {
			material_shader.set_uniform(MaterialShaderGLES2::TIME, Math::fmod(last_time, shader_time_rollback));
			draw_next_frame = true;
		}
		//if uses TIME - draw_next_frame=true

	} else {

		material_shader.set_custom_shader(0);
		rebind = material_shader.bind();

		DEBUG_TEST_ERROR("Shader bind2");
	}

	if (shadow) {

		float zofs = shadow->base->vars[VS::LIGHT_PARAM_SHADOW_Z_OFFSET];
		float zslope = shadow->base->vars[VS::LIGHT_PARAM_SHADOW_Z_SLOPE_SCALE];
		if (shadow_pass >= 1 && shadow->base->type == VS::LIGHT_DIRECTIONAL) {
			float m = Math::pow(shadow->base->directional_shadow_param[VS::LIGHT_DIRECTIONAL_SHADOW_PARAM_PSSM_ZOFFSET_SCALE], shadow_pass);
			zofs *= m;
			zslope *= m;
		}
		material_shader.set_uniform(MaterialShaderGLES2::SHADOW_Z_OFFSET, zofs);
		material_shader.set_uniform(MaterialShaderGLES2::SHADOW_Z_SLOPE_SCALE, zslope);
		if (shadow->base->type == VS::LIGHT_OMNI)
			material_shader.set_uniform(MaterialShaderGLES2::DUAL_PARABOLOID, shadow->dp);
		DEBUG_TEST_ERROR("Shadow uniforms");
	}

	if (current_env && current_env->fx_enabled[VS::ENV_FX_FOG]) {

		Color col_begin = current_env->fx_param[VS::ENV_FX_PARAM_FOG_BEGIN_COLOR];
		Color col_end = current_env->fx_param[VS::ENV_FX_PARAM_FOG_END_COLOR];
		col_begin = _convert_color(col_begin);
		col_end = _convert_color(col_end);
		float from = current_env->fx_param[VS::ENV_FX_PARAM_FOG_BEGIN];
		float zf = camera_z_far;
		float curve = current_env->fx_param[VS::ENV_FX_PARAM_FOG_ATTENUATION];
		material_shader.set_uniform(MaterialShaderGLES2::FOG_PARAMS, Vector3(from, zf, curve));
		material_shader.set_uniform(MaterialShaderGLES2::FOG_COLOR_BEGIN, Vector3(col_begin.r, col_begin.g, col_begin.b));
		material_shader.set_uniform(MaterialShaderGLES2::FOG_COLOR_END, Vector3(col_end.r, col_end.g, col_end.b));
	}

	//material_shader.set_uniform(MaterialShaderGLES2::TIME,Math::fmod(last_time,300.0));
	//if uses TIME - draw_next_frame=true

	return rebind;
}

void RasterizerGLES2::_setup_light(uint16_t p_light) {

	if (shadow)
		return;

	if (p_light == 0xFFFF)
		return;

	enum {
		VL_LIGHT_POS,
		VL_LIGHT_DIR,
		VL_LIGHT_ATTENUATION,
		VL_LIGHT_SPOT_ATTENUATION,
		VL_LIGHT_DIFFUSE,
		VL_LIGHT_SPECULAR,
		VL_LIGHT_MAX
	};

	static const MaterialShaderGLES2::Uniforms light_uniforms[VL_LIGHT_MAX] = {
		MaterialShaderGLES2::LIGHT_POS,
		MaterialShaderGLES2::LIGHT_DIRECTION,
		MaterialShaderGLES2::LIGHT_ATTENUATION,
		MaterialShaderGLES2::LIGHT_SPOT_ATTENUATION,
		MaterialShaderGLES2::LIGHT_DIFFUSE,
		MaterialShaderGLES2::LIGHT_SPECULAR,
	};

	GLfloat light_data[VL_LIGHT_MAX][3];
	memset(light_data, 0, (VL_LIGHT_MAX)*3 * sizeof(GLfloat));

	LightInstance *li = light_instances[p_light];
	Light *l = li->base;

	Color col_diffuse = _convert_color(l->colors[VS::LIGHT_COLOR_DIFFUSE]);
	Color col_specular = _convert_color(l->colors[VS::LIGHT_COLOR_SPECULAR]);

	for (int j = 0; j < 3; j++) {
		light_data[VL_LIGHT_DIFFUSE][j] = col_diffuse[j];
		light_data[VL_LIGHT_SPECULAR][j] = col_specular[j];
	}

	if (l->type != VS::LIGHT_OMNI) {

		Vector3 dir = -li->transform.get_basis().get_axis(2);
		dir = camera_transform_inverse.basis.xform(dir).normalized();
		for (int j = 0; j < 3; j++)
			light_data[VL_LIGHT_DIR][j] = dir[j];
	}

	if (l->type != VS::LIGHT_DIRECTIONAL) {

		Vector3 pos = li->transform.get_origin();
		pos = camera_transform_inverse.xform(pos);
		for (int j = 0; j < 3; j++)
			light_data[VL_LIGHT_POS][j] = pos[j];
	}

	if (li->near_shadow_buffer) {

		glActiveTexture(GL_TEXTURE0 + max_texture_units - 1);
		glBindTexture(GL_TEXTURE_2D, li->near_shadow_buffer->depth);

		material_shader.set_uniform(MaterialShaderGLES2::SHADOW_MATRIX, li->shadow_projection[0]);
		material_shader.set_uniform(MaterialShaderGLES2::SHADOW_TEXEL_SIZE, Vector2(1.0, 1.0) / li->near_shadow_buffer->size);
		material_shader.set_uniform(MaterialShaderGLES2::SHADOW_TEXTURE, max_texture_units - 1);
		if (shadow_filter == SHADOW_FILTER_ESM)
			material_shader.set_uniform(MaterialShaderGLES2::ESM_MULTIPLIER, float(li->base->vars[VS::LIGHT_PARAM_SHADOW_ESM_MULTIPLIER]));

		if (li->base->type == VS::LIGHT_DIRECTIONAL) {

			if (li->base->directional_shadow_mode == VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS) {

				material_shader.set_uniform(MaterialShaderGLES2::SHADOW_MATRIX2, li->shadow_projection[1]);
				material_shader.set_uniform(MaterialShaderGLES2::LIGHT_PSSM_SPLIT, Vector3(li->shadow_split[0], li->shadow_split[1], li->shadow_split[2]));
			} else if (li->base->directional_shadow_mode == VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS) {

				material_shader.set_uniform(MaterialShaderGLES2::SHADOW_MATRIX2, li->shadow_projection[1]);
				material_shader.set_uniform(MaterialShaderGLES2::SHADOW_MATRIX3, li->shadow_projection[2]);
				material_shader.set_uniform(MaterialShaderGLES2::SHADOW_MATRIX4, li->shadow_projection[3]);
				material_shader.set_uniform(MaterialShaderGLES2::LIGHT_PSSM_SPLIT, Vector3(li->shadow_split[0], li->shadow_split[1], li->shadow_split[2]));
			}
			//print_line("shadow split: "+rtos(li->shadow_split));
		}

		material_shader.set_uniform(MaterialShaderGLES2::SHADOW_DARKENING, li->base->vars[VS::LIGHT_PARAM_SHADOW_DARKENING]);
		//matrix
	}

	light_data[VL_LIGHT_ATTENUATION][0] = l->vars[VS::LIGHT_PARAM_ENERGY];

	if (l->type == VS::LIGHT_DIRECTIONAL) {
		light_data[VL_LIGHT_ATTENUATION][1] = l->directional_shadow_param[VS::LIGHT_DIRECTIONAL_SHADOW_PARAM_MAX_DISTANCE];
	} else {
		light_data[VL_LIGHT_ATTENUATION][1] = l->vars[VS::LIGHT_PARAM_RADIUS];
	}

	light_data[VL_LIGHT_ATTENUATION][2] = l->vars[VS::LIGHT_PARAM_ATTENUATION];

	light_data[VL_LIGHT_SPOT_ATTENUATION][0] = Math::cos(Math::deg2rad(l->vars[VS::LIGHT_PARAM_SPOT_ANGLE]));
	light_data[VL_LIGHT_SPOT_ATTENUATION][1] = l->vars[VS::LIGHT_PARAM_SPOT_ATTENUATION];

	//int uf = material_shader.get_uniform(MaterialShaderGLES2::LIGHT_PARAMS);
	for (int i = 0; i < VL_LIGHT_MAX; i++) {
		glUniform3f(material_shader.get_uniform(light_uniforms[i]), light_data[i][0], light_data[i][1], light_data[i][2]);
	}
}

template <bool USE_NORMAL, bool USE_TANGENT, bool INPLACE>
void RasterizerGLES2::_skeleton_xform(const uint8_t *p_src_array, int p_src_stride, uint8_t *p_dst_array, int p_dst_stride, int p_elements, const uint8_t *p_src_bones, const uint8_t *p_src_weights, const Skeleton::Bone *p_bone_xforms) {

	uint32_t basesize = 3;
	if (USE_NORMAL)
		basesize += 3;
	if (USE_TANGENT)
		basesize += 4;

	uint32_t extra = (p_dst_stride - basesize * 4);
	const int dstvec_size = 3 + (USE_NORMAL ? 3 : 0) + (USE_TANGENT ? 4 : 0);
	float dstcopy[dstvec_size];

	for (int i = 0; i < p_elements; i++) {

		uint32_t ss = p_src_stride * i;
		uint32_t ds = p_dst_stride * i;
		const uint16_t *bi = (const uint16_t *)&p_src_bones[ss];
		const float *bw = (const float *)&p_src_weights[ss];
		const float *src_vec = (const float *)&p_src_array[ss];
		float *dst_vec;
		if (INPLACE)
			dst_vec = dstcopy;
		else
			dst_vec = (float *)&p_dst_array[ds];

		dst_vec[0] = 0.0;
		dst_vec[1] = 0.0;
		dst_vec[2] = 0.0;
		//conditionals simply removed by optimizer
		if (USE_NORMAL) {

			dst_vec[3] = 0.0;
			dst_vec[4] = 0.0;
			dst_vec[5] = 0.0;

			if (USE_TANGENT) {

				dst_vec[6] = 0.0;
				dst_vec[7] = 0.0;
				dst_vec[8] = 0.0;
				dst_vec[9] = src_vec[9];
			}
		} else {

			if (USE_TANGENT) {

				dst_vec[3] = 0.0;
				dst_vec[4] = 0.0;
				dst_vec[5] = 0.0;
				dst_vec[6] = src_vec[6];
			}
		}

#define _XFORM_BONE(m_idx)                                                                     \
	if (bw[m_idx] == 0)                                                                        \
		goto end;                                                                              \
	p_bone_xforms[bi[m_idx]].transform_add_mul3(&src_vec[0], &dst_vec[0], bw[m_idx]);          \
	if (USE_NORMAL) {                                                                          \
		p_bone_xforms[bi[m_idx]].transform3_add_mul3(&src_vec[3], &dst_vec[3], bw[m_idx]);     \
		if (USE_TANGENT) {                                                                     \
			p_bone_xforms[bi[m_idx]].transform3_add_mul3(&src_vec[6], &dst_vec[6], bw[m_idx]); \
		}                                                                                      \
	} else {                                                                                   \
		if (USE_TANGENT) {                                                                     \
			p_bone_xforms[bi[m_idx]].transform3_add_mul3(&src_vec[3], &dst_vec[3], bw[m_idx]); \
		}                                                                                      \
	}

		_XFORM_BONE(0);
		_XFORM_BONE(1);
		_XFORM_BONE(2);
		_XFORM_BONE(3);

	end:

		if (INPLACE) {

			const uint8_t *esp = (const uint8_t *)dstcopy;
			uint8_t *edp = (uint8_t *)&p_dst_array[ds];

			for (uint32_t j = 0; j < dstvec_size * 4; j++) {

				edp[j] = esp[j];
			}

		} else {
			//copy extra stuff
			const uint8_t *esp = (const uint8_t *)&src_vec[basesize];
			uint8_t *edp = (uint8_t *)&dst_vec[basesize];

			for (uint32_t j = 0; j < extra; j++) {

				edp[j] = esp[j];
			}
		}
	}
}

Error RasterizerGLES2::_setup_geometry(const Geometry *p_geometry, const Material *p_material, const Skeleton *p_skeleton, const float *p_morphs) {

	switch (p_geometry->type) {

		case Geometry::GEOMETRY_MULTISURFACE:
		case Geometry::GEOMETRY_SURFACE: {

			const Surface *surf = NULL;
			if (p_geometry->type == Geometry::GEOMETRY_SURFACE)
				surf = static_cast<const Surface *>(p_geometry);
			else if (p_geometry->type == Geometry::GEOMETRY_MULTISURFACE)
				surf = static_cast<const MultiMeshSurface *>(p_geometry)->surface;

			if (surf->format != surf->configured_format) {
				if (OS::get_singleton()->is_stdout_verbose()) {

					print_line("has format: " + itos(surf->format));
					print_line("configured format: " + itos(surf->configured_format));
				}
				ERR_EXPLAIN("Missing arrays (not set) in surface");
			}
			ERR_FAIL_COND_V(surf->format != surf->configured_format, ERR_UNCONFIGURED);
			uint8_t *base = 0;
			int stride = surf->stride;
			bool use_VBO = (surf->array_local == 0);
			_setup_geometry_vinfo = surf->array_len;

			bool skeleton_valid = p_skeleton && (surf->format & VS::ARRAY_FORMAT_BONES) && (surf->format & VS::ARRAY_FORMAT_WEIGHTS) && !p_skeleton->bones.empty() && p_skeleton->bones.size() > surf->max_bone;
			/*
			if (surf->packed) {
				float scales[4]={surf->vertex_scale,surf->uv_scale,surf->uv2_scale,0.0};
				glVertexAttrib4fv( 7, scales );
			} else {
				glVertexAttrib4f( 7, 1,1,1,1 );

			}*/

			if (!use_VBO) {

				DEBUG_TEST_ERROR("Draw NO VBO");

				base = surf->array_local;
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				bool can_copy_to_local = surf->local_stride * surf->array_len <= skinned_buffer_size;
				if (p_morphs && surf->stride * surf->array_len > skinned_buffer_size)
					can_copy_to_local = false;

				if (!can_copy_to_local)
					skeleton_valid = false;

				/* compute morphs */

				if (p_morphs && surf->morph_target_count && can_copy_to_local) {

					base = skinned_buffer;
					stride = surf->local_stride;

					//copy all first
					float coef = 1.0;

					for (int i = 0; i < surf->morph_target_count; i++) {
						if (surf->mesh->morph_target_mode == VS::MORPH_MODE_NORMALIZED)
							coef -= p_morphs[i];
						ERR_FAIL_COND_V(surf->morph_format != surf->morph_targets_local[i].configured_format, ERR_INVALID_DATA);
					}

					int16_t coeffp = CLAMP(coef * 255, 0, 255);

					for (int i = 0; i < VS::ARRAY_MAX - 1; i++) {

						const Surface::ArrayData &ad = surf->array[i];
						if (ad.size == 0)
							continue;

						int ofs = ad.ofs;
						int src_stride = surf->stride;
						int dst_stride = skeleton_valid ? surf->stride : surf->local_stride;
						int count = surf->array_len;

						if (!skeleton_valid && i >= VS::ARRAY_MAX - 3)
							break;

						switch (i) {

							case VS::ARRAY_VERTEX:
							case VS::ARRAY_NORMAL:
							case VS::ARRAY_TANGENT: {

								for (int k = 0; k < count; k++) {

									const float *src = (const float *)&surf->array_local[ofs + k * src_stride];
									float *dst = (float *)&base[ofs + k * dst_stride];

									dst[0] = src[0] * coef;
									dst[1] = src[1] * coef;
									dst[2] = src[2] * coef;
								};

							} break;
							case VS::ARRAY_COLOR: {

								for (int k = 0; k < count; k++) {

									const uint8_t *src = (const uint8_t *)&surf->array_local[ofs + k * src_stride];
									uint8_t *dst = (uint8_t *)&base[ofs + k * dst_stride];

									dst[0] = (src[0] * coeffp) >> 8;
									dst[1] = (src[1] * coeffp) >> 8;
									dst[2] = (src[2] * coeffp) >> 8;
									dst[3] = (src[3] * coeffp) >> 8;
								}

							} break;
							case VS::ARRAY_TEX_UV:
							case VS::ARRAY_TEX_UV2: {

								for (int k = 0; k < count; k++) {

									const float *src = (const float *)&surf->array_local[ofs + k * src_stride];
									float *dst = (float *)&base[ofs + k * dst_stride];

									dst[0] = src[0] * coef;
									dst[1] = src[1] * coef;
								}

							} break;
							case VS::ARRAY_BONES:
							case VS::ARRAY_WEIGHTS: {

								for (int k = 0; k < count; k++) {

									const float *src = (const float *)&surf->array_local[ofs + k * src_stride];
									float *dst = (float *)&base[ofs + k * dst_stride];

									dst[0] = src[0];
									dst[1] = src[1];
									dst[2] = src[2];
									dst[3] = src[3];
								}

							} break;
						}
					}

					for (int j = 0; j < surf->morph_target_count; j++) {

						for (int i = 0; i < VS::ARRAY_MAX - 3; i++) {

							const Surface::ArrayData &ad = surf->array[i];
							if (ad.size == 0)
								continue;

							int ofs = ad.ofs;
							int src_stride = surf->local_stride;
							int dst_stride = skeleton_valid ? surf->stride : surf->local_stride;
							int count = surf->array_len;
							const uint8_t *morph = surf->morph_targets_local[j].array;
							float w = p_morphs[j];
							int16_t wfp = CLAMP(w * 255, 0, 255);

							switch (i) {

								case VS::ARRAY_VERTEX:
								case VS::ARRAY_NORMAL:
								case VS::ARRAY_TANGENT: {

									for (int k = 0; k < count; k++) {

										const float *src_morph = (const float *)&morph[ofs + k * src_stride];
										float *dst = (float *)&base[ofs + k * dst_stride];

										dst[0] += src_morph[0] * w;
										dst[1] += src_morph[1] * w;
										dst[2] += src_morph[2] * w;
									}

								} break;
								case VS::ARRAY_COLOR: {
									for (int k = 0; k < count; k++) {

										const uint8_t *src = (const uint8_t *)&morph[ofs + k * src_stride];
										uint8_t *dst = (uint8_t *)&base[ofs + k * dst_stride];

										dst[0] = (src[0] * wfp) >> 8;
										dst[1] = (src[1] * wfp) >> 8;
										dst[2] = (src[2] * wfp) >> 8;
										dst[3] = (src[3] * wfp) >> 8;
									}

								} break;
								case VS::ARRAY_TEX_UV:
								case VS::ARRAY_TEX_UV2: {

									for (int k = 0; k < count; k++) {

										const float *src_morph = (const float *)&morph[ofs + k * src_stride];
										float *dst = (float *)&base[ofs + k * dst_stride];

										dst[0] += src_morph[0] * w;
										dst[1] += src_morph[1] * w;
									}

								} break;
							}
						}
					}

					if (skeleton_valid) {

						const uint8_t *src_weights = &surf->array_local[surf->array[VS::ARRAY_WEIGHTS].ofs];
						const uint8_t *src_bones = &surf->array_local[surf->array[VS::ARRAY_BONES].ofs];
						const Skeleton::Bone *skeleton = &p_skeleton->bones[0];

						if (surf->format & VS::ARRAY_FORMAT_NORMAL && surf->format & VS::ARRAY_FORMAT_TANGENT)
							_skeleton_xform<true, true, true>(base, surf->stride, base, surf->stride, surf->array_len, src_bones, src_weights, skeleton);
						else if (surf->format & (VS::ARRAY_FORMAT_NORMAL))
							_skeleton_xform<true, false, true>(base, surf->stride, base, surf->stride, surf->array_len, src_bones, src_weights, skeleton);
						else if (surf->format & (VS::ARRAY_FORMAT_TANGENT))
							_skeleton_xform<false, true, true>(base, surf->stride, base, surf->stride, surf->array_len, src_bones, src_weights, skeleton);
						else
							_skeleton_xform<false, false, true>(base, surf->stride, base, surf->stride, surf->array_len, src_bones, src_weights, skeleton);
					}

					stride = skeleton_valid ? surf->stride : surf->local_stride;

#if 0
					{
						//in-place skeleton tansformation, only used for morphs, slow.
						//should uptimize some day....

						const uint8_t *src_weights=&surf->array_local[surf->array[VS::ARRAY_WEIGHTS].ofs];
						const uint8_t *src_bones=&surf->array_local[surf->array[VS::ARRAY_BONES].ofs];
						int src_stride = surf->stride;
						int count = surf->array_len;
						const Transform *skeleton = &p_skeleton->bones[0];

						for(int i=0;i<VS::ARRAY_MAX-1;i++) {

							const Surface::ArrayData& ad=surf->array[i];
							if (ad.size==0)
								continue;

							int ofs = ad.ofs;


							switch(i) {

								case VS::ARRAY_VERTEX: {
									for(int k=0;k<count;k++) {

										float *ptr=  (float*)&base[ofs+k*stride];
										const GLfloat* weights = reinterpret_cast<const GLfloat*>(&src_weights[k*src_stride]);
										const GLfloat *bones = reinterpret_cast<const GLfloat*>(&src_bones[k*src_stride]);

										Vector3 src( ptr[0], ptr[1], ptr[2] );
										Vector3 dst;
										for(int j=0;j<VS::ARRAY_WEIGHTS_SIZE;j++) {

											float w = weights[j];
											if (w==0)
												break;

											//print_line("accum "+itos(i)+" += "+rtos(Math::ftoi(bones[j]))+" * "+skeleton[ Math::ftoi(bones[j]) ]+" * "+rtos(w));
											int bidx = Math::fast_ftoi(bones[j]);
											dst+=skeleton[ bidx ].xform(src) * w;
										}

										ptr[0]=dst.x;
										ptr[1]=dst.y;
										ptr[2]=dst.z;

									} break;

								} break;
								case VS::ARRAY_NORMAL:
								case VS::ARRAY_TANGENT: {
									for(int k=0;k<count;k++) {

										float *ptr=  (float*)&base[ofs+k*stride];
										const GLfloat* weights = reinterpret_cast<const GLfloat*>(&src_weights[k*src_stride]);
										const GLfloat *bones = reinterpret_cast<const GLfloat*>(&src_bones[k*src_stride]);

										Vector3 src( ptr[0], ptr[1], ptr[2] );
										Vector3 dst;
										for(int j=0;j<VS::ARRAY_WEIGHTS_SIZE;j++) {

											float w = weights[j];
											if (w==0)
												break;

											//print_line("accum "+itos(i)+" += "+rtos(Math::ftoi(bones[j]))+" * "+skeleton[ Math::ftoi(bones[j]) ]+" * "+rtos(w));
											int bidx=Math::fast_ftoi(bones[j]);
											dst+=skeleton[ bidx ].basis.xform(src) * w;
										}

										ptr[0]=dst.x;
										ptr[1]=dst.y;
										ptr[2]=dst.z;

									} break;

								} break;
							}
						}
					}
#endif

				} else if (skeleton_valid) {

					base = skinned_buffer;
					//copy stuff and get it ready for the skeleton

					int dst_stride = surf->stride - (surf->array[VS::ARRAY_BONES].size + surf->array[VS::ARRAY_WEIGHTS].size);
					const uint8_t *src_weights = &surf->array_local[surf->array[VS::ARRAY_WEIGHTS].ofs];
					const uint8_t *src_bones = &surf->array_local[surf->array[VS::ARRAY_BONES].ofs];
					const Skeleton::Bone *skeleton = &p_skeleton->bones[0];

					if (surf->format & VS::ARRAY_FORMAT_NORMAL && surf->format & VS::ARRAY_FORMAT_TANGENT)
						_skeleton_xform<true, true, false>(surf->array_local, surf->stride, base, dst_stride, surf->array_len, src_bones, src_weights, skeleton);
					else if (surf->format & (VS::ARRAY_FORMAT_NORMAL))
						_skeleton_xform<true, false, false>(surf->array_local, surf->stride, base, dst_stride, surf->array_len, src_bones, src_weights, skeleton);
					else if (surf->format & (VS::ARRAY_FORMAT_TANGENT))
						_skeleton_xform<false, true, false>(surf->array_local, surf->stride, base, dst_stride, surf->array_len, src_bones, src_weights, skeleton);
					else
						_skeleton_xform<false, false, false>(surf->array_local, surf->stride, base, dst_stride, surf->array_len, src_bones, src_weights, skeleton);

					stride = dst_stride;
				}

			} else {

				glBindBuffer(GL_ARRAY_BUFFER, surf->vertex_id);
			};

			for (int i = 0; i < (VS::ARRAY_MAX - 1); i++) {

				const Surface::ArrayData &ad = surf->array[i];

				/*
				if (!gl_texcoord_shader[i])
					continue;
				*/

				if (ad.size == 0 || !ad.bind) {
					glDisableVertexAttribArray(i);
					if (i == VS::ARRAY_COLOR) {
						_set_color_attrib(Color(1, 1, 1, 1));
					};
					//print_line("disable: "+itos(i));
					continue; // this one is disabled.
				}

				glEnableVertexAttribArray(i);
				//print_line("set: "+itos(i)+" - count: "+itos(ad.count)+" datatype: "+itos(ad.datatype)+" ofs: "+itos(ad.ofs)+" stride: "+itos(stride)+" total len: "+itos(surf->array_len));
				glVertexAttribPointer(i, ad.count, ad.datatype, ad.normalize, stride, &base[ad.ofs]);
			}
#ifdef GLEW_ENABLED
			//"desktop" opengl needs this.
			if (surf->primitive == VS::PRIMITIVE_POINTS) {
				glEnable(GL_POINT_SPRITE);
				glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

			} else {
				glDisable(GL_POINT_SPRITE);
				glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
			}
#endif
		} break;

		default: break;
	};

	return OK;
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

void RasterizerGLES2::_render(const Geometry *p_geometry, const Material *p_material, const Skeleton *p_skeleton, const GeometryOwner *p_owner, const Transform &p_xform) {

	_rinfo.object_count++;

	switch (p_geometry->type) {

		case Geometry::GEOMETRY_SURFACE: {

			Surface *s = (Surface *)p_geometry;

			_rinfo.vertex_count += s->array_len;

			if (s->index_array_len > 0) {

				if (s->index_array_local) {

					//print_line("LOCAL F: "+itos(s->format)+" C: "+itos(s->index_array_len)+" VC: "+itos(s->array_len));
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
					glDrawElements(gl_primitive[s->primitive], s->index_array_len, (s->array_len > (1 << 16)) ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT, s->index_array_local);

				} else {
					//print_line("indices: "+itos(s->index_array_local) );

					//print_line("VBO F: "+itos(s->format)+" C: "+itos(s->index_array_len)+" VC: "+itos(s->array_len));
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->index_id);
					glDrawElements(gl_primitive[s->primitive], s->index_array_len, (s->array_len > (1 << 16)) ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT, 0);
				}

			} else {

				glDrawArrays(gl_primitive[s->primitive], 0, s->array_len);
			};

			_rinfo.draw_calls++;
		} break;

		case Geometry::GEOMETRY_MULTISURFACE: {

			material_shader.bind_uniforms();
			Surface *s = static_cast<const MultiMeshSurface *>(p_geometry)->surface;
			const MultiMesh *mm = static_cast<const MultiMesh *>(p_owner);
			int element_count = mm->elements.size();

			if (element_count == 0)
				return;

			if (mm->visible >= 0) {
				element_count = MIN(element_count, mm->visible);
			}

			const MultiMesh::Element *elements = &mm->elements[0];

			_rinfo.vertex_count += s->array_len * element_count;

			_rinfo.draw_calls += element_count;

			if (use_texture_instancing) {
				//this is probably the fastest all around way if vertex texture fetch is supported

				float twd = (1.0 / mm->tw) * 4.0;
				float thd = 1.0 / mm->th;
				float parm[3] = { 0.0, 01.0, (1.0f / mm->tw) };
				glActiveTexture(GL_TEXTURE0 + max_texture_units - 2);
				glDisableVertexAttribArray(6);
				glBindTexture(GL_TEXTURE_2D, mm->tex_id);
				material_shader.set_uniform(MaterialShaderGLES2::INSTANCE_MATRICES, GL_TEXTURE0 + max_texture_units - 2);

				if (s->index_array_len > 0) {

					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->index_id);
					for (int i = 0; i < element_count; i++) {
						parm[0] = (i % (mm->tw >> 2)) * twd;
						parm[1] = (i / (mm->tw >> 2)) * thd;
						glVertexAttrib3fv(6, parm);
						glDrawElements(gl_primitive[s->primitive], s->index_array_len, (s->array_len > (1 << 16)) ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT, 0);
					}

				} else {

					for (int i = 0; i < element_count; i++) {
						//parm[0]=(i%(mm->tw>>2))*twd;
						//parm[1]=(i/(mm->tw>>2))*thd;
						glVertexAttrib3fv(6, parm);
						glDrawArrays(gl_primitive[s->primitive], 0, s->array_len);
					}
				};

			} else if (use_attribute_instancing) {
				//if not, using attributes instead of uniforms can be really fast in forward rendering architectures
				if (s->index_array_len > 0) {

					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->index_id);
					for (int i = 0; i < element_count; i++) {
						glVertexAttrib4fv(8, &elements[i].matrix[0]);
						glVertexAttrib4fv(9, &elements[i].matrix[4]);
						glVertexAttrib4fv(10, &elements[i].matrix[8]);
						glVertexAttrib4fv(11, &elements[i].matrix[12]);
						glDrawElements(gl_primitive[s->primitive], s->index_array_len, (s->array_len > (1 << 16)) ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT, 0);
					}

				} else {

					for (int i = 0; i < element_count; i++) {
						glVertexAttrib4fv(8, &elements[i].matrix[0]);
						glVertexAttrib4fv(9, &elements[i].matrix[4]);
						glVertexAttrib4fv(10, &elements[i].matrix[8]);
						glVertexAttrib4fv(11, &elements[i].matrix[12]);
						glDrawArrays(gl_primitive[s->primitive], 0, s->array_len);
					}
				};

			} else {

				//nothing to do, slow path (hope no hardware has to use it... but you never know)

				if (s->index_array_len > 0) {

					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->index_id);
					for (int i = 0; i < element_count; i++) {

						glUniformMatrix4fv(material_shader.get_uniform_location(MaterialShaderGLES2::INSTANCE_TRANSFORM), 1, false, elements[i].matrix);
						glDrawElements(gl_primitive[s->primitive], s->index_array_len, (s->array_len > (1 << 16)) ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT, 0);
					}

				} else {

					for (int i = 0; i < element_count; i++) {
						glUniformMatrix4fv(material_shader.get_uniform_location(MaterialShaderGLES2::INSTANCE_TRANSFORM), 1, false, elements[i].matrix);
						glDrawArrays(gl_primitive[s->primitive], 0, s->array_len);
					}
				};
			}
		} break;
		case Geometry::GEOMETRY_IMMEDIATE: {

			bool restore_tex = false;
			const Immediate *im = static_cast<const Immediate *>(p_geometry);
			if (im->building) {
				return;
			}

			glBindBuffer(GL_ARRAY_BUFFER, 0);

			for (const List<Immediate::Chunk>::Element *E = im->chunks.front(); E; E = E->next()) {

				const Immediate::Chunk &c = E->get();
				if (c.vertices.empty()) {
					continue;
				}
				for (int i = 0; i < c.vertices.size(); i++)

					if (c.texture.is_valid() && texture_owner.owns(c.texture)) {

						const Texture *t = texture_owner.get(c.texture);
						glActiveTexture(GL_TEXTURE0 + tc0_idx);
						glBindTexture(t->target, t->tex_id);
						restore_tex = true;

					} else if (restore_tex) {

						glActiveTexture(GL_TEXTURE0 + tc0_idx);
						glBindTexture(GL_TEXTURE_2D, tc0_id_cache);
						restore_tex = false;
					}

				if (!c.normals.empty()) {

					glEnableVertexAttribArray(VS::ARRAY_NORMAL);
					glVertexAttribPointer(VS::ARRAY_NORMAL, 3, GL_FLOAT, false, sizeof(Vector3), c.normals.ptr());

				} else {

					glDisableVertexAttribArray(VS::ARRAY_NORMAL);
				}

				if (!c.tangents.empty()) {

					glEnableVertexAttribArray(VS::ARRAY_TANGENT);
					glVertexAttribPointer(VS::ARRAY_TANGENT, 4, GL_FLOAT, false, sizeof(Plane), c.tangents.ptr());

				} else {

					glDisableVertexAttribArray(VS::ARRAY_TANGENT);
				}

				if (!c.colors.empty()) {

					glEnableVertexAttribArray(VS::ARRAY_COLOR);
					glVertexAttribPointer(VS::ARRAY_COLOR, 4, GL_FLOAT, false, sizeof(Color), c.colors.ptr());

				} else {

					glDisableVertexAttribArray(VS::ARRAY_COLOR);
					_set_color_attrib(Color(1, 1, 1, 1));
				}

				if (!c.uvs.empty()) {

					glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
					glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, false, sizeof(Vector2), c.uvs.ptr());

				} else {

					glDisableVertexAttribArray(VS::ARRAY_TEX_UV);
				}

				if (!c.uvs2.empty()) {

					glEnableVertexAttribArray(VS::ARRAY_TEX_UV2);
					glVertexAttribPointer(VS::ARRAY_TEX_UV2, 2, GL_FLOAT, false, sizeof(Vector2), c.uvs2.ptr());

				} else {

					glDisableVertexAttribArray(VS::ARRAY_TEX_UV2);
				}

				glEnableVertexAttribArray(VS::ARRAY_VERTEX);
				glVertexAttribPointer(VS::ARRAY_VERTEX, 3, GL_FLOAT, false, sizeof(Vector3), c.vertices.ptr());
				glDrawArrays(gl_primitive[c.primitive], 0, c.vertices.size());
			}

			if (restore_tex) {

				glActiveTexture(GL_TEXTURE0 + tc0_idx);
				glBindTexture(GL_TEXTURE_2D, tc0_id_cache);
				restore_tex = false;
			}

		} break;
		case Geometry::GEOMETRY_PARTICLES: {

			//print_line("particulinas");
			const Particles *particles = static_cast<const Particles *>(p_geometry);
			ERR_FAIL_COND(!p_owner);
			ParticlesInstance *particles_instance = (ParticlesInstance *)p_owner;

			ParticleSystemProcessSW &pp = particles_instance->particles_process;
			float td = time_delta; //MIN(time_delta,1.0/10.0);
			pp.process(&particles->data, particles_instance->transform, td);
			ERR_EXPLAIN("A parameter in the particle system is not correct.");
			ERR_FAIL_COND(!pp.valid);

			Transform camera;
			if (shadow)
				camera = shadow->transform;
			else
				camera = camera_transform;

			particle_draw_info.prepare(&particles->data, &pp, particles_instance->transform, camera);
			_rinfo.draw_calls += particles->data.amount;

			_rinfo.vertex_count += 4 * particles->data.amount;

			{
				static const Vector3 points[4] = {
					Vector3(-1.0, 1.0, 0),
					Vector3(1.0, 1.0, 0),
					Vector3(1.0, -1.0, 0),
					Vector3(-1.0, -1.0, 0)
				};
				static const Vector3 uvs[4] = {
					Vector3(0.0, 0.0, 0.0),
					Vector3(1.0, 0.0, 0.0),
					Vector3(1.0, 1.0, 0.0),
					Vector3(0, 1.0, 0.0)
				};
				static const Vector3 normals[4] = {
					Vector3(0, 0, 1),
					Vector3(0, 0, 1),
					Vector3(0, 0, 1),
					Vector3(0, 0, 1)
				};

				static const Plane tangents[4] = {
					Plane(Vector3(1, 0, 0), 0),
					Plane(Vector3(1, 0, 0), 0),
					Plane(Vector3(1, 0, 0), 0),
					Plane(Vector3(1, 0, 0), 0)
				};

				for (int i = 0; i < particles->data.amount; i++) {

					ParticleSystemDrawInfoSW::ParticleDrawInfo &pinfo = *particle_draw_info.draw_info_order[i];
					if (!pinfo.data->active)
						continue;

					material_shader.set_uniform(MaterialShaderGLES2::WORLD_TRANSFORM, pinfo.transform);
					_set_color_attrib(pinfo.color);
					_draw_primitive(4, points, normals, NULL, uvs, tangents);
				}
			}

		} break;
		default: break;
	};
};

void RasterizerGLES2::_setup_shader_params(const Material *p_material) {

#if 0
	int idx=0;
	int tex_idx=0;
	for(Map<StringName,Variant>::Element *E=p_material->shader_cache->params.front();E;E=E->next(),idx++) {

		Variant v; //
		v = E->get();
		const Map<StringName,Variant>::Element *F=p_material->shader_params.find(E->key());
		if (F)
			v=F->get();

		switch(v.get_type() ) {
			case Variant::OBJECT:
			case Variant::_RID: {

				RID tex=v;
				if (!tex.is_valid())
					break;

				Texture *texture = texture_owner.get(tex);
				if (!texture)
					break;
				glUniform1i( material_shader.get_custom_uniform_location(idx), tex_idx);
				glActiveTexture(tex_idx);
				glBindTexture(texture->target,texture->tex_id);

			} break;
			case Variant::COLOR: {

				Color c=v;
				material_shader.set_custom_uniform(idx,Vector3(c.r,c.g,c.b));
			} break;
			default: {

				material_shader.set_custom_uniform(idx,v);
			} break;
		}

	}
#endif
}

void RasterizerGLES2::_setup_skeleton(const Skeleton *p_skeleton) {

	material_shader.set_conditional(MaterialShaderGLES2::USE_SKELETON, p_skeleton != NULL);
	if (p_skeleton && p_skeleton->tex_id) {

		glActiveTexture(GL_TEXTURE0 + max_texture_units - 2);
		glBindTexture(GL_TEXTURE_2D, p_skeleton->tex_id);
	}
}

void RasterizerGLES2::_render_list_forward(RenderList *p_render_list, const Transform &p_view_transform, const Transform &p_view_transform_inverse, const CameraMatrix &p_projection, bool p_reverse_cull, bool p_fragment_light, bool p_alpha_pass) {

	if (current_rt && current_rt_vflip) {
		//p_reverse_cull=!p_reverse_cull;
		glFrontFace(GL_CCW);
	}

	const Material *prev_material = NULL;
	uint16_t prev_light = 0x777E;
	const Geometry *prev_geometry_cmp = NULL;
	uint8_t prev_light_type = 0xEF;
	const Skeleton *prev_skeleton = NULL;
	uint8_t prev_sort_flags = 0xFF;
	const BakedLightData *prev_baked_light = NULL;
	RID prev_baked_light_texture;
	const float *prev_morph_values = NULL;
	int prev_receive_shadows_state = -1;

	material_shader.set_conditional(MaterialShaderGLES2::USE_VERTEX_LIGHTING, !shadow && !p_fragment_light);
	material_shader.set_conditional(MaterialShaderGLES2::USE_FRAGMENT_LIGHTING, !shadow && p_fragment_light);
	material_shader.set_conditional(MaterialShaderGLES2::USE_SKELETON, false);

	if (shadow) {
		material_shader.set_conditional(MaterialShaderGLES2::LIGHT_TYPE_DIRECTIONAL, false);
		material_shader.set_conditional(MaterialShaderGLES2::LIGHT_TYPE_OMNI, false);
		material_shader.set_conditional(MaterialShaderGLES2::LIGHT_TYPE_SPOT, false);
		material_shader.set_conditional(MaterialShaderGLES2::LIGHT_USE_SHADOW, false);
		material_shader.set_conditional(MaterialShaderGLES2::LIGHT_USE_PSSM, false);
		material_shader.set_conditional(MaterialShaderGLES2::LIGHT_USE_PSSM4, false);
		material_shader.set_conditional(MaterialShaderGLES2::SHADELESS, false);
		material_shader.set_conditional(MaterialShaderGLES2::ENABLE_AMBIENT_OCTREE, false);
		material_shader.set_conditional(MaterialShaderGLES2::ENABLE_AMBIENT_LIGHTMAP, false);
		//material_shader.set_conditional(MaterialShaderGLES2::ENABLE_AMBIENT_TEXTURE,false);
	}

	bool stores_glow = !shadow && (current_env && current_env->fx_enabled[VS::ENV_FX_GLOW]) && !p_alpha_pass;
	float sampled_light_dp_multiplier = 1.0;

	bool prev_blend = false;
	glDisable(GL_BLEND);
	for (int i = 0; i < p_render_list->element_count; i++) {

		RenderList::Element *e = p_render_list->elements[i];
		const Material *material = e->material;
		uint16_t light = e->light;
		uint8_t light_type = e->light_type;
		uint8_t sort_flags = e->sort_flags;
		const Skeleton *skeleton = e->skeleton;
		const Geometry *geometry_cmp = e->geometry_cmp;
		const BakedLightData *baked_light = e->instance->baked_light;
		const float *morph_values = e->instance->morph_values.ptr();
		int receive_shadows_state = e->instance->receive_shadows == true ? 1 : 0;

		bool rebind = false;
		bool bind_baked_light_octree = false;
		bool bind_baked_lightmap = false;
		bool additive = false;
		bool bind_dp_sampler = false;

		if (!shadow) {

			if (texscreen_used && !texscreen_copied && material->shader_cache && material->shader_cache->valid && material->shader_cache->has_texscreen) {
				texscreen_copied = true;
				_copy_to_texscreen();

				//force reset state
				prev_material = NULL;
				prev_light = 0x777E;
				prev_geometry_cmp = NULL;
				prev_light_type = 0xEF;
				prev_skeleton = NULL;
				prev_sort_flags = 0xFF;
				prev_morph_values = NULL;
				prev_receive_shadows_state = -1;
				glEnable(GL_BLEND);
				glDepthMask(GL_TRUE);
				glEnable(GL_DEPTH_TEST);
				glDisable(GL_SCISSOR_TEST);
			}

			if (light_type != prev_light_type || receive_shadows_state != prev_receive_shadows_state) {

				if (material->flags[VS::MATERIAL_FLAG_UNSHADED] || current_debug == VS::SCENARIO_DEBUG_SHADELESS) {
					material_shader.set_conditional(MaterialShaderGLES2::LIGHT_TYPE_DIRECTIONAL, false);
					material_shader.set_conditional(MaterialShaderGLES2::LIGHT_TYPE_OMNI, false);
					material_shader.set_conditional(MaterialShaderGLES2::LIGHT_TYPE_SPOT, false);
					material_shader.set_conditional(MaterialShaderGLES2::LIGHT_USE_SHADOW, false);
					material_shader.set_conditional(MaterialShaderGLES2::LIGHT_USE_PSSM, false);
					material_shader.set_conditional(MaterialShaderGLES2::LIGHT_USE_PSSM4, false);
					material_shader.set_conditional(MaterialShaderGLES2::SHADELESS, true);
				} else {
					material_shader.set_conditional(MaterialShaderGLES2::LIGHT_TYPE_DIRECTIONAL, (light_type & 0x3) == VS::LIGHT_DIRECTIONAL);
					material_shader.set_conditional(MaterialShaderGLES2::LIGHT_TYPE_OMNI, (light_type & 0x3) == VS::LIGHT_OMNI);
					material_shader.set_conditional(MaterialShaderGLES2::LIGHT_TYPE_SPOT, (light_type & 0x3) == VS::LIGHT_SPOT);
					if (receive_shadows_state == 1) {
						material_shader.set_conditional(MaterialShaderGLES2::LIGHT_USE_SHADOW, (light_type & 0x8));
						material_shader.set_conditional(MaterialShaderGLES2::LIGHT_USE_PSSM, (light_type & 0x10));
						material_shader.set_conditional(MaterialShaderGLES2::LIGHT_USE_PSSM4, (light_type & 0x20));
					} else {
						material_shader.set_conditional(MaterialShaderGLES2::LIGHT_USE_SHADOW, false);
						material_shader.set_conditional(MaterialShaderGLES2::LIGHT_USE_PSSM, false);
						material_shader.set_conditional(MaterialShaderGLES2::LIGHT_USE_PSSM4, false);
					}
					material_shader.set_conditional(MaterialShaderGLES2::SHADELESS, false);
				}

				rebind = true;
			}

			if (!*e->additive_ptr) {

				additive = false;
				*e->additive_ptr = true;
			} else {
				additive = true;
			}

			if (stores_glow)
				material_shader.set_conditional(MaterialShaderGLES2::USE_GLOW, !additive);

			bool desired_blend = false;
			VS::MaterialBlendMode desired_blend_mode = VS::MATERIAL_BLEND_MODE_MIX;

			if (additive) {
				desired_blend = true;
				desired_blend_mode = VS::MATERIAL_BLEND_MODE_ADD;
			} else {
				desired_blend = p_alpha_pass;
				desired_blend_mode = material->blend_mode;
			}

			if (prev_blend != desired_blend) {

				if (desired_blend) {
					glEnable(GL_BLEND);
					if (!current_rt || !current_rt_transparent)
						glColorMask(1, 1, 1, 0);
				} else {
					glDisable(GL_BLEND);
					glColorMask(1, 1, 1, 1);
				}

				prev_blend = desired_blend;
			}

			if (desired_blend && desired_blend_mode != current_blend_mode) {

				switch (desired_blend_mode) {

					case VS::MATERIAL_BLEND_MODE_MIX: {
						glBlendEquation(GL_FUNC_ADD);
						if (current_rt && current_rt_transparent) {
							glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
						} else {
							glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
						}

					} break;
					case VS::MATERIAL_BLEND_MODE_ADD: {

						glBlendEquation(GL_FUNC_ADD);
						glBlendFunc(p_alpha_pass ? GL_SRC_ALPHA : GL_ONE, GL_ONE);

					} break;
					case VS::MATERIAL_BLEND_MODE_SUB: {

						glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
						glBlendFunc(GL_SRC_ALPHA, GL_ONE);
					} break;
					case VS::MATERIAL_BLEND_MODE_MUL: {
						glBlendEquation(GL_FUNC_ADD);
						if (current_rt && current_rt_transparent) {
							glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
						} else {
							glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
						}

					} break;
				}

				current_blend_mode = desired_blend_mode;
			}

			material_shader.set_conditional(MaterialShaderGLES2::ENABLE_AMBIENT_OCTREE, false);
			material_shader.set_conditional(MaterialShaderGLES2::ENABLE_AMBIENT_LIGHTMAP, false);
			material_shader.set_conditional(MaterialShaderGLES2::ENABLE_AMBIENT_DP_SAMPLER, false);

			material_shader.set_conditional(MaterialShaderGLES2::ENABLE_AMBIENT_COLOR, false);

			if (material->flags[VS::MATERIAL_FLAG_UNSHADED] == false && current_debug != VS::SCENARIO_DEBUG_SHADELESS) {

				if (baked_light != NULL) {
					if (baked_light->realtime_color_enabled) {
						float realtime_energy = baked_light->realtime_energy;
						material_shader.set_conditional(MaterialShaderGLES2::ENABLE_AMBIENT_COLOR, true);
						material_shader.set_uniform(MaterialShaderGLES2::AMBIENT_COLOR, Vector3(baked_light->realtime_color.r * realtime_energy, baked_light->realtime_color.g * realtime_energy, baked_light->realtime_color.b * realtime_energy));
					}
				}

				if (e->instance->sampled_light.is_valid()) {

					SampledLight *sl = sampled_light_owner.get(e->instance->sampled_light);
					if (sl) {

						baked_light = NULL; //can't mix
						material_shader.set_conditional(MaterialShaderGLES2::ENABLE_AMBIENT_DP_SAMPLER, true);
						glActiveTexture(GL_TEXTURE0 + max_texture_units - 3);
						glBindTexture(GL_TEXTURE_2D, sl->texture); //bind the texture
						sampled_light_dp_multiplier = sl->multiplier;
						bind_dp_sampler = true;
					}
				}

				if (!additive && baked_light) {

					if (baked_light->mode == VS::BAKED_LIGHT_OCTREE && baked_light->octree_texture.is_valid() && e->instance->baked_light_octree_xform) {
						material_shader.set_conditional(MaterialShaderGLES2::ENABLE_AMBIENT_OCTREE, true);
						bind_baked_light_octree = true;
						if (prev_baked_light != baked_light) {
							Texture *tex = texture_owner.get(baked_light->octree_texture);
							if (tex) {

								glActiveTexture(GL_TEXTURE0 + max_texture_units - 3);
								glBindTexture(tex->target, tex->tex_id); //bind the texture
							}
							if (baked_light->light_texture.is_valid()) {
								Texture *texl = texture_owner.get(baked_light->light_texture);
								if (texl) {
									glActiveTexture(GL_TEXTURE0 + max_texture_units - 4);
									glBindTexture(texl->target, texl->tex_id); //bind the light texture
								}
							}
						}
					} else if (baked_light->mode == VS::BAKED_LIGHT_LIGHTMAPS) {

						int lightmap_idx = e->instance->baked_lightmap_id;

						material_shader.set_conditional(MaterialShaderGLES2::ENABLE_AMBIENT_LIGHTMAP, false);
						bind_baked_lightmap = false;

						if (baked_light->lightmaps.has(lightmap_idx)) {

							RID texid = baked_light->lightmaps[lightmap_idx];

							if (prev_baked_light != baked_light || texid != prev_baked_light_texture) {

								Texture *tex = texture_owner.get(texid);
								if (tex) {

									glActiveTexture(GL_TEXTURE0 + max_texture_units - 3);
									glBindTexture(tex->target, tex->tex_id); //bind the texture
								}

								prev_baked_light_texture = texid;
							}

							if (texid.is_valid()) {
								material_shader.set_conditional(MaterialShaderGLES2::ENABLE_AMBIENT_LIGHTMAP, true);
								bind_baked_lightmap = true;
							}
						}
					}
				}

				if (int(prev_baked_light != NULL) ^ int(baked_light != NULL)) {
					rebind = true;
				}
			}
		}

		if (sort_flags != prev_sort_flags) {

			if (sort_flags & RenderList::SORT_FLAG_INSTANCING) {
				material_shader.set_conditional(MaterialShaderGLES2::USE_UNIFORM_INSTANCING, !use_texture_instancing && !use_attribute_instancing);
				material_shader.set_conditional(MaterialShaderGLES2::USE_ATTRIBUTE_INSTANCING, use_attribute_instancing);
				material_shader.set_conditional(MaterialShaderGLES2::USE_TEXTURE_INSTANCING, use_texture_instancing);
			} else {
				material_shader.set_conditional(MaterialShaderGLES2::USE_UNIFORM_INSTANCING, false);
				material_shader.set_conditional(MaterialShaderGLES2::USE_ATTRIBUTE_INSTANCING, false);
				material_shader.set_conditional(MaterialShaderGLES2::USE_TEXTURE_INSTANCING, false);
			}
			rebind = true;
		}

		if (use_hw_skeleton_xform && (skeleton != prev_skeleton || morph_values != prev_morph_values)) {
			if (!prev_skeleton || !skeleton)
				rebind = true; //went from skeleton <-> no skeleton, needs rebind

			if (morph_values == NULL)
				_setup_skeleton(skeleton);
			else
				_setup_skeleton(NULL);
		}

		if (material != prev_material || rebind) {

			rebind = _setup_material(e->geometry, material, additive, !p_alpha_pass);

			DEBUG_TEST_ERROR("Setup material");
			_rinfo.mat_change_count++;
			//_setup_material_overrides(e->material,NULL,material_overrides);
			//_setup_material_skeleton(material,skeleton);
		} else {

			if (prev_skeleton != skeleton) {
				//_setup_material_skeleton(material,skeleton);
			};
		}

		if (geometry_cmp != prev_geometry_cmp || prev_skeleton != skeleton) {

			_setup_geometry(e->geometry, material, e->skeleton, e->instance->morph_values.ptr());
			_rinfo.surface_count++;
			DEBUG_TEST_ERROR("Setup geometry");
		};

		if (i == 0 || light != prev_light || rebind) {
			if (e->light != 0xFFFF) {
				_setup_light(e->light);
			}
		}

		if (bind_baked_light_octree && (baked_light != prev_baked_light || rebind)) {

			material_shader.set_uniform(MaterialShaderGLES2::AMBIENT_OCTREE_INVERSE_TRANSFORM, *e->instance->baked_light_octree_xform);
			material_shader.set_uniform(MaterialShaderGLES2::AMBIENT_OCTREE_LATTICE_SIZE, baked_light->octree_lattice_size);
			material_shader.set_uniform(MaterialShaderGLES2::AMBIENT_OCTREE_LATTICE_DIVIDE, baked_light->octree_lattice_divide);
			material_shader.set_uniform(MaterialShaderGLES2::AMBIENT_OCTREE_STEPS, baked_light->octree_steps);
			material_shader.set_uniform(MaterialShaderGLES2::AMBIENT_OCTREE_TEX, max_texture_units - 3);
			if (baked_light->light_texture.is_valid()) {

				material_shader.set_uniform(MaterialShaderGLES2::AMBIENT_OCTREE_LIGHT_TEX, max_texture_units - 4);
				material_shader.set_uniform(MaterialShaderGLES2::AMBIENT_OCTREE_LIGHT_PIX_SIZE, baked_light->light_tex_pixel_size);
			} else {
				material_shader.set_uniform(MaterialShaderGLES2::AMBIENT_OCTREE_LIGHT_TEX, max_texture_units - 3);
				material_shader.set_uniform(MaterialShaderGLES2::AMBIENT_OCTREE_LIGHT_PIX_SIZE, baked_light->octree_tex_pixel_size);
			}
			material_shader.set_uniform(MaterialShaderGLES2::AMBIENT_OCTREE_MULTIPLIER, baked_light->texture_multiplier);
			material_shader.set_uniform(MaterialShaderGLES2::AMBIENT_OCTREE_PIX_SIZE, baked_light->octree_tex_pixel_size);
		}

		if (bind_baked_lightmap && (baked_light != prev_baked_light || rebind)) {

			material_shader.set_uniform(MaterialShaderGLES2::AMBIENT_LIGHTMAP, max_texture_units - 3);
			material_shader.set_uniform(MaterialShaderGLES2::AMBIENT_LIGHTMAP_MULTIPLIER, baked_light->lightmap_multiplier);
		}

		if (bind_dp_sampler) {

			material_shader.set_uniform(MaterialShaderGLES2::AMBIENT_DP_SAMPLER_MULTIPLIER, sampled_light_dp_multiplier);
			material_shader.set_uniform(MaterialShaderGLES2::AMBIENT_DP_SAMPLER, max_texture_units - 3);
		}

		_set_cull(e->mirror, p_reverse_cull);

		if (i == 0 || rebind) {
			material_shader.set_uniform(MaterialShaderGLES2::CAMERA_INVERSE_TRANSFORM, p_view_transform_inverse);
			material_shader.set_uniform(MaterialShaderGLES2::PROJECTION_TRANSFORM, p_projection);
			if (!shadow) {

				if (!additive && current_env && current_env->fx_enabled[VS::ENV_FX_AMBIENT_LIGHT]) {
					Color ambcolor = _convert_color(current_env->fx_param[VS::ENV_FX_PARAM_AMBIENT_LIGHT_COLOR]);
					float ambnrg = current_env->fx_param[VS::ENV_FX_PARAM_AMBIENT_LIGHT_ENERGY];
					material_shader.set_uniform(MaterialShaderGLES2::AMBIENT_LIGHT, Vector3(ambcolor.r * ambnrg, ambcolor.g * ambnrg, ambcolor.b * ambnrg));
				} else {
					material_shader.set_uniform(MaterialShaderGLES2::AMBIENT_LIGHT, Vector3());
				}
			}

			_rinfo.shader_change_count++;
		}

		if (skeleton != prev_skeleton || rebind) {
			if (skeleton && morph_values == NULL) {
				material_shader.set_uniform(MaterialShaderGLES2::SKELETON_MATRICES, max_texture_units - 2);
				material_shader.set_uniform(MaterialShaderGLES2::SKELTEX_PIXEL_SIZE, skeleton->pixel_size);
			}
		}

		if (e->instance->billboard || e->instance->billboard_y || e->instance->depth_scale) {

			Transform xf = e->instance->transform;
			if (e->instance->depth_scale) {

				if (p_projection.matrix[3][3]) {
					//orthogonal matrix, try to do about the same
					//with viewport size
					//real_t w = Math::abs( 1.0/(2.0*(p_projection.matrix[0][0])) );
					real_t h = Math::abs(1.0 / (2.0 * p_projection.matrix[1][1]));
					float sc = (h * 2.0); //consistent with Y-fov
					xf.basis.scale(Vector3(sc, sc, sc));
				} else {
					//just scale by depth
					real_t sc = -camera_plane.distance_to(xf.origin);
					xf.basis.scale(Vector3(sc, sc, sc));
				}
			}

			if (e->instance->billboard) {

				Vector3 scale = xf.basis.get_scale();

				if (current_rt && current_rt_vflip) {
					xf.set_look_at(xf.origin, xf.origin + p_view_transform.get_basis().get_axis(2), -p_view_transform.get_basis().get_axis(1));
				} else {
					xf.set_look_at(xf.origin, xf.origin + p_view_transform.get_basis().get_axis(2), p_view_transform.get_basis().get_axis(1));
				}

				xf.basis.scale(scale);
			}

			if (e->instance->billboard_y) {

				Vector3 scale = xf.basis.get_scale();
				Vector3 look_at = p_view_transform.get_origin();
				look_at.y = 0.0;
				Vector3 look_at_norm = look_at.normalized();

				if (current_rt && current_rt_vflip) {
					xf.set_look_at(xf.origin, xf.origin + look_at_norm, Vector3(0.0, -1.0, 0.0));
				} else {
					xf.set_look_at(xf.origin, xf.origin + look_at_norm, Vector3(0.0, 1.0, 0.0));
				}
				xf.basis.scale(scale);
			}
			material_shader.set_uniform(MaterialShaderGLES2::WORLD_TRANSFORM, xf);

		} else {
			material_shader.set_uniform(MaterialShaderGLES2::WORLD_TRANSFORM, e->instance->transform);
		}

		material_shader.set_uniform(MaterialShaderGLES2::NORMAL_MULT, e->mirror ? -1.0 : 1.0);
		material_shader.set_uniform(MaterialShaderGLES2::CONST_LIGHT_MULT, additive ? 0.0 : 1.0);

		_render(e->geometry, material, skeleton, e->owner, e->instance->transform);
		DEBUG_TEST_ERROR("Rendering");

		prev_material = material;
		prev_skeleton = skeleton;
		prev_geometry_cmp = geometry_cmp;
		prev_light = e->light;
		prev_light_type = e->light_type;
		prev_sort_flags = sort_flags;
		prev_baked_light = baked_light;
		prev_morph_values = morph_values;
		prev_receive_shadows_state = receive_shadows_state;
	}

	//print_line("shaderchanges: "+itos(p_alpha_pass)+": "+itos(_rinfo.shader_change_count));

	if (current_rt && current_rt_vflip) {
		glFrontFace(GL_CW);
	}
};

void RasterizerGLES2::_copy_to_texscreen() {

	//what am i missing?
	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_SCISSOR_TEST);
#ifdef GLEW_ENABLED
	glDisable(GL_POINT_SPRITE);
	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
#endif
	glDisable(GL_BLEND);
	glBlendEquation(GL_FUNC_ADD);
	if (current_rt && current_rt_transparent) {
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	} else {
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	//glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	for (int i = 0; i < VS::ARRAY_MAX; i++) {
		glDisableVertexAttribArray(i);
	}

	glActiveTexture(GL_TEXTURE0);

	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.sample_fbo);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, framebuffer.color);
	copy_shader.bind();
	_copy_screen_quad();
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.fbo);
}

void RasterizerGLES2::_copy_screen_quad() {

	Vector2 dst_pos[4] = {
		Vector2(-1, 1),
		Vector2(1, 1),
		Vector2(1, -1),
		Vector2(-1, -1)
	};

	Size2 uvscale(
			(viewport.width / float(framebuffer.scale)) / framebuffer.width,
			(viewport.height / float(framebuffer.scale)) / framebuffer.height);

	Vector2 src_uv[4] = {
		Vector2(0, 1) * uvscale,
		Vector2(1, 1) * uvscale,
		Vector2(1, 0) * uvscale,
		Vector2(0, 0) * uvscale
	};

	Vector2 full_uv[4] = {
		Vector2(0, 1),
		Vector2(1, 1),
		Vector2(1, 0),
		Vector2(0, 0)
	};

	_draw_gui_primitive2(4, dst_pos, NULL, src_uv, full_uv);
}

void RasterizerGLES2::_process_glow_bloom() {

	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.blur[0].fbo);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, framebuffer.color);
	copy_shader.set_conditional(CopyShaderGLES2::USE_GLOW_COPY, true);
	if (current_vd && current_env->fx_enabled[VS::ENV_FX_HDR]) {

		copy_shader.set_conditional(CopyShaderGLES2::USE_HDR, true);
	}

	copy_shader.bind();
	copy_shader.set_uniform(CopyShaderGLES2::BLOOM, float(current_env->fx_param[VS::ENV_FX_PARAM_GLOW_BLOOM]));
	copy_shader.set_uniform(CopyShaderGLES2::BLOOM_TRESHOLD, float(current_env->fx_param[VS::ENV_FX_PARAM_GLOW_BLOOM_TRESHOLD]));
	glUniform1i(copy_shader.get_uniform_location(CopyShaderGLES2::SOURCE), 0);

	if (current_vd && current_env->fx_enabled[VS::ENV_FX_HDR]) {
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, current_vd->lum_color);
		glUniform1i(copy_shader.get_uniform_location(CopyShaderGLES2::HDR_SOURCE), 2);
		copy_shader.set_uniform(CopyShaderGLES2::TONEMAP_EXPOSURE, float(current_env->fx_param[VS::ENV_FX_PARAM_HDR_EXPOSURE]));
		copy_shader.set_uniform(CopyShaderGLES2::TONEMAP_WHITE, float(current_env->fx_param[VS::ENV_FX_PARAM_HDR_WHITE]));
		//copy_shader.set_uniform(CopyShaderGLES2::TONEMAP_WHITE,1.0);
		copy_shader.set_uniform(CopyShaderGLES2::HDR_GLOW_TRESHOLD, float(current_env->fx_param[VS::ENV_FX_PARAM_HDR_GLOW_TRESHOLD]));
		copy_shader.set_uniform(CopyShaderGLES2::HDR_GLOW_SCALE, float(current_env->fx_param[VS::ENV_FX_PARAM_HDR_GLOW_SCALE]));

		glActiveTexture(GL_TEXTURE0);
	}

	glViewport(0, 0, framebuffer.blur_size, framebuffer.blur_size);
	_copy_screen_quad();

	copy_shader.set_conditional(CopyShaderGLES2::USE_GLOW_COPY, false);
	copy_shader.set_conditional(CopyShaderGLES2::USE_HDR, false);
	int passes = current_env->fx_param[VS::ENV_FX_PARAM_GLOW_BLUR_PASSES];
	Vector2 psize(1.0 / framebuffer.blur_size, 1.0 / framebuffer.blur_size);
	float pscale = current_env->fx_param[VS::ENV_FX_PARAM_GLOW_BLUR_SCALE];
	float pmag = current_env->fx_param[VS::ENV_FX_PARAM_GLOW_BLUR_STRENGTH];

	for (int i = 0; i < passes; i++) {

		static const Vector2 src_uv[4] = {
			Vector2(0, 1),
			Vector2(1, 1),
			Vector2(1, 0),
			Vector2(0, 0)
		};
		static const Vector2 dst_pos[4] = {
			Vector2(-1, 1),
			Vector2(1, 1),
			Vector2(1, -1),
			Vector2(-1, -1)
		};

		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.blur[1].fbo);
		glBindTexture(GL_TEXTURE_2D, framebuffer.blur[0].color);
		copy_shader.set_conditional(CopyShaderGLES2::BLUR_V_PASS, true);
		copy_shader.set_conditional(CopyShaderGLES2::BLUR_H_PASS, false);
		copy_shader.bind();
		copy_shader.set_uniform(CopyShaderGLES2::PIXEL_SIZE, psize);
		copy_shader.set_uniform(CopyShaderGLES2::PIXEL_SCALE, pscale);
		copy_shader.set_uniform(CopyShaderGLES2::BLUR_MAGNITUDE, pmag);

		_draw_gui_primitive(4, dst_pos, NULL, src_uv);

		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.blur[0].fbo);
		glBindTexture(GL_TEXTURE_2D, framebuffer.blur[1].color);
		copy_shader.set_conditional(CopyShaderGLES2::BLUR_V_PASS, false);
		copy_shader.set_conditional(CopyShaderGLES2::BLUR_H_PASS, true);
		copy_shader.bind();
		copy_shader.set_uniform(CopyShaderGLES2::PIXEL_SIZE, psize);
		copy_shader.set_uniform(CopyShaderGLES2::PIXEL_SCALE, pscale);
		copy_shader.set_uniform(CopyShaderGLES2::BLUR_MAGNITUDE, pmag);

		_draw_gui_primitive(4, dst_pos, NULL, src_uv);
	}

	copy_shader.set_conditional(CopyShaderGLES2::BLUR_V_PASS, false);
	copy_shader.set_conditional(CopyShaderGLES2::BLUR_H_PASS, false);
	copy_shader.set_conditional(CopyShaderGLES2::USE_HDR, false);

	//blur it
}

void RasterizerGLES2::_process_hdr() {

	if (framebuffer.luminance.empty()) {
		return;
	}

	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.luminance[0].fbo);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, framebuffer.color);
	copy_shader.set_conditional(CopyShaderGLES2::USE_HDR_COPY, true);
	copy_shader.bind();
	glUniform1i(copy_shader.get_uniform_location(CopyShaderGLES2::SOURCE), 0);
	glViewport(0, 0, framebuffer.luminance[0].size, framebuffer.luminance[0].size);
	_copy_screen_quad();

	copy_shader.set_conditional(CopyShaderGLES2::USE_HDR_COPY, false);
	//int passes = current_env->fx_param[VS::ENV_FX_PARAM_GLOW_BLUR_PASSES];

	copy_shader.set_conditional(CopyShaderGLES2::USE_HDR_REDUCE, true);
	copy_shader.bind();

	for (int i = 1; i < framebuffer.luminance.size(); i++) {

		static const Vector2 src_uv[4] = {
			Vector2(0, 1),
			Vector2(1, 1),
			Vector2(1, 0),
			Vector2(0, 0)
		};
		static const Vector2 dst_pos[4] = {
			Vector2(-1, 1),
			Vector2(1, 1),
			Vector2(1, -1),
			Vector2(-1, -1)
		};

		Vector2 psize(1.0 / framebuffer.luminance[i - 1].size, 1.0 / framebuffer.luminance[i - 1].size);
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.luminance[i].fbo);
		glBindTexture(GL_TEXTURE_2D, framebuffer.luminance[i - 1].color);
		glViewport(0, 0, framebuffer.luminance[i].size, framebuffer.luminance[i].size);
		glUniform1i(copy_shader.get_uniform_location(CopyShaderGLES2::SOURCE), 0);

		if (framebuffer.luminance[i].size == 1) {
			//last step
			copy_shader.set_conditional(CopyShaderGLES2::USE_HDR_STORE, true);
			copy_shader.bind();
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, current_vd->lum_color);
			glUniform1i(copy_shader.get_uniform_location(CopyShaderGLES2::SOURCE_VD_LUM), 1);
			copy_shader.set_uniform(CopyShaderGLES2::HDR_TIME_DELTA, time_delta);
			copy_shader.set_uniform(CopyShaderGLES2::HDR_EXP_ADJ_SPEED, float(current_env->fx_param[VS::ENV_FX_PARAM_HDR_EXPOSURE_ADJUST_SPEED]));
			copy_shader.set_uniform(CopyShaderGLES2::MIN_LUMINANCE, float(current_env->fx_param[VS::ENV_FX_PARAM_HDR_MIN_LUMINANCE]));
			copy_shader.set_uniform(CopyShaderGLES2::MAX_LUMINANCE, float(current_env->fx_param[VS::ENV_FX_PARAM_HDR_MAX_LUMINANCE]));
			glUniform1i(copy_shader.get_uniform_location(CopyShaderGLES2::SOURCE), 0);

			//swap them
			SWAP(current_vd->lum_color, framebuffer.luminance[i].color);
			SWAP(current_vd->lum_fbo, framebuffer.luminance[i].fbo);
		}

		copy_shader.set_uniform(CopyShaderGLES2::PIXEL_SIZE, psize);

		_draw_gui_primitive(4, dst_pos, NULL, src_uv);
	}

	copy_shader.set_conditional(CopyShaderGLES2::USE_HDR_REDUCE, false);
	copy_shader.set_conditional(CopyShaderGLES2::USE_HDR_STORE, false);

	draw_next_frame = true;
}

void RasterizerGLES2::_draw_tex_bg() {

	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDisable(GL_BLEND);
	glColorMask(1, 1, 1, 1);

	RID texture;

	if (current_env->bg_mode == VS::ENV_BG_TEXTURE) {
		texture = current_env->bg_param[VS::ENV_BG_PARAM_TEXTURE];
	} else {
		texture = current_env->bg_param[VS::ENV_BG_PARAM_CUBEMAP];
	}

	if (!texture_owner.owns(texture)) {
		return;
	}

	Texture *t = texture_owner.get(texture);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(t->target, t->tex_id);

	copy_shader.set_conditional(CopyShaderGLES2::USE_ENERGY, true);

	if (current_env->bg_mode == VS::ENV_BG_TEXTURE) {
		copy_shader.set_conditional(CopyShaderGLES2::USE_CUBEMAP, false);

	} else {
		copy_shader.set_conditional(CopyShaderGLES2::USE_CUBEMAP, true);
	}

	copy_shader.set_conditional(CopyShaderGLES2::USE_CUSTOM_ALPHA, true);

	copy_shader.bind();

	if (current_env->bg_mode == VS::ENV_BG_TEXTURE) {
		glUniform1i(copy_shader.get_uniform_location(CopyShaderGLES2::SOURCE), 0);
	} else {
		glUniform1i(copy_shader.get_uniform_location(CopyShaderGLES2::SOURCE_CUBE), 0);
	}

	float nrg = float(current_env->bg_param[VS::ENV_BG_PARAM_ENERGY]);
	if (current_env->fx_enabled[VS::ENV_FX_HDR] && !use_fp16_fb)
		nrg *= 0.25; //go down a quarter for hdr
	copy_shader.set_uniform(CopyShaderGLES2::ENERGY, nrg);
	copy_shader.set_uniform(CopyShaderGLES2::CUSTOM_ALPHA, float(current_env->bg_param[VS::ENV_BG_PARAM_GLOW]));

	float flip_sign = (current_env->bg_mode == VS::ENV_BG_TEXTURE && current_rt && current_rt_vflip) ? -1 : 1;

	Vector3 vertices[4] = {
		Vector3(-1, -1 * flip_sign, 1),
		Vector3(1, -1 * flip_sign, 1),
		Vector3(1, 1 * flip_sign, 1),
		Vector3(-1, 1 * flip_sign, 1)
	};

	Vector3 src_uv[4] = {
		Vector3(0, 1, 0),
		Vector3(1, 1, 0),
		Vector3(1, 0, 0),
		Vector3(0, 0, 0)
	};

	if (current_env->bg_mode == VS::ENV_BG_TEXTURE) {

		//regular texture
		//adjust aspect

		float aspect_t = t->width / float(t->height);
		float aspect_v = viewport.width / float(viewport.height);

		if (aspect_v > aspect_t) {
			//wider than texture
			for (int i = 0; i < 4; i++) {
				src_uv[i].y = (src_uv[i].y - 0.5) * (aspect_t / aspect_v) + 0.5;
			}

		} else {
			//narrower than texture
			for (int i = 0; i < 4; i++) {
				src_uv[i].x = (src_uv[i].x - 0.5) * (aspect_v / aspect_t) + 0.5;
			}
		}

		float scale = current_env->bg_param[VS::ENV_BG_PARAM_SCALE];
		for (int i = 0; i < 4; i++) {

			src_uv[i].x *= scale;
			src_uv[i].y *= scale;
		}
	} else {

		//skybox uv vectors
		float vw, vh, zn;
		camera_projection.get_viewport_size(vw, vh);
		zn = camera_projection.get_z_near();

		float scale = current_env->bg_param[VS::ENV_BG_PARAM_SCALE];

		for (int i = 0; i < 4; i++) {

			Vector3 uv = src_uv[i];
			uv.x = (uv.x * 2.0 - 1.0) * vw * scale;
			uv.y = -(uv.y * 2.0 - 1.0) * vh * scale;
			uv.z = -zn;
			src_uv[i] = camera_transform.basis.xform(uv).normalized();
			src_uv[i].z = -src_uv[i].z;
		}
	}

	_draw_primitive(4, vertices, NULL, NULL, src_uv);

	copy_shader.set_conditional(CopyShaderGLES2::USE_ENERGY, false);
	copy_shader.set_conditional(CopyShaderGLES2::USE_RGBE, false);
	copy_shader.set_conditional(CopyShaderGLES2::USE_CUBEMAP, false);
	copy_shader.set_conditional(CopyShaderGLES2::USE_CUSTOM_ALPHA, false);
}

void RasterizerGLES2::end_scene() {

	glEnable(GL_BLEND);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_SCISSOR_TEST);

	bool use_fb = false;

	if (framebuffer.active) {

		//detect when to use the framebuffer object
		if (using_canvas_bg || texscreen_used || framebuffer.scale != 1) {
			use_fb = true;
		} else if (current_env) {
			use_fb = false;
			for (int i = 0; i < VS::ENV_FX_MAX; i++) {

				if (i == VS::ENV_FX_FOG) //does not need fb
					continue;

				if (current_env->fx_enabled[i]) {
					use_fb = true;
					break;
				}
			}
		}
	}

	if (use_fb) {

		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.fbo);
		glViewport(0, 0, viewport.width / framebuffer.scale, viewport.height / framebuffer.scale);
		glScissor(0, 0, viewport.width / framebuffer.scale, viewport.height / framebuffer.scale);

		material_shader.set_conditional(MaterialShaderGLES2::USE_8BIT_HDR, !use_fp16_fb && current_env && current_env->fx_enabled[VS::ENV_FX_HDR]);

	} else {
		if (current_rt) {
			glScissor(0, 0, viewport.width, viewport.height);
		} else {
			glScissor(viewport.x, window_size.height - (viewport.height + viewport.y), viewport.width, viewport.height);
		}
	}

	glEnable(GL_SCISSOR_TEST);
	_glClearDepth(1.0);

	bool draw_tex_background = false;

	if (current_debug == VS::SCENARIO_DEBUG_OVERDRAW) {

		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	} else if (current_rt && current_rt_transparent) {

		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	} else if (current_env) {

		switch (current_env->bg_mode) {

			case VS::ENV_BG_CANVAS:
			case VS::ENV_BG_KEEP: {
				//copy from framebuffer if framebuffer
				glClear(GL_DEPTH_BUFFER_BIT);
			} break;
			case VS::ENV_BG_DEFAULT_COLOR:
			case VS::ENV_BG_COLOR: {

				Color bgcolor;
				if (current_env->bg_mode == VS::ENV_BG_COLOR)
					bgcolor = current_env->bg_param[VS::ENV_BG_PARAM_COLOR];
				else
					bgcolor = GlobalConfig::get_singleton()->get("render/default_clear_color");
				bgcolor = _convert_color(bgcolor);
				float a = use_fb ? float(current_env->bg_param[VS::ENV_BG_PARAM_GLOW]) : 1.0;
				glClearColor(bgcolor.r, bgcolor.g, bgcolor.b, a);
				_glClearDepth(1.0);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			} break;
			case VS::ENV_BG_TEXTURE:
			case VS::ENV_BG_CUBEMAP: {

				glClear(GL_DEPTH_BUFFER_BIT);
				draw_tex_background = true;
			} break;
		}
	} else {

		Color c = _convert_color(Color(0.3, 0.3, 0.3));
		glClearColor(c.r, c.g, c.b, 0.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	glDisable(GL_SCISSOR_TEST);

	//material_shader.set_uniform_camera(MaterialShaderGLES2::PROJECTION_MATRIX, camera_projection);

	/*
	printf("setting projection to ");
	for (int i=0; i<16; i++) {
		printf("%f, ", ((float*)camera_projection.matrix)[i]);
	};
	printf("\n");

	print_line(String("setting camera to ")+camera_transform_inverse);
	*/
	//material_shader.set_uniform_default(MaterialShaderGLES2::CAMERA_INVERSE, camera_transform_inverse);

	current_depth_test = true;
	current_depth_mask = true;
	texscreen_copied = false;
	glBlendEquation(GL_FUNC_ADD);
	if (current_rt && current_rt_transparent) {
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	} else {
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	glDisable(GL_BLEND);
	current_blend_mode = VS::MATERIAL_BLEND_MODE_MIX;

	//material_shader.set_conditional(MaterialShaderGLES2::USE_GLOW,current_env && current_env->fx_enabled[VS::ENV_FX_GLOW]);
	opaque_render_list.sort_mat_light_type_flags();
	_render_list_forward(&opaque_render_list, camera_transform, camera_transform_inverse, camera_projection, false, fragment_lighting);

	if (draw_tex_background) {

		//most 3D vendors recommend drawing a texture bg or skybox here,
		//after opaque geometry has been drawn
		//so the zbuffer can get rid of most pixels
		_draw_tex_bg();
	}

	glBlendEquation(GL_FUNC_ADD);
	if (current_rt && current_rt_transparent) {
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	} else {
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	glDisable(GL_BLEND);
	current_blend_mode = VS::MATERIAL_BLEND_MODE_MIX;
	material_shader.set_conditional(MaterialShaderGLES2::USE_GLOW, false);
	if (current_env && current_env->fx_enabled[VS::ENV_FX_GLOW]) {
		glColorMask(1, 1, 1, 0); //don't touch alpha
	}

	alpha_render_list.sort_z();
	_render_list_forward(&alpha_render_list, camera_transform, camera_transform_inverse, camera_projection, false, fragment_lighting, true);
	glColorMask(1, 1, 1, 1);

	//material_shader.set_conditional( MaterialShaderGLES2::USE_FOG,false);

	DEBUG_TEST_ERROR("Drawing Scene");

#ifdef GLEW_ENABLED
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
#endif

	if (use_fb) {

		for (int i = 0; i < VS::ARRAY_MAX; i++) {
			glDisableVertexAttribArray(i);
		}
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glDisable(GL_BLEND);
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);
		glDisable(GL_SCISSOR_TEST);
		glDepthMask(false);

		if (current_env && current_env->fx_enabled[VS::ENV_FX_HDR]) {

			int hdr_tm = current_env->fx_param[VS::ENV_FX_PARAM_HDR_TONEMAPPER];
			switch (hdr_tm) {
				case VS::ENV_FX_HDR_TONE_MAPPER_LINEAR: {

				} break;
				case VS::ENV_FX_HDR_TONE_MAPPER_LOG: {
					copy_shader.set_conditional(CopyShaderGLES2::USE_LOG_TONEMAPPER, true);

				} break;
				case VS::ENV_FX_HDR_TONE_MAPPER_REINHARDT: {
					copy_shader.set_conditional(CopyShaderGLES2::USE_REINHARDT_TONEMAPPER, true);
				} break;
				case VS::ENV_FX_HDR_TONE_MAPPER_REINHARDT_AUTOWHITE: {

					copy_shader.set_conditional(CopyShaderGLES2::USE_REINHARDT_TONEMAPPER, true);
					copy_shader.set_conditional(CopyShaderGLES2::USE_AUTOWHITE, true);
				} break;
			}

			_process_hdr();
		}
		if (current_env && current_env->fx_enabled[VS::ENV_FX_GLOW]) {
			_process_glow_bloom();
			int glow_transfer_mode = current_env->fx_param[VS::ENV_FX_PARAM_GLOW_BLUR_BLEND_MODE];
			if (glow_transfer_mode == 1)
				copy_shader.set_conditional(CopyShaderGLES2::USE_GLOW_SCREEN, true);
			if (glow_transfer_mode == 2)
				copy_shader.set_conditional(CopyShaderGLES2::USE_GLOW_SOFTLIGHT, true);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, current_rt ? current_rt->fbo : base_framebuffer);

		Size2 size;
		if (current_rt) {
			glBindFramebuffer(GL_FRAMEBUFFER, current_rt->fbo);
			glViewport(0, 0, viewport.width, viewport.height);
			size = Size2(viewport.width, viewport.height);
		} else {
			glBindFramebuffer(GL_FRAMEBUFFER, base_framebuffer);
			glViewport(viewport.x, window_size.height - (viewport.height + viewport.y), viewport.width, viewport.height);
			size = Size2(viewport.width, viewport.height);
		}

		//time to copy!!!
		copy_shader.set_conditional(CopyShaderGLES2::USE_BCS, current_env && current_env->fx_enabled[VS::ENV_FX_BCS]);
		copy_shader.set_conditional(CopyShaderGLES2::USE_SRGB, current_env && current_env->fx_enabled[VS::ENV_FX_SRGB]);
		copy_shader.set_conditional(CopyShaderGLES2::USE_GLOW, current_env && current_env->fx_enabled[VS::ENV_FX_GLOW]);
		copy_shader.set_conditional(CopyShaderGLES2::USE_HDR, current_env && current_env->fx_enabled[VS::ENV_FX_HDR]);
		copy_shader.set_conditional(CopyShaderGLES2::USE_NO_ALPHA, true);
		copy_shader.set_conditional(CopyShaderGLES2::USE_FXAA, current_env && current_env->fx_enabled[VS::ENV_FX_FXAA]);

		copy_shader.bind();
		//copy_shader.set_uniform(CopyShaderGLES2::SOURCE,0);

		if (current_env && current_env->fx_enabled[VS::ENV_FX_GLOW]) {

			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, framebuffer.blur[0].color);
			glUniform1i(copy_shader.get_uniform_location(CopyShaderGLES2::GLOW_SOURCE), 1);
		}

		if (current_env && current_env->fx_enabled[VS::ENV_FX_HDR]) {

			glActiveTexture(GL_TEXTURE2);
			glBindTexture(GL_TEXTURE_2D, current_vd->lum_color);
			glUniform1i(copy_shader.get_uniform_location(CopyShaderGLES2::HDR_SOURCE), 2);
			copy_shader.set_uniform(CopyShaderGLES2::TONEMAP_EXPOSURE, float(current_env->fx_param[VS::ENV_FX_PARAM_HDR_EXPOSURE]));
			copy_shader.set_uniform(CopyShaderGLES2::TONEMAP_WHITE, float(current_env->fx_param[VS::ENV_FX_PARAM_HDR_WHITE]));
		}

		if (current_env && current_env->fx_enabled[VS::ENV_FX_FXAA])
			copy_shader.set_uniform(CopyShaderGLES2::PIXEL_SIZE, Size2(1.0 / size.x, 1.0 / size.y));

		if (current_env && current_env->fx_enabled[VS::ENV_FX_BCS]) {

			Vector3 bcs;
			bcs.x = current_env->fx_param[VS::ENV_FX_PARAM_BCS_BRIGHTNESS];
			bcs.y = current_env->fx_param[VS::ENV_FX_PARAM_BCS_CONTRAST];
			bcs.z = current_env->fx_param[VS::ENV_FX_PARAM_BCS_SATURATION];
			copy_shader.set_uniform(CopyShaderGLES2::BCS, bcs);
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, framebuffer.color);
		glUniform1i(copy_shader.get_uniform_location(CopyShaderGLES2::SOURCE), 0);

		_copy_screen_quad();

		copy_shader.set_conditional(CopyShaderGLES2::USE_BCS, false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_SRGB, false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_GLOW, false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_HDR, false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_NO_ALPHA, false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_FXAA, false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_GLOW_SCREEN, false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_GLOW_SOFTLIGHT, false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_REINHARDT_TONEMAPPER, false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_AUTOWHITE, false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_LOG_TONEMAPPER, false);

		material_shader.set_conditional(MaterialShaderGLES2::USE_8BIT_HDR, false);

		if (current_env && current_env->fx_enabled[VS::ENV_FX_HDR] && GLOBAL_DEF("rasterizer/debug_hdr", false)) {
			_debug_luminances();
		}
	}

	current_env = NULL;
	current_debug = VS::SCENARIO_DEBUG_DISABLED;
	if (GLOBAL_DEF("rasterizer/debug_shadow_maps", false)) {
		_debug_shadows();
	}
	//_debug_luminances();
	//_debug_samplers();

	if (using_canvas_bg) {
		using_canvas_bg = false;
		glColorMask(1, 1, 1, 1); //don't touch alpha
	}
}
void RasterizerGLES2::end_shadow_map() {

	ERR_FAIL_COND(!shadow);

	glDisable(GL_BLEND);
	glDisable(GL_SCISSOR_TEST);
	glDisable(GL_DITHER);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(true);

	ShadowBuffer *sb = shadow->near_shadow_buffer;

	ERR_FAIL_COND(!sb);

	glBindFramebuffer(GL_FRAMEBUFFER, sb->fbo);

	if (!use_rgba_shadowmaps)
		glColorMask(0, 0, 0, 0);

	//glEnable(GL_POLYGON_OFFSET_FILL);
	//glPolygonOffset( 8.0f, 16.0f);

	CameraMatrix cm;
	float z_near, z_far;
	Transform light_transform;

	float dp_direction = 0.0;
	bool flip_facing = false;
	Rect2 vp_rect;

	switch (shadow->base->type) {

		case VS::LIGHT_DIRECTIONAL: {

			if (shadow->base->directional_shadow_mode == VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS) {

				cm = shadow->custom_projection[shadow_pass];
				light_transform = shadow->custom_transform[shadow_pass];

				if (shadow_pass == 0) {

					vp_rect = Rect2(0, sb->size / 2, sb->size / 2, sb->size / 2);
					glViewport(0, sb->size / 2, sb->size / 2, sb->size / 2);
					glScissor(0, sb->size / 2, sb->size / 2, sb->size / 2);
				} else if (shadow_pass == 1) {

					vp_rect = Rect2(0, 0, sb->size / 2, sb->size / 2);
					glViewport(0, 0, sb->size / 2, sb->size / 2);
					glScissor(0, 0, sb->size / 2, sb->size / 2);

				} else if (shadow_pass == 2) {

					vp_rect = Rect2(sb->size / 2, sb->size / 2, sb->size / 2, sb->size / 2);
					glViewport(sb->size / 2, sb->size / 2, sb->size / 2, sb->size / 2);
					glScissor(sb->size / 2, sb->size / 2, sb->size / 2, sb->size / 2);
				} else if (shadow_pass == 3) {

					vp_rect = Rect2(sb->size / 2, 0, sb->size / 2, sb->size / 2);
					glViewport(sb->size / 2, 0, sb->size / 2, sb->size / 2);
					glScissor(sb->size / 2, 0, sb->size / 2, sb->size / 2);
				}

				glEnable(GL_SCISSOR_TEST);

			} else if (shadow->base->directional_shadow_mode == VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS) {

				if (shadow_pass == 0) {

					cm = shadow->custom_projection[0];
					light_transform = shadow->custom_transform[0];
					vp_rect = Rect2(0, sb->size / 2, sb->size, sb->size / 2);
					glViewport(0, sb->size / 2, sb->size, sb->size / 2);
					glScissor(0, sb->size / 2, sb->size, sb->size / 2);
				} else {

					cm = shadow->custom_projection[1];
					light_transform = shadow->custom_transform[1];
					vp_rect = Rect2(0, 0, sb->size, sb->size / 2);
					glViewport(0, 0, sb->size, sb->size / 2);
					glScissor(0, 0, sb->size, sb->size / 2);
				}

				glEnable(GL_SCISSOR_TEST);

			} else {
				cm = shadow->custom_projection[0];
				light_transform = shadow->custom_transform[0];
				vp_rect = Rect2(0, 0, sb->size, sb->size);
				glViewport(0, 0, sb->size, sb->size);
			}

			z_near = cm.get_z_near();
			z_far = cm.get_z_far();

			_glClearDepth(1.0f);
			glClearColor(1, 1, 1, 1);

			if (use_rgba_shadowmaps)
				glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
			else
				glClear(GL_DEPTH_BUFFER_BIT);

			glDisable(GL_SCISSOR_TEST);

		} break;
		case VS::LIGHT_OMNI: {

			material_shader.set_conditional(MaterialShaderGLES2::USE_DUAL_PARABOLOID, true);
			dp_direction = shadow_pass ? 1.0 : -1.0;
			flip_facing = (shadow_pass == 1);
			light_transform = shadow->transform;
			z_near = 0;
			z_far = shadow->base->vars[VS::LIGHT_PARAM_RADIUS];
			shadow->dp.x = 1.0 / z_far;
			shadow->dp.y = dp_direction;

			if (shadow_pass == 0) {
				vp_rect = Rect2(0, sb->size / 2, sb->size, sb->size / 2);
				glViewport(0, sb->size / 2, sb->size, sb->size / 2);
				glScissor(0, sb->size / 2, sb->size, sb->size / 2);
			} else {
				vp_rect = Rect2(0, 0, sb->size, sb->size / 2);
				glViewport(0, 0, sb->size, sb->size / 2);
				glScissor(0, 0, sb->size, sb->size / 2);
			}
			glEnable(GL_SCISSOR_TEST);
			shadow->projection = cm;

			glClearColor(1, 1, 1, 1);
			_glClearDepth(1.0f);
			if (use_rgba_shadowmaps)
				glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
			else
				glClear(GL_DEPTH_BUFFER_BIT);
			glDisable(GL_SCISSOR_TEST);

		} break;
		case VS::LIGHT_SPOT: {

			float far = shadow->base->vars[VS::LIGHT_PARAM_RADIUS];
			ERR_FAIL_COND(far <= 0);
			float near = far / 200.0;
			if (near < 0.05)
				near = 0.05;

			float angle = shadow->base->vars[VS::LIGHT_PARAM_SPOT_ANGLE];

			cm.set_perspective(angle * 2.0, 1.0, near, far);

			shadow->projection = cm; // cache
			light_transform = shadow->transform;
			z_near = cm.get_z_near();
			z_far = cm.get_z_far();

			glViewport(0, 0, sb->size, sb->size);
			vp_rect = Rect2(0, 0, sb->size, sb->size);
			_glClearDepth(1.0f);
			glClearColor(1, 1, 1, 1);
			if (use_rgba_shadowmaps)
				glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
			else
				glClear(GL_DEPTH_BUFFER_BIT);

		} break;
	}

	Transform light_transform_inverse = light_transform.affine_inverse();

	opaque_render_list.sort_mat_geom();
	_render_list_forward(&opaque_render_list, light_transform, light_transform_inverse, cm, flip_facing, false);

	material_shader.set_conditional(MaterialShaderGLES2::USE_DUAL_PARABOLOID, false);

	//if (!use_rgba_shadowmaps)

	if (shadow_filter == SHADOW_FILTER_ESM) {

		copy_shader.set_conditional(CopyShaderGLES2::USE_RGBA_DEPTH, use_rgba_shadowmaps);
		copy_shader.set_conditional(CopyShaderGLES2::USE_HIGHP_SOURCE, !use_rgba_shadowmaps);

		Vector2 psize(1.0 / sb->size, 1.0 / sb->size);
		float pscale = 1.0;
		int passes = shadow->base->vars[VS::LIGHT_PARAM_SHADOW_BLUR_PASSES];
		glDisable(GL_BLEND);
		glDisable(GL_CULL_FACE);
#ifdef GLEW_ENABLED
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
#endif

		for (int i = 0; i < VS::ARRAY_MAX; i++) {
			glDisableVertexAttribArray(i);
		}
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glDisable(GL_SCISSOR_TEST);

		if (!use_rgba_shadowmaps) {
			glEnable(GL_DEPTH_TEST);
			glDepthFunc(GL_ALWAYS);
			glDepthMask(true);
		} else {
			glDisable(GL_DEPTH_TEST);
		}

		for (int i = 0; i < passes; i++) {

			Vector2 src_sb_uv[4] = {
				(vp_rect.pos + Vector2(0, vp_rect.size.y)) / sb->size,
				(vp_rect.pos + vp_rect.size) / sb->size,
				(vp_rect.pos + Vector2(vp_rect.size.x, 0)) / sb->size,
				(vp_rect.pos) / sb->size
			};
			/*
			Vector2 src_uv[4]={
				Vector2( 0, 1),
				Vector2( 1, 1),
				Vector2( 1, 0),
				Vector2( 0, 0)
			};
*/
			static const Vector2 dst_pos[4] = {
				Vector2(-1, 1),
				Vector2(1, 1),
				Vector2(1, -1),
				Vector2(-1, -1)
			};

			glBindFramebuffer(GL_FRAMEBUFFER, blur_shadow_buffer.fbo);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, sb->depth);
#ifdef GLEW_ENABLED
//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
#endif

			copy_shader.set_conditional(CopyShaderGLES2::SHADOW_BLUR_V_PASS, true);
			copy_shader.set_conditional(CopyShaderGLES2::SHADOW_BLUR_H_PASS, false);

			copy_shader.bind();
			copy_shader.set_uniform(CopyShaderGLES2::PIXEL_SIZE, psize);
			copy_shader.set_uniform(CopyShaderGLES2::PIXEL_SCALE, pscale);
			copy_shader.set_uniform(CopyShaderGLES2::BLUR_MAGNITUDE, 1);
			//copy_shader.set_uniform(CopyShaderGLES2::SOURCE,0);
			glUniform1i(copy_shader.get_uniform_location(CopyShaderGLES2::SOURCE), 0);

			_draw_gui_primitive(4, dst_pos, NULL, src_sb_uv);

			Vector2 src_bb_uv[4] = {
				(vp_rect.pos + Vector2(0, vp_rect.size.y)) / blur_shadow_buffer.size,
				(vp_rect.pos + vp_rect.size) / blur_shadow_buffer.size,
				(vp_rect.pos + Vector2(vp_rect.size.x, 0)) / blur_shadow_buffer.size,
				(vp_rect.pos) / blur_shadow_buffer.size,
			};

			glBindFramebuffer(GL_FRAMEBUFFER, sb->fbo);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, blur_shadow_buffer.depth);
#ifdef GLEW_ENABLED

//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
#endif

			copy_shader.set_conditional(CopyShaderGLES2::SHADOW_BLUR_V_PASS, false);
			copy_shader.set_conditional(CopyShaderGLES2::SHADOW_BLUR_H_PASS, true);
			copy_shader.bind();
			copy_shader.set_uniform(CopyShaderGLES2::PIXEL_SIZE, psize);
			copy_shader.set_uniform(CopyShaderGLES2::PIXEL_SCALE, pscale);
			copy_shader.set_uniform(CopyShaderGLES2::BLUR_MAGNITUDE, 1);
			glUniform1i(copy_shader.get_uniform_location(CopyShaderGLES2::SOURCE), 0);

			_draw_gui_primitive(4, dst_pos, NULL, src_bb_uv);
		}

		glDepthFunc(GL_LEQUAL);
		copy_shader.set_conditional(CopyShaderGLES2::USE_RGBA_DEPTH, false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_HIGHP_SOURCE, false);
		copy_shader.set_conditional(CopyShaderGLES2::SHADOW_BLUR_V_PASS, false);
		copy_shader.set_conditional(CopyShaderGLES2::SHADOW_BLUR_H_PASS, false);
	}

	DEBUG_TEST_ERROR("Drawing Shadow");
	shadow = NULL;
	glBindFramebuffer(GL_FRAMEBUFFER, current_rt ? current_rt->fbo : base_framebuffer);
	glColorMask(1, 1, 1, 1);
	//glDisable(GL_POLYGON_OFFSET_FILL);
}

void RasterizerGLES2::_debug_draw_shadow(GLuint tex, const Rect2 &p_rect) {

	Transform2D modelview;
	modelview.translate(p_rect.pos.x, p_rect.pos.y);
	canvas_shader.set_uniform(CanvasShaderGLES2::MODELVIEW_MATRIX, modelview);
	glBindTexture(GL_TEXTURE_2D, tex);

	Vector3 coords[4] = {
		Vector3(p_rect.pos.x, p_rect.pos.y, 0),
		Vector3(p_rect.pos.x + p_rect.size.width,
				p_rect.pos.y, 0),
		Vector3(p_rect.pos.x + p_rect.size.width,
				p_rect.pos.y + p_rect.size.height, 0),
		Vector3(p_rect.pos.x,
				p_rect.pos.y + p_rect.size.height, 0)
	};

	Vector3 texcoords[4] = {
		Vector3(0.0f, 0.0f, 0),
		Vector3(1.0f, 0.0f, 0),
		Vector3(1.0f, 1.0f, 0),
		Vector3(0.0f, 1.0f, 0),
	};

	_draw_primitive(4, coords, 0, 0, texcoords);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
}

void RasterizerGLES2::_debug_draw_shadows_type(Vector<ShadowBuffer> &p_shadows, Point2 &ofs) {

	Size2 debug_size(128, 128);
	//Size2 debug_size(512,512);

	int useblur = shadow_filter == SHADOW_FILTER_ESM ? 1 : 0;
	for (int i = 0; i < p_shadows.size() + useblur; i++) {

		ShadowBuffer *sb = i == p_shadows.size() ? &blur_shadow_buffer : &p_shadows[i];

		if (!sb->owner && i != p_shadows.size())
			continue;

		_debug_draw_shadow(sb->depth, Rect2(ofs, debug_size));
		ofs.x += debug_size.x;
		if ((ofs.x + debug_size.x) > viewport.width) {

			ofs.x = 0;
			ofs.y += debug_size.y;
		}
	}
}

void RasterizerGLES2::_debug_luminances() {

	canvas_shader.set_conditional(CanvasShaderGLES2::DEBUG_ENCODED_32, !use_fp16_fb);
	canvas_begin();
	glDisable(GL_BLEND);
	canvas_shader.bind();

	Size2 debug_size(128, 128);
	Size2 ofs;

	for (int i = 0; i <= framebuffer.luminance.size(); i++) {

		if (i == framebuffer.luminance.size()) {
			if (!current_vd)
				break;
			_debug_draw_shadow(current_vd->lum_color, Rect2(ofs, debug_size));
		} else {
			_debug_draw_shadow(framebuffer.luminance[i].color, Rect2(ofs, debug_size));
		}
		ofs.x += debug_size.x / 2;
		if ((ofs.x + debug_size.x) > viewport.width) {

			ofs.x = 0;
			ofs.y += debug_size.y;
		}
	}

	canvas_shader.set_conditional(CanvasShaderGLES2::DEBUG_ENCODED_32, false);
}

void RasterizerGLES2::_debug_samplers() {
	canvas_shader.set_conditional(CanvasShaderGLES2::DEBUG_ENCODED_32, false);
	canvas_begin();
	glDisable(GL_BLEND);
	_set_color_attrib(Color(1, 1, 1, 1));
	canvas_shader.bind();

	List<RID> samplers;
	sampled_light_owner.get_owned_list(&samplers);

	Size2 debug_size(128, 128);
	Size2 ofs;

	for (List<RID>::Element *E = samplers.front(); E; E = E->next()) {

		SampledLight *sl = sampled_light_owner.get(E->get());

		_debug_draw_shadow(sl->texture, Rect2(ofs, debug_size));

		ofs.x += debug_size.x / 2;
		if ((ofs.x + debug_size.x) > viewport.width) {

			ofs.x = 0;
			ofs.y += debug_size.y;
		}
	}
}
void RasterizerGLES2::_debug_shadows() {

	canvas_begin();
	glDisable(GL_BLEND);
	Size2 ofs;

	/*
	for(int i=0;i<16;i++) {
		glActiveTexture(GL_TEXTURE0+i);
		//glDisable(GL_TEXTURE_2D);
	}
	glActiveTexture(GL_TEXTURE0);
	//glEnable(GL_TEXTURE_2D);
	*/

	_debug_draw_shadows_type(near_shadow_buffers, ofs);
	//_debug_draw_shadows_type(far_shadow_buffers,ofs);
}

void RasterizerGLES2::end_frame() {

	//print_line("VTX: "+itos(_rinfo.vertex_count)+" OBJ: "+itos(_rinfo.object_count)+" MAT: "+itos(_rinfo.mat_change_count)+" SHD: "+itos(_rinfo.shader_change_count)+" CI: "+itos(_rinfo.ci_draw_commands));

	//print_line("TOTAL VTX: "+itos(_rinfo.vertex_count));
	OS::get_singleton()->swap_buffers();
}

void RasterizerGLES2::flush_frame() {

	glFlush();
}

/* CANVAS API */

void RasterizerGLES2::begin_canvas_bg() {

	if (framebuffer.active) {
		using_canvas_bg = true;
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.fbo);
		glViewport(0, 0, viewport.width, viewport.height);
	} else {
		using_canvas_bg = false;
	}
}

void RasterizerGLES2::canvas_begin() {

	if (using_canvas_bg) {
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.fbo);
		glColorMask(1, 1, 1, 0); //don't touch alpha
	}

	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_SCISSOR_TEST);
#ifdef GLEW_ENABLED
	glDisable(GL_POINT_SPRITE);
	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
#endif
	glEnable(GL_BLEND);
	glBlendEquation(GL_FUNC_ADD);
	if (current_rt && current_rt_transparent) {
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	} else {
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	//glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	glLineWidth(1.0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	for (int i = 0; i < VS::ARRAY_MAX; i++) {
		glDisableVertexAttribArray(i);
	}

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, white_tex);
	canvas_tex = RID();
	//material_shader.unbind();
	canvas_shader.unbind();
	canvas_shader.set_custom_shader(0);
	canvas_shader.set_conditional(CanvasShaderGLES2::USE_MODULATE, false);
	canvas_shader.bind();
	canvas_shader.set_uniform(CanvasShaderGLES2::TEXTURE, 0);
	canvas_use_modulate = false;
	_set_color_attrib(Color(1, 1, 1));
	canvas_transform = Transform();
	canvas_transform.translate(-(viewport.width / 2.0f), -(viewport.height / 2.0f), 0.0f);
	float csy = 1.0;
	if (current_rt && current_rt_vflip)
		csy = -1.0;

	canvas_transform.scale(Vector3(2.0f / viewport.width, csy * -2.0f / viewport.height, 1.0f));
	canvas_shader.set_uniform(CanvasShaderGLES2::PROJECTION_MATRIX, canvas_transform);
	canvas_shader.set_uniform(CanvasShaderGLES2::MODELVIEW_MATRIX, Transform2D());
	canvas_shader.set_uniform(CanvasShaderGLES2::EXTRA_MATRIX, Transform2D());

	canvas_opacity = 1.0;
	canvas_blend_mode = VS::MATERIAL_BLEND_MODE_MIX;
	canvas_texscreen_used = false;
	uses_texpixel_size = false;

	canvas_last_material = NULL;
}

void RasterizerGLES2::canvas_disable_blending() {

	glDisable(GL_BLEND);
}

void RasterizerGLES2::canvas_set_opacity(float p_opacity) {

	canvas_opacity = p_opacity;
}

void RasterizerGLES2::canvas_set_blend_mode(VS::MaterialBlendMode p_mode) {

	if (p_mode == canvas_blend_mode)
		return;
	switch (p_mode) {

		case VS::MATERIAL_BLEND_MODE_MIX: {
			glBlendEquation(GL_FUNC_ADD);
			if (current_rt && current_rt_transparent) {
				glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
			} else {
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			}

		} break;
		case VS::MATERIAL_BLEND_MODE_ADD: {

			glBlendEquation(GL_FUNC_ADD);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE);

		} break;
		case VS::MATERIAL_BLEND_MODE_SUB: {

			glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE);
		} break;
		case VS::MATERIAL_BLEND_MODE_MUL: {
			glBlendEquation(GL_FUNC_ADD);
			glBlendFunc(GL_DST_COLOR, GL_ZERO);
		} break;
		case VS::MATERIAL_BLEND_MODE_PREMULT_ALPHA: {
			glBlendEquation(GL_FUNC_ADD);
			glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
		} break;
	}

	canvas_blend_mode = p_mode;
}

void RasterizerGLES2::canvas_begin_rect(const Transform2D &p_transform) {

	canvas_shader.set_uniform(CanvasShaderGLES2::MODELVIEW_MATRIX, p_transform);
	canvas_shader.set_uniform(CanvasShaderGLES2::EXTRA_MATRIX, Transform2D());
}

void RasterizerGLES2::canvas_set_clip(bool p_clip, const Rect2 &p_rect) {

	if (p_clip) {

		glEnable(GL_SCISSOR_TEST);
		//glScissor(viewport.x+p_rect.pos.x,viewport.y+ (viewport.height-(p_rect.pos.y+p_rect.size.height)),

		int x = p_rect.pos.x;
		int y = window_size.height - (p_rect.pos.y + p_rect.size.y);
		int w = p_rect.size.x;
		int h = p_rect.size.y;

		glScissor(x, y, w, h);

	} else {

		glDisable(GL_SCISSOR_TEST);
	}
}

void RasterizerGLES2::canvas_end_rect() {

	//glPopMatrix();
}

RasterizerGLES2::Texture *RasterizerGLES2::_bind_canvas_texture(const RID &p_texture) {

	if (p_texture == canvas_tex && !rebind_texpixel_size) {
		if (canvas_tex.is_valid()) {
			Texture *texture = texture_owner.get(p_texture);
			return texture;
		}
		return NULL;
	}

	rebind_texpixel_size = false;

	if (p_texture.is_valid()) {

		Texture *texture = texture_owner.get(p_texture);
		if (!texture) {
			canvas_tex = RID();
			glBindTexture(GL_TEXTURE_2D, white_tex);

			return NULL;
		}

		if (texture->render_target)
			texture->render_target->last_pass = frame;

		glBindTexture(GL_TEXTURE_2D, texture->tex_id);
		canvas_tex = p_texture;
		if (uses_texpixel_size) {
			canvas_shader.set_uniform(CanvasShaderGLES2::TEXPIXEL_SIZE, Size2(1.0 / texture->width, 1.0 / texture->height));
		}
		return texture;

	} else {

		glBindTexture(GL_TEXTURE_2D, white_tex);
		canvas_tex = p_texture;
	}

	return NULL;
}

void RasterizerGLES2::canvas_draw_line(const Point2 &p_from, const Point2 &p_to, const Color &p_color, float p_width, bool p_antialiased) {

	_bind_canvas_texture(RID());
	Color c = p_color;
	c.a *= canvas_opacity;
	_set_color_attrib(c);

	Vector3 verts[2] = {
		Vector3(p_from.x, p_from.y, 0),
		Vector3(p_to.x, p_to.y, 0)
	};

#ifdef GLEW_ENABLED
	if (p_antialiased)
		glEnable(GL_LINE_SMOOTH);
#endif
	glLineWidth(p_width);
	_draw_primitive(2, verts, 0, 0, 0);

#ifdef GLEW_ENABLED
	if (p_antialiased)
		glDisable(GL_LINE_SMOOTH);
#endif

	_rinfo.ci_draw_commands++;
}

void RasterizerGLES2::_draw_gui_primitive(int p_points, const Vector2 *p_vertices, const Color *p_colors, const Vector2 *p_uvs) {

	static const GLenum prim[5] = { GL_POINTS, GL_POINTS, GL_LINES, GL_TRIANGLES, GL_TRIANGLE_FAN };

//#define GLES_USE_PRIMITIVE_BUFFER

#ifndef GLES_NO_CLIENT_ARRAYS

	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, false, sizeof(Vector2), p_vertices);

	if (p_colors) {

		glEnableVertexAttribArray(VS::ARRAY_COLOR);
		glVertexAttribPointer(VS::ARRAY_COLOR, 4, GL_FLOAT, false, sizeof(Color), p_colors);
	} else {
		glDisableVertexAttribArray(VS::ARRAY_COLOR);
	}

	if (p_uvs) {

		glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
		glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, false, sizeof(Vector2), p_uvs);
	} else {
		glDisableVertexAttribArray(VS::ARRAY_TEX_UV);
	}

	glDrawArrays(prim[p_points], 0, p_points);

#else

	glBindBuffer(GL_ARRAY_BUFFER, gui_quad_buffer);
	float b[32];
	int ofs = 0;
	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, false, sizeof(float) * 2, ((float *)0) + ofs);
	for (int i = 0; i < p_points; i++) {
		b[ofs++] = p_vertices[i].x;
		b[ofs++] = p_vertices[i].y;
	}

	if (p_colors) {

		glEnableVertexAttribArray(VS::ARRAY_COLOR);
		glVertexAttribPointer(VS::ARRAY_COLOR, 4, GL_FLOAT, false, sizeof(float) * 4, ((float *)0) + ofs);
		for (int i = 0; i < p_points; i++) {
			b[ofs++] = p_colors[i].r;
			b[ofs++] = p_colors[i].g;
			b[ofs++] = p_colors[i].b;
			b[ofs++] = p_colors[i].a;
		}

	} else {
		glDisableVertexAttribArray(VS::ARRAY_COLOR);
	}

	if (p_uvs) {

		glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
		glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, false, sizeof(float) * 2, ((float *)0) + ofs);
		for (int i = 0; i < p_points; i++) {
			b[ofs++] = p_uvs[i].x;
			b[ofs++] = p_uvs[i].y;
		}

	} else {
		glDisableVertexAttribArray(VS::ARRAY_TEX_UV);
	}

	glBufferSubData(GL_ARRAY_BUFFER, 0, ofs * 4, &b[0]);
	glDrawArrays(prim[p_points], 0, p_points);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

#endif
	_rinfo.ci_draw_commands++;
}

void RasterizerGLES2::_draw_gui_primitive2(int p_points, const Vector2 *p_vertices, const Color *p_colors, const Vector2 *p_uvs, const Vector2 *p_uvs2) {

	static const GLenum prim[5] = { GL_POINTS, GL_POINTS, GL_LINES, GL_TRIANGLES, GL_TRIANGLE_FAN };

	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, false, sizeof(Vector2), p_vertices);
	if (p_colors) {

		glEnableVertexAttribArray(VS::ARRAY_COLOR);
		glVertexAttribPointer(VS::ARRAY_COLOR, 4, GL_FLOAT, false, sizeof(Color), p_colors);
	} else {
		glDisableVertexAttribArray(VS::ARRAY_COLOR);
	}

	if (p_uvs) {

		glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
		glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, false, sizeof(Vector2), p_uvs);
	} else {
		glDisableVertexAttribArray(VS::ARRAY_TEX_UV);
	}

	if (p_uvs2) {

		glEnableVertexAttribArray(VS::ARRAY_TEX_UV2);
		glVertexAttribPointer(VS::ARRAY_TEX_UV2, 2, GL_FLOAT, false, sizeof(Vector2), p_uvs2);
	} else {
		glDisableVertexAttribArray(VS::ARRAY_TEX_UV2);
	}

	glDrawArrays(prim[p_points], 0, p_points);
	_rinfo.ci_draw_commands++;
}

void RasterizerGLES2::_draw_textured_quad(const Rect2 &p_rect, const Rect2 &p_src_region, const Size2 &p_tex_size, bool p_h_flip, bool p_v_flip, bool p_transpose) {

	Vector2 texcoords[4] = {
		Vector2(p_src_region.pos.x / p_tex_size.width,
				p_src_region.pos.y / p_tex_size.height),

		Vector2((p_src_region.pos.x + p_src_region.size.width) / p_tex_size.width,
				p_src_region.pos.y / p_tex_size.height),

		Vector2((p_src_region.pos.x + p_src_region.size.width) / p_tex_size.width,
				(p_src_region.pos.y + p_src_region.size.height) / p_tex_size.height),

		Vector2(p_src_region.pos.x / p_tex_size.width,
				(p_src_region.pos.y + p_src_region.size.height) / p_tex_size.height)
	};

	if (p_transpose) {
		SWAP(texcoords[1], texcoords[3]);
	}
	if (p_h_flip) {
		SWAP(texcoords[0], texcoords[1]);
		SWAP(texcoords[2], texcoords[3]);
	}
	if (p_v_flip) {
		SWAP(texcoords[1], texcoords[2]);
		SWAP(texcoords[0], texcoords[3]);
	}

	Vector2 coords[4] = {
		Vector2(p_rect.pos.x, p_rect.pos.y),
		Vector2(p_rect.pos.x + p_rect.size.width, p_rect.pos.y),
		Vector2(p_rect.pos.x + p_rect.size.width, p_rect.pos.y + p_rect.size.height),
		Vector2(p_rect.pos.x, p_rect.pos.y + p_rect.size.height)
	};

	_draw_gui_primitive(4, coords, 0, texcoords);
	_rinfo.ci_draw_commands++;
}

void RasterizerGLES2::_draw_quad(const Rect2 &p_rect) {

	Vector2 coords[4] = {
		Vector2(p_rect.pos.x, p_rect.pos.y),
		Vector2(p_rect.pos.x + p_rect.size.width, p_rect.pos.y),
		Vector2(p_rect.pos.x + p_rect.size.width, p_rect.pos.y + p_rect.size.height),
		Vector2(p_rect.pos.x, p_rect.pos.y + p_rect.size.height)
	};

	_draw_gui_primitive(4, coords, 0, 0);
	_rinfo.ci_draw_commands++;
}

void RasterizerGLES2::canvas_draw_rect(const Rect2 &p_rect, int p_flags, const Rect2 &p_source, RID p_texture, const Color &p_modulate) {

	Color m = p_modulate;
	m.a *= canvas_opacity;
	_set_color_attrib(m);
	Texture *texture = _bind_canvas_texture(p_texture);

	if (texture) {

		bool untile = false;

		if (p_flags & CANVAS_RECT_TILE && !(texture->flags & VS::TEXTURE_FLAG_REPEAT)) {
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			untile = true;
		}

		if (!(p_flags & CANVAS_RECT_REGION)) {

			Rect2 region = Rect2(0, 0, texture->width, texture->height);
			_draw_textured_quad(p_rect, region, region.size, p_flags & CANVAS_RECT_FLIP_H, p_flags & CANVAS_RECT_FLIP_V, p_flags & CANVAS_RECT_TRANSPOSE);

		} else {

			_draw_textured_quad(p_rect, p_source, Size2(texture->width, texture->height), p_flags & CANVAS_RECT_FLIP_H, p_flags & CANVAS_RECT_FLIP_V, p_flags & CANVAS_RECT_TRANSPOSE);
		}

		if (untile) {
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		}

	} else {

		//glDisable(GL_TEXTURE_2D);
		_draw_quad(p_rect);
		//print_line("rect: "+p_rect);
	}

	_rinfo.ci_draw_commands++;
}

void RasterizerGLES2::canvas_draw_style_box(const Rect2 &p_rect, const Rect2 &p_src_region, RID p_texture, const float *p_margin, bool p_draw_center, const Color &p_modulate) {

	Color m = p_modulate;
	m.a *= canvas_opacity;
	_set_color_attrib(m);

	Texture *texture = _bind_canvas_texture(p_texture);
	ERR_FAIL_COND(!texture);

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

	_rinfo.ci_draw_commands++;
}

void RasterizerGLES2::canvas_draw_primitive(const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, RID p_texture, float p_width) {

	ERR_FAIL_COND(p_points.size() < 1);
	_set_color_attrib(Color(1, 1, 1, canvas_opacity));
	_bind_canvas_texture(p_texture);
	_draw_gui_primitive(p_points.size(), p_points.ptr(), p_colors.ptr(), p_uvs.ptr());

	_rinfo.ci_draw_commands++;
}

void RasterizerGLES2::canvas_draw_polygon(int p_vertex_count, const int *p_indices, const Vector2 *p_vertices, const Vector2 *p_uvs, const Color *p_colors, const RID &p_texture, bool p_singlecolor) {

	bool do_colors = false;
	Color m;
	if (p_singlecolor) {
		m = *p_colors;
		m.a *= canvas_opacity;
		_set_color_attrib(m);
	} else if (!p_colors) {
		m = Color(1, 1, 1, canvas_opacity);
		_set_color_attrib(m);
	} else
		do_colors = true;

	Texture *texture = _bind_canvas_texture(p_texture);

#ifndef GLES_NO_CLIENT_ARRAYS
	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, false, sizeof(Vector2), p_vertices);
	if (do_colors) {

		glEnableVertexAttribArray(VS::ARRAY_COLOR);
		glVertexAttribPointer(VS::ARRAY_COLOR, 4, GL_FLOAT, false, sizeof(Color), p_colors);
	} else {
		glDisableVertexAttribArray(VS::ARRAY_COLOR);
	}

	if (texture && p_uvs) {

		glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
		glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, false, sizeof(Vector2), p_uvs);
	} else {
		glDisableVertexAttribArray(VS::ARRAY_TEX_UV);
	}

	if (p_indices) {
#ifdef GLEW_ENABLED
		glDrawElements(GL_TRIANGLES, p_vertex_count, GL_UNSIGNED_INT, p_indices);
#else
		static const int _max_draw_poly_indices = 16 * 1024; // change this size if needed!!!
		ERR_FAIL_COND(p_vertex_count > _max_draw_poly_indices);
		static uint16_t _draw_poly_indices[_max_draw_poly_indices];
		for (int i = 0; i < p_vertex_count; i++) {
			_draw_poly_indices[i] = p_indices[i];
		};
		glDrawElements(GL_TRIANGLES, p_vertex_count, GL_UNSIGNED_SHORT, _draw_poly_indices);
#endif
	} else {
		glDrawArrays(GL_TRIANGLES, 0, p_vertex_count);
	}

#else //WebGL specific impl.
	glBindBuffer(GL_ARRAY_BUFFER, gui_quad_buffer);
	float *b = GlobalVertexBuffer;
	int ofs = 0;
	if (p_vertex_count > MAX_POLYGON_VERTICES) {
		print_line("Too many vertices to render");
		return;
	}
	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	glVertexAttribPointer(VS::ARRAY_VERTEX, 2, GL_FLOAT, false, sizeof(float) * 2, ((float *)0) + ofs);
	for (int i = 0; i < p_vertex_count; i++) {
		b[ofs++] = p_vertices[i].x;
		b[ofs++] = p_vertices[i].y;
	}

	if (p_colors && do_colors) {

		glEnableVertexAttribArray(VS::ARRAY_COLOR);
		glVertexAttribPointer(VS::ARRAY_COLOR, 4, GL_FLOAT, false, sizeof(float) * 4, ((float *)0) + ofs);
		for (int i = 0; i < p_vertex_count; i++) {
			b[ofs++] = p_colors[i].r;
			b[ofs++] = p_colors[i].g;
			b[ofs++] = p_colors[i].b;
			b[ofs++] = p_colors[i].a;
		}

	} else {
		glDisableVertexAttribArray(VS::ARRAY_COLOR);
	}

	if (p_uvs) {

		glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
		glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, false, sizeof(float) * 2, ((float *)0) + ofs);
		for (int i = 0; i < p_vertex_count; i++) {
			b[ofs++] = p_uvs[i].x;
			b[ofs++] = p_uvs[i].y;
		}

	} else {
		glDisableVertexAttribArray(VS::ARRAY_TEX_UV);
	}

	glBufferSubData(GL_ARRAY_BUFFER, 0, ofs * 4, &b[0]);

	//bind the indices buffer.
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices_buffer);

	static const int _max_draw_poly_indices = 16 * 1024; // change this size if needed!!!
	ERR_FAIL_COND(p_vertex_count > _max_draw_poly_indices);
	static uint16_t _draw_poly_indices[_max_draw_poly_indices];
	for (int i = 0; i < p_vertex_count; i++) {
		_draw_poly_indices[i] = p_indices[i];
		//OS::get_singleton()->print("ind: %d ", p_indices[i]);
	};

	//copy the data to GPU.
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, p_vertex_count * sizeof(uint16_t), &_draw_poly_indices[0]);

	//draw the triangles.
	glDrawElements(GL_TRIANGLES, p_vertex_count, GL_UNSIGNED_SHORT, 0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
#endif

	_rinfo.ci_draw_commands++;
};

void RasterizerGLES2::canvas_set_transform(const Transform2D &p_transform) {

	canvas_shader.set_uniform(CanvasShaderGLES2::EXTRA_MATRIX, p_transform);

	//canvas_transform = Variant(p_transform);
}

RID RasterizerGLES2::canvas_light_occluder_create() {

	CanvasOccluder *co = memnew(CanvasOccluder);
	co->index_id = 0;
	co->vertex_id = 0;
	co->len = 0;

	return canvas_occluder_owner.make_rid(co);
}

void RasterizerGLES2::canvas_light_occluder_set_polylines(RID p_occluder, const PoolVector<Vector2> &p_lines) {

	CanvasOccluder *co = canvas_occluder_owner.get(p_occluder);
	ERR_FAIL_COND(!co);

	co->lines = p_lines;

	if (p_lines.size() != co->len) {

		if (co->index_id)
			glDeleteBuffers(1, &co->index_id);
		if (co->vertex_id)
			glDeleteBuffers(1, &co->vertex_id);

		co->index_id = 0;
		co->vertex_id = 0;
		co->len = 0;
	}

	if (p_lines.size()) {

		PoolVector<float> geometry;
		PoolVector<uint16_t> indices;
		int lc = p_lines.size();

		geometry.resize(lc * 6);
		indices.resize(lc * 3);

		PoolVector<float>::Write vw = geometry.write();
		PoolVector<uint16_t>::Write iw = indices.write();

		PoolVector<Vector2>::Read lr = p_lines.read();

		const int POLY_HEIGHT = 16384;

		for (int i = 0; i < lc / 2; i++) {

			vw[i * 12 + 0] = lr[i * 2 + 0].x;
			vw[i * 12 + 1] = lr[i * 2 + 0].y;
			vw[i * 12 + 2] = POLY_HEIGHT;

			vw[i * 12 + 3] = lr[i * 2 + 1].x;
			vw[i * 12 + 4] = lr[i * 2 + 1].y;
			vw[i * 12 + 5] = POLY_HEIGHT;

			vw[i * 12 + 6] = lr[i * 2 + 1].x;
			vw[i * 12 + 7] = lr[i * 2 + 1].y;
			vw[i * 12 + 8] = -POLY_HEIGHT;

			vw[i * 12 + 9] = lr[i * 2 + 0].x;
			vw[i * 12 + 10] = lr[i * 2 + 0].y;
			vw[i * 12 + 11] = -POLY_HEIGHT;

			iw[i * 6 + 0] = i * 4 + 0;
			iw[i * 6 + 1] = i * 4 + 1;
			iw[i * 6 + 2] = i * 4 + 2;

			iw[i * 6 + 3] = i * 4 + 2;
			iw[i * 6 + 4] = i * 4 + 3;
			iw[i * 6 + 5] = i * 4 + 0;
		}

		//if same buffer len is being set, just use BufferSubData to avoid a pipeline flush

		if (!co->vertex_id) {
			glGenBuffers(1, &co->vertex_id);
			glBindBuffer(GL_ARRAY_BUFFER, co->vertex_id);
			glBufferData(GL_ARRAY_BUFFER, lc * 6 * sizeof(real_t), vw.ptr(), GL_STATIC_DRAW);
		} else {

			glBindBuffer(GL_ARRAY_BUFFER, co->vertex_id);
			glBufferSubData(GL_ARRAY_BUFFER, 0, lc * 6 * sizeof(real_t), vw.ptr());
		}

		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind

		if (!co->index_id) {

			glGenBuffers(1, &co->index_id);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, co->index_id);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, lc * 3 * sizeof(uint16_t), iw.ptr(), GL_STATIC_DRAW);
		} else {

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, co->index_id);
			glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, lc * 3 * sizeof(uint16_t), iw.ptr());
		}

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); //unbind

		co->len = lc;
	}
}

RID RasterizerGLES2::canvas_light_shadow_buffer_create(int p_width) {

	CanvasLightShadow *cls = memnew(CanvasLightShadow);
	if (p_width > max_texture_size)
		p_width = max_texture_size;

	cls->size = p_width;
	glActiveTexture(GL_TEXTURE0);

	glGenFramebuffers(1, &cls->fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, cls->fbo);

	// Create a render buffer
	glGenRenderbuffers(1, &cls->rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, cls->rbo);

	// Create a texture for storing the depth
	glGenTextures(1, &cls->depth);
	glBindTexture(GL_TEXTURE_2D, cls->depth);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Remove artifact on the edges of the shadowmap
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	cls->height = 16;

	//print_line("ERROR? "+itos(glGetError()));
	if (read_depth_supported) {

		// We'll use a depth texture to store the depths in the shadow map
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, cls->size, cls->height, 0,
				GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);

#ifdef GLEW_ENABLED
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#endif

		// Attach the depth texture to FBO depth attachment point
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
				GL_TEXTURE_2D, cls->depth, 0);

#ifdef GLEW_ENABLED
		glDrawBuffer(GL_NONE);
#endif

	} else {
		// We'll use a RGBA texture into which we pack the depth info
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cls->size, cls->height, 0,
				GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		// Attach the RGBA texture to FBO color attachment point
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
				GL_TEXTURE_2D, cls->depth, 0);
		cls->rgba = cls->depth;

		// Allocate 16-bit depth buffer
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, cls->size, cls->height);

		// Attach the render buffer as depth buffer - will be ignored
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
				GL_RENDERBUFFER, cls->rbo);
	}

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
//printf("errnum: %x\n",status);
#ifdef GLEW_ENABLED
	if (read_depth_supported) {
		//glDrawBuffer(GL_BACK);
	}
#endif
	glBindFramebuffer(GL_FRAMEBUFFER, base_framebuffer);
	DEBUG_TEST_ERROR("2D Shadow Buffer Init");
	ERR_FAIL_COND_V(status != GL_FRAMEBUFFER_COMPLETE, RID());

#ifdef GLEW_ENABLED
	if (read_depth_supported) {
		//glDrawBuffer(GL_BACK);
	}
#endif

	return canvas_light_shadow_owner.make_rid(cls);
}

void RasterizerGLES2::canvas_light_shadow_buffer_update(RID p_buffer, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, CanvasLightOccluderInstance *p_occluders, CameraMatrix *p_xform_cache) {

	CanvasLightShadow *cls = canvas_light_shadow_owner.get(p_buffer);
	ERR_FAIL_COND(!cls);

	glDisable(GL_BLEND);
	glDisable(GL_SCISSOR_TEST);
	glDisable(GL_DITHER);
	glDisable(GL_CULL_FACE);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(true);

	glBindFramebuffer(GL_FRAMEBUFFER, cls->fbo);

	if (!use_rgba_shadowmaps)
		glColorMask(0, 0, 0, 0);

	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	canvas_shadow_shader.bind();

	glViewport(0, 0, cls->size, cls->height);
	_glClearDepth(1.0f);
	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	VS::CanvasOccluderPolygonCullMode cull = VS::CANVAS_OCCLUDER_POLYGON_CULL_DISABLED;

	for (int i = 0; i < 4; i++) {

		//make sure it remains orthogonal, makes easy to read angle later

		Transform light;
		light.origin[0] = p_light_xform[2][0];
		light.origin[1] = p_light_xform[2][1];
		light.basis[0][0] = p_light_xform[0][0];
		light.basis[0][1] = p_light_xform[1][0];
		light.basis[1][0] = p_light_xform[0][1];
		light.basis[1][1] = p_light_xform[1][1];

		//light.basis.scale(Vector3(to_light.elements[0].length(),to_light.elements[1].length(),1));

		/ //p_near=1;
				CameraMatrix projection;
		{
			real_t fov = 90;
			real_t near = p_near;
			real_t far = p_far;
			real_t aspect = 1.0;

			real_t ymax = near * Math::tan(Math::deg2rad(fov * 0.5));
			real_t ymin = -ymax;
			real_t xmin = ymin * aspect;
			real_t xmax = ymax * aspect;

			projection.set_frustum(xmin, xmax, ymin, ymax, near, far);
		}

		Vector3 cam_target = Matrix3(Vector3(0, 0, Math_PI * 2 * (i / 4.0))).xform(Vector3(0, 1, 0));
		projection = projection * CameraMatrix(Transform().looking_at(cam_target, Vector3(0, 0, -1)).affine_inverse());

		canvas_shadow_shader.set_uniform(CanvasShadowShaderGLES2::PROJECTION_MATRIX, projection);
		canvas_shadow_shader.set_uniform(CanvasShadowShaderGLES2::LIGHT_MATRIX, light);

		if (i == 0)
			*p_xform_cache = projection;

		glViewport(0, (cls->height / 4) * i, cls->size, cls->height / 4);

		CanvasLightOccluderInstance *instance = p_occluders;

		while (instance) {

			CanvasOccluder *cc = canvas_occluder_owner.get(instance->polygon_buffer);
			if (!cc || cc->len == 0 || !(p_light_mask & instance->light_mask)) {

				instance = instance->next;
				continue;
			}

			canvas_shadow_shader.set_uniform(CanvasShadowShaderGLES2::WORLD_MATRIX, instance->xform_cache);
			if (cull != instance->cull_cache) {

				cull = instance->cull_cache;
				switch (cull) {
					case VS::CANVAS_OCCLUDER_POLYGON_CULL_DISABLED: {

						glDisable(GL_CULL_FACE);

					} break;
					case VS::CANVAS_OCCLUDER_POLYGON_CULL_CLOCKWISE: {

						glEnable(GL_CULL_FACE);
						glCullFace(GL_FRONT);
					} break;
					case VS::CANVAS_OCCLUDER_POLYGON_CULL_COUNTER_CLOCKWISE: {

						glEnable(GL_CULL_FACE);
						glCullFace(GL_BACK);

					} break;
				}
			}
			/*
			if (i==0) {
				for(int i=0;i<cc->lines.size();i++) {
					Vector2 p = instance->xform_cache.xform(cc->lines.get(i));
					Plane pp(Vector3(p.x,p.y,0),1);
					pp.normal = light.xform(pp.normal);
					pp = projection.xform4(pp);
					print_line(itos(i)+": "+pp.normal/pp.d);
					//pp=light_mat.xform4(pp);
					//print_line(itos(i)+": "+pp.normal/pp.d);
				}
			}
*/
			glBindBuffer(GL_ARRAY_BUFFER, cc->vertex_id);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cc->index_id);
			glVertexAttribPointer(VS::ARRAY_VERTEX, 3, GL_FLOAT, false, 0, 0);
			glDrawElements(GL_TRIANGLES, cc->len * 3, GL_UNSIGNED_SHORT, 0);

			instance = instance->next;
		}
	}

	glDisableVertexAttribArray(VS::ARRAY_VERTEX);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	if (shadow_filter == SHADOW_FILTER_ESM) {
//blur the buffer
#if 0
	//this is ignord, it did not make any difference..
		if (read_depth_supported) {
			glDepthFunc(GL_ALWAYS);
		} else {
			glDisable(GL_DEPTH_TEST);
			glDepthMask(false);
		}
		glDisable(GL_CULL_FACE);
		glViewport(0, 0, cls->size,cls->height);

		int passes=1;
		CanvasLightShadow *blur = canvas_light_shadow_owner.get(canvas_shadow_blur);

		copy_shader.set_conditional(CopyShaderGLES2::SHADOW_BLUR_H_PASS,true);
		copy_shader.bind();
		copy_shader.set_uniform(CopyShaderGLES2::PIXEL_SCALE,1);
		copy_shader.set_uniform(CopyShaderGLES2::BLUR_MAGNITUDE,1);
		glUniform1i(copy_shader.get_uniform_location(CopyShaderGLES2::SOURCE),0);

		for(int i=0;i<passes;i++) {

			glBindFramebuffer(GL_FRAMEBUFFER, blur->fbo);
			glActiveTexture(GL_TEXTURE0);

			if (read_depth_supported)
				glBindTexture(GL_TEXTURE_2D,cls->depth);
			else
				glBindTexture(GL_TEXTURE_2D,cls->rgba);


			{
				Vector2 src_sb_uv[4]={
					Vector2( 0, 1),
					Vector2( 1, 1),
					Vector2( 1, 0),
					Vector2( 0, 0)
				};
				static const Vector2 dst_pos[4]={
					Vector2(-1, 1),
					Vector2( 1, 1),
					Vector2( 1,-1),
					Vector2(-1,-1)
				};



				copy_shader.set_uniform(CopyShaderGLES2::PIXEL_SIZE,Vector2(1.0,1.0)/cls->size);
				_draw_gui_primitive(4,dst_pos,NULL,src_sb_uv);
			}

			glActiveTexture(GL_TEXTURE0);
			if (read_depth_supported)
				glBindTexture(GL_TEXTURE_2D,blur->depth);
			else
				glBindTexture(GL_TEXTURE_2D,blur->rgba);

			glBindFramebuffer(GL_FRAMEBUFFER, cls->fbo);

			{
				float hlimit = float(cls->size) / blur->size;
				//hlimit*=2.0;
				Vector2 src_sb_uv[4]={
					Vector2( 0, 1),
					Vector2( hlimit, 1),
					Vector2( hlimit, 0),
					Vector2( 0, 0)
				};
				static const Vector2 dst_pos[4]={
					Vector2(-1, 1),
					Vector2( 1, 1),
					Vector2( 1,-1),
					Vector2(-1,-1)
				};


				copy_shader.set_uniform(CopyShaderGLES2::PIXEL_SIZE,Vector2(1.0,1.0)/blur->size);
				_draw_gui_primitive(4,dst_pos,NULL,src_sb_uv);
			}

		}
		copy_shader.set_conditional(CopyShaderGLES2::SHADOW_BLUR_H_PASS,false);
		glDepthFunc(GL_LEQUAL);
#endif
	}

	glBindFramebuffer(GL_FRAMEBUFFER, current_rt ? current_rt->fbo : base_framebuffer);
	glColorMask(1, 1, 1, 1);
}

void RasterizerGLES2::canvas_debug_viewport_shadows(CanvasLight *p_lights_with_shadow) {

	CanvasLight *light = p_lights_with_shadow;

	canvas_begin(); //reset

	int h = 10;
	int w = viewport.width;
	int ofs = h;

	//print_line(" debug lights ");
	while (light) {

		//print_line("debug light");
		if (light->shadow_buffer.is_valid()) {

			//print_line("sb is valid");
			CanvasLightShadow *sb = canvas_light_shadow_owner.get(light->shadow_buffer);
			if (sb) {
				glActiveTexture(GL_TEXTURE0);
				if (read_depth_supported)
					glBindTexture(GL_TEXTURE_2D, sb->depth);
				else
					glBindTexture(GL_TEXTURE_2D, sb->rgba);
				_draw_textured_quad(Rect2(h, ofs, w - h * 2, h), Rect2(0, 0, sb->size, 10), Size2(sb->size, 10), false, false);
				ofs += h * 2;
			}
		}

		light = light->shadows_next_ptr;
	}
}

void RasterizerGLES2::_canvas_normal_set_flip(const Vector2 &p_flip) {

	if (p_flip == normal_flip)
		return;
	normal_flip = p_flip;
	canvas_shader.set_uniform(CanvasShaderGLES2::NORMAL_FLIP, normal_flip);
}

template <bool use_normalmap>
void RasterizerGLES2::_canvas_item_render_commands(CanvasItem *p_item, CanvasItem *current_clip, bool &reclip) {

	int cc = p_item->commands.size();
	CanvasItem::Command **commands = p_item->commands.ptr();

	for (int i = 0; i < cc; i++) {

		CanvasItem::Command *c = commands[i];

		switch (c->type) {
			case CanvasItem::Command::TYPE_LINE: {

				CanvasItem::CommandLine *line = static_cast<CanvasItem::CommandLine *>(c);
				canvas_draw_line(line->from, line->to, line->color, line->width, line->antialiased);
			} break;
			case CanvasItem::Command::TYPE_RECT: {

				CanvasItem::CommandRect *rect = static_cast<CanvasItem::CommandRect *>(c);
//canvas_draw_rect(rect->rect,rect->region,rect->source,rect->flags&CanvasItem::CommandRect::FLAG_TILE,rect->flags&CanvasItem::CommandRect::FLAG_FLIP_H,rect->flags&CanvasItem::CommandRect::FLAG_FLIP_V,rect->texture,rect->modulate);
#if 0
				int flags=0;

				if (rect->flags&CanvasItem::CommandRect::FLAG_REGION) {
					flags|=Rasterizer::CANVAS_RECT_REGION;
				}
				if (rect->flags&CanvasItem::CommandRect::FLAG_TILE) {
					flags|=Rasterizer::CANVAS_RECT_TILE;
				}
				if (rect->flags&CanvasItem::CommandRect::FLAG_FLIP_H) {

					flags|=Rasterizer::CANVAS_RECT_FLIP_H;
				}
				if (rect->flags&CanvasItem::CommandRect::FLAG_FLIP_V) {

					flags|=Rasterizer::CANVAS_RECT_FLIP_V;
				}
#else

				int flags = rect->flags;
#endif
				if (use_normalmap)
					_canvas_normal_set_flip(Vector2((flags & CANVAS_RECT_FLIP_H) ? -1 : 1, (flags & CANVAS_RECT_FLIP_V) ? -1 : 1));
				canvas_draw_rect(rect->rect, flags, rect->source, rect->texture, rect->modulate);

			} break;
			case CanvasItem::Command::TYPE_STYLE: {

				CanvasItem::CommandStyle *style = static_cast<CanvasItem::CommandStyle *>(c);
				if (use_normalmap)
					_canvas_normal_set_flip(Vector2(1, 1));
				canvas_draw_style_box(style->rect, style->source, style->texture, style->margin, style->draw_center, style->color);

			} break;
			case CanvasItem::Command::TYPE_PRIMITIVE: {

				if (use_normalmap)
					_canvas_normal_set_flip(Vector2(1, 1));
				CanvasItem::CommandPrimitive *primitive = static_cast<CanvasItem::CommandPrimitive *>(c);
				canvas_draw_primitive(primitive->points, primitive->colors, primitive->uvs, primitive->texture, primitive->width);
			} break;
			case CanvasItem::Command::TYPE_POLYGON: {

				if (use_normalmap)
					_canvas_normal_set_flip(Vector2(1, 1));
				CanvasItem::CommandPolygon *polygon = static_cast<CanvasItem::CommandPolygon *>(c);
				canvas_draw_polygon(polygon->count, polygon->indices.ptr(), polygon->points.ptr(), polygon->uvs.ptr(), polygon->colors.ptr(), polygon->texture, polygon->colors.size() == 1);

			} break;

			case CanvasItem::Command::TYPE_POLYGON_PTR: {

				if (use_normalmap)
					_canvas_normal_set_flip(Vector2(1, 1));
				CanvasItem::CommandPolygonPtr *polygon = static_cast<CanvasItem::CommandPolygonPtr *>(c);
				canvas_draw_polygon(polygon->count, polygon->indices, polygon->points, polygon->uvs, polygon->colors, polygon->texture, false);
			} break;
			case CanvasItem::Command::TYPE_CIRCLE: {

				CanvasItem::CommandCircle *circle = static_cast<CanvasItem::CommandCircle *>(c);
				static const int numpoints = 32;
				Vector2 points[numpoints + 1];
				points[numpoints] = circle->pos;
				int indices[numpoints * 3];

				for (int i = 0; i < numpoints; i++) {

					points[i] = circle->pos + Vector2(Math::sin(i * Math_PI * 2.0 / numpoints), Math::cos(i * Math_PI * 2.0 / numpoints)) * circle->radius;
					indices[i * 3 + 0] = i;
					indices[i * 3 + 1] = (i + 1) % numpoints;
					indices[i * 3 + 2] = numpoints;
				}
				canvas_draw_polygon(numpoints * 3, indices, points, NULL, &circle->color, RID(), true);
				//canvas_draw_circle(circle->indices.size(),circle->indices.ptr(),circle->points.ptr(),circle->uvs.ptr(),circle->colors.ptr(),circle->texture,circle->colors.size()==1);
			} break;
			case CanvasItem::Command::TYPE_TRANSFORM: {

				CanvasItem::CommandTransform *transform = static_cast<CanvasItem::CommandTransform *>(c);
				canvas_set_transform(transform->xform);
			} break;
			case CanvasItem::Command::TYPE_BLEND_MODE: {

				CanvasItem::CommandBlendMode *bm = static_cast<CanvasItem::CommandBlendMode *>(c);
				canvas_set_blend_mode(bm->blend_mode);

			} break;
			case CanvasItem::Command::TYPE_CLIP_IGNORE: {

				CanvasItem::CommandClipIgnore *ci = static_cast<CanvasItem::CommandClipIgnore *>(c);
				if (current_clip) {

					if (ci->ignore != reclip) {
						if (ci->ignore) {

							glDisable(GL_SCISSOR_TEST);
							reclip = true;
						} else {

							glEnable(GL_SCISSOR_TEST);
							//glScissor(viewport.x+current_clip->final_clip_rect.pos.x,viewport.y+ (viewport.height-(current_clip->final_clip_rect.pos.y+current_clip->final_clip_rect.size.height)),
							//current_clip->final_clip_rect.size.width,current_clip->final_clip_rect.size.height);

							int x;
							int y;
							int w;
							int h;

							if (current_rt) {
								x = current_clip->final_clip_rect.pos.x;
								y = current_clip->final_clip_rect.pos.y;
								w = current_clip->final_clip_rect.size.x;
								h = current_clip->final_clip_rect.size.y;
							} else {
								x = current_clip->final_clip_rect.pos.x;
								y = window_size.height - (current_clip->final_clip_rect.pos.y + current_clip->final_clip_rect.size.y);
								w = current_clip->final_clip_rect.size.x;
								h = current_clip->final_clip_rect.size.y;
							}

							glScissor(x, y, w, h);

							reclip = false;
						}
					}
				}

			} break;
		}
	}
}

void RasterizerGLES2::_canvas_item_setup_shader_params(ShaderMaterial *material, Shader *shader) {

	if (canvas_shader.bind())
		rebind_texpixel_size = true;

	if (material->shader_version != shader->version) {
		//todo optimize uniforms
		material->shader_version = shader->version;
	}

	if (shader->has_texscreen && framebuffer.active) {

		int x = viewport.x;
		int y = window_size.height - (viewport.height + viewport.y);

		canvas_shader.set_uniform(CanvasShaderGLES2::TEXSCREEN_SCREEN_MULT, Vector2(float(viewport.width) / framebuffer.width, float(viewport.height) / framebuffer.height));
		canvas_shader.set_uniform(CanvasShaderGLES2::TEXSCREEN_SCREEN_CLAMP, Color(float(x) / framebuffer.width, float(y) / framebuffer.height, float(x + viewport.width) / framebuffer.width, float(y + viewport.height) / framebuffer.height));
		canvas_shader.set_uniform(CanvasShaderGLES2::TEXSCREEN_TEX, max_texture_units - 1);
		glActiveTexture(GL_TEXTURE0 + max_texture_units - 1);
		glBindTexture(GL_TEXTURE_2D, framebuffer.sample_color);
		if (framebuffer.scale == 1 && !canvas_texscreen_used) {
#ifdef GLEW_ENABLED
			if (current_rt) {
				glReadBuffer(GL_COLOR_ATTACHMENT0);
			} else {
				glReadBuffer(GL_BACK);
			}
#endif
			if (current_rt) {
				glCopyTexSubImage2D(GL_TEXTURE_2D, 0, viewport.x, viewport.y, viewport.x, viewport.y, viewport.width, viewport.height);
				canvas_shader.set_uniform(CanvasShaderGLES2::TEXSCREEN_SCREEN_CLAMP, Color(float(x) / framebuffer.width, float(viewport.y) / framebuffer.height, float(x + viewport.width) / framebuffer.width, float(y + viewport.height) / framebuffer.height));
				//window_size.height-(viewport.height+viewport.y)
			} else {
				glCopyTexSubImage2D(GL_TEXTURE_2D, 0, x, y, x, y, viewport.width, viewport.height);
			}

			canvas_texscreen_used = true;
		}

		glActiveTexture(GL_TEXTURE0);
	}

	if (shader->has_screen_uv) {
		canvas_shader.set_uniform(CanvasShaderGLES2::SCREEN_UV_MULT, Vector2(1.0 / viewport.width, 1.0 / viewport.height));
	}

	uses_texpixel_size = shader->uses_texpixel_size;
}

void RasterizerGLES2::_canvas_item_setup_shader_uniforms(ShaderMaterial *material, Shader *shader) {

	//this can be optimized..
	int tex_id = 1;
	int idx = 0;
	for (Map<StringName, ShaderLanguage::Uniform>::Element *E = shader->uniforms.front(); E; E = E->next()) {

		Map<StringName, Variant>::Element *F = material->shader_param.find(E->key());

		if ((E->get().type == ShaderLanguage::TYPE_TEXTURE || E->get().type == ShaderLanguage::TYPE_CUBEMAP)) {

			RID rid;
			if (F) {
				rid = F->get();
			}

			if (!rid.is_valid()) {

				Map<StringName, RID>::Element *DT = shader->default_textures.find(E->key());
				if (DT) {
					rid = DT->get();
				}
			}

			if (rid.is_valid()) {

				int loc = canvas_shader.get_custom_uniform_location(idx); //should be automatic..

				glActiveTexture(GL_TEXTURE0 + tex_id);
				Texture *t = texture_owner.get(rid);
				if (!t)
					glBindTexture(GL_TEXTURE_2D, white_tex);
				else
					glBindTexture(t->target, t->tex_id);

				glUniform1i(loc, tex_id);
				tex_id++;
			}
		} else {
			Variant &v = F ? F->get() : E->get().default_value;
			canvas_shader.set_custom_uniform(idx, v);
		}

		idx++;
	}

	if (tex_id > 1) {
		glActiveTexture(GL_TEXTURE0);
	}

	if (shader->uses_time) {
		canvas_shader.set_uniform(CanvasShaderGLES2::TIME, Math::fmod(last_time, shader_time_rollback));
		draw_next_frame = true;
	}
	//if uses TIME - draw_next_frame=true
}

void RasterizerGLES2::canvas_render_items(CanvasItem *p_item_list, int p_z, const Color &p_modulate, CanvasLight *p_light) {

	CanvasItem *current_clip = NULL;
	Shader *shader_cache = NULL;

	bool rebind_shader = true;

	canvas_opacity = 1.0;
	canvas_use_modulate = p_modulate != Color(1, 1, 1, 1);
	canvas_modulate = p_modulate;
	canvas_shader.set_conditional(CanvasShaderGLES2::USE_MODULATE, canvas_use_modulate);
	canvas_shader.set_conditional(CanvasShaderGLES2::USE_DISTANCE_FIELD, false);

	bool reset_modulate = false;
	bool prev_distance_field = false;

	while (p_item_list) {

		CanvasItem *ci = p_item_list;

		if (ci->vp_render) {
			if (draw_viewport_func) {
				draw_viewport_func(ci->vp_render->owner, ci->vp_render->udata, ci->vp_render->rect);
			}
			memdelete(ci->vp_render);
			ci->vp_render = NULL;
			canvas_last_material = NULL;
			canvas_use_modulate = p_modulate != Color(1, 1, 1, 1);
			canvas_modulate = p_modulate;
			canvas_shader.set_conditional(CanvasShaderGLES2::USE_MODULATE, canvas_use_modulate);
			canvas_shader.set_conditional(CanvasShaderGLES2::USE_DISTANCE_FIELD, false);
			prev_distance_field = false;
			rebind_shader = true;
			reset_modulate = true;
		}

		if (prev_distance_field != ci->distance_field) {

			canvas_shader.set_conditional(CanvasShaderGLES2::USE_DISTANCE_FIELD, ci->distance_field);
			prev_distance_field = ci->distance_field;
			rebind_shader = true;
		}

		if (current_clip != ci->final_clip_owner) {

			current_clip = ci->final_clip_owner;

			//setup clip
			if (current_clip) {

				glEnable(GL_SCISSOR_TEST);
				//glScissor(viewport.x+current_clip->final_clip_rect.pos.x,viewport.y+ (viewport.height-(current_clip->final_clip_rect.pos.y+current_clip->final_clip_rect.size.height)),
				//current_clip->final_clip_rect.size.width,current_clip->final_clip_rect.size.height);

				/*				int x = viewport.x+current_clip->final_clip_rect.pos.x;
				int y = window_size.height-(viewport.y+current_clip->final_clip_rect.pos.y+current_clip->final_clip_rect.size.y);
				int w = current_clip->final_clip_rect.size.x;
				int h = current_clip->final_clip_rect.size.y;
*/
				int x;
				int y;
				int w;
				int h;

				if (current_rt) {
					x = current_clip->final_clip_rect.pos.x;
					y = current_clip->final_clip_rect.pos.y;
					w = current_clip->final_clip_rect.size.x;
					h = current_clip->final_clip_rect.size.y;
				} else {
					x = current_clip->final_clip_rect.pos.x;
					y = window_size.height - (current_clip->final_clip_rect.pos.y + current_clip->final_clip_rect.size.y);
					w = current_clip->final_clip_rect.size.x;
					h = current_clip->final_clip_rect.size.y;
				}

				glScissor(x, y, w, h);

			} else {

				glDisable(GL_SCISSOR_TEST);
			}
		}

		if (ci->copy_back_buffer && framebuffer.active && framebuffer.scale == 1) {

			Rect2 rect;
			int x, y;

			if (ci->copy_back_buffer->full) {

				x = viewport.x;
				y = window_size.height - (viewport.height + viewport.y);
			} else {
				x = viewport.x + ci->copy_back_buffer->screen_rect.pos.x;
				y = window_size.height - (viewport.y + ci->copy_back_buffer->screen_rect.pos.y + ci->copy_back_buffer->screen_rect.size.y);
			}
			glActiveTexture(GL_TEXTURE0 + max_texture_units - 1);
			glBindTexture(GL_TEXTURE_2D, framebuffer.sample_color);

#ifdef GLEW_ENABLED
			if (current_rt) {
				glReadBuffer(GL_COLOR_ATTACHMENT0);
			} else {
				glReadBuffer(GL_BACK);
			}
#endif
			if (current_rt) {
				glCopyTexSubImage2D(GL_TEXTURE_2D, 0, viewport.x, viewport.y, viewport.x, viewport.y, viewport.width, viewport.height);
				//window_size.height-(viewport.height+viewport.y)
			} else {
				glCopyTexSubImage2D(GL_TEXTURE_2D, 0, x, y, x, y, viewport.width, viewport.height);
			}

			canvas_texscreen_used = true;
			glActiveTexture(GL_TEXTURE0);
		}

		//begin rect
		CanvasItem *material_owner = ci->material_owner ? ci->material_owner : ci;
		ShaderMaterial *material = material_owner->material;

		if (material != canvas_last_material || rebind_shader) {

			Shader *shader = NULL;
			if (material && material->shader.is_valid()) {
				shader = shader_owner.get(material->shader);
				if (shader && !shader->valid) {
					shader = NULL;
				}
			}

			shader_cache = shader;

			if (shader) {
				canvas_shader.set_custom_shader(shader->custom_code_id);
				_canvas_item_setup_shader_params(material, shader);
			} else {
				shader_cache = NULL;
				canvas_shader.set_custom_shader(0);
				canvas_shader.bind();
				uses_texpixel_size = false;
			}

			canvas_shader.set_uniform(CanvasShaderGLES2::PROJECTION_MATRIX, canvas_transform);
			if (canvas_use_modulate)
				reset_modulate = true;
			canvas_last_material = material;
			rebind_shader = false;
		}

		if (material && shader_cache) {

			_canvas_item_setup_shader_uniforms(material, shader_cache);
		}

		bool unshaded = (material && material->shading_mode == VS::CANVAS_ITEM_SHADING_UNSHADED) || ci->blend_mode != VS::MATERIAL_BLEND_MODE_MIX;

		if (unshaded) {
			canvas_shader.set_uniform(CanvasShaderGLES2::MODULATE, Color(1, 1, 1, 1));
			reset_modulate = true;
		} else if (reset_modulate) {
			canvas_shader.set_uniform(CanvasShaderGLES2::MODULATE, canvas_modulate);
			reset_modulate = false;
		}

		canvas_shader.set_uniform(CanvasShaderGLES2::MODELVIEW_MATRIX, ci->final_transform);
		canvas_shader.set_uniform(CanvasShaderGLES2::EXTRA_MATRIX, Transform2D());

		bool reclip = false;

		if (ci == p_item_list || ci->blend_mode != canvas_blend_mode) {

			switch (ci->blend_mode) {

				case VS::MATERIAL_BLEND_MODE_MIX: {
					glBlendEquation(GL_FUNC_ADD);
					if (current_rt && current_rt_transparent) {
						glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
					} else {
						glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
					}

				} break;
				case VS::MATERIAL_BLEND_MODE_ADD: {

					glBlendEquation(GL_FUNC_ADD);
					glBlendFunc(GL_SRC_ALPHA, GL_ONE);

				} break;
				case VS::MATERIAL_BLEND_MODE_SUB: {

					glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
					glBlendFunc(GL_SRC_ALPHA, GL_ONE);
				} break;
				case VS::MATERIAL_BLEND_MODE_MUL: {
					glBlendEquation(GL_FUNC_ADD);
					glBlendFunc(GL_DST_COLOR, GL_ZERO);
				} break;
				case VS::MATERIAL_BLEND_MODE_PREMULT_ALPHA: {
					glBlendEquation(GL_FUNC_ADD);
					glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
				} break;
			}

			canvas_blend_mode = ci->blend_mode;
		}

		canvas_opacity = ci->final_opacity;

		if (unshaded || (p_modulate.a > 0.001 && (!material || material->shading_mode != VS::CANVAS_ITEM_SHADING_ONLY_LIGHT) && !ci->light_masked))
			_canvas_item_render_commands<false>(ci, current_clip, reclip);

		if (canvas_blend_mode == VS::MATERIAL_BLEND_MODE_MIX && p_light && !unshaded) {

			CanvasLight *light = p_light;
			bool light_used = false;
			VS::CanvasLightMode mode = VS::CANVAS_LIGHT_MODE_ADD;

			while (light) {

				if (ci->light_mask & light->item_mask && p_z >= light->z_min && p_z <= light->z_max && ci->global_rect_cache.intersects_transformed(light->xform_cache, light->rect_cache)) {

					//intersects this light

					if (!light_used || mode != light->mode) {

						mode = light->mode;

						switch (mode) {

							case VS::CANVAS_LIGHT_MODE_ADD: {
								glBlendEquation(GL_FUNC_ADD);
								glBlendFunc(GL_SRC_ALPHA, GL_ONE);

							} break;
							case VS::CANVAS_LIGHT_MODE_SUB: {
								glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
								glBlendFunc(GL_SRC_ALPHA, GL_ONE);
							} break;
							case VS::CANVAS_LIGHT_MODE_MIX:
							case VS::CANVAS_LIGHT_MODE_MASK: {
								glBlendEquation(GL_FUNC_ADD);
								glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

							} break;
						}
					}

					if (!light_used) {

						canvas_shader.set_conditional(CanvasShaderGLES2::USE_LIGHTING, true);
						canvas_shader.set_conditional(CanvasShaderGLES2::USE_MODULATE, false);
						light_used = true;
						normal_flip = Vector2(1, 1);
					}

					bool has_shadow = light->shadow_buffer.is_valid() && ci->light_mask & light->item_shadow_mask;

					canvas_shader.set_conditional(CanvasShaderGLES2::USE_SHADOWS, has_shadow);

					bool light_rebind = canvas_shader.bind();

					if (light_rebind) {

						if (material && shader_cache) {
							_canvas_item_setup_shader_params(material, shader_cache);
							_canvas_item_setup_shader_uniforms(material, shader_cache);
						}

						canvas_shader.set_uniform(CanvasShaderGLES2::MODELVIEW_MATRIX, ci->final_transform);
						canvas_shader.set_uniform(CanvasShaderGLES2::EXTRA_MATRIX, Transform2D());
						canvas_shader.set_uniform(CanvasShaderGLES2::PROJECTION_MATRIX, canvas_transform);
						if (canvas_use_modulate)
							canvas_shader.set_uniform(CanvasShaderGLES2::MODULATE, canvas_modulate);
						canvas_shader.set_uniform(CanvasShaderGLES2::NORMAL_FLIP, Vector2(1, 1));
						canvas_shader.set_uniform(CanvasShaderGLES2::SHADOWPIXEL_SIZE, 1.0 / light->shadow_buffer_size);
					}

					canvas_shader.set_uniform(CanvasShaderGLES2::LIGHT_MATRIX, light->light_shader_xform);
					canvas_shader.set_uniform(CanvasShaderGLES2::LIGHT_POS, light->light_shader_pos);
					canvas_shader.set_uniform(CanvasShaderGLES2::LIGHT_COLOR, Color(light->color.r * light->energy, light->color.g * light->energy, light->color.b * light->energy, light->color.a));
					canvas_shader.set_uniform(CanvasShaderGLES2::LIGHT_HEIGHT, light->height);
					canvas_shader.set_uniform(CanvasShaderGLES2::LIGHT_LOCAL_MATRIX, light->xform_cache.affine_inverse());
					canvas_shader.set_uniform(CanvasShaderGLES2::LIGHT_OUTSIDE_ALPHA, light->mode == VS::CANVAS_LIGHT_MODE_MASK ? 1.0 : 0.0);

					if (has_shadow) {

						CanvasLightShadow *cls = canvas_light_shadow_owner.get(light->shadow_buffer);
						glActiveTexture(GL_TEXTURE0 + max_texture_units - 3);
						if (read_depth_supported)
							glBindTexture(GL_TEXTURE_2D, cls->depth);
						else
							glBindTexture(GL_TEXTURE_2D, cls->rgba);

						canvas_shader.set_uniform(CanvasShaderGLES2::SHADOW_TEXTURE, max_texture_units - 3);
						canvas_shader.set_uniform(CanvasShaderGLES2::SHADOW_MATRIX, light->shadow_matrix_cache);
						canvas_shader.set_uniform(CanvasShaderGLES2::SHADOW_ESM_MULTIPLIER, light->shadow_esm_mult);
						canvas_shader.set_uniform(CanvasShaderGLES2::LIGHT_SHADOW_COLOR, light->shadow_color);
					}

					glActiveTexture(GL_TEXTURE0 + max_texture_units - 2);
					canvas_shader.set_uniform(CanvasShaderGLES2::LIGHT_TEXTURE, max_texture_units - 2);
					Texture *t = texture_owner.get(light->texture);
					if (!t) {
						glBindTexture(GL_TEXTURE_2D, white_tex);
					} else {

						glBindTexture(t->target, t->tex_id);
					}

					glActiveTexture(GL_TEXTURE0);
					_canvas_item_render_commands<true>(ci, current_clip, reclip); //redraw using light
				}

				light = light->next_ptr;
			}

			if (light_used) {

				canvas_shader.set_conditional(CanvasShaderGLES2::USE_LIGHTING, false);
				canvas_shader.set_conditional(CanvasShaderGLES2::USE_MODULATE, canvas_use_modulate);
				canvas_shader.set_conditional(CanvasShaderGLES2::USE_SHADOWS, false);

				canvas_shader.bind();

				if (material && shader_cache) {
					_canvas_item_setup_shader_params(material, shader_cache);
					_canvas_item_setup_shader_uniforms(material, shader_cache);
				}

				canvas_shader.set_uniform(CanvasShaderGLES2::MODELVIEW_MATRIX, ci->final_transform);
				canvas_shader.set_uniform(CanvasShaderGLES2::EXTRA_MATRIX, Transform2D());
				if (canvas_use_modulate)
					canvas_shader.set_uniform(CanvasShaderGLES2::MODULATE, canvas_modulate);

				glBlendEquation(GL_FUNC_ADD);
				if (current_rt && current_rt_transparent) {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
				} else {
					glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
				}
			}
		}

		if (reclip) {

			glEnable(GL_SCISSOR_TEST);
			//glScissor(viewport.x+current_clip->final_clip_rect.pos.x,viewport.y+ (viewport.height-(current_clip->final_clip_rect.pos.y+current_clip->final_clip_rect.size.height)),
			//current_clip->final_clip_rect.size.width,current_clip->final_clip_rect.size.height);

			int x;
			int y;
			int w;
			int h;

			if (current_rt) {
				x = current_clip->final_clip_rect.pos.x;
				y = current_clip->final_clip_rect.pos.y;
				w = current_clip->final_clip_rect.size.x;
				h = current_clip->final_clip_rect.size.y;
			} else {
				x = current_clip->final_clip_rect.pos.x;
				y = window_size.height - (current_clip->final_clip_rect.pos.y + current_clip->final_clip_rect.size.y);
				w = current_clip->final_clip_rect.size.x;
				h = current_clip->final_clip_rect.size.y;
			}

			glScissor(x, y, w, h);
		}

		p_item_list = p_item_list->next;
	}

	if (current_clip) {
		glDisable(GL_SCISSOR_TEST);
	}
}

/* ENVIRONMENT */

RID RasterizerGLES2::environment_create() {

	Environment *env = memnew(Environment);
	return environment_owner.make_rid(env);
}

void RasterizerGLES2::environment_set_background(RID p_env, VS::EnvironmentBG p_bg) {

	ERR_FAIL_INDEX(p_bg, VS::ENV_BG_MAX);
	Environment *env = environment_owner.get(p_env);
	ERR_FAIL_COND(!env);
	env->bg_mode = p_bg;
}

VS::EnvironmentBG RasterizerGLES2::environment_get_background(RID p_env) const {

	const Environment *env = environment_owner.get(p_env);
	ERR_FAIL_COND_V(!env, VS::ENV_BG_MAX);
	return env->bg_mode;
}

void RasterizerGLES2::environment_set_background_param(RID p_env, VS::EnvironmentBGParam p_param, const Variant &p_value) {

	ERR_FAIL_INDEX(p_param, VS::ENV_BG_PARAM_MAX);
	Environment *env = environment_owner.get(p_env);
	ERR_FAIL_COND(!env);
	env->bg_param[p_param] = p_value;
}
Variant RasterizerGLES2::environment_get_background_param(RID p_env, VS::EnvironmentBGParam p_param) const {

	ERR_FAIL_INDEX_V(p_param, VS::ENV_BG_PARAM_MAX, Variant());
	const Environment *env = environment_owner.get(p_env);
	ERR_FAIL_COND_V(!env, Variant());
	return env->bg_param[p_param];
}

void RasterizerGLES2::environment_set_enable_fx(RID p_env, VS::EnvironmentFx p_effect, bool p_enabled) {

	ERR_FAIL_INDEX(p_effect, VS::ENV_FX_MAX);
	Environment *env = environment_owner.get(p_env);
	ERR_FAIL_COND(!env);
	env->fx_enabled[p_effect] = p_enabled;
}
bool RasterizerGLES2::environment_is_fx_enabled(RID p_env, VS::EnvironmentFx p_effect) const {

	ERR_FAIL_INDEX_V(p_effect, VS::ENV_FX_MAX, false);
	const Environment *env = environment_owner.get(p_env);
	ERR_FAIL_COND_V(!env, false);
	return env->fx_enabled[p_effect];
}

void RasterizerGLES2::environment_fx_set_param(RID p_env, VS::EnvironmentFxParam p_param, const Variant &p_value) {

	ERR_FAIL_INDEX(p_param, VS::ENV_FX_PARAM_MAX);
	Environment *env = environment_owner.get(p_env);
	ERR_FAIL_COND(!env);
	env->fx_param[p_param] = p_value;
}
Variant RasterizerGLES2::environment_fx_get_param(RID p_env, VS::EnvironmentFxParam p_param) const {

	ERR_FAIL_INDEX_V(p_param, VS::ENV_FX_PARAM_MAX, Variant());
	const Environment *env = environment_owner.get(p_env);
	ERR_FAIL_COND_V(!env, Variant());
	return env->fx_param[p_param];
}

RID RasterizerGLES2::sampled_light_dp_create(int p_width, int p_height) {

	SampledLight *slight = memnew(SampledLight);
	slight->w = p_width;
	slight->h = p_height;
	slight->multiplier = 1.0;
	slight->is_float = float_linear_supported;

	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &slight->texture);
	glBindTexture(GL_TEXTURE_2D, slight->texture);
	// for debug, but glitchy
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Remove artifact on the edges of the shadowmap
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	if (slight->is_float) {
#ifdef GLEW_ENABLED
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, p_width, p_height, 0, GL_RGBA, GL_FLOAT, NULL);
#else
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, p_width, p_height, 0, GL_RGBA, GL_FLOAT, NULL);
#endif
	} else {

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, p_width, p_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	}

	return sampled_light_owner.make_rid(slight);
}

void RasterizerGLES2::sampled_light_dp_update(RID p_sampled_light, const Color *p_data, float p_multiplier) {

	SampledLight *slight = sampled_light_owner.get(p_sampled_light);
	ERR_FAIL_COND(!slight);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, slight->texture);

	if (slight->is_float) {

#ifdef GLEW_ENABLED
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, slight->w, slight->h, GL_RGBA, GL_FLOAT, p_data);
#else
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, slight->w, slight->h, GL_RGBA, GL_FLOAT, p_data);
#endif

	} else {
		//convert to bytes
		uint8_t *tex8 = (uint8_t *)alloca(slight->w * slight->h * 4);
		const float *src = (const float *)p_data;

		for (int i = 0; i < slight->w * slight->h * 4; i++) {

			tex8[i] = Math::fast_ftoi(CLAMP(src[i] * 255.0, 0.0, 255.0));
		}

		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, slight->w, slight->h, GL_RGBA, GL_UNSIGNED_BYTE, p_data);
	}

	slight->multiplier = p_multiplier;
}

/*MISC*/

bool RasterizerGLES2::is_texture(const RID &p_rid) const {

	return texture_owner.owns(p_rid);
}
bool RasterizerGLES2::is_material(const RID &p_rid) const {

	return material_owner.owns(p_rid);
}
bool RasterizerGLES2::is_mesh(const RID &p_rid) const {

	return mesh_owner.owns(p_rid);
}
bool RasterizerGLES2::is_immediate(const RID &p_rid) const {

	return immediate_owner.owns(p_rid);
}
bool RasterizerGLES2::is_multimesh(const RID &p_rid) const {

	return multimesh_owner.owns(p_rid);
}
bool RasterizerGLES2::is_particles(const RID &p_beam) const {

	return particles_owner.owns(p_beam);
}

bool RasterizerGLES2::is_light(const RID &p_rid) const {

	return light_owner.owns(p_rid);
}
bool RasterizerGLES2::is_light_instance(const RID &p_rid) const {

	return light_instance_owner.owns(p_rid);
}
bool RasterizerGLES2::is_particles_instance(const RID &p_rid) const {

	return particles_instance_owner.owns(p_rid);
}
bool RasterizerGLES2::is_skeleton(const RID &p_rid) const {

	return skeleton_owner.owns(p_rid);
}
bool RasterizerGLES2::is_environment(const RID &p_rid) const {

	return environment_owner.owns(p_rid);
}
bool RasterizerGLES2::is_shader(const RID &p_rid) const {

	return shader_owner.owns(p_rid);
}

bool RasterizerGLES2::is_canvas_light_occluder(const RID &p_rid) const {

	return false;
}

void RasterizerGLES2::free(const RID &p_rid) {
	if (texture_owner.owns(p_rid)) {

		// delete the texture
		Texture *texture = texture_owner.get(p_rid);

		//glDeleteTextures( 1,&texture->tex_id );
		_rinfo.texture_mem -= texture->total_data_size;
		texture_owner.free(p_rid);
		memdelete(texture);

	} else if (shader_owner.owns(p_rid)) {

		// delete the texture
		Shader *shader = shader_owner.get(p_rid);

		switch (shader->mode) {
			case VS::SHADER_MATERIAL: {
				material_shader.free_custom_shader(shader->custom_code_id);
			} break;
			case VS::SHADER_POST_PROCESS: {
				//postprocess_shader.free_custom_shader(shader->custom_code_id);
			} break;
		}

		if (shader->dirty_list.in_list())
			_shader_dirty_list.remove(&shader->dirty_list);

		//material_shader.free_custom_shader(shader->custom_code_id);
		shader_owner.free(p_rid);
		memdelete(shader);

	} else if (material_owner.owns(p_rid)) {

		Material *material = material_owner.get(p_rid);
		ERR_FAIL_COND(!material);

		_free_fixed_material(p_rid); //just in case
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

			if (mesh->morph_target_count > 0) {

				for (int i = 0; i < mesh->morph_target_count; i++) {

					memdelete_arr(surface->morph_targets_local[i].array);
				}
				memdelete_arr(surface->morph_targets_local);
				surface->morph_targets_local = NULL;
			}

			if (surface->vertex_id)
				glDeleteBuffers(1, &surface->vertex_id);
			if (surface->index_id)
				glDeleteBuffers(1, &surface->index_id);

			memdelete(surface);
		};

		mesh->surfaces.clear();

		mesh_owner.free(p_rid);
		memdelete(mesh);

	} else if (multimesh_owner.owns(p_rid)) {

		MultiMesh *multimesh = multimesh_owner.get(p_rid);
		ERR_FAIL_COND(!multimesh);

		if (multimesh->tex_id) {
			glDeleteTextures(1, &multimesh->tex_id);
		}

		multimesh_owner.free(p_rid);
		memdelete(multimesh);

	} else if (immediate_owner.owns(p_rid)) {

		Immediate *immediate = immediate_owner.get(p_rid);
		ERR_FAIL_COND(!immediate);

		immediate_owner.free(p_rid);
		memdelete(immediate);
	} else if (particles_owner.owns(p_rid)) {

		Particles *particles = particles_owner.get(p_rid);
		ERR_FAIL_COND(!particles);

		particles_owner.free(p_rid);
		memdelete(particles);
	} else if (particles_instance_owner.owns(p_rid)) {

		ParticlesInstance *particles_isntance = particles_instance_owner.get(p_rid);
		ERR_FAIL_COND(!particles_isntance);

		particles_instance_owner.free(p_rid);
		memdelete(particles_isntance);

	} else if (skeleton_owner.owns(p_rid)) {

		Skeleton *skeleton = skeleton_owner.get(p_rid);
		ERR_FAIL_COND(!skeleton);

		if (skeleton->dirty_list.in_list())
			_skeleton_dirty_list.remove(&skeleton->dirty_list);
		if (skeleton->tex_id) {
			glDeleteTextures(1, &skeleton->tex_id);
		}
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
		light_instance->clear_shadow_buffers();
		light_instance_owner.free(p_rid);
		memdelete(light_instance);

	} else if (environment_owner.owns(p_rid)) {

		Environment *env = environment_owner.get(p_rid);
		ERR_FAIL_COND(!env);

		environment_owner.free(p_rid);
		memdelete(env);

	} else if (viewport_data_owner.owns(p_rid)) {

		ViewportData *viewport_data = viewport_data_owner.get(p_rid);
		ERR_FAIL_COND(!viewport_data);
		glDeleteFramebuffers(1, &viewport_data->lum_fbo);
		glDeleteTextures(1, &viewport_data->lum_color);
		viewport_data_owner.free(p_rid);
		memdelete(viewport_data);

	} else if (render_target_owner.owns(p_rid)) {

		RenderTarget *render_target = render_target_owner.get(p_rid);
		ERR_FAIL_COND(!render_target);
		render_target_set_size(p_rid, 0, 0); //clears framebuffer
		texture_owner.free(render_target->texture);
		memdelete(render_target->texture_ptr);
		render_target_owner.free(p_rid);
		memdelete(render_target);
	} else if (sampled_light_owner.owns(p_rid)) {

		SampledLight *sampled_light = sampled_light_owner.get(p_rid);
		ERR_FAIL_COND(!sampled_light);
		glDeleteTextures(1, &sampled_light->texture);
		sampled_light_owner.free(p_rid);
		memdelete(sampled_light);
	} else if (canvas_occluder_owner.owns(p_rid)) {

		CanvasOccluder *co = canvas_occluder_owner.get(p_rid);
		if (co->index_id)
			glDeleteBuffers(1, &co->index_id);
		if (co->vertex_id)
			glDeleteBuffers(1, &co->vertex_id);

		canvas_occluder_owner.free(p_rid);
		memdelete(co);

	} else if (canvas_light_shadow_owner.owns(p_rid)) {

		CanvasLightShadow *cls = canvas_light_shadow_owner.get(p_rid);
		glDeleteFramebuffers(1, &cls->fbo);
		glDeleteRenderbuffers(1, &cls->rbo);
		glDeleteTextures(1, &cls->depth);
		/*
		if (!read_depth_supported) {
			glDeleteTextures(1,&cls->rgba);
		}
		*/

		canvas_light_shadow_owner.free(p_rid);
		memdelete(cls);
	};
}

bool RasterizerGLES2::ShadowBuffer::init(int p_size, bool p_use_depth) {

	size = p_size;
	// Create a framebuffer object
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	// Create a render buffer
	glGenRenderbuffers(1, &rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo);

	// Create a texture for storing the depth
	glGenTextures(1, &depth);
	glBindTexture(GL_TEXTURE_2D, depth);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Remove artifact on the edges of the shadowmap
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	//print_line("ERROR? "+itos(glGetError()));
	if (p_use_depth) {

		// We'll use a depth texture to store the depths in the shadow map
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, size, size, 0,
				GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);

#ifdef GLEW_ENABLED
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#endif

		// Attach the depth texture to FBO depth attachment point
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
				GL_TEXTURE_2D, depth, 0);

#ifdef GLEW_ENABLED
		glDrawBuffer(GL_NONE);
#endif
	} else {
		// We'll use a RGBA texture into which we pack the depth info
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0,
				GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		// Attach the RGBA texture to FBO color attachment point
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
				GL_TEXTURE_2D, depth, 0);

		// Allocate 16-bit depth buffer
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, size, size);

		// Attach the render buffer as depth buffer - will be ignored
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
				GL_RENDERBUFFER, rbo);
	}

#if 0

	if (!p_use_depth) {


		print_line("try no depth!");

		glGenTextures(1, &rgba);
		glBindTexture(GL_TEXTURE_2D, rgba);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rgba, 0);
/*
		glGenRenderbuffers(1, &depth);
		glBindRenderbuffer(GL_RENDERBUFFER, depth);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, p_size, p_size);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
*/
		glGenTextures(1, &depth);
		glBindTexture(GL_TEXTURE_2D, depth);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, size, size, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);

	} else {

		//glGenRenderbuffers(1, &rbo);
		//glBindRenderbuffer(GL_RENDERBUFFER, rbo);

		glGenTextures(1, &depth);
		glBindTexture(GL_TEXTURE_2D, depth);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, size, size, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);

	}

#endif
	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
//printf("errnum: %x\n",status);
#ifdef GLEW_ENABLED
	if (p_use_depth) {
		//glDrawBuffer(GL_BACK);
	}
#endif
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	DEBUG_TEST_ERROR("Shadow Buffer Init");
	ERR_FAIL_COND_V(status != GL_FRAMEBUFFER_COMPLETE, false);

#ifdef GLEW_ENABLED
	if (p_use_depth) {
		//glDrawBuffer(GL_BACK);
	}
#endif

#if 0
	glGenFramebuffers(1, &fbo_blur);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo_blur);

	glGenRenderbuffers(1, &rbo_blur);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo_blur);

	glGenTextures(1, &blur);
	glBindTexture(GL_TEXTURE_2D, blur);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);


	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, size, size, 0,
	//GL_DEPTH_COMPONENT16, GL_UNSIGNED_SHORT, NULL);

	// Attach the RGBA texture to FBO color attachment point
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			GL_TEXTURE_2D, blur, 0);

	// Allocate 16-bit depth buffer
	/*
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, size, size);

	// Attach the render buffer as depth buffer - will be ignored
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
			GL_RENDERBUFFER, rbo_blur);
	*/
	status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	OS::get_singleton()->print("Status: %x\n",status);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	DEBUG_TEST_ERROR("Shadow Blur Buffer Init");
	ERR_FAIL_COND_V( status != GL_FRAMEBUFFER_COMPLETE,false );
#endif

	return true;
}

void RasterizerGLES2::_update_framebuffer() {

	if (!use_framebuffers)
		return;

	int scale = GLOBAL_DEF("rasterizer/framebuffer_shrink", 1);
	if (scale < 1)
		scale = 1;

	int dwidth = OS::get_singleton()->get_video_mode().width / scale;
	int dheight = OS::get_singleton()->get_video_mode().height / scale;

	if (framebuffer.fbo && dwidth == framebuffer.width && dheight == framebuffer.height)
		return;

	bool use_fbo = true;

	if (framebuffer.fbo != 0) {

		glDeleteFramebuffers(1, &framebuffer.fbo);
#if 0
		glDeleteTextures(1,&framebuffer.depth);
#else
		glDeleteRenderbuffers(1, &framebuffer.depth);

#endif
		glDeleteTextures(1, &framebuffer.color);

		for (int i = 0; i < framebuffer.luminance.size(); i++) {

			glDeleteTextures(1, &framebuffer.luminance[i].color);
			glDeleteFramebuffers(1, &framebuffer.luminance[i].fbo);
		}

		for (int i = 0; i < 3; i++) {

			glDeleteTextures(1, &framebuffer.blur[i].color);
			glDeleteFramebuffers(1, &framebuffer.blur[i].fbo);
		}

		glDeleteTextures(1, &framebuffer.sample_color);
		glDeleteFramebuffers(1, &framebuffer.sample_fbo);
		framebuffer.luminance.clear();
		framebuffer.blur_size = 0;
		framebuffer.fbo = 0;
	}

#ifdef TOOLS_ENABLED
	framebuffer.active = use_fbo;
#else
	framebuffer.active = use_fbo && !low_memory_2d;
#endif
	framebuffer.width = dwidth;
	framebuffer.height = dheight;
	framebuffer.scale = scale;

	if (!framebuffer.active)
		return;

	glGenFramebuffers(1, &framebuffer.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.fbo);

//print_line("generating fbo, id: "+itos(framebuffer.fbo));
//depth

// Create a render buffer

#if 0
	glGenTextures(1, &framebuffer.depth);
	glBindTexture(GL_TEXTURE_2D, framebuffer.depth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24,  framebuffer.width, framebuffer.height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE );
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, framebuffer.depth, 0);

#else

	glGenRenderbuffers(1, &framebuffer.depth);
	glBindRenderbuffer(GL_RENDERBUFFER, framebuffer.depth);

	glRenderbufferStorage(GL_RENDERBUFFER, use_depth24 ? _DEPTH_COMPONENT24_OES : GL_DEPTH_COMPONENT16, framebuffer.width, framebuffer.height);

	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, framebuffer.depth);

#endif
	//color

	//GLuint format_rgba = use_fp16_fb?_GL_RGBA16F_EXT:GL_RGBA;
	GLuint format_rgba = GL_RGBA;
	GLuint format_type = use_fp16_fb ? _GL_HALF_FLOAT_OES : GL_UNSIGNED_BYTE;
	GLuint format_internal = GL_RGBA;

	if (use_16bits_fbo) {
		format_type = GL_UNSIGNED_SHORT_5_6_5;
		format_rgba = GL_RGB;
		format_internal = GL_RGB;
	}
	/*GLuint format_luminance = use_fp16_fb?GL_RGB16F:GL_RGBA;
	GLuint format_luminance_type = use_fp16_fb?(use_fu_GL_HALF_FLOAT_OES):GL_UNSIGNED_BYTE;
	GLuint format_luminance_components = use_fp16_fb?GL_RGB:GL_RGBA;*/

	GLuint format_luminance = use_fp16_fb ? _GL_RG_EXT : GL_RGBA;
	GLuint format_luminance_type = use_fp16_fb ? (full_float_fb_supported ? GL_FLOAT : _GL_HALF_FLOAT_OES) : GL_UNSIGNED_BYTE;
	GLuint format_luminance_components = use_fp16_fb ? _GL_RG_EXT : GL_RGBA;

	glGenTextures(1, &framebuffer.color);
	glBindTexture(GL_TEXTURE_2D, framebuffer.color);
	glTexImage2D(GL_TEXTURE_2D, 0, format_rgba, framebuffer.width, framebuffer.height, 0, format_internal, format_type, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, framebuffer.color, 0);
#
	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	if (status != GL_FRAMEBUFFER_COMPLETE) {

		glDeleteFramebuffers(1, &framebuffer.fbo);
#if 0
		glDeleteTextures(1,&framebuffer.depth);
#else
		glDeleteRenderbuffers(1, &framebuffer.depth);

#endif
		glDeleteTextures(1, &framebuffer.color);
		framebuffer.fbo = 0;
		framebuffer.active = false;
		//print_line("**************** NO FAMEBUFFEEEERRRR????");
		WARN_PRINT(String("Could not create framebuffer!!, code: " + itos(status)).ascii().get_data());
	}

	//sample

	glGenFramebuffers(1, &framebuffer.sample_fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.sample_fbo);
	glGenTextures(1, &framebuffer.sample_color);
	glBindTexture(GL_TEXTURE_2D, framebuffer.sample_color);
	glTexImage2D(GL_TEXTURE_2D, 0, format_rgba, framebuffer.width, framebuffer.height, 0, format_internal, format_type, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, framebuffer.sample_color, 0);
#
	status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	if (status != GL_FRAMEBUFFER_COMPLETE) {

		glDeleteFramebuffers(1, &framebuffer.fbo);
#if 0
		glDeleteTextures(1,&framebuffer.depth);
#else
		glDeleteRenderbuffers(1, &framebuffer.depth);

#endif
		glDeleteTextures(1, &framebuffer.color);
		glDeleteTextures(1, &framebuffer.sample_color);
		glDeleteFramebuffers(1, &framebuffer.sample_fbo);
		framebuffer.fbo = 0;
		framebuffer.active = false;
		//print_line("**************** NO FAMEBUFFEEEERRRR????");
		WARN_PRINT("Could not create framebuffer!!");
	}
	//blur

	int size = GLOBAL_DEF("rasterizer/blur_buffer_size", 256);

	if (size != framebuffer.blur_size) {

		for (int i = 0; i < 3; i++) {

			if (framebuffer.blur[i].fbo) {
				glDeleteFramebuffers(1, &framebuffer.blur[i].fbo);
				glDeleteTextures(1, &framebuffer.blur[i].color);
				framebuffer.blur[i].fbo = 0;
				framebuffer.blur[i].color = 0;
			}
		}

		framebuffer.blur_size = size;

		for (int i = 0; i < 3; i++) {

			glGenFramebuffers(1, &framebuffer.blur[i].fbo);
			glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.blur[i].fbo);

			glGenTextures(1, &framebuffer.blur[i].color);
			glBindTexture(GL_TEXTURE_2D, framebuffer.blur[i].color);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexImage2D(GL_TEXTURE_2D, 0, format_rgba, size, size, 0,
					format_internal, format_type, NULL);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
					GL_TEXTURE_2D, framebuffer.blur[i].color, 0);

			GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			DEBUG_TEST_ERROR("Shadow Buffer Init");
			ERR_CONTINUE(status != GL_FRAMEBUFFER_COMPLETE);
		}
	}

	// luminance

	int base_size = GLOBAL_DEF("rasterizer/luminance_buffer_size", 81);

	if (framebuffer.luminance.empty() || framebuffer.luminance[0].size != base_size) {

		for (int i = 0; i < framebuffer.luminance.size(); i++) {

			glDeleteFramebuffers(1, &framebuffer.luminance[i].fbo);
			glDeleteTextures(1, &framebuffer.luminance[i].color);
		}

		framebuffer.luminance.clear();

		while (base_size > 0) {

			FrameBuffer::Luminance lb;
			lb.size = base_size;

			glGenFramebuffers(1, &lb.fbo);
			glBindFramebuffer(GL_FRAMEBUFFER, lb.fbo);

			glGenTextures(1, &lb.color);
			glBindTexture(GL_TEXTURE_2D, lb.color);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexImage2D(GL_TEXTURE_2D, 0, format_luminance, lb.size, lb.size, 0,
					format_luminance_components, format_luminance_type, NULL);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
					GL_TEXTURE_2D, lb.color, 0);

			GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

			glBindFramebuffer(GL_FRAMEBUFFER, 0);

			base_size /= 3;

			DEBUG_TEST_ERROR("Shadow Buffer Init");
			ERR_CONTINUE(status != GL_FRAMEBUFFER_COMPLETE);

			framebuffer.luminance.push_back(lb);
		}
	}
}

void RasterizerGLES2::set_base_framebuffer(GLuint p_id, Vector2 p_size) {

	base_framebuffer = p_id;

	if (p_size.x != 0) {
		window_size = p_size;
	};
}

#if 0
void RasterizerGLES2::_update_blur_buffer() {

	int size = GLOBAL_DEF("rasterizer/blur_buffer_size",256);
	if (size!=framebuffer.blur_size) {

		for(int i=0;i<3;i++) {

			if (framebuffer.blur[i].fbo) {
				glDeleteFramebuffers(1,&framebuffer.blur[i].fbo);
				glDeleteTextures(1,&framebuffer.blur[i].color);
				framebuffer.blur[i].fbo=0;
				framebuffer.blur[i].color=0;
			}
		}

		framebuffer.blur_size=size;

		for(int i=0;i<3;i++) {

			glGenFramebuffers(1, &framebuffer.blur[i].fbo);
			glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.blur[i].fbo);

			glGenTextures(1, &framebuffer.blur[i].color);
			glBindTexture(GL_TEXTURE_2D, framebuffer.blur[i].color);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0,
					GL_RGBA, GL_UNSIGNED_BYTE, NULL);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
					GL_TEXTURE_2D, framebuffer.blur[i].color, 0);


			GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			DEBUG_TEST_ERROR("Shadow Buffer Init");
			ERR_CONTINUE( status != GL_FRAMEBUFFER_COMPLETE );


		}

	}





}
#endif

bool RasterizerGLES2::_test_depth_shadow_buffer() {

	int size = 16;

	GLuint fbo;
	GLuint rbo;
	GLuint depth;

	glActiveTexture(GL_TEXTURE0);

	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	// Create a render buffer
	glGenRenderbuffers(1, &rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo);

	// Create a texture for storing the depth
	glGenTextures(1, &depth);
	glBindTexture(GL_TEXTURE_2D, depth);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Remove artifact on the edges of the shadowmap
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	// We'll use a depth texture to store the depths in the shadow map
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, size, size, 0,
			GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);

#ifdef GLEW_ENABLED
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#endif

	// Attach the depth texture to FBO depth attachment point
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
			GL_TEXTURE_2D, depth, 0);

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

	glDeleteFramebuffers(1, &fbo);
	glDeleteRenderbuffers(1, &rbo);
	glDeleteTextures(1, &depth);

	return status == GL_FRAMEBUFFER_COMPLETE;
}

void RasterizerGLES2::init() {

	if (OS::get_singleton()->is_stdout_verbose()) {
		print_line("Using GLES2 video driver");
	}

#ifdef GLEW_ENABLED
	GLuint res = glewInit();
	ERR_FAIL_COND(res != GLEW_OK);
	if (OS::get_singleton()->is_stdout_verbose()) {
		print_line(String("GLES2: Using GLEW ") + (const char *)glewGetString(GLEW_VERSION));
	}

	// Godot makes use of functions from ARB_framebuffer_object extension which is not implemented by all drivers.
	// On the other hand, these drivers might implement the older EXT_framebuffer_object extension
	// with which current source code is backward compatible.

	bool framebuffer_object_is_supported = glewIsSupported("GL_ARB_framebuffer_object");

	if (!framebuffer_object_is_supported) {
		WARN_PRINT("GL_ARB_framebuffer_object not supported by your graphics card.");

		if (glewIsSupported("GL_EXT_framebuffer_object")) {
			// falling-back to the older EXT function if present
			WARN_PRINT("Falling-back to GL_EXT_framebuffer_object.");

			glIsRenderbuffer = glIsRenderbufferEXT;
			glBindRenderbuffer = glBindRenderbufferEXT;
			glDeleteRenderbuffers = glDeleteRenderbuffersEXT;
			glGenRenderbuffers = glGenRenderbuffersEXT;
			glRenderbufferStorage = glRenderbufferStorageEXT;
			glGetRenderbufferParameteriv = glGetRenderbufferParameterivEXT;
			glIsFramebuffer = glIsFramebufferEXT;
			glBindFramebuffer = glBindFramebufferEXT;
			glDeleteFramebuffers = glDeleteFramebuffersEXT;
			glGenFramebuffers = glGenFramebuffersEXT;
			glCheckFramebufferStatus = glCheckFramebufferStatusEXT;
			glFramebufferTexture1D = glFramebufferTexture1DEXT;
			glFramebufferTexture2D = glFramebufferTexture2DEXT;
			glFramebufferTexture3D = glFramebufferTexture3DEXT;
			glFramebufferRenderbuffer = glFramebufferRenderbufferEXT;
			glGetFramebufferAttachmentParameteriv = glGetFramebufferAttachmentParameterivEXT;
			glGenerateMipmap = glGenerateMipmapEXT;

			framebuffer_object_is_supported = true;
		} else {
			ERR_PRINT("Framebuffer Object is not supported by your graphics card.");
		}
	}

	// Check for GL 2.1 compatibility, if not bail out
	if (!(glewIsSupported("GL_VERSION_2_1") && framebuffer_object_is_supported)) {
		ERR_PRINT("Your system's graphic drivers seem not to support OpenGL 2.1 / GLES 2.0, sorry :(\n"
				  "Try a drivers update, buy a new GPU or try software rendering on Linux; Godot is now going to terminate.");
		OS::get_singleton()->alert("Your system's graphic drivers seem not to support OpenGL 2.1 / GLES 2.0, sorry :(\n"
								   "Godot Engine will self-destruct as soon as you acknowledge this error message.",
				"Fatal error: Insufficient OpenGL / GLES drivers");
		exit(1);
	}
#endif

	scene_pass = 1;

	if (extensions.size() == 0) {

		set_extensions((const char *)glGetString(GL_EXTENSIONS));
	}

	GLint tmp = 0;
	glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &tmp);
	//print_line("GL_MAX_VERTEX_ATTRIBS "+itos(tmp));

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glFrontFace(GL_CW);
	//glEnable(GL_TEXTURE_2D);

	default_material = create_default_material();

	material_shader.init();
	canvas_shader.init();
	copy_shader.init();
	canvas_shadow_shader.init();

#ifdef GLEW_ENABLED
	material_shader.set_conditional(MaterialShaderGLES2::USE_GLES_OVER_GL, true);
	canvas_shader.set_conditional(CanvasShaderGLES2::USE_GLES_OVER_GL, true);
	canvas_shadow_shader.set_conditional(CanvasShadowShaderGLES2::USE_GLES_OVER_GL, true);
	copy_shader.set_conditional(CopyShaderGLES2::USE_GLES_OVER_GL, true);
#endif

#ifdef ANGLE_ENABLED
	// Fix for ANGLE
	material_shader.set_conditional(MaterialShaderGLES2::DISABLE_FRONT_FACING, true);
#endif

	shadow = NULL;
	shadow_pass = 0;

	framebuffer.fbo = 0;
	framebuffer.width = 0;
	framebuffer.height = 0;
	//framebuffer.buff16=false;
	//framebuffer.blur[0].fbo=false;
	//framebuffer.blur[1].fbo=false;
	framebuffer.active = false;

	//do a single initial clear
	glClearColor(0, 0, 0, 1);
	//glClearDepth(1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glGenTextures(1, &white_tex);
	unsigned char whitetexdata[8 * 8 * 3];
	for (int i = 0; i < 8 * 8 * 3; i++) {
		whitetexdata[i] = 255;
	}
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, white_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 8, 8, 0, GL_RGB, GL_UNSIGNED_BYTE, whitetexdata);
	glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

#ifdef GLEW_ENABLED

	pvr_supported = false;
	etc_supported = false;
	use_depth24 = true;
	s3tc_supported = true;
	atitc_supported = false;
	//use_texture_instancing=false;
	//use_attribute_instancing=true;
	use_texture_instancing = false;
	use_attribute_instancing = true;
	full_float_fb_supported = true;
	srgb_supported = true;
	latc_supported = true;
	s3tc_srgb_supported = true;
	use_anisotropic_filter = true;
	float_linear_supported = true;

	GLint vtf;
	glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS, &vtf);
	float_supported = extensions.has("GL_OES_texture_float") || extensions.has("GL_ARB_texture_float");
	use_hw_skeleton_xform = vtf > 0 && float_supported;

	read_depth_supported = _test_depth_shadow_buffer();
	use_rgba_shadowmaps = !read_depth_supported;
	//print_line("read depth support? "+itos(read_depth_supported));

	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &anisotropic_level);
	anisotropic_level = MIN(anisotropic_level, float(GLOBAL_DEF("rasterizer/anisotropic_filter_level", 4.0)));
#ifdef OSX_ENABLED
	use_rgba_shadowmaps = true;
	use_fp16_fb = false;
#else

#endif
	use_half_float = true;

#else

	for (Set<String>::Element *E = extensions.front(); E; E = E->next()) {
		print_line(E->get());
	}
	read_depth_supported = extensions.has("GL_OES_depth_texture");
	use_rgba_shadowmaps = !read_depth_supported;
	if (shadow_filter >= SHADOW_FILTER_ESM && !extensions.has("GL_EXT_frag_depth")) {
		use_rgba_shadowmaps = true; //no other way, go back to rgba
	}
	pvr_supported = extensions.has("GL_IMG_texture_compression_pvrtc");
	pvr_srgb_supported = extensions.has("GL_EXT_pvrtc_sRGB");
	etc_supported = extensions.has("GL_OES_compressed_ETC1_RGB8_texture");
	use_depth24 = extensions.has("GL_OES_depth24");
	s3tc_supported = extensions.has("GL_EXT_texture_compression_dxt1") || extensions.has("GL_EXT_texture_compression_s3tc") || extensions.has("WEBGL_compressed_texture_s3tc");
	use_half_float = extensions.has("GL_OES_vertex_half_float");
	atitc_supported = extensions.has("GL_AMD_compressed_ATC_texture");

	srgb_supported = extensions.has("GL_EXT_sRGB");
#ifndef ANGLE_ENABLED
	s3tc_srgb_supported = s3tc_supported && extensions.has("GL_EXT_texture_compression_s3tc");
#else
	s3tc_srgb_supported = s3tc_supported;
#endif
	latc_supported = extensions.has("GL_EXT_texture_compression_latc");
	anisotropic_level = 1.0;
	use_anisotropic_filter = extensions.has("GL_EXT_texture_filter_anisotropic");
	if (use_anisotropic_filter) {
		glGetFloatv(_GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &anisotropic_level);
		anisotropic_level = MIN(anisotropic_level, float(GLOBAL_DEF("rasterizer/anisotropic_filter_level", 4.0)));
	}

	print_line("S3TC: " + itos(s3tc_supported) + " ATITC: " + itos(atitc_supported));

	GLint vtf;
	glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS, &vtf);
	float_supported = extensions.has("GL_OES_texture_float") || extensions.has("GL_ARB_texture_float");
	use_hw_skeleton_xform = vtf > 0 && float_supported;
	float_linear_supported = extensions.has("GL_OES_texture_float_linear");

	/*
	if (extensions.has("GL_QCOM_tiled_rendering"))
		use_hw_skeleton_xform=false;
	*/
	GLint mva;
	glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &mva);
	if (vtf == 0 && mva > 8) {
		//tegra 3, mali 400
		use_attribute_instancing = true;
		use_texture_instancing = false;
	} else if (vtf > 0 && extensions.has("GL_OES_texture_float")) {
		//use_texture_instancing=true;
		use_texture_instancing = false; // i don't get it, uniforms are faster.
		use_attribute_instancing = false;

	} else {

		use_texture_instancing = false;
		use_attribute_instancing = false;
	}

	if (use_fp16_fb) {
		use_fp16_fb = extensions.has("GL_OES_texture_half_float") && extensions.has("GL_EXT_color_buffer_half_float") && extensions.has("GL_EXT_texture_rg");
	}

	full_float_fb_supported = extensions.has("GL_EXT_color_buffer_float");

//etc_supported=false;

#endif

	//use_rgba_shadowmaps=true;

	glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &max_texture_units);
	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture_size);
	//read_depth_supported=false;

	canvas_shadow_blur = canvas_light_shadow_buffer_create(max_texture_size);

	{
		//shadowmaps

		//don't use a shadowbuffer too big in GLES, this should be the maximum
		int max_shadow_size = GLOBAL_DEF("rasterizer/max_shadow_buffer_size", 1024);
		int smsize = max_shadow_size;
		while (smsize >= 16) {

			ShadowBuffer sb;
			bool s = sb.init(smsize, !use_rgba_shadowmaps);
			if (s)
				near_shadow_buffers.push_back(sb);
			smsize /= 2;
		}

		blur_shadow_buffer.init(max_shadow_size, !use_rgba_shadowmaps);

		//material_shader
		material_shader.set_conditional(MaterialShaderGLES2::USE_DEPTH_SHADOWS, !use_rgba_shadowmaps);
		canvas_shadow_shader.set_conditional(CanvasShadowShaderGLES2::USE_DEPTH_SHADOWS, !use_rgba_shadowmaps);
	}

	shadow_material = material_create(); //empty with nothing
	shadow_mat_ptr = material_owner.get(shadow_material);

	// Now create a second shadow material for double-sided shadow instances
	shadow_material_double_sided = material_create();
	shadow_mat_double_sided_ptr = material_owner.get(shadow_material_double_sided);
	shadow_mat_double_sided_ptr->flags[VS::MATERIAL_FLAG_DOUBLE_SIDED] = true;

	overdraw_material = create_overdraw_debug_material();
	copy_shader.set_conditional(CopyShaderGLES2::USE_8BIT_HDR, !use_fp16_fb);
	canvas_shader.set_conditional(CanvasShaderGLES2::USE_DEPTH_SHADOWS, read_depth_supported);

	canvas_shader.set_conditional(CanvasShaderGLES2::USE_PIXEL_SNAP, GLOBAL_DEF("display/use_2d_pixel_snap", false));

	npo2_textures_available = true;
	//fragment_lighting=false;
	_rinfo.texture_mem = 0;
	current_env = NULL;
	current_rt = NULL;
	current_vd = NULL;
	current_debug = VS::SCENARIO_DEBUG_DISABLED;
	camera_ortho = false;

	glGenBuffers(1, &gui_quad_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, gui_quad_buffer);
#ifdef GLES_NO_CLIENT_ARRAYS //WebGL specific implementation.
	glBufferData(GL_ARRAY_BUFFER, 8 * MAX_POLYGON_VERTICES, NULL, GL_DYNAMIC_DRAW);
#else
	glBufferData(GL_ARRAY_BUFFER, 128, NULL, GL_DYNAMIC_DRAW);
#endif
	glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind

#ifdef GLES_NO_CLIENT_ARRAYS //webgl indices buffer
	glGenBuffers(1, &indices_buffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices_buffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 16 * 1024, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); // unbind
#endif

	shader_time_rollback = GLOBAL_DEF("rasterizer/shader_time_rollback", 300);

	using_canvas_bg = false;
	_update_framebuffer();
	DEBUG_TEST_ERROR("Initializing");
}

void RasterizerGLES2::finish() {

	free(default_material);
	free(shadow_material);
	free(shadow_material_double_sided);
	free(canvas_shadow_blur);
	free(overdraw_material);
}

int RasterizerGLES2::get_render_info(VS::RenderInfo p_info) {

	switch (p_info) {

		case VS::INFO_OBJECTS_IN_FRAME: {

			return _rinfo.object_count;
		} break;
		case VS::INFO_VERTICES_IN_FRAME: {

			return _rinfo.vertex_count;
		} break;
		case VS::INFO_MATERIAL_CHANGES_IN_FRAME: {

			return _rinfo.mat_change_count;
		} break;
		case VS::INFO_SHADER_CHANGES_IN_FRAME: {

			return _rinfo.shader_change_count;
		} break;
		case VS::INFO_DRAW_CALLS_IN_FRAME: {

			return _rinfo.draw_calls;
		} break;
		case VS::INFO_SURFACE_CHANGES_IN_FRAME: {

			return _rinfo.surface_count;
		} break;
		case VS::INFO_USAGE_VIDEO_MEM_TOTAL: {

			return 0;
		} break;
		case VS::INFO_VIDEO_MEM_USED: {

			return get_render_info(VS::INFO_TEXTURE_MEM_USED) + get_render_info(VS::INFO_VERTEX_MEM_USED);
		} break;
		case VS::INFO_TEXTURE_MEM_USED: {

			return _rinfo.texture_mem;
		} break;
		case VS::INFO_VERTEX_MEM_USED: {

			return 0;
		} break;
	}

	return 0;
}

void RasterizerGLES2::set_extensions(const char *p_strings) {

	Vector<String> strings = String(p_strings).split(" ", false);
	for (int i = 0; i < strings.size(); i++) {

		extensions.insert(strings[i]);
		//print_line(strings[i]);
	}
}

bool RasterizerGLES2::needs_to_draw_next_frame() const {

	return draw_next_frame;
}

bool RasterizerGLES2::has_feature(VS::Features p_feature) const {

	switch (p_feature) {
		case VS::FEATURE_SHADERS: return true;
		case VS::FEATURE_NEEDS_RELOAD_HOOK: return use_reload_hooks;
		default: return false;
	}
}

void RasterizerGLES2::reload_vram() {

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glFrontFace(GL_CW);

	//do a single initial clear
	glClearColor(0, 0, 0, 1);
	//glClearDepth(1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glGenTextures(1, &white_tex);
	unsigned char whitetexdata[8 * 8 * 3];
	for (int i = 0; i < 8 * 8 * 3; i++) {
		whitetexdata[i] = 255;
	}
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, white_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 8, 8, 0, GL_RGB, GL_UNSIGNED_BYTE, whitetexdata);
	glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	List<RID> textures;
	texture_owner.get_owned_list(&textures);
	keep_copies = false;
	for (List<RID>::Element *E = textures.front(); E; E = E->next()) {

		RID tid = E->get();
		Texture *t = texture_owner.get(tid);
		ERR_CONTINUE(!t);
		t->tex_id = 0;
		t->data_size = 0;
		glGenTextures(1, &t->tex_id);
		t->active = false;
		if (t->render_target)
			continue;
		texture_allocate(tid, t->width, t->height, t->format, t->flags);
		bool had_image = false;
		for (int i = 0; i < 6; i++) {
			if (!t->image[i].empty()) {
				texture_set_data(tid, t->image[i], VS::CubeMapSide(i));
				had_image = true;
			}
		}

		if (!had_image && t->reloader) {
			Object *rl = ObjectDB::get_instance(t->reloader);
			if (rl)
				rl->call(t->reloader_func, tid);
		}
	}
	keep_copies = true;

	List<RID> render_targets;
	render_target_owner.get_owned_list(&render_targets);
	for (List<RID>::Element *E = render_targets.front(); E; E = E->next()) {
		RenderTarget *rt = render_target_owner.get(E->get());

		int w = rt->width;
		int h = rt->height;
		rt->width = 0;
		rt->height = 0;
		render_target_set_size(E->get(), w, h);
	}

	List<RID> meshes;
	mesh_owner.get_owned_list(&meshes);
	for (List<RID>::Element *E = meshes.front(); E; E = E->next()) {

		Mesh *mesh = mesh_owner.get(E->get());
		Vector<Surface *> surfaces = mesh->surfaces;
		mesh->surfaces.clear();
		for (int i = 0; i < surfaces.size(); i++) {
			mesh_add_surface(E->get(), surfaces[i]->primitive, surfaces[i]->data, surfaces[i]->morph_data, surfaces[i]->alpha_sort);
			mesh_surface_set_material(E->get(), i, surfaces[i]->material);

			if (surfaces[i]->array_local != 0) {
				memfree(surfaces[i]->array_local);
			};
			if (surfaces[i]->index_array_local != 0) {
				memfree(surfaces[i]->index_array_local);
			};

			memdelete(surfaces[i]);
		}
	}

	List<RID> skeletons;
	skeleton_owner.get_owned_list(&skeletons);
	for (List<RID>::Element *E = skeletons.front(); E; E = E->next()) {

		Skeleton *sk = skeleton_owner.get(E->get());
		if (!sk->tex_id)
			continue; //does not use hw transform, leave alone

		Vector<Skeleton::Bone> bones = sk->bones;
		sk->bones.clear();
		sk->tex_id = 0;
		sk->pixel_size = 1.0;
		skeleton_resize(E->get(), bones.size());
		sk->bones = bones;
	}

	List<RID> multimeshes;
	multimesh_owner.get_owned_list(&multimeshes);
	for (List<RID>::Element *E = multimeshes.front(); E; E = E->next()) {

		MultiMesh *mm = multimesh_owner.get(E->get());
		if (!mm->tex_id)
			continue; //does not use hw transform, leave alone

		Vector<MultiMesh::Element> elements = mm->elements;
		mm->elements.clear();

		mm->tw = 1;
		mm->th = 1;
		mm->tex_id = 0;
		mm->last_pass = 0;
		mm->visible = -1;

		multimesh_set_instance_count(E->get(), elements.size());
		mm->elements = elements;
	}

	if (framebuffer.fbo != 0) {

		framebuffer.fbo = 0;
		framebuffer.depth = 0;
		framebuffer.color = 0;

		for (int i = 0; i < 3; i++) {
			framebuffer.blur[i].fbo = 0;
			framebuffer.blur[i].color = 0;
		}

		framebuffer.luminance.clear();
	}

	for (int i = 0; i < near_shadow_buffers.size(); i++) {
		near_shadow_buffers[i].init(near_shadow_buffers[i].size, !use_rgba_shadowmaps);
	}

	blur_shadow_buffer.init(near_shadow_buffers[0].size, !use_rgba_shadowmaps);

	canvas_shader.clear_caches();
	material_shader.clear_caches();
	blur_shader.clear_caches();
	copy_shader.clear_caches();

	List<RID> shaders;
	shader_owner.get_owned_list(&shaders);
	for (List<RID>::Element *E = shaders.front(); E; E = E->next()) {

		Shader *s = shader_owner.get(E->get());
		s->custom_code_id = 0;
		s->version = 1;
		s->valid = false;
		shader_set_mode(E->get(), s->mode);
	}

	List<RID> materials;
	material_owner.get_owned_list(&materials);
	for (List<RID>::Element *E = materials.front(); E; E = E->next()) {

		Material *m = material_owner.get(E->get());
		RID shader = m->shader;
		m->shader_version = 0;
		material_set_shader(E->get(), shader);
	}
}

void RasterizerGLES2::set_use_framebuffers(bool p_use) {

	use_framebuffers = p_use;
}

RasterizerGLES2 *RasterizerGLES2::get_singleton() {

	return _singleton;
};

int RasterizerGLES2::RenderList::max_elements = RenderList::DEFAULT_MAX_ELEMENTS;

void RasterizerGLES2::set_force_16_bits_fbo(bool p_force) {

	use_16bits_fbo = p_force;
}

RasterizerGLES2::RasterizerGLES2(bool p_compress_arrays, bool p_keep_ram_copy, bool p_default_fragment_lighting, bool p_use_reload_hooks) {

	_singleton = this;
	shrink_textures_x2 = false;
	RenderList::max_elements = GLOBAL_DEF("rasterizer/max_render_elements", (int)RenderList::DEFAULT_MAX_ELEMENTS);
	if (RenderList::max_elements > 64000)
		RenderList::max_elements = 64000;
	if (RenderList::max_elements < 1024)
		RenderList::max_elements = 1024;

	opaque_render_list.init();
	alpha_render_list.init();

	skinned_buffer_size = GLOBAL_DEF("rasterizer/skeleton_buffer_size_kb", DEFAULT_SKINNED_BUFFER_SIZE);
	if (skinned_buffer_size < 256)
		skinned_buffer_size = 256;
	if (skinned_buffer_size > 16384)
		skinned_buffer_size = 16384;
	skinned_buffer_size *= 1024;
	skinned_buffer = memnew_arr(uint8_t, skinned_buffer_size);

	keep_copies = p_keep_ram_copy;
	use_reload_hooks = p_use_reload_hooks;
	pack_arrays = p_compress_arrays;
	p_default_fragment_lighting = false;
	fragment_lighting = GLOBAL_DEF("rasterizer/use_fragment_lighting", true);
	read_depth_supported = true; //todo check for extension
	shadow_filter = ShadowFilterTechnique((int)(GLOBAL_DEF("rasterizer/shadow_filter", SHADOW_FILTER_PCF5)));
	GlobalConfig::get_singleton()->set_custom_property_info("rasterizer/shadow_filter", PropertyInfo(Variant::INT, "rasterizer/shadow_filter", PROPERTY_HINT_ENUM, "None,PCF5,PCF13,ESM"));
	use_fp16_fb = bool(GLOBAL_DEF("rasterizer/fp16_framebuffer", true));
	use_shadow_mapping = true;
	use_fast_texture_filter = !bool(GLOBAL_DEF("rasterizer/trilinear_mipmap_filter", true));
	low_memory_2d = bool(GLOBAL_DEF("rasterizer/low_memory_2d_mode", false));
	skel_default.resize(1024 * 4);
	for (int i = 0; i < 1024 / 3; i++) {

		float *ptr = skel_default.ptr();
		ptr += i * 4 * 4;
		ptr[0] = 1.0;
		ptr[1] = 0.0;
		ptr[2] = 0.0;
		ptr[3] = 0.0;

		ptr[4] = 0.0;
		ptr[5] = 1.0;
		ptr[6] = 0.0;
		ptr[7] = 0.0;

		ptr[8] = 0.0;
		ptr[9] = 0.0;
		ptr[10] = 1.0;
		ptr[12] = 0.0;
	}

	base_framebuffer = 0;
	frame = 0;
	draw_next_frame = false;
	use_framebuffers = true;
	framebuffer.active = false;
	tc0_id_cache = 0;
	tc0_idx = 0;
	use_16bits_fbo = false;
};

void RasterizerGLES2::restore_framebuffer() {

	glBindFramebuffer(GL_FRAMEBUFFER, base_framebuffer);
}

RasterizerGLES2::~RasterizerGLES2() {

	memdelete_arr(skinned_buffer);
};

#endif
