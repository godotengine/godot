/*************************************************************************/
/*  material_storage.cpp                                                 */
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

#include "core/config/project_settings.h"

#include "config.h"
#include "material_storage.h"
#include "texture_storage.h"

#include "drivers/gles3/rasterizer_canvas_gles3.h"

using namespace GLES3;

///////////////////////////////////////////////////////////////////////////
// UBI helper functions

_FORCE_INLINE_ static void _fill_std140_variant_ubo_value(ShaderLanguage::DataType type, int p_array_size, const Variant &value, uint8_t *data) {
	switch (type) {
		case ShaderLanguage::TYPE_BOOL: {
			uint32_t *gui = (uint32_t *)data;

			if (p_array_size > 0) {
				const PackedInt32Array &ba = value;
				int s = ba.size();
				const int *r = ba.ptr();

				for (int i = 0, j = 0; i < p_array_size; i++, j += 4) {
					if (i < s) {
						gui[j] = (r[i] != 0) ? 1 : 0;
					} else {
						gui[j] = 0;
					}
					gui[j + 1] = 0; // ignored
					gui[j + 2] = 0; // ignored
					gui[j + 3] = 0; // ignored
				}
			} else {
				bool v = value;
				gui[0] = v ? 1 : 0;
			}
		} break;
		case ShaderLanguage::TYPE_BVEC2: {
			uint32_t *gui = (uint32_t *)data;

			if (p_array_size > 0) {
				const PackedInt32Array &ba = value;
				int s = ba.size();
				const int *r = ba.ptr();
				int count = 2 * p_array_size;

				for (int i = 0, j = 0; i < count; i += 2, j += 4) {
					if (i < s) {
						gui[j] = r[i] ? 1 : 0;
						gui[j + 1] = r[i + 1] ? 1 : 0;
					} else {
						gui[j] = 0;
						gui[j + 1] = 0;
					}
					gui[j + 2] = 0; // ignored
					gui[j + 3] = 0; // ignored
				}
			} else {
				int v = value;
				gui[0] = v & 1 ? 1 : 0;
				gui[1] = v & 2 ? 1 : 0;
			}
		} break;
		case ShaderLanguage::TYPE_BVEC3: {
			uint32_t *gui = (uint32_t *)data;

			if (p_array_size > 0) {
				const PackedInt32Array &ba = value;
				int s = ba.size();
				const int *r = ba.ptr();
				int count = 3 * p_array_size;

				for (int i = 0, j = 0; i < count; i += 3, j += 4) {
					if (i < s) {
						gui[j] = r[i] ? 1 : 0;
						gui[j + 1] = r[i + 1] ? 1 : 0;
						gui[j + 2] = r[i + 2] ? 1 : 0;
					} else {
						gui[j] = 0;
						gui[j + 1] = 0;
						gui[j + 2] = 0;
					}
					gui[j + 3] = 0; // ignored
				}
			} else {
				int v = value;
				gui[0] = (v & 1) ? 1 : 0;
				gui[1] = (v & 2) ? 1 : 0;
				gui[2] = (v & 4) ? 1 : 0;
			}
		} break;
		case ShaderLanguage::TYPE_BVEC4: {
			uint32_t *gui = (uint32_t *)data;

			if (p_array_size > 0) {
				const PackedInt32Array &ba = value;
				int s = ba.size();
				const int *r = ba.ptr();
				int count = 4 * p_array_size;

				for (int i = 0; i < count; i += 4) {
					if (i < s) {
						gui[i] = r[i] ? 1 : 0;
						gui[i + 1] = r[i + 1] ? 1 : 0;
						gui[i + 2] = r[i + 2] ? 1 : 0;
						gui[i + 3] = r[i + 3] ? 1 : 0;
					} else {
						gui[i] = 0;
						gui[i + 1] = 0;
						gui[i + 2] = 0;
						gui[i + 3] = 0;
					}
				}
			} else {
				int v = value;
				gui[0] = (v & 1) ? 1 : 0;
				gui[1] = (v & 2) ? 1 : 0;
				gui[2] = (v & 4) ? 1 : 0;
				gui[3] = (v & 8) ? 1 : 0;
			}
		} break;
		case ShaderLanguage::TYPE_INT: {
			int32_t *gui = (int32_t *)data;

			if (p_array_size > 0) {
				Vector<int> iv = value;
				int s = iv.size();
				const int *r = iv.ptr();

				for (int i = 0, j = 0; i < p_array_size; i++, j += 4) {
					if (i < s) {
						gui[j] = r[i];
					} else {
						gui[j] = 0;
					}
					gui[j + 1] = 0; // ignored
					gui[j + 2] = 0; // ignored
					gui[j + 3] = 0; // ignored
				}
			} else {
				int v = value;
				gui[0] = v;
			}
		} break;
		case ShaderLanguage::TYPE_IVEC2: {
			Vector<int> iv = value;
			int s = iv.size();
			int32_t *gui = (int32_t *)data;

			if (p_array_size <= 0) {
				p_array_size = 1;
			}
			int count = 2 * p_array_size;

			const int *r = iv.ptr();
			for (int i = 0, j = 0; i < count; i += 2, j += 4) {
				if (i < s) {
					gui[j] = r[i];
					gui[j + 1] = r[i + 1];
				} else {
					gui[j] = 0;
					gui[j + 1] = 0;
				}
				gui[j + 2] = 0; // ignored
				gui[j + 3] = 0; // ignored
			}
		} break;
		case ShaderLanguage::TYPE_IVEC3: {
			Vector<int> iv = value;
			int s = iv.size();
			int32_t *gui = (int32_t *)data;

			if (p_array_size <= 0) {
				p_array_size = 1;
			}
			int count = 3 * p_array_size;

			const int *r = iv.ptr();
			for (int i = 0, j = 0; i < count; i += 3, j += 4) {
				if (i < s) {
					gui[j] = r[i];
					gui[j + 1] = r[i + 1];
					gui[j + 2] = r[i + 2];
				} else {
					gui[j] = 0;
					gui[j + 1] = 0;
					gui[j + 2] = 0;
				}
				gui[j + 3] = 0; // ignored
			}
		} break;
		case ShaderLanguage::TYPE_IVEC4: {
			Vector<int> iv = value;
			int s = iv.size();
			int32_t *gui = (int32_t *)data;

			if (p_array_size <= 0) {
				p_array_size = 1;
			}
			int count = 4 * p_array_size;

			const int *r = iv.ptr();
			for (int i = 0; i < count; i += 4) {
				if (i < s) {
					gui[i] = r[i];
					gui[i + 1] = r[i + 1];
					gui[i + 2] = r[i + 2];
					gui[i + 3] = r[i + 3];
				} else {
					gui[i] = 0;
					gui[i + 1] = 0;
					gui[i + 2] = 0;
					gui[i + 3] = 0;
				}
			}
		} break;
		case ShaderLanguage::TYPE_UINT: {
			uint32_t *gui = (uint32_t *)data;

			if (p_array_size > 0) {
				Vector<int> iv = value;
				int s = iv.size();
				const int *r = iv.ptr();

				for (int i = 0, j = 0; i < p_array_size; i++, j += 4) {
					if (i < s) {
						gui[j] = r[i];
					} else {
						gui[j] = 0;
					}
					gui[j + 1] = 0; // ignored
					gui[j + 2] = 0; // ignored
					gui[j + 3] = 0; // ignored
				}
			} else {
				int v = value;
				gui[0] = v;
			}
		} break;
		case ShaderLanguage::TYPE_UVEC2: {
			Vector<int> iv = value;
			int s = iv.size();
			uint32_t *gui = (uint32_t *)data;

			if (p_array_size <= 0) {
				p_array_size = 1;
			}
			int count = 2 * p_array_size;

			const int *r = iv.ptr();
			for (int i = 0, j = 0; i < count; i += 2, j += 4) {
				if (i < s) {
					gui[j] = r[i];
					gui[j + 1] = r[i + 1];
				} else {
					gui[j] = 0;
					gui[j + 1] = 0;
				}
				gui[j + 2] = 0; // ignored
				gui[j + 3] = 0; // ignored
			}
		} break;
		case ShaderLanguage::TYPE_UVEC3: {
			Vector<int> iv = value;
			int s = iv.size();
			uint32_t *gui = (uint32_t *)data;

			if (p_array_size <= 0) {
				p_array_size = 1;
			}
			int count = 3 * p_array_size;

			const int *r = iv.ptr();
			for (int i = 0, j = 0; i < count; i += 3, j += 4) {
				if (i < s) {
					gui[j] = r[i];
					gui[j + 1] = r[i + 1];
					gui[j + 2] = r[i + 2];
				} else {
					gui[j] = 0;
					gui[j + 1] = 0;
					gui[j + 2] = 0;
				}
				gui[j + 3] = 0; // ignored
			}
		} break;
		case ShaderLanguage::TYPE_UVEC4: {
			Vector<int> iv = value;
			int s = iv.size();
			uint32_t *gui = (uint32_t *)data;

			if (p_array_size <= 0) {
				p_array_size = 1;
			}
			int count = 4 * p_array_size;

			const int *r = iv.ptr();
			for (int i = 0; i < count; i++) {
				if (i < s) {
					gui[i] = r[i];
					gui[i + 1] = r[i + 1];
					gui[i + 2] = r[i + 2];
					gui[i + 3] = r[i + 3];
				} else {
					gui[i] = 0;
					gui[i + 1] = 0;
					gui[i + 2] = 0;
					gui[i + 3] = 0;
				}
			}
		} break;
		case ShaderLanguage::TYPE_FLOAT: {
			float *gui = (float *)data;

			if (p_array_size > 0) {
				const PackedFloat32Array &a = value;
				int s = a.size();

				for (int i = 0, j = 0; i < p_array_size; i++, j += 4) {
					if (i < s) {
						gui[j] = a[i];
					} else {
						gui[j] = 0;
					}
					gui[j + 1] = 0; // ignored
					gui[j + 2] = 0; // ignored
					gui[j + 3] = 0; // ignored
				}
			} else {
				float v = value;
				gui[0] = v;
			}
		} break;
		case ShaderLanguage::TYPE_VEC2: {
			float *gui = (float *)data;

			if (p_array_size > 0) {
				const PackedVector2Array &a = value;
				int s = a.size();

				for (int i = 0, j = 0; i < p_array_size; i++, j += 4) {
					if (i < s) {
						gui[j] = a[i].x;
						gui[j + 1] = a[i].y;
					} else {
						gui[j] = 0;
						gui[j + 1] = 0;
					}
					gui[j + 2] = 0; // ignored
					gui[j + 3] = 0; // ignored
				}
			} else {
				Vector2 v = value;
				gui[0] = v.x;
				gui[1] = v.y;
			}
		} break;
		case ShaderLanguage::TYPE_VEC3: {
			float *gui = (float *)data;

			if (p_array_size > 0) {
				if (value.get_type() == Variant::PACKED_COLOR_ARRAY) {
					const PackedColorArray &a = value;
					int s = a.size();

					for (int i = 0, j = 0; i < p_array_size; i++, j += 4) {
						if (i < s) {
							Color color = a[i];
							gui[j] = color.r;
							gui[j + 1] = color.g;
							gui[j + 2] = color.b;
						} else {
							gui[j] = 0;
							gui[j + 1] = 0;
							gui[j + 2] = 0;
						}
						gui[j + 3] = 0; // ignored
					}
				} else {
					const PackedVector3Array &a = value;
					int s = a.size();

					for (int i = 0, j = 0; i < p_array_size; i++, j += 4) {
						if (i < s) {
							gui[j] = a[i].x;
							gui[j + 1] = a[i].y;
							gui[j + 2] = a[i].z;
						} else {
							gui[j] = 0;
							gui[j + 1] = 0;
							gui[j + 2] = 0;
						}
						gui[j + 3] = 0; // ignored
					}
				}
			} else {
				if (value.get_type() == Variant::COLOR) {
					Color v = value;

					gui[0] = v.r;
					gui[1] = v.g;
					gui[2] = v.b;
				} else {
					Vector3 v = value;
					gui[0] = v.x;
					gui[1] = v.y;
					gui[2] = v.z;
				}
			}
		} break;
		case ShaderLanguage::TYPE_VEC4: {
			float *gui = (float *)data;

			if (p_array_size > 0) {
				if (value.get_type() == Variant::PACKED_COLOR_ARRAY) {
					const PackedColorArray &a = value;
					int s = a.size();

					for (int i = 0, j = 0; i < p_array_size; i++, j += 4) {
						if (i < s) {
							Color color = a[i];
							gui[j] = color.r;
							gui[j + 1] = color.g;
							gui[j + 2] = color.b;
							gui[j + 3] = color.a;
						} else {
							gui[j] = 0;
							gui[j + 1] = 0;
							gui[j + 2] = 0;
							gui[j + 3] = 0;
						}
					}
				} else {
					const PackedFloat32Array &a = value;
					int s = a.size();
					int count = 4 * p_array_size;

					for (int i = 0; i < count; i += 4) {
						if (i + 3 < s) {
							gui[i] = a[i];
							gui[i + 1] = a[i + 1];
							gui[i + 2] = a[i + 2];
							gui[i + 3] = a[i + 3];
						} else {
							gui[i] = 0;
							gui[i + 1] = 0;
							gui[i + 2] = 0;
							gui[i + 3] = 0;
						}
					}
				}
			} else {
				if (value.get_type() == Variant::COLOR) {
					Color v = value;

					gui[0] = v.r;
					gui[1] = v.g;
					gui[2] = v.b;
					gui[3] = v.a;
				} else if (value.get_type() == Variant::RECT2) {
					Rect2 v = value;

					gui[0] = v.position.x;
					gui[1] = v.position.y;
					gui[2] = v.size.x;
					gui[3] = v.size.y;
				} else if (value.get_type() == Variant::QUATERNION) {
					Quaternion v = value;

					gui[0] = v.x;
					gui[1] = v.y;
					gui[2] = v.z;
					gui[3] = v.w;
				} else {
					Plane v = value;

					gui[0] = v.normal.x;
					gui[1] = v.normal.y;
					gui[2] = v.normal.z;
					gui[3] = v.d;
				}
			}
		} break;
		case ShaderLanguage::TYPE_MAT2: {
			float *gui = (float *)data;

			if (p_array_size > 0) {
				const PackedFloat32Array &a = value;
				int s = a.size();

				for (int i = 0, j = 0; i < p_array_size * 4; i += 4, j += 8) {
					if (i + 3 < s) {
						gui[j] = a[i];
						gui[j + 1] = a[i + 1];

						gui[j + 4] = a[i + 2];
						gui[j + 5] = a[i + 3];
					} else {
						gui[j] = 1;
						gui[j + 1] = 0;

						gui[j + 4] = 0;
						gui[j + 5] = 1;
					}
					gui[j + 2] = 0; // ignored
					gui[j + 3] = 0; // ignored
					gui[j + 6] = 0; // ignored
					gui[j + 7] = 0; // ignored
				}
			} else {
				Transform2D v = value;

				//in std140 members of mat2 are treated as vec4s
				gui[0] = v.columns[0][0];
				gui[1] = v.columns[0][1];
				gui[2] = 0; // ignored
				gui[3] = 0; // ignored

				gui[4] = v.columns[1][0];
				gui[5] = v.columns[1][1];
				gui[6] = 0; // ignored
				gui[7] = 0; // ignored
			}
		} break;
		case ShaderLanguage::TYPE_MAT3: {
			float *gui = (float *)data;

			if (p_array_size > 0) {
				const PackedFloat32Array &a = value;
				int s = a.size();

				for (int i = 0, j = 0; i < p_array_size * 9; i += 9, j += 12) {
					if (i + 8 < s) {
						gui[j] = a[i];
						gui[j + 1] = a[i + 1];
						gui[j + 2] = a[i + 2];

						gui[j + 4] = a[i + 3];
						gui[j + 5] = a[i + 4];
						gui[j + 6] = a[i + 5];

						gui[j + 8] = a[i + 6];
						gui[j + 9] = a[i + 7];
						gui[j + 10] = a[i + 8];
					} else {
						gui[j] = 1;
						gui[j + 1] = 0;
						gui[j + 2] = 0;

						gui[j + 4] = 0;
						gui[j + 5] = 1;
						gui[j + 6] = 0;

						gui[j + 8] = 0;
						gui[j + 9] = 0;
						gui[j + 10] = 1;
					}
					gui[j + 3] = 0; // ignored
					gui[j + 7] = 0; // ignored
					gui[j + 11] = 0; // ignored
				}
			} else {
				Basis v = value;
				gui[0] = v.rows[0][0];
				gui[1] = v.rows[1][0];
				gui[2] = v.rows[2][0];
				gui[3] = 0; // ignored

				gui[4] = v.rows[0][1];
				gui[5] = v.rows[1][1];
				gui[6] = v.rows[2][1];
				gui[7] = 0; // ignored

				gui[8] = v.rows[0][2];
				gui[9] = v.rows[1][2];
				gui[10] = v.rows[2][2];
				gui[11] = 0; // ignored
			}
		} break;
		case ShaderLanguage::TYPE_MAT4: {
			float *gui = (float *)data;

			if (p_array_size > 0) {
				const PackedFloat32Array &a = value;
				int s = a.size();

				for (int i = 0; i < p_array_size * 16; i += 16) {
					if (i + 15 < s) {
						gui[i] = a[i];
						gui[i + 1] = a[i + 1];
						gui[i + 2] = a[i + 2];
						gui[i + 3] = a[i + 3];

						gui[i + 4] = a[i + 4];
						gui[i + 5] = a[i + 5];
						gui[i + 6] = a[i + 6];
						gui[i + 7] = a[i + 7];

						gui[i + 8] = a[i + 8];
						gui[i + 9] = a[i + 9];
						gui[i + 10] = a[i + 10];
						gui[i + 11] = a[i + 11];

						gui[i + 12] = a[i + 12];
						gui[i + 13] = a[i + 13];
						gui[i + 14] = a[i + 14];
						gui[i + 15] = a[i + 15];
					} else {
						gui[i] = 1;
						gui[i + 1] = 0;
						gui[i + 2] = 0;
						gui[i + 3] = 0;

						gui[i + 4] = 0;
						gui[i + 5] = 1;
						gui[i + 6] = 0;
						gui[i + 7] = 0;

						gui[i + 8] = 0;
						gui[i + 9] = 0;
						gui[i + 10] = 1;
						gui[i + 11] = 0;

						gui[i + 12] = 0;
						gui[i + 13] = 0;
						gui[i + 14] = 0;
						gui[i + 15] = 1;
					}
				}
			} else {
				Transform3D v = value;
				gui[0] = v.basis.rows[0][0];
				gui[1] = v.basis.rows[1][0];
				gui[2] = v.basis.rows[2][0];
				gui[3] = 0;

				gui[4] = v.basis.rows[0][1];
				gui[5] = v.basis.rows[1][1];
				gui[6] = v.basis.rows[2][1];
				gui[7] = 0;

				gui[8] = v.basis.rows[0][2];
				gui[9] = v.basis.rows[1][2];
				gui[10] = v.basis.rows[2][2];
				gui[11] = 0;

				gui[12] = v.origin.x;
				gui[13] = v.origin.y;
				gui[14] = v.origin.z;
				gui[15] = 1;
			}
		} break;
		default: {
		}
	}
}

_FORCE_INLINE_ static void _fill_std140_ubo_value(ShaderLanguage::DataType type, const Vector<ShaderLanguage::ConstantNode::Value> &value, uint8_t *data) {
	switch (type) {
		case ShaderLanguage::TYPE_BOOL: {
			uint32_t *gui = (uint32_t *)data;
			*gui = value[0].boolean ? 1 : 0;
		} break;
		case ShaderLanguage::TYPE_BVEC2: {
			uint32_t *gui = (uint32_t *)data;
			gui[0] = value[0].boolean ? 1 : 0;
			gui[1] = value[1].boolean ? 1 : 0;

		} break;
		case ShaderLanguage::TYPE_BVEC3: {
			uint32_t *gui = (uint32_t *)data;
			gui[0] = value[0].boolean ? 1 : 0;
			gui[1] = value[1].boolean ? 1 : 0;
			gui[2] = value[2].boolean ? 1 : 0;

		} break;
		case ShaderLanguage::TYPE_BVEC4: {
			uint32_t *gui = (uint32_t *)data;
			gui[0] = value[0].boolean ? 1 : 0;
			gui[1] = value[1].boolean ? 1 : 0;
			gui[2] = value[2].boolean ? 1 : 0;
			gui[3] = value[3].boolean ? 1 : 0;

		} break;
		case ShaderLanguage::TYPE_INT: {
			int32_t *gui = (int32_t *)data;
			gui[0] = value[0].sint;

		} break;
		case ShaderLanguage::TYPE_IVEC2: {
			int32_t *gui = (int32_t *)data;

			for (int i = 0; i < 2; i++) {
				gui[i] = value[i].sint;
			}

		} break;
		case ShaderLanguage::TYPE_IVEC3: {
			int32_t *gui = (int32_t *)data;

			for (int i = 0; i < 3; i++) {
				gui[i] = value[i].sint;
			}

		} break;
		case ShaderLanguage::TYPE_IVEC4: {
			int32_t *gui = (int32_t *)data;

			for (int i = 0; i < 4; i++) {
				gui[i] = value[i].sint;
			}

		} break;
		case ShaderLanguage::TYPE_UINT: {
			uint32_t *gui = (uint32_t *)data;
			gui[0] = value[0].uint;

		} break;
		case ShaderLanguage::TYPE_UVEC2: {
			int32_t *gui = (int32_t *)data;

			for (int i = 0; i < 2; i++) {
				gui[i] = value[i].uint;
			}
		} break;
		case ShaderLanguage::TYPE_UVEC3: {
			int32_t *gui = (int32_t *)data;

			for (int i = 0; i < 3; i++) {
				gui[i] = value[i].uint;
			}

		} break;
		case ShaderLanguage::TYPE_UVEC4: {
			int32_t *gui = (int32_t *)data;

			for (int i = 0; i < 4; i++) {
				gui[i] = value[i].uint;
			}
		} break;
		case ShaderLanguage::TYPE_FLOAT: {
			float *gui = (float *)data;
			gui[0] = value[0].real;

		} break;
		case ShaderLanguage::TYPE_VEC2: {
			float *gui = (float *)data;

			for (int i = 0; i < 2; i++) {
				gui[i] = value[i].real;
			}

		} break;
		case ShaderLanguage::TYPE_VEC3: {
			float *gui = (float *)data;

			for (int i = 0; i < 3; i++) {
				gui[i] = value[i].real;
			}

		} break;
		case ShaderLanguage::TYPE_VEC4: {
			float *gui = (float *)data;

			for (int i = 0; i < 4; i++) {
				gui[i] = value[i].real;
			}
		} break;
		case ShaderLanguage::TYPE_MAT2: {
			float *gui = (float *)data;

			//in std140 members of mat2 are treated as vec4s
			gui[0] = value[0].real;
			gui[1] = value[1].real;
			gui[2] = 0;
			gui[3] = 0;
			gui[4] = value[2].real;
			gui[5] = value[3].real;
			gui[6] = 0;
			gui[7] = 0;
		} break;
		case ShaderLanguage::TYPE_MAT3: {
			float *gui = (float *)data;

			gui[0] = value[0].real;
			gui[1] = value[1].real;
			gui[2] = value[2].real;
			gui[3] = 0;
			gui[4] = value[3].real;
			gui[5] = value[4].real;
			gui[6] = value[5].real;
			gui[7] = 0;
			gui[8] = value[6].real;
			gui[9] = value[7].real;
			gui[10] = value[8].real;
			gui[11] = 0;
		} break;
		case ShaderLanguage::TYPE_MAT4: {
			float *gui = (float *)data;

			for (int i = 0; i < 16; i++) {
				gui[i] = value[i].real;
			}
		} break;
		default: {
		}
	}
}

_FORCE_INLINE_ static void _fill_std140_ubo_empty(ShaderLanguage::DataType type, int p_array_size, uint8_t *data) {
	if (p_array_size <= 0) {
		p_array_size = 1;
	}

	switch (type) {
		case ShaderLanguage::TYPE_BOOL:
		case ShaderLanguage::TYPE_INT:
		case ShaderLanguage::TYPE_UINT:
		case ShaderLanguage::TYPE_FLOAT: {
			memset(data, 0, 4 * p_array_size);
		} break;
		case ShaderLanguage::TYPE_BVEC2:
		case ShaderLanguage::TYPE_IVEC2:
		case ShaderLanguage::TYPE_UVEC2:
		case ShaderLanguage::TYPE_VEC2: {
			memset(data, 0, 8 * p_array_size);
		} break;
		case ShaderLanguage::TYPE_BVEC3:
		case ShaderLanguage::TYPE_IVEC3:
		case ShaderLanguage::TYPE_UVEC3:
		case ShaderLanguage::TYPE_VEC3:
		case ShaderLanguage::TYPE_BVEC4:
		case ShaderLanguage::TYPE_IVEC4:
		case ShaderLanguage::TYPE_UVEC4:
		case ShaderLanguage::TYPE_VEC4: {
			memset(data, 0, 16 * p_array_size);
		} break;
		case ShaderLanguage::TYPE_MAT2: {
			memset(data, 0, 32 * p_array_size);
		} break;
		case ShaderLanguage::TYPE_MAT3: {
			memset(data, 0, 48 * p_array_size);
		} break;
		case ShaderLanguage::TYPE_MAT4: {
			memset(data, 0, 64 * p_array_size);
		} break;

		default: {
		}
	}
}

///////////////////////////////////////////////////////////////////////////
// MaterialData

// Look up table to translate ShaderLanguage::DataType to GL_TEXTURE_*
static const GLenum target_from_type[ShaderLanguage::TYPE_MAX] = {
	GL_TEXTURE_2D, // TYPE_VOID,
	GL_TEXTURE_2D, // TYPE_BOOL,
	GL_TEXTURE_2D, // TYPE_BVEC2,
	GL_TEXTURE_2D, // TYPE_BVEC3,
	GL_TEXTURE_2D, // TYPE_BVEC4,
	GL_TEXTURE_2D, // TYPE_INT,
	GL_TEXTURE_2D, // TYPE_IVEC2,
	GL_TEXTURE_2D, // TYPE_IVEC3,
	GL_TEXTURE_2D, // TYPE_IVEC4,
	GL_TEXTURE_2D, // TYPE_UINT,
	GL_TEXTURE_2D, // TYPE_UVEC2,
	GL_TEXTURE_2D, // TYPE_UVEC3,
	GL_TEXTURE_2D, // TYPE_UVEC4,
	GL_TEXTURE_2D, // TYPE_FLOAT,
	GL_TEXTURE_2D, // TYPE_VEC2,
	GL_TEXTURE_2D, // TYPE_VEC3,
	GL_TEXTURE_2D, // TYPE_VEC4,
	GL_TEXTURE_2D, // TYPE_MAT2,
	GL_TEXTURE_2D, // TYPE_MAT3,
	GL_TEXTURE_2D, // TYPE_MAT4,
	GL_TEXTURE_2D, // TYPE_SAMPLER2D,
	GL_TEXTURE_2D, // TYPE_ISAMPLER2D,
	GL_TEXTURE_2D, // TYPE_USAMPLER2D,
	GL_TEXTURE_2D_ARRAY, // TYPE_SAMPLER2DARRAY,
	GL_TEXTURE_2D_ARRAY, // TYPE_ISAMPLER2DARRAY,
	GL_TEXTURE_2D_ARRAY, // TYPE_USAMPLER2DARRAY,
	GL_TEXTURE_3D, // TYPE_SAMPLER3D,
	GL_TEXTURE_3D, // TYPE_ISAMPLER3D,
	GL_TEXTURE_3D, // TYPE_USAMPLER3D,
	GL_TEXTURE_CUBE_MAP, // TYPE_SAMPLERCUBE,
	GL_TEXTURE_CUBE_MAP, // TYPE_SAMPLERCUBEARRAY,
	GL_TEXTURE_2D, // TYPE_STRUCT
};

void MaterialData::update_uniform_buffer(const Map<StringName, ShaderLanguage::ShaderNode::Uniform> &p_uniforms, const uint32_t *p_uniform_offsets, const Map<StringName, Variant> &p_parameters, uint8_t *p_buffer, uint32_t p_buffer_size, bool p_use_linear_color) {
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	bool uses_global_buffer = false;

	for (const KeyValue<StringName, ShaderLanguage::ShaderNode::Uniform> &E : p_uniforms) {
		if (E.value.order < 0) {
			continue; // texture, does not go here
		}

		if (E.value.scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			continue; //instance uniforms don't appear in the buffer
		}

		if (E.value.scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_GLOBAL) {
			//this is a global variable, get the index to it
			GlobalVariables::Variable *gv = material_storage->global_variables.variables.getptr(E.key);
			uint32_t index = 0;
			if (gv) {
				index = gv->buffer_index;
			} else {
				WARN_PRINT("Shader uses global uniform '" + E.key + "', but it was removed at some point. Material will not display correctly.");
			}

			uint32_t offset = p_uniform_offsets[E.value.order];
			uint32_t *intptr = (uint32_t *)&p_buffer[offset];
			*intptr = index;
			uses_global_buffer = true;
			continue;
		}

		//regular uniform
		uint32_t offset = p_uniform_offsets[E.value.order];
#ifdef DEBUG_ENABLED
		uint32_t size = 0U;
		// The following code enforces a 16-byte alignment of uniform arrays.
		if (E.value.array_size > 0) {
			size = ShaderLanguage::get_datatype_size(E.value.type) * E.value.array_size;
			int m = (16 * E.value.array_size);
			if ((size % m) != 0U) {
				size += m - (size % m);
			}
		} else {
			size = ShaderLanguage::get_datatype_size(E.value.type);
		}
		ERR_CONTINUE(offset + size > p_buffer_size);
#endif
		uint8_t *data = &p_buffer[offset];
		const Map<StringName, Variant>::Element *V = p_parameters.find(E.key);

		if (V) {
			//user provided
			_fill_std140_variant_ubo_value(E.value.type, E.value.array_size, V->get(), data);

		} else if (E.value.default_value.size()) {
			//default value
			_fill_std140_ubo_value(E.value.type, E.value.default_value, data);
			//value=E.value.default_value;
		} else {
			//zero because it was not provided
			if ((E.value.type == ShaderLanguage::TYPE_VEC3 || E.value.type == ShaderLanguage::TYPE_VEC4) && E.value.hint == ShaderLanguage::ShaderNode::Uniform::HINT_COLOR) {
				//colors must be set as black, with alpha as 1.0
				_fill_std140_variant_ubo_value(E.value.type, E.value.array_size, Color(0, 0, 0, 1), data);
			} else {
				//else just zero it out
				_fill_std140_ubo_empty(E.value.type, E.value.array_size, data);
			}
		}
	}

	if (uses_global_buffer != (global_buffer_E != nullptr)) {
		if (uses_global_buffer) {
			global_buffer_E = material_storage->global_variables.materials_using_buffer.push_back(self);
		} else {
			material_storage->global_variables.materials_using_buffer.erase(global_buffer_E);
			global_buffer_E = nullptr;
		}
	}
}

MaterialData::~MaterialData() {
	MaterialStorage *material_storage = MaterialStorage::get_singleton();

	if (global_buffer_E) {
		//unregister global buffers
		material_storage->global_variables.materials_using_buffer.erase(global_buffer_E);
	}

	if (global_texture_E) {
		//unregister global textures

		for (const KeyValue<StringName, uint64_t> &E : used_global_textures) {
			GlobalVariables::Variable *v = material_storage->global_variables.variables.getptr(E.key);
			if (v) {
				v->texture_materials.erase(self);
			}
		}
		//unregister material from those using global textures
		material_storage->global_variables.materials_using_texture.erase(global_texture_E);
	}

	if (uniform_buffer) {
		glDeleteBuffers(1, &uniform_buffer);
	}
}

void MaterialData::update_textures(const Map<StringName, Variant> &p_parameters, const Map<StringName, Map<int, RID>> &p_default_textures, const Vector<ShaderCompiler::GeneratedCode::Texture> &p_texture_uniforms, RID *p_textures, bool p_use_linear_color) {
	TextureStorage *texture_storage = TextureStorage::get_singleton();
	MaterialStorage *material_storage = MaterialStorage::get_singleton();

#ifdef TOOLS_ENABLED
	Texture *roughness_detect_texture = nullptr;
	RS::TextureDetectRoughnessChannel roughness_channel = RS::TEXTURE_DETECT_ROUGHNESS_R;
	Texture *normal_detect_texture = nullptr;
#endif

	bool uses_global_textures = false;
	global_textures_pass++;

	for (int i = 0, k = 0; i < p_texture_uniforms.size(); i++) {
		const StringName &uniform_name = p_texture_uniforms[i].name;
		int uniform_array_size = p_texture_uniforms[i].array_size;

		Vector<RID> textures;

		if (p_texture_uniforms[i].global) {
			uses_global_textures = true;

			GlobalVariables::Variable *v = material_storage->global_variables.variables.getptr(uniform_name);
			if (v) {
				if (v->buffer_index >= 0) {
					WARN_PRINT("Shader uses global uniform texture '" + String(uniform_name) + "', but it changed type and is no longer a texture!.");

				} else {
					Map<StringName, uint64_t>::Element *E = used_global_textures.find(uniform_name);
					if (!E) {
						E = used_global_textures.insert(uniform_name, global_textures_pass);
						v->texture_materials.insert(self);
					} else {
						E->get() = global_textures_pass;
					}

					textures.push_back(v->override.get_type() != Variant::NIL ? v->override : v->value);
				}

			} else {
				WARN_PRINT("Shader uses global uniform texture '" + String(uniform_name) + "', but it was removed at some point. Material will not display correctly.");
			}
		} else {
			const Map<StringName, Variant>::Element *V = p_parameters.find(uniform_name);
			if (V) {
				if (V->get().is_array()) {
					Array array = (Array)V->get();
					if (uniform_array_size > 0) {
						for (int j = 0; j < array.size(); j++) {
							textures.push_back(array[j]);
						}
					} else {
						if (array.size() > 0) {
							textures.push_back(array[0]);
						}
					}
				} else {
					textures.push_back(V->get());
				}
			}

			if (uniform_array_size > 0) {
				if (textures.size() < uniform_array_size) {
					const Map<StringName, Map<int, RID>>::Element *W = p_default_textures.find(uniform_name);
					for (int j = textures.size(); j < uniform_array_size; j++) {
						if (W && W->get().has(j)) {
							textures.push_back(W->get()[j]);
						} else {
							textures.push_back(RID());
						}
					}
				}
			} else if (textures.is_empty()) {
				const Map<StringName, Map<int, RID>>::Element *W = p_default_textures.find(uniform_name);
				if (W && W->get().has(0)) {
					textures.push_back(W->get()[0]);
				}
			}
		}

		RID gl_texture = texture_storage->texture_gl_get_default(DEFAULT_GL_TEXTURE_WHITE);

		if (textures.is_empty()) {
			//check default usage
			switch (p_texture_uniforms[i].type) {
				case ShaderLanguage::TYPE_ISAMPLER2D:
				case ShaderLanguage::TYPE_USAMPLER2D:
				case ShaderLanguage::TYPE_SAMPLER2D: {
					switch (p_texture_uniforms[i].hint) {
						case ShaderLanguage::ShaderNode::Uniform::HINT_BLACK:
						case ShaderLanguage::ShaderNode::Uniform::HINT_BLACK_ALBEDO: {
							gl_texture = texture_storage->texture_gl_get_default(DEFAULT_GL_TEXTURE_BLACK);
						} break;
						case ShaderLanguage::ShaderNode::Uniform::HINT_ANISOTROPY: {
							gl_texture = texture_storage->texture_gl_get_default(DEFAULT_GL_TEXTURE_ANISO);
						} break;
						case ShaderLanguage::ShaderNode::Uniform::HINT_NORMAL: {
							gl_texture = texture_storage->texture_gl_get_default(DEFAULT_GL_TEXTURE_NORMAL);
						} break;
						case ShaderLanguage::ShaderNode::Uniform::HINT_ROUGHNESS_NORMAL: {
							gl_texture = texture_storage->texture_gl_get_default(DEFAULT_GL_TEXTURE_NORMAL);
						} break;
						default: {
							gl_texture = texture_storage->texture_gl_get_default(DEFAULT_GL_TEXTURE_WHITE);
						} break;
					}
				} break;

				case ShaderLanguage::TYPE_SAMPLERCUBE: {
					switch (p_texture_uniforms[i].hint) {
						case ShaderLanguage::ShaderNode::Uniform::HINT_BLACK:
						case ShaderLanguage::ShaderNode::Uniform::HINT_BLACK_ALBEDO: {
							gl_texture = texture_storage->texture_gl_get_default(DEFAULT_GL_TEXTURE_CUBEMAP_BLACK);
						} break;
						default: {
							gl_texture = texture_storage->texture_gl_get_default(DEFAULT_GL_TEXTURE_CUBEMAP_WHITE);
						} break;
					}
				} break;
				case ShaderLanguage::TYPE_SAMPLERCUBEARRAY: {
					ERR_PRINT_ONCE("Type: SamplerCubeArray not supported in OpenGL renderer, please use another type.");
				} break;

				case ShaderLanguage::TYPE_ISAMPLER3D:
				case ShaderLanguage::TYPE_USAMPLER3D:
				case ShaderLanguage::TYPE_SAMPLER3D: {
					switch (p_texture_uniforms[i].hint) {
						case ShaderLanguage::ShaderNode::Uniform::HINT_BLACK:
						case ShaderLanguage::ShaderNode::Uniform::HINT_BLACK_ALBEDO: {
							gl_texture = texture_storage->texture_gl_get_default(DEFAULT_GL_TEXTURE_3D_BLACK);
						} break;
						default: {
							gl_texture = texture_storage->texture_gl_get_default(DEFAULT_GL_TEXTURE_3D_WHITE);
						} break;
					}
				} break;

				case ShaderLanguage::TYPE_ISAMPLER2DARRAY:
				case ShaderLanguage::TYPE_USAMPLER2DARRAY:
				case ShaderLanguage::TYPE_SAMPLER2DARRAY: {
					gl_texture = texture_storage->texture_gl_get_default(DEFAULT_GL_TEXTURE_2D_ARRAY_WHITE);
				} break;

				default: {
				}
			}
#ifdef TOOLS_ENABLED
			if (roughness_detect_texture && normal_detect_texture && !normal_detect_texture->path.is_empty()) {
				roughness_detect_texture->detect_roughness_callback(roughness_detect_texture->detect_roughness_callback_ud, normal_detect_texture->path, roughness_channel);
			}
#endif
			if (uniform_array_size > 0) {
				for (int j = 0; j < uniform_array_size; j++) {
					p_textures[k++] = gl_texture;
				}
			} else {
				p_textures[k++] = gl_texture;
			}
		} else {
			//bool srgb = p_use_linear_color && (p_texture_uniforms[i].hint == ShaderLanguage::ShaderNode::Uniform::HINT_ALBEDO || p_texture_uniforms[i].hint == ShaderLanguage::ShaderNode::Uniform::HINT_BLACK_ALBEDO);

			for (int j = 0; j < textures.size(); j++) {
				Texture *tex = TextureStorage::get_singleton()->get_texture(textures[j]);

				if (tex) {
					gl_texture = textures[j];
#ifdef TOOLS_ENABLED
					if (tex->detect_3d_callback && p_use_linear_color) {
						tex->detect_3d_callback(tex->detect_3d_callback_ud);
					}
					if (tex->detect_normal_callback && (p_texture_uniforms[i].hint == ShaderLanguage::ShaderNode::Uniform::HINT_NORMAL || p_texture_uniforms[i].hint == ShaderLanguage::ShaderNode::Uniform::HINT_ROUGHNESS_NORMAL)) {
						if (p_texture_uniforms[i].hint == ShaderLanguage::ShaderNode::Uniform::HINT_ROUGHNESS_NORMAL) {
							normal_detect_texture = tex;
						}
						tex->detect_normal_callback(tex->detect_normal_callback_ud);
					}
					if (tex->detect_roughness_callback && (p_texture_uniforms[i].hint >= ShaderLanguage::ShaderNode::Uniform::HINT_ROUGHNESS_R || p_texture_uniforms[i].hint <= ShaderLanguage::ShaderNode::Uniform::HINT_ROUGHNESS_GRAY)) {
						//find the normal texture
						roughness_detect_texture = tex;
						roughness_channel = RS::TextureDetectRoughnessChannel(p_texture_uniforms[i].hint - ShaderLanguage::ShaderNode::Uniform::HINT_ROUGHNESS_R);
					}
#endif
				}
#ifdef TOOLS_ENABLED
				if (roughness_detect_texture && normal_detect_texture && !normal_detect_texture->path.is_empty()) {
					roughness_detect_texture->detect_roughness_callback(roughness_detect_texture->detect_roughness_callback_ud, normal_detect_texture->path, roughness_channel);
				}
#endif
				p_textures[k++] = gl_texture;
			}
		}
	}
	{
		//for textures no longer used, unregister them
		List<Map<StringName, uint64_t>::Element *> to_delete;
		for (Map<StringName, uint64_t>::Element *E = used_global_textures.front(); E; E = E->next()) {
			if (E->get() != global_textures_pass) {
				to_delete.push_back(E);

				GlobalVariables::Variable *v = material_storage->global_variables.variables.getptr(E->key());
				if (v) {
					v->texture_materials.erase(self);
				}
			}
		}

		while (to_delete.front()) {
			used_global_textures.erase(to_delete.front()->get());
			to_delete.pop_front();
		}
		//handle registering/unregistering global textures
		if (uses_global_textures != (global_texture_E != nullptr)) {
			if (uses_global_textures) {
				global_texture_E = material_storage->global_variables.materials_using_texture.push_back(self);
			} else {
				material_storage->global_variables.materials_using_texture.erase(global_texture_E);
				global_texture_E = nullptr;
			}
		}
	}
}

void MaterialData::update_parameters_internal(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty, const Map<StringName, ShaderLanguage::ShaderNode::Uniform> &p_uniforms, const uint32_t *p_uniform_offsets, const Vector<ShaderCompiler::GeneratedCode::Texture> &p_texture_uniforms, const Map<StringName, Map<int, RID>> &p_default_texture_params, uint32_t p_ubo_size) {
	if ((uint32_t)ubo_data.size() != p_ubo_size) {
		p_uniform_dirty = true;
		if (!uniform_buffer) {
			glGenBuffers(1, &uniform_buffer);
		}

		ubo_data.resize(p_ubo_size);
		if (ubo_data.size()) {
			memset(ubo_data.ptrw(), 0, ubo_data.size()); //clear
		}
	}

	//check whether buffer changed
	if (p_uniform_dirty && ubo_data.size()) {
		update_uniform_buffer(p_uniforms, p_uniform_offsets, p_parameters, ubo_data.ptrw(), ubo_data.size(), true);
		glBindBuffer(GL_UNIFORM_BUFFER, uniform_buffer);
		glBufferData(GL_UNIFORM_BUFFER, ubo_data.size(), ubo_data.ptrw(), GL_DYNAMIC_DRAW);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}

	uint32_t tex_uniform_count = 0U;
	for (int i = 0; i < p_texture_uniforms.size(); i++) {
		tex_uniform_count += uint32_t(p_texture_uniforms[i].array_size > 0 ? p_texture_uniforms[i].array_size : 1);
	}

	if ((uint32_t)texture_cache.size() != tex_uniform_count || p_textures_dirty) {
		texture_cache.resize(tex_uniform_count);
		p_textures_dirty = true;
	}

	if (p_textures_dirty && tex_uniform_count) {
		update_textures(p_parameters, p_default_texture_params, p_texture_uniforms, texture_cache.ptrw(), true);
	}
}

///////////////////////////////////////////////////////////////////////////
// Material Storage

MaterialStorage *MaterialStorage::singleton = nullptr;

MaterialStorage *MaterialStorage::get_singleton() {
	return singleton;
}

MaterialStorage::MaterialStorage() {
	singleton = this;

	shader_data_request_func[RS::SHADER_SPATIAL] = _create_scene_shader_func;
	shader_data_request_func[RS::SHADER_CANVAS_ITEM] = _create_canvas_shader_func;
	shader_data_request_func[RS::SHADER_PARTICLES] = nullptr;
	shader_data_request_func[RS::SHADER_SKY] = _create_sky_shader_func;
	shader_data_request_func[RS::SHADER_FOG] = nullptr;

	material_data_request_func[RS::SHADER_SPATIAL] = _create_scene_material_func;
	material_data_request_func[RS::SHADER_CANVAS_ITEM] = _create_canvas_material_func;
	material_data_request_func[RS::SHADER_PARTICLES] = nullptr;
	material_data_request_func[RS::SHADER_SKY] = _create_sky_material_func;
	material_data_request_func[RS::SHADER_FOG] = nullptr;

	static_assert(sizeof(GlobalVariables::Value) == 16);

	global_variables.buffer_size = MAX(4096, (int)GLOBAL_GET("rendering/limits/global_shader_variables/buffer_size"));
	if (global_variables.buffer_size > uint32_t(Config::get_singleton()->max_uniform_buffer_size)) {
		global_variables.buffer_size = uint32_t(Config::get_singleton()->max_uniform_buffer_size);
		WARN_PRINT("Project setting: rendering/limits/global_shader_variables/buffer_size exceeds maximum uniform buffer size of: " + itos(Config::get_singleton()->max_uniform_buffer_size));
	}

	global_variables.buffer_values = memnew_arr(GlobalVariables::Value, global_variables.buffer_size);
	memset(global_variables.buffer_values, 0, sizeof(GlobalVariables::Value) * global_variables.buffer_size);
	global_variables.buffer_usage = memnew_arr(GlobalVariables::ValueUsage, global_variables.buffer_size);
	global_variables.buffer_dirty_regions = memnew_arr(bool, global_variables.buffer_size / GlobalVariables::BUFFER_DIRTY_REGION_SIZE);
	memset(global_variables.buffer_dirty_regions, 0, sizeof(bool) * global_variables.buffer_size / GlobalVariables::BUFFER_DIRTY_REGION_SIZE);
	glGenBuffers(1, &global_variables.buffer);
	glBindBuffer(GL_UNIFORM_BUFFER, global_variables.buffer);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(GlobalVariables::Value) * global_variables.buffer_size, nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	{
		// Setup CanvasItem compiler
		ShaderCompiler::DefaultIdentifierActions actions;

		actions.renames["VERTEX"] = "vertex";
		actions.renames["LIGHT_VERTEX"] = "light_vertex";
		actions.renames["SHADOW_VERTEX"] = "shadow_vertex";
		actions.renames["UV"] = "uv";
		actions.renames["POINT_SIZE"] = "gl_PointSize";

		actions.renames["MODEL_MATRIX"] = "model_matrix";
		actions.renames["CANVAS_MATRIX"] = "canvas_data.canvas_transform";
		actions.renames["SCREEN_MATRIX"] = "canvas_data.screen_transform";
		actions.renames["TIME"] = "canvas_data.time";
		actions.renames["PI"] = _MKSTR(Math_PI);
		actions.renames["TAU"] = _MKSTR(Math_TAU);
		actions.renames["E"] = _MKSTR(Math_E);
		actions.renames["AT_LIGHT_PASS"] = "false";
		actions.renames["INSTANCE_CUSTOM"] = "instance_custom";

		actions.renames["COLOR"] = "color";
		actions.renames["NORMAL"] = "normal";
		actions.renames["NORMAL_MAP"] = "normal_map";
		actions.renames["NORMAL_MAP_DEPTH"] = "normal_map_depth";
		actions.renames["TEXTURE"] = "color_texture";
		actions.renames["TEXTURE_PIXEL_SIZE"] = "draw_data.color_texture_pixel_size";
		actions.renames["NORMAL_TEXTURE"] = "normal_texture";
		actions.renames["SPECULAR_SHININESS_TEXTURE"] = "specular_texture";
		actions.renames["SPECULAR_SHININESS"] = "specular_shininess";
		actions.renames["SCREEN_UV"] = "screen_uv";
		actions.renames["SCREEN_TEXTURE"] = "screen_texture";
		actions.renames["SCREEN_PIXEL_SIZE"] = "canvas_data.screen_pixel_size";
		actions.renames["FRAGCOORD"] = "gl_FragCoord";
		actions.renames["POINT_COORD"] = "gl_PointCoord";
		actions.renames["INSTANCE_ID"] = "gl_InstanceIndex";
		actions.renames["VERTEX_ID"] = "gl_VertexIndex";

		actions.renames["LIGHT_POSITION"] = "light_position";
		actions.renames["LIGHT_COLOR"] = "light_color";
		actions.renames["LIGHT_ENERGY"] = "light_energy";
		actions.renames["LIGHT"] = "light";
		actions.renames["SHADOW_MODULATE"] = "shadow_modulate";

		actions.renames["texture_sdf"] = "texture_sdf";
		actions.renames["texture_sdf_normal"] = "texture_sdf_normal";
		actions.renames["sdf_to_screen_uv"] = "sdf_to_screen_uv";
		actions.renames["screen_uv_to_sdf"] = "screen_uv_to_sdf";

		actions.usage_defines["COLOR"] = "#define COLOR_USED\n";
		actions.usage_defines["SCREEN_TEXTURE"] = "#define SCREEN_TEXTURE_USED\n";
		actions.usage_defines["SCREEN_UV"] = "#define SCREEN_UV_USED\n";
		actions.usage_defines["SCREEN_PIXEL_SIZE"] = "@SCREEN_UV";
		actions.usage_defines["NORMAL"] = "#define NORMAL_USED\n";
		actions.usage_defines["NORMAL_MAP"] = "#define NORMAL_MAP_USED\n";
		actions.usage_defines["LIGHT"] = "#define LIGHT_SHADER_CODE_USED\n";

		actions.render_mode_defines["skip_vertex_transform"] = "#define SKIP_TRANSFORM_USED\n";
		actions.render_mode_defines["unshaded"] = "#define MODE_UNSHADED\n";
		actions.render_mode_defines["light_only"] = "#define MODE_LIGHT_ONLY\n";

		shaders.compiler_canvas.initialize(actions);
	}

	{
		// Setup Scene compiler

		//shader compiler
		ShaderCompiler::DefaultIdentifierActions actions;

		actions.renames["MODEL_MATRIX"] = "model_matrix";
		actions.renames["MODEL_NORMAL_MATRIX"] = "model_normal_matrix";
		actions.renames["VIEW_MATRIX"] = "scene_data.view_matrix";
		actions.renames["INV_VIEW_MATRIX"] = "scene_data.inv_view_matrix";
		actions.renames["PROJECTION_MATRIX"] = "projection_matrix";
		actions.renames["INV_PROJECTION_MATRIX"] = "inv_projection_matrix";
		actions.renames["MODELVIEW_MATRIX"] = "modelview";
		actions.renames["MODELVIEW_NORMAL_MATRIX"] = "modelview_normal";

		actions.renames["VERTEX"] = "vertex";
		actions.renames["NORMAL"] = "normal";
		actions.renames["TANGENT"] = "tangent";
		actions.renames["BINORMAL"] = "binormal";
		actions.renames["POSITION"] = "position";
		actions.renames["UV"] = "uv_interp";
		actions.renames["UV2"] = "uv2_interp";
		actions.renames["COLOR"] = "color_interp";
		actions.renames["POINT_SIZE"] = "gl_PointSize";
		actions.renames["INSTANCE_ID"] = "gl_InstanceIndex";
		actions.renames["VERTEX_ID"] = "gl_VertexIndex";

		actions.renames["ALPHA_SCISSOR_THRESHOLD"] = "alpha_scissor_threshold";
		actions.renames["ALPHA_HASH_SCALE"] = "alpha_hash_scale";
		actions.renames["ALPHA_ANTIALIASING_EDGE"] = "alpha_antialiasing_edge";
		actions.renames["ALPHA_TEXTURE_COORDINATE"] = "alpha_texture_coordinate";

		//builtins

		actions.renames["TIME"] = "scene_data.time";
		actions.renames["PI"] = _MKSTR(Math_PI);
		actions.renames["TAU"] = _MKSTR(Math_TAU);
		actions.renames["E"] = _MKSTR(Math_E);
		actions.renames["VIEWPORT_SIZE"] = "scene_data.viewport_size";

		actions.renames["FRAGCOORD"] = "gl_FragCoord";
		actions.renames["FRONT_FACING"] = "gl_FrontFacing";
		actions.renames["NORMAL_MAP"] = "normal_map";
		actions.renames["NORMAL_MAP_DEPTH"] = "normal_map_depth";
		actions.renames["ALBEDO"] = "albedo";
		actions.renames["ALPHA"] = "alpha";
		actions.renames["METALLIC"] = "metallic";
		actions.renames["SPECULAR"] = "specular";
		actions.renames["ROUGHNESS"] = "roughness";
		actions.renames["RIM"] = "rim";
		actions.renames["RIM_TINT"] = "rim_tint";
		actions.renames["CLEARCOAT"] = "clearcoat";
		actions.renames["CLEARCOAT_ROUGHNESS"] = "clearcoat_roughness";
		actions.renames["ANISOTROPY"] = "anisotropy";
		actions.renames["ANISOTROPY_FLOW"] = "anisotropy_flow";
		actions.renames["SSS_STRENGTH"] = "sss_strength";
		actions.renames["SSS_TRANSMITTANCE_COLOR"] = "transmittance_color";
		actions.renames["SSS_TRANSMITTANCE_DEPTH"] = "transmittance_depth";
		actions.renames["SSS_TRANSMITTANCE_BOOST"] = "transmittance_boost";
		actions.renames["BACKLIGHT"] = "backlight";
		actions.renames["AO"] = "ao";
		actions.renames["AO_LIGHT_AFFECT"] = "ao_light_affect";
		actions.renames["EMISSION"] = "emission";
		actions.renames["POINT_COORD"] = "gl_PointCoord";
		actions.renames["INSTANCE_CUSTOM"] = "instance_custom";
		actions.renames["SCREEN_UV"] = "screen_uv";
		actions.renames["SCREEN_TEXTURE"] = "color_buffer";
		actions.renames["DEPTH_TEXTURE"] = "depth_buffer";
		actions.renames["NORMAL_ROUGHNESS_TEXTURE"] = "normal_roughness_buffer";
		actions.renames["DEPTH"] = "gl_FragDepth";
		actions.renames["OUTPUT_IS_SRGB"] = "true";
		actions.renames["FOG"] = "custom_fog";
		actions.renames["RADIANCE"] = "custom_radiance";
		actions.renames["IRRADIANCE"] = "custom_irradiance";
		actions.renames["BONE_INDICES"] = "bone_attrib";
		actions.renames["BONE_WEIGHTS"] = "weight_attrib";
		actions.renames["CUSTOM0"] = "custom0_attrib";
		actions.renames["CUSTOM1"] = "custom1_attrib";
		actions.renames["CUSTOM2"] = "custom2_attrib";
		actions.renames["CUSTOM3"] = "custom3_attrib";
		actions.renames["OUTPUT_IS_SRGB"] = "SHADER_IS_SRGB";

		actions.renames["VIEW_INDEX"] = "ViewIndex";
		actions.renames["VIEW_MONO_LEFT"] = "0";
		actions.renames["VIEW_RIGHT"] = "1";

		//for light
		actions.renames["VIEW"] = "view";
		actions.renames["LIGHT_COLOR"] = "light_color";
		actions.renames["LIGHT"] = "light";
		actions.renames["ATTENUATION"] = "attenuation";
		actions.renames["DIFFUSE_LIGHT"] = "diffuse_light";
		actions.renames["SPECULAR_LIGHT"] = "specular_light";

		actions.usage_defines["NORMAL"] = "#define NORMAL_USED\n";
		actions.usage_defines["TANGENT"] = "#define TANGENT_USED\n";
		actions.usage_defines["BINORMAL"] = "@TANGENT";
		actions.usage_defines["RIM"] = "#define LIGHT_RIM_USED\n";
		actions.usage_defines["RIM_TINT"] = "@RIM";
		actions.usage_defines["CLEARCOAT"] = "#define LIGHT_CLEARCOAT_USED\n";
		actions.usage_defines["CLEARCOAT_ROUGHNESS"] = "@CLEARCOAT";
		actions.usage_defines["ANISOTROPY"] = "#define LIGHT_ANISOTROPY_USED\n";
		actions.usage_defines["ANISOTROPY_FLOW"] = "@ANISOTROPY";
		actions.usage_defines["AO"] = "#define AO_USED\n";
		actions.usage_defines["AO_LIGHT_AFFECT"] = "#define AO_USED\n";
		actions.usage_defines["UV"] = "#define UV_USED\n";
		actions.usage_defines["UV2"] = "#define UV2_USED\n";
		actions.usage_defines["BONE_INDICES"] = "#define BONES_USED\n";
		actions.usage_defines["BONE_WEIGHTS"] = "#define WEIGHTS_USED\n";
		actions.usage_defines["CUSTOM0"] = "#define CUSTOM0_USED\n";
		actions.usage_defines["CUSTOM1"] = "#define CUSTOM1_USED\n";
		actions.usage_defines["CUSTOM2"] = "#define CUSTOM2_USED\n";
		actions.usage_defines["CUSTOM3"] = "#define CUSTOM3_USED\n";
		actions.usage_defines["NORMAL_MAP"] = "#define NORMAL_MAP_USED\n";
		actions.usage_defines["NORMAL_MAP_DEPTH"] = "@NORMAL_MAP";
		actions.usage_defines["COLOR"] = "#define COLOR_USED\n";
		actions.usage_defines["INSTANCE_CUSTOM"] = "#define ENABLE_INSTANCE_CUSTOM\n";
		actions.usage_defines["POSITION"] = "#define OVERRIDE_POSITION\n";

		actions.usage_defines["ALPHA_SCISSOR_THRESHOLD"] = "#define ALPHA_SCISSOR_USED\n";
		actions.usage_defines["ALPHA_HASH_SCALE"] = "#define ALPHA_HASH_USED\n";
		actions.usage_defines["ALPHA_ANTIALIASING_EDGE"] = "#define ALPHA_ANTIALIASING_EDGE_USED\n";
		actions.usage_defines["ALPHA_TEXTURE_COORDINATE"] = "@ALPHA_ANTIALIASING_EDGE";

		actions.usage_defines["SSS_STRENGTH"] = "#define ENABLE_SSS\n";
		actions.usage_defines["SSS_TRANSMITTANCE_DEPTH"] = "#define ENABLE_TRANSMITTANCE\n";
		actions.usage_defines["BACKLIGHT"] = "#define LIGHT_BACKLIGHT_USED\n";
		actions.usage_defines["SCREEN_TEXTURE"] = "#define SCREEN_TEXTURE_USED\n";
		actions.usage_defines["SCREEN_UV"] = "#define SCREEN_UV_USED\n";

		actions.usage_defines["DIFFUSE_LIGHT"] = "#define USE_LIGHT_SHADER_CODE\n";
		actions.usage_defines["SPECULAR_LIGHT"] = "#define USE_LIGHT_SHADER_CODE\n";

		actions.usage_defines["FOG"] = "#define CUSTOM_FOG_USED\n";
		actions.usage_defines["RADIANCE"] = "#define CUSTOM_RADIANCE_USED\n";
		actions.usage_defines["IRRADIANCE"] = "#define CUSTOM_IRRADIANCE_USED\n";

		actions.render_mode_defines["skip_vertex_transform"] = "#define SKIP_TRANSFORM_USED\n";
		actions.render_mode_defines["world_vertex_coords"] = "#define VERTEX_WORLD_COORDS_USED\n";
		actions.render_mode_defines["ensure_correct_normals"] = "#define ENSURE_CORRECT_NORMALS\n";
		actions.render_mode_defines["cull_front"] = "#define DO_SIDE_CHECK\n";
		actions.render_mode_defines["cull_disabled"] = "#define DO_SIDE_CHECK\n";
		actions.render_mode_defines["particle_trails"] = "#define USE_PARTICLE_TRAILS\n";
		actions.render_mode_defines["depth_draw_opaque"] = "#define USE_OPAQUE_PREPASS\n";

		bool force_lambert = GLOBAL_GET("rendering/shading/overrides/force_lambert_over_burley");

		if (!force_lambert) {
			actions.render_mode_defines["diffuse_burley"] = "#define DIFFUSE_BURLEY\n";
		}

		actions.render_mode_defines["diffuse_lambert_wrap"] = "#define DIFFUSE_LAMBERT_WRAP\n";
		actions.render_mode_defines["diffuse_toon"] = "#define DIFFUSE_TOON\n";

		actions.render_mode_defines["sss_mode_skin"] = "#define SSS_MODE_SKIN\n";

		actions.render_mode_defines["specular_schlick_ggx"] = "#define SPECULAR_SCHLICK_GGX\n";
		actions.render_mode_defines["specular_toon"] = "#define SPECULAR_TOON\n";
		actions.render_mode_defines["specular_disabled"] = "#define SPECULAR_DISABLED\n";
		actions.render_mode_defines["shadows_disabled"] = "#define SHADOWS_DISABLED\n";
		actions.render_mode_defines["ambient_light_disabled"] = "#define AMBIENT_LIGHT_DISABLED\n";
		actions.render_mode_defines["shadow_to_opacity"] = "#define USE_SHADOW_TO_OPACITY\n";
		actions.render_mode_defines["unshaded"] = "#define MODE_UNSHADED\n";

		actions.default_filter = ShaderLanguage::FILTER_LINEAR_MIPMAP;
		actions.default_repeat = ShaderLanguage::REPEAT_ENABLE;

		shaders.compiler_scene.initialize(actions);
	}

	{
		// Setup Particles compiler
		/*
ShaderCompiler::DefaultIdentifierActions actions;

		actions.renames["COLOR"] = "PARTICLE.color";
		actions.renames["VELOCITY"] = "PARTICLE.velocity";
		//actions.renames["MASS"] = "mass"; ?
		actions.renames["ACTIVE"] = "particle_active";
		actions.renames["RESTART"] = "restart";
		actions.renames["CUSTOM"] = "PARTICLE.custom";
		for (int i = 0; i < ParticlesShader::MAX_USERDATAS; i++) {
			String udname = "USERDATA" + itos(i + 1);
			actions.renames[udname] = "PARTICLE.userdata" + itos(i + 1);
			actions.usage_defines[udname] = "#define USERDATA" + itos(i + 1) + "_USED\n";
		}
		actions.renames["TRANSFORM"] = "PARTICLE.xform";
		actions.renames["TIME"] = "frame_history.data[0].time";
		actions.renames["PI"] = _MKSTR(Math_PI);
		actions.renames["TAU"] = _MKSTR(Math_TAU);
		actions.renames["E"] = _MKSTR(Math_E);
		actions.renames["LIFETIME"] = "params.lifetime";
		actions.renames["DELTA"] = "local_delta";
		actions.renames["NUMBER"] = "particle_number";
		actions.renames["INDEX"] = "index";
		//actions.renames["GRAVITY"] = "current_gravity";
		actions.renames["EMISSION_TRANSFORM"] = "FRAME.emission_transform";
		actions.renames["RANDOM_SEED"] = "FRAME.random_seed";
		actions.renames["FLAG_EMIT_POSITION"] = "EMISSION_FLAG_HAS_POSITION";
		actions.renames["FLAG_EMIT_ROT_SCALE"] = "EMISSION_FLAG_HAS_ROTATION_SCALE";
		actions.renames["FLAG_EMIT_VELOCITY"] = "EMISSION_FLAG_HAS_VELOCITY";
		actions.renames["FLAG_EMIT_COLOR"] = "EMISSION_FLAG_HAS_COLOR";
		actions.renames["FLAG_EMIT_CUSTOM"] = "EMISSION_FLAG_HAS_CUSTOM";
		actions.renames["RESTART_POSITION"] = "restart_position";
		actions.renames["RESTART_ROT_SCALE"] = "restart_rotation_scale";
		actions.renames["RESTART_VELOCITY"] = "restart_velocity";
		actions.renames["RESTART_COLOR"] = "restart_color";
		actions.renames["RESTART_CUSTOM"] = "restart_custom";
		actions.renames["emit_subparticle"] = "emit_subparticle";
		actions.renames["COLLIDED"] = "collided";
		actions.renames["COLLISION_NORMAL"] = "collision_normal";
		actions.renames["COLLISION_DEPTH"] = "collision_depth";
		actions.renames["ATTRACTOR_FORCE"] = "attractor_force";

		actions.render_mode_defines["disable_force"] = "#define DISABLE_FORCE\n";
		actions.render_mode_defines["disable_velocity"] = "#define DISABLE_VELOCITY\n";
		actions.render_mode_defines["keep_data"] = "#define ENABLE_KEEP_DATA\n";
		actions.render_mode_defines["collision_use_scale"] = "#define USE_COLLISON_SCALE\n";

		actions.sampler_array_name = "material_samplers";
		actions.base_texture_binding_index = 1;
		actions.texture_layout_set = 3;
		actions.base_uniform_string = "material.";
		actions.base_varying_index = 10;

		actions.default_filter = ShaderLanguage::FILTER_LINEAR_MIPMAP;
		actions.default_repeat = ShaderLanguage::REPEAT_ENABLE;
		actions.global_buffer_array_variable = "global_variables.data";

		particles_shader.compiler.initialize(actions);
		*/
	}

	{
		// Setup Sky compiler
		ShaderCompiler::DefaultIdentifierActions actions;

		actions.renames["COLOR"] = "color";
		actions.renames["ALPHA"] = "alpha";
		actions.renames["EYEDIR"] = "cube_normal";
		actions.renames["POSITION"] = "position";
		actions.renames["SKY_COORDS"] = "panorama_coords";
		actions.renames["SCREEN_UV"] = "uv";
		actions.renames["TIME"] = "time";
		actions.renames["PI"] = _MKSTR(Math_PI);
		actions.renames["TAU"] = _MKSTR(Math_TAU);
		actions.renames["E"] = _MKSTR(Math_E);
		actions.renames["HALF_RES_COLOR"] = "half_res_color";
		actions.renames["QUARTER_RES_COLOR"] = "quarter_res_color";
		actions.renames["RADIANCE"] = "radiance";
		actions.renames["FOG"] = "custom_fog";
		actions.renames["LIGHT0_ENABLED"] = "directional_lights.data[0].enabled";
		actions.renames["LIGHT0_DIRECTION"] = "directional_lights.data[0].direction_energy.xyz";
		actions.renames["LIGHT0_ENERGY"] = "directional_lights.data[0].direction_energy.w";
		actions.renames["LIGHT0_COLOR"] = "directional_lights.data[0].color_size.xyz";
		actions.renames["LIGHT0_SIZE"] = "directional_lights.data[0].color_size.w";
		actions.renames["LIGHT1_ENABLED"] = "directional_lights.data[1].enabled";
		actions.renames["LIGHT1_DIRECTION"] = "directional_lights.data[1].direction_energy.xyz";
		actions.renames["LIGHT1_ENERGY"] = "directional_lights.data[1].direction_energy.w";
		actions.renames["LIGHT1_COLOR"] = "directional_lights.data[1].color_size.xyz";
		actions.renames["LIGHT1_SIZE"] = "directional_lights.data[1].color_size.w";
		actions.renames["LIGHT2_ENABLED"] = "directional_lights.data[2].enabled";
		actions.renames["LIGHT2_DIRECTION"] = "directional_lights.data[2].direction_energy.xyz";
		actions.renames["LIGHT2_ENERGY"] = "directional_lights.data[2].direction_energy.w";
		actions.renames["LIGHT2_COLOR"] = "directional_lights.data[2].color_size.xyz";
		actions.renames["LIGHT2_SIZE"] = "directional_lights.data[2].color_size.w";
		actions.renames["LIGHT3_ENABLED"] = "directional_lights.data[3].enabled";
		actions.renames["LIGHT3_DIRECTION"] = "directional_lights.data[3].direction_energy.xyz";
		actions.renames["LIGHT3_ENERGY"] = "directional_lights.data[3].direction_energy.w";
		actions.renames["LIGHT3_COLOR"] = "directional_lights.data[3].color_size.xyz";
		actions.renames["LIGHT3_SIZE"] = "directional_lights.data[3].color_size.w";
		actions.renames["AT_CUBEMAP_PASS"] = "AT_CUBEMAP_PASS";
		actions.renames["AT_HALF_RES_PASS"] = "AT_HALF_RES_PASS";
		actions.renames["AT_QUARTER_RES_PASS"] = "AT_QUARTER_RES_PASS";
		actions.usage_defines["HALF_RES_COLOR"] = "\n#define USES_HALF_RES_COLOR\n";
		actions.usage_defines["QUARTER_RES_COLOR"] = "\n#define USES_QUARTER_RES_COLOR\n";
		actions.render_mode_defines["disable_fog"] = "#define DISABLE_FOG\n";

		actions.default_filter = ShaderLanguage::FILTER_LINEAR_MIPMAP;
		actions.default_repeat = ShaderLanguage::REPEAT_ENABLE;

		shaders.compiler_sky.initialize(actions);
	}

	//shaders.copy.initialize();
	//shaders.copy_version = shaders.copy.version_create(); //TODO
	//shaders.copy.version_bind_shader(shaders.copy_version, CopyShaderGLES3::MODE_COPY_SECTION);
	//shaders.cubemap_filter.init();
	//bool ggx_hq = GLOBAL_GET("rendering/quality/reflections/high_quality_ggx");
	//shaders.cubemap_filter.set_conditional(CubemapFilterShaderGLES3::LOW_QUALITY, !ggx_hq);
}

MaterialStorage::~MaterialStorage() {
	//shaders.copy.version_free(shaders.copy_version);

	memdelete_arr(global_variables.buffer_values);
	memdelete_arr(global_variables.buffer_usage);
	memdelete_arr(global_variables.buffer_dirty_regions);
	glDeleteBuffers(1, &global_variables.buffer);

	singleton = nullptr;
}

/* GLOBAL VARIABLE API */

int32_t MaterialStorage::_global_variable_allocate(uint32_t p_elements) {
	int32_t idx = 0;
	while (idx + p_elements <= global_variables.buffer_size) {
		if (global_variables.buffer_usage[idx].elements == 0) {
			bool valid = true;
			for (uint32_t i = 1; i < p_elements; i++) {
				if (global_variables.buffer_usage[idx + i].elements > 0) {
					valid = false;
					idx += i + global_variables.buffer_usage[idx + i].elements;
					break;
				}
			}

			if (!valid) {
				continue; //if not valid, idx is in new position
			}

			return idx;
		} else {
			idx += global_variables.buffer_usage[idx].elements;
		}
	}

	return -1;
}

void MaterialStorage::_global_variable_store_in_buffer(int32_t p_index, RS::GlobalVariableType p_type, const Variant &p_value) {
	switch (p_type) {
		case RS::GLOBAL_VAR_TYPE_BOOL: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			bool b = p_value;
			bv.x = b ? 1.0 : 0.0;
			bv.y = 0.0;
			bv.z = 0.0;
			bv.w = 0.0;

		} break;
		case RS::GLOBAL_VAR_TYPE_BVEC2: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			uint32_t bvec = p_value;
			bv.x = (bvec & 1) ? 1.0 : 0.0;
			bv.y = (bvec & 2) ? 1.0 : 0.0;
			bv.z = 0.0;
			bv.w = 0.0;
		} break;
		case RS::GLOBAL_VAR_TYPE_BVEC3: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			uint32_t bvec = p_value;
			bv.x = (bvec & 1) ? 1.0 : 0.0;
			bv.y = (bvec & 2) ? 1.0 : 0.0;
			bv.z = (bvec & 4) ? 1.0 : 0.0;
			bv.w = 0.0;
		} break;
		case RS::GLOBAL_VAR_TYPE_BVEC4: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			uint32_t bvec = p_value;
			bv.x = (bvec & 1) ? 1.0 : 0.0;
			bv.y = (bvec & 2) ? 1.0 : 0.0;
			bv.z = (bvec & 4) ? 1.0 : 0.0;
			bv.w = (bvec & 8) ? 1.0 : 0.0;
		} break;
		case RS::GLOBAL_VAR_TYPE_INT: {
			GlobalVariables::ValueInt &bv = *(GlobalVariables::ValueInt *)&global_variables.buffer_values[p_index];
			int32_t v = p_value;
			bv.x = v;
			bv.y = 0;
			bv.z = 0;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_IVEC2: {
			GlobalVariables::ValueInt &bv = *(GlobalVariables::ValueInt *)&global_variables.buffer_values[p_index];
			Vector2i v = p_value;
			bv.x = v.x;
			bv.y = v.y;
			bv.z = 0;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_IVEC3: {
			GlobalVariables::ValueInt &bv = *(GlobalVariables::ValueInt *)&global_variables.buffer_values[p_index];
			Vector3i v = p_value;
			bv.x = v.x;
			bv.y = v.y;
			bv.z = v.z;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_IVEC4: {
			GlobalVariables::ValueInt &bv = *(GlobalVariables::ValueInt *)&global_variables.buffer_values[p_index];
			Vector<int32_t> v = p_value;
			bv.x = v.size() >= 1 ? v[0] : 0;
			bv.y = v.size() >= 2 ? v[1] : 0;
			bv.z = v.size() >= 3 ? v[2] : 0;
			bv.w = v.size() >= 4 ? v[3] : 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_RECT2I: {
			GlobalVariables::ValueInt &bv = *(GlobalVariables::ValueInt *)&global_variables.buffer_values[p_index];
			Rect2i v = p_value;
			bv.x = v.position.x;
			bv.y = v.position.y;
			bv.z = v.size.x;
			bv.w = v.size.y;
		} break;
		case RS::GLOBAL_VAR_TYPE_UINT: {
			GlobalVariables::ValueUInt &bv = *(GlobalVariables::ValueUInt *)&global_variables.buffer_values[p_index];
			uint32_t v = p_value;
			bv.x = v;
			bv.y = 0;
			bv.z = 0;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_UVEC2: {
			GlobalVariables::ValueUInt &bv = *(GlobalVariables::ValueUInt *)&global_variables.buffer_values[p_index];
			Vector2i v = p_value;
			bv.x = v.x;
			bv.y = v.y;
			bv.z = 0;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_UVEC3: {
			GlobalVariables::ValueUInt &bv = *(GlobalVariables::ValueUInt *)&global_variables.buffer_values[p_index];
			Vector3i v = p_value;
			bv.x = v.x;
			bv.y = v.y;
			bv.z = v.z;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_UVEC4: {
			GlobalVariables::ValueUInt &bv = *(GlobalVariables::ValueUInt *)&global_variables.buffer_values[p_index];
			Vector<int32_t> v = p_value;
			bv.x = v.size() >= 1 ? v[0] : 0;
			bv.y = v.size() >= 2 ? v[1] : 0;
			bv.z = v.size() >= 3 ? v[2] : 0;
			bv.w = v.size() >= 4 ? v[3] : 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_FLOAT: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			float v = p_value;
			bv.x = v;
			bv.y = 0;
			bv.z = 0;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_VEC2: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			Vector2 v = p_value;
			bv.x = v.x;
			bv.y = v.y;
			bv.z = 0;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_VEC3: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			Vector3 v = p_value;
			bv.x = v.x;
			bv.y = v.y;
			bv.z = v.z;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_VEC4: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			Plane v = p_value;
			bv.x = v.normal.x;
			bv.y = v.normal.y;
			bv.z = v.normal.z;
			bv.w = v.d;
		} break;
		case RS::GLOBAL_VAR_TYPE_COLOR: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			Color v = p_value;
			bv.x = v.r;
			bv.y = v.g;
			bv.z = v.b;
			bv.w = v.a;

			GlobalVariables::Value &bv_linear = global_variables.buffer_values[p_index + 1];
			//v = v.srgb_to_linear();
			bv_linear.x = v.r;
			bv_linear.y = v.g;
			bv_linear.z = v.b;
			bv_linear.w = v.a;

		} break;
		case RS::GLOBAL_VAR_TYPE_RECT2: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			Rect2 v = p_value;
			bv.x = v.position.x;
			bv.y = v.position.y;
			bv.z = v.size.x;
			bv.w = v.size.y;
		} break;
		case RS::GLOBAL_VAR_TYPE_MAT2: {
			GlobalVariables::Value *bv = &global_variables.buffer_values[p_index];
			Vector<float> m2 = p_value;
			if (m2.size() < 4) {
				m2.resize(4);
			}
			bv[0].x = m2[0];
			bv[0].y = m2[1];
			bv[0].z = 0;
			bv[0].w = 0;

			bv[1].x = m2[2];
			bv[1].y = m2[3];
			bv[1].z = 0;
			bv[1].w = 0;

		} break;
		case RS::GLOBAL_VAR_TYPE_MAT3: {
			GlobalVariables::Value *bv = &global_variables.buffer_values[p_index];
			Basis v = p_value;
			bv[0].x = v.rows[0][0];
			bv[0].y = v.rows[1][0];
			bv[0].z = v.rows[2][0];
			bv[0].w = 0;

			bv[1].x = v.rows[0][1];
			bv[1].y = v.rows[1][1];
			bv[1].z = v.rows[2][1];
			bv[1].w = 0;

			bv[2].x = v.rows[0][2];
			bv[2].y = v.rows[1][2];
			bv[2].z = v.rows[2][2];
			bv[2].w = 0;

		} break;
		case RS::GLOBAL_VAR_TYPE_MAT4: {
			GlobalVariables::Value *bv = &global_variables.buffer_values[p_index];

			Vector<float> m2 = p_value;
			if (m2.size() < 16) {
				m2.resize(16);
			}

			bv[0].x = m2[0];
			bv[0].y = m2[1];
			bv[0].z = m2[2];
			bv[0].w = m2[3];

			bv[1].x = m2[4];
			bv[1].y = m2[5];
			bv[1].z = m2[6];
			bv[1].w = m2[7];

			bv[2].x = m2[8];
			bv[2].y = m2[9];
			bv[2].z = m2[10];
			bv[2].w = m2[11];

			bv[3].x = m2[12];
			bv[3].y = m2[13];
			bv[3].z = m2[14];
			bv[3].w = m2[15];

		} break;
		case RS::GLOBAL_VAR_TYPE_TRANSFORM_2D: {
			GlobalVariables::Value *bv = &global_variables.buffer_values[p_index];
			Transform2D v = p_value;
			bv[0].x = v.columns[0][0];
			bv[0].y = v.columns[0][1];
			bv[0].z = 0;
			bv[0].w = 0;

			bv[1].x = v.columns[1][0];
			bv[1].y = v.columns[1][1];
			bv[1].z = 0;
			bv[1].w = 0;

			bv[2].x = v.columns[2][0];
			bv[2].y = v.columns[2][1];
			bv[2].z = 1;
			bv[2].w = 0;

		} break;
		case RS::GLOBAL_VAR_TYPE_TRANSFORM: {
			GlobalVariables::Value *bv = &global_variables.buffer_values[p_index];
			Transform3D v = p_value;
			bv[0].x = v.basis.rows[0][0];
			bv[0].y = v.basis.rows[1][0];
			bv[0].z = v.basis.rows[2][0];
			bv[0].w = 0;

			bv[1].x = v.basis.rows[0][1];
			bv[1].y = v.basis.rows[1][1];
			bv[1].z = v.basis.rows[2][1];
			bv[1].w = 0;

			bv[2].x = v.basis.rows[0][2];
			bv[2].y = v.basis.rows[1][2];
			bv[2].z = v.basis.rows[2][2];
			bv[2].w = 0;

			bv[3].x = v.origin.x;
			bv[3].y = v.origin.y;
			bv[3].z = v.origin.z;
			bv[3].w = 1;

		} break;
		default: {
			ERR_FAIL();
		}
	}
}

void MaterialStorage::_global_variable_mark_buffer_dirty(int32_t p_index, int32_t p_elements) {
	int32_t prev_chunk = -1;

	for (int32_t i = 0; i < p_elements; i++) {
		int32_t chunk = (p_index + i) / GlobalVariables::BUFFER_DIRTY_REGION_SIZE;
		if (chunk != prev_chunk) {
			if (!global_variables.buffer_dirty_regions[chunk]) {
				global_variables.buffer_dirty_regions[chunk] = true;
				global_variables.buffer_dirty_region_count++;
			}
		}

		prev_chunk = chunk;
	}
}

void MaterialStorage::global_variable_add(const StringName &p_name, RS::GlobalVariableType p_type, const Variant &p_value) {
	ERR_FAIL_COND(global_variables.variables.has(p_name));
	GlobalVariables::Variable gv;
	gv.type = p_type;
	gv.value = p_value;
	gv.buffer_index = -1;

	if (p_type >= RS::GLOBAL_VAR_TYPE_SAMPLER2D) {
		//is texture
		global_variables.must_update_texture_materials = true; //normally there are none
	} else {
		gv.buffer_elements = 1;
		if (p_type == RS::GLOBAL_VAR_TYPE_COLOR || p_type == RS::GLOBAL_VAR_TYPE_MAT2) {
			//color needs to elements to store srgb and linear
			gv.buffer_elements = 2;
		}
		if (p_type == RS::GLOBAL_VAR_TYPE_MAT3 || p_type == RS::GLOBAL_VAR_TYPE_TRANSFORM_2D) {
			//color needs to elements to store srgb and linear
			gv.buffer_elements = 3;
		}
		if (p_type == RS::GLOBAL_VAR_TYPE_MAT4 || p_type == RS::GLOBAL_VAR_TYPE_TRANSFORM) {
			//color needs to elements to store srgb and linear
			gv.buffer_elements = 4;
		}

		//is vector, allocate in buffer and update index
		gv.buffer_index = _global_variable_allocate(gv.buffer_elements);
		ERR_FAIL_COND_MSG(gv.buffer_index < 0, vformat("Failed allocating global variable '%s' out of buffer memory. Consider increasing it in the Project Settings.", String(p_name)));
		global_variables.buffer_usage[gv.buffer_index].elements = gv.buffer_elements;
		_global_variable_store_in_buffer(gv.buffer_index, gv.type, gv.value);
		_global_variable_mark_buffer_dirty(gv.buffer_index, gv.buffer_elements);

		global_variables.must_update_buffer_materials = true; //normally there are none
	}

	global_variables.variables[p_name] = gv;
}

void MaterialStorage::global_variable_remove(const StringName &p_name) {
	if (!global_variables.variables.has(p_name)) {
		return;
	}
	GlobalVariables::Variable &gv = global_variables.variables[p_name];

	if (gv.buffer_index >= 0) {
		global_variables.buffer_usage[gv.buffer_index].elements = 0;
		global_variables.must_update_buffer_materials = true;
	} else {
		global_variables.must_update_texture_materials = true;
	}

	global_variables.variables.erase(p_name);
}

Vector<StringName> MaterialStorage::global_variable_get_list() const {
	if (!Engine::get_singleton()->is_editor_hint()) {
		ERR_FAIL_V_MSG(Vector<StringName>(), "This function should never be used outside the editor, it can severely damage performance.");
	}

	Vector<StringName> names;
	for (const KeyValue<StringName, GlobalVariables::Variable> &E : global_variables.variables) {
		names.push_back(E.key);
	}
	names.sort_custom<StringName::AlphCompare>();
	return names;
}

void MaterialStorage::global_variable_set(const StringName &p_name, const Variant &p_value) {
	ERR_FAIL_COND(!global_variables.variables.has(p_name));
	GlobalVariables::Variable &gv = global_variables.variables[p_name];
	gv.value = p_value;
	if (gv.override.get_type() == Variant::NIL) {
		if (gv.buffer_index >= 0) {
			//buffer
			_global_variable_store_in_buffer(gv.buffer_index, gv.type, gv.value);
			_global_variable_mark_buffer_dirty(gv.buffer_index, gv.buffer_elements);
		} else {
			//texture
			MaterialStorage *material_storage = MaterialStorage::get_singleton();
			for (Set<RID>::Element *E = gv.texture_materials.front(); E; E = E->next()) {
				Material *material = material_storage->get_material(E->get());
				ERR_CONTINUE(!material);
				material_storage->_material_queue_update(material, false, true);
			}
		}
	}
}

void MaterialStorage::global_variable_set_override(const StringName &p_name, const Variant &p_value) {
	if (!global_variables.variables.has(p_name)) {
		return; //variable may not exist
	}

	ERR_FAIL_COND(p_value.get_type() == Variant::OBJECT);

	GlobalVariables::Variable &gv = global_variables.variables[p_name];

	gv.override = p_value;

	if (gv.buffer_index >= 0) {
		//buffer
		if (gv.override.get_type() == Variant::NIL) {
			_global_variable_store_in_buffer(gv.buffer_index, gv.type, gv.value);
		} else {
			_global_variable_store_in_buffer(gv.buffer_index, gv.type, gv.override);
		}

		_global_variable_mark_buffer_dirty(gv.buffer_index, gv.buffer_elements);
	} else {
		//texture
		MaterialStorage *material_storage = MaterialStorage::get_singleton();
		for (Set<RID>::Element *E = gv.texture_materials.front(); E; E = E->next()) {
			Material *material = material_storage->get_material(E->get());
			ERR_CONTINUE(!material);
			material_storage->_material_queue_update(material, false, true);
		}
	}
}

Variant MaterialStorage::global_variable_get(const StringName &p_name) const {
	if (!Engine::get_singleton()->is_editor_hint()) {
		ERR_FAIL_V_MSG(Variant(), "This function should never be used outside the editor, it can severely damage performance.");
	}

	if (!global_variables.variables.has(p_name)) {
		return Variant();
	}

	return global_variables.variables[p_name].value;
}

RS::GlobalVariableType MaterialStorage::global_variable_get_type_internal(const StringName &p_name) const {
	if (!global_variables.variables.has(p_name)) {
		return RS::GLOBAL_VAR_TYPE_MAX;
	}

	return global_variables.variables[p_name].type;
}

RS::GlobalVariableType MaterialStorage::global_variable_get_type(const StringName &p_name) const {
	if (!Engine::get_singleton()->is_editor_hint()) {
		ERR_FAIL_V_MSG(RS::GLOBAL_VAR_TYPE_MAX, "This function should never be used outside the editor, it can severely damage performance.");
	}

	return global_variable_get_type_internal(p_name);
}

void MaterialStorage::global_variables_load_settings(bool p_load_textures) {
	List<PropertyInfo> settings;
	ProjectSettings::get_singleton()->get_property_list(&settings);

	for (const PropertyInfo &E : settings) {
		if (E.name.begins_with("shader_globals/")) {
			StringName name = E.name.get_slice("/", 1);
			Dictionary d = ProjectSettings::get_singleton()->get(E.name);

			ERR_CONTINUE(!d.has("type"));
			ERR_CONTINUE(!d.has("value"));

			String type = d["type"];

			static const char *global_var_type_names[RS::GLOBAL_VAR_TYPE_MAX] = {
				"bool",
				"bvec2",
				"bvec3",
				"bvec4",
				"int",
				"ivec2",
				"ivec3",
				"ivec4",
				"rect2i",
				"uint",
				"uvec2",
				"uvec3",
				"uvec4",
				"float",
				"vec2",
				"vec3",
				"vec4",
				"color",
				"rect2",
				"mat2",
				"mat3",
				"mat4",
				"transform_2d",
				"transform",
				"sampler2D",
				"sampler2DArray",
				"sampler3D",
				"samplerCube",
			};

			RS::GlobalVariableType gvtype = RS::GLOBAL_VAR_TYPE_MAX;

			for (int i = 0; i < RS::GLOBAL_VAR_TYPE_MAX; i++) {
				if (global_var_type_names[i] == type) {
					gvtype = RS::GlobalVariableType(i);
					break;
				}
			}

			ERR_CONTINUE(gvtype == RS::GLOBAL_VAR_TYPE_MAX); //type invalid

			Variant value = d["value"];

			if (gvtype >= RS::GLOBAL_VAR_TYPE_SAMPLER2D) {
				//textire
				if (!p_load_textures) {
					value = RID();
					continue;
				}

				String path = value;
				Ref<Resource> resource = ResourceLoader::load(path);
				ERR_CONTINUE(resource.is_null());
				value = resource;
			}

			if (global_variables.variables.has(name)) {
				//has it, update it
				global_variable_set(name, value);
			} else {
				global_variable_add(name, gvtype, value);
			}
		}
	}
}

void MaterialStorage::global_variables_clear() {
	global_variables.variables.clear();
}

GLuint MaterialStorage::global_variables_get_uniform_buffer() const {
	return global_variables.buffer;
}

int32_t MaterialStorage::global_variables_instance_allocate(RID p_instance) {
	ERR_FAIL_COND_V(global_variables.instance_buffer_pos.has(p_instance), -1);
	int32_t pos = _global_variable_allocate(ShaderLanguage::MAX_INSTANCE_UNIFORM_INDICES);
	global_variables.instance_buffer_pos[p_instance] = pos; //save anyway
	ERR_FAIL_COND_V_MSG(pos < 0, -1, "Too many instances using shader instance variables. Increase buffer size in Project Settings.");
	global_variables.buffer_usage[pos].elements = ShaderLanguage::MAX_INSTANCE_UNIFORM_INDICES;
	return pos;
}

void MaterialStorage::global_variables_instance_free(RID p_instance) {
	ERR_FAIL_COND(!global_variables.instance_buffer_pos.has(p_instance));
	int32_t pos = global_variables.instance_buffer_pos[p_instance];
	if (pos >= 0) {
		global_variables.buffer_usage[pos].elements = 0;
	}
	global_variables.instance_buffer_pos.erase(p_instance);
}

void MaterialStorage::global_variables_instance_update(RID p_instance, int p_index, const Variant &p_value) {
	if (!global_variables.instance_buffer_pos.has(p_instance)) {
		return; //just not allocated, ignore
	}
	int32_t pos = global_variables.instance_buffer_pos[p_instance];

	if (pos < 0) {
		return; //again, not allocated, ignore
	}
	ERR_FAIL_INDEX(p_index, ShaderLanguage::MAX_INSTANCE_UNIFORM_INDICES);
	ERR_FAIL_COND_MSG(p_value.get_type() > Variant::COLOR, "Unsupported variant type for instance parameter: " + Variant::get_type_name(p_value.get_type())); //anything greater not supported

	ShaderLanguage::DataType datatype_from_value[Variant::COLOR + 1] = {
		ShaderLanguage::TYPE_MAX, //nil
		ShaderLanguage::TYPE_BOOL, //bool
		ShaderLanguage::TYPE_INT, //int
		ShaderLanguage::TYPE_FLOAT, //float
		ShaderLanguage::TYPE_MAX, //string
		ShaderLanguage::TYPE_VEC2, //vec2
		ShaderLanguage::TYPE_IVEC2, //vec2i
		ShaderLanguage::TYPE_VEC4, //rect2
		ShaderLanguage::TYPE_IVEC4, //rect2i
		ShaderLanguage::TYPE_VEC3, // vec3
		ShaderLanguage::TYPE_IVEC3, //vec3i
		ShaderLanguage::TYPE_MAX, //xform2d not supported here
		ShaderLanguage::TYPE_VEC4, //plane
		ShaderLanguage::TYPE_VEC4, //quat
		ShaderLanguage::TYPE_MAX, //aabb not supported here
		ShaderLanguage::TYPE_MAX, //basis not supported here
		ShaderLanguage::TYPE_MAX, //xform not supported here
		ShaderLanguage::TYPE_VEC4 //color
	};

	ShaderLanguage::DataType datatype = datatype_from_value[p_value.get_type()];

	ERR_FAIL_COND_MSG(datatype == ShaderLanguage::TYPE_MAX, "Unsupported variant type for instance parameter: " + Variant::get_type_name(p_value.get_type())); //anything greater not supported

	pos += p_index;

	_fill_std140_variant_ubo_value(datatype, 0, p_value, (uint8_t *)&global_variables.buffer_values[pos]);
	_global_variable_mark_buffer_dirty(pos, 1);
}

void MaterialStorage::_update_global_variables() {
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	if (global_variables.buffer_dirty_region_count > 0) {
		uint32_t total_regions = global_variables.buffer_size / GlobalVariables::BUFFER_DIRTY_REGION_SIZE;
		if (total_regions / global_variables.buffer_dirty_region_count <= 4) {
			// 25% of regions dirty, just update all buffer
			glBindBuffer(GL_UNIFORM_BUFFER, global_variables.buffer);
			glBufferData(GL_UNIFORM_BUFFER, sizeof(GlobalVariables::Value) * global_variables.buffer_size, global_variables.buffer_values, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_UNIFORM_BUFFER, 0);
			memset(global_variables.buffer_dirty_regions, 0, sizeof(bool) * total_regions);
		} else {
			uint32_t region_byte_size = sizeof(GlobalVariables::Value) * GlobalVariables::BUFFER_DIRTY_REGION_SIZE;
			glBindBuffer(GL_UNIFORM_BUFFER, global_variables.buffer);
			for (uint32_t i = 0; i < total_regions; i++) {
				if (global_variables.buffer_dirty_regions[i]) {
					glBufferSubData(GL_UNIFORM_BUFFER, i * region_byte_size, region_byte_size, &global_variables.buffer_values[i * GlobalVariables::BUFFER_DIRTY_REGION_SIZE]);
					global_variables.buffer_dirty_regions[i] = false;
				}
			}
			glBindBuffer(GL_UNIFORM_BUFFER, 0);
		}

		global_variables.buffer_dirty_region_count = 0;
	}

	if (global_variables.must_update_buffer_materials) {
		// only happens in the case of a buffer variable added or removed,
		// so not often.
		for (const RID &E : global_variables.materials_using_buffer) {
			Material *material = material_storage->get_material(E);
			ERR_CONTINUE(!material); //wtf

			material_storage->_material_queue_update(material, true, false);
		}

		global_variables.must_update_buffer_materials = false;
	}

	if (global_variables.must_update_texture_materials) {
		// only happens in the case of a buffer variable added or removed,
		// so not often.
		for (const RID &E : global_variables.materials_using_texture) {
			Material *material = material_storage->get_material(E);
			ERR_CONTINUE(!material); //wtf

			material_storage->_material_queue_update(material, false, true);
		}

		global_variables.must_update_texture_materials = false;
	}
}

/* SHADER API */

RID MaterialStorage::shader_allocate() {
	return shader_owner.allocate_rid();
}

void MaterialStorage::shader_initialize(RID p_rid) {
	Shader shader;
	shader.data = nullptr;
	shader.mode = RS::SHADER_MAX;

	shader_owner.initialize_rid(p_rid, shader);
}

void MaterialStorage::shader_free(RID p_rid) {
	GLES3::Shader *shader = shader_owner.get_or_null(p_rid);
	ERR_FAIL_COND(!shader);

	//make material unreference this
	while (shader->owners.size()) {
		material_set_shader(shader->owners.front()->get()->self, RID());
	}

	//clear data if exists
	if (shader->data) {
		memdelete(shader->data);
	}
	shader_owner.free(p_rid);
}

void MaterialStorage::shader_set_code(RID p_shader, const String &p_code) {
	GLES3::Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND(!shader);

	shader->code = p_code;

	String mode_string = ShaderLanguage::get_shader_type(p_code);

	RS::ShaderMode new_mode;
	if (mode_string == "canvas_item") {
		new_mode = RS::SHADER_CANVAS_ITEM;
		//} else if (mode_string == "particles") {
		//	new_mode = RS::SHADER_PARTICLES;
	} else if (mode_string == "spatial") {
		new_mode = RS::SHADER_SPATIAL;
	} else if (mode_string == "sky") {
		new_mode = RS::SHADER_SKY;
		//} else if (mode_string == "fog") {
		//	new_mode = RS::SHADER_FOG;
	} else {
		new_mode = RS::SHADER_MAX;
		ERR_PRINT("shader type " + mode_string + " not supported in OpenGL renderer");
	}

	if (new_mode != shader->mode) {
		if (shader->data) {
			memdelete(shader->data);
			shader->data = nullptr;
		}

		for (Set<Material *>::Element *E = shader->owners.front(); E; E = E->next()) {
			Material *material = E->get();
			material->shader_mode = new_mode;
			if (material->data) {
				memdelete(material->data);
				material->data = nullptr;
			}
		}

		shader->mode = new_mode;

		if (new_mode < RS::SHADER_MAX && shader_data_request_func[new_mode]) {
			shader->data = shader_data_request_func[new_mode]();
		} else {
			shader->mode = RS::SHADER_MAX; //invalid
		}

		for (Set<Material *>::Element *E = shader->owners.front(); E; E = E->next()) {
			Material *material = E->get();
			if (shader->data) {
				material->data = material_data_request_func[new_mode](shader->data);
				material->data->self = material->self;
				material->data->set_next_pass(material->next_pass);
				material->data->set_render_priority(material->priority);
			}
			material->shader_mode = new_mode;
		}

		if (shader->data) {
			for (const KeyValue<StringName, Map<int, RID>> &E : shader->default_texture_parameter) {
				for (const KeyValue<int, RID> &E2 : E.value) {
					shader->data->set_default_texture_param(E.key, E2.value, E2.key);
				}
			}
		}
	}

	if (shader->data) {
		shader->data->set_code(p_code);
	}

	for (Set<Material *>::Element *E = shader->owners.front(); E; E = E->next()) {
		Material *material = E->get();
		material->dependency.changed_notify(RendererStorage::DEPENDENCY_CHANGED_MATERIAL);
		_material_queue_update(material, true, true);
	}
}

String MaterialStorage::shader_get_code(RID p_shader) const {
	const GLES3::Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND_V(!shader, String());
	return shader->code;
}

void MaterialStorage::shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const {
	GLES3::Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND(!shader);
	if (shader->data) {
		return shader->data->get_param_list(p_param_list);
	}
}

void MaterialStorage::shader_set_default_texture_param(RID p_shader, const StringName &p_name, RID p_texture, int p_index) {
	GLES3::Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND(!shader);

	if (p_texture.is_valid() && TextureStorage::get_singleton()->owns_texture(p_texture)) {
		if (!shader->default_texture_parameter.has(p_name)) {
			shader->default_texture_parameter[p_name] = Map<int, RID>();
		}
		shader->default_texture_parameter[p_name][p_index] = p_texture;
	} else {
		if (shader->default_texture_parameter.has(p_name) && shader->default_texture_parameter[p_name].has(p_index)) {
			shader->default_texture_parameter[p_name].erase(p_index);

			if (shader->default_texture_parameter[p_name].is_empty()) {
				shader->default_texture_parameter.erase(p_name);
			}
		}
	}
	if (shader->data) {
		shader->data->set_default_texture_param(p_name, p_texture, p_index);
	}
	for (Set<Material *>::Element *E = shader->owners.front(); E; E = E->next()) {
		Material *material = E->get();
		_material_queue_update(material, false, true);
	}
}

RID MaterialStorage::shader_get_default_texture_param(RID p_shader, const StringName &p_name, int p_index) const {
	const GLES3::Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND_V(!shader, RID());
	if (shader->default_texture_parameter.has(p_name) && shader->default_texture_parameter[p_name].has(p_index)) {
		return shader->default_texture_parameter[p_name][p_index];
	}

	return RID();
}

Variant MaterialStorage::shader_get_param_default(RID p_shader, const StringName &p_param) const {
	Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND_V(!shader, Variant());
	if (shader->data) {
		return shader->data->get_default_parameter(p_param);
	}
	return Variant();
}

RS::ShaderNativeSourceCode MaterialStorage::shader_get_native_source_code(RID p_shader) const {
	Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND_V(!shader, RS::ShaderNativeSourceCode());
	if (shader->data) {
		return shader->data->get_native_source_code();
	}
	return RS::ShaderNativeSourceCode();
}

/* MATERIAL API */

void MaterialStorage::_material_queue_update(GLES3::Material *material, bool p_uniform, bool p_texture) {
	material->uniform_dirty = material->uniform_dirty || p_uniform;
	material->texture_dirty = material->texture_dirty || p_texture;

	if (material->update_element.in_list()) {
		return;
	}

	material_update_list.add(&material->update_element);
}

void MaterialStorage::_update_queued_materials() {
	while (material_update_list.first()) {
		Material *material = material_update_list.first()->self();

		if (material->data) {
			material->data->update_parameters(material->params, material->uniform_dirty, material->texture_dirty);
		}
		material->texture_dirty = false;
		material->uniform_dirty = false;

		material_update_list.remove(&material->update_element);
	}
}

RID MaterialStorage::material_allocate() {
	return material_owner.allocate_rid();
}

void MaterialStorage::material_initialize(RID p_rid) {
	material_owner.initialize_rid(p_rid);
	Material *material = material_owner.get_or_null(p_rid);
	material->self = p_rid;
}

void MaterialStorage::material_free(RID p_rid) {
	Material *material = material_owner.get_or_null(p_rid);
	ERR_FAIL_COND(!material);

	material_set_shader(p_rid, RID()); //clean up shader
	material->dependency.deleted_notify(p_rid);

	material_owner.free(p_rid);
}

void MaterialStorage::material_set_shader(RID p_material, RID p_shader) {
	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);

	if (material->data) {
		memdelete(material->data);
		material->data = nullptr;
	}

	if (material->shader) {
		material->shader->owners.erase(material);
		material->shader = nullptr;
		material->shader_mode = RS::SHADER_MAX;
	}

	if (p_shader.is_null()) {
		material->dependency.changed_notify(RendererStorage::DEPENDENCY_CHANGED_MATERIAL);
		material->shader_id = 0;
		return;
	}

	Shader *shader = get_shader(p_shader);
	ERR_FAIL_COND(!shader);
	material->shader = shader;
	material->shader_mode = shader->mode;
	material->shader_id = p_shader.get_local_index();
	shader->owners.insert(material);

	if (shader->mode == RS::SHADER_MAX) {
		return;
	}

	ERR_FAIL_COND(shader->data == nullptr);

	material->data = material_data_request_func[shader->mode](shader->data);
	material->data->self = p_material;
	material->data->set_next_pass(material->next_pass);
	material->data->set_render_priority(material->priority);
	//updating happens later
	material->dependency.changed_notify(RendererStorage::DEPENDENCY_CHANGED_MATERIAL);
	_material_queue_update(material, true, true);
}

void MaterialStorage::material_set_param(RID p_material, const StringName &p_param, const Variant &p_value) {
	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);

	if (p_value.get_type() == Variant::NIL) {
		material->params.erase(p_param);
	} else {
		ERR_FAIL_COND(p_value.get_type() == Variant::OBJECT); //object not allowed
		material->params[p_param] = p_value;
	}

	if (material->shader && material->shader->data) { //shader is valid
		bool is_texture = material->shader->data->is_param_texture(p_param);
		_material_queue_update(material, !is_texture, is_texture);
	} else {
		_material_queue_update(material, true, true);
	}
}

Variant MaterialStorage::material_get_param(RID p_material, const StringName &p_param) const {
	const GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND_V(!material, Variant());
	if (material->params.has(p_param)) {
		return material->params[p_param];
	} else {
		return Variant();
	}
}

void MaterialStorage::material_set_next_pass(RID p_material, RID p_next_material) {
	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);

	if (material->next_pass == p_next_material) {
		return;
	}

	material->next_pass = p_next_material;
	if (material->data) {
		material->data->set_next_pass(p_next_material);
	}

	material->dependency.changed_notify(RendererStorage::DEPENDENCY_CHANGED_MATERIAL);
}

void MaterialStorage::material_set_render_priority(RID p_material, int priority) {
	ERR_FAIL_COND(priority < RS::MATERIAL_RENDER_PRIORITY_MIN);
	ERR_FAIL_COND(priority > RS::MATERIAL_RENDER_PRIORITY_MAX);

	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);
	material->priority = priority;
	if (material->data) {
		material->data->set_render_priority(priority);
	}
}

bool MaterialStorage::material_is_animated(RID p_material) {
	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND_V(!material, false);
	if (material->shader && material->shader->data) {
		if (material->shader->data->is_animated()) {
			return true;
		} else if (material->next_pass.is_valid()) {
			return material_is_animated(material->next_pass);
		}
	}
	return false; //by default nothing is animated
}

bool MaterialStorage::material_casts_shadows(RID p_material) {
	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND_V(!material, true);
	if (material->shader && material->shader->data) {
		if (material->shader->data->casts_shadows()) {
			return true;
		} else if (material->next_pass.is_valid()) {
			return material_casts_shadows(material->next_pass);
		}
	}
	return true; //by default everything casts shadows
}

void MaterialStorage::material_get_instance_shader_parameters(RID p_material, List<InstanceShaderParam> *r_parameters) {
	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);
	if (material->shader && material->shader->data) {
		material->shader->data->get_instance_param_list(r_parameters);

		if (material->next_pass.is_valid()) {
			material_get_instance_shader_parameters(material->next_pass, r_parameters);
		}
	}
}

void MaterialStorage::material_update_dependency(RID p_material, RendererStorage::DependencyTracker *p_instance) {
	Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);
	p_instance->update_dependency(&material->dependency);
	if (material->next_pass.is_valid()) {
		material_update_dependency(material->next_pass, p_instance);
	}
}

/* Canvas Shader Data */

void CanvasShaderData::set_code(const String &p_code) {
	// compile the shader

	code = p_code;
	valid = false;
	ubo_size = 0;
	uniforms.clear();
	uses_screen_texture = false;
	uses_sdf = false;
	uses_time = false;

	if (code.is_empty()) {
		return; //just invalid, but no error
	}

	ShaderCompiler::GeneratedCode gen_code;

	int blend_mode = BLEND_MODE_MIX;
	uses_screen_texture = false;

	ShaderCompiler::IdentifierActions actions;
	actions.entry_point_stages["vertex"] = ShaderCompiler::STAGE_VERTEX;
	actions.entry_point_stages["fragment"] = ShaderCompiler::STAGE_FRAGMENT;
	actions.entry_point_stages["light"] = ShaderCompiler::STAGE_FRAGMENT;

	actions.render_mode_values["blend_add"] = Pair<int *, int>(&blend_mode, BLEND_MODE_ADD);
	actions.render_mode_values["blend_mix"] = Pair<int *, int>(&blend_mode, BLEND_MODE_MIX);
	actions.render_mode_values["blend_sub"] = Pair<int *, int>(&blend_mode, BLEND_MODE_SUB);
	actions.render_mode_values["blend_mul"] = Pair<int *, int>(&blend_mode, BLEND_MODE_MUL);
	actions.render_mode_values["blend_premul_alpha"] = Pair<int *, int>(&blend_mode, BLEND_MODE_PMALPHA);
	actions.render_mode_values["blend_disabled"] = Pair<int *, int>(&blend_mode, BLEND_MODE_DISABLED);

	actions.usage_flag_pointers["SCREEN_TEXTURE"] = &uses_screen_texture;
	actions.usage_flag_pointers["texture_sdf"] = &uses_sdf;
	actions.usage_flag_pointers["TIME"] = &uses_time;

	actions.uniforms = &uniforms;
	Error err = MaterialStorage::get_singleton()->shaders.compiler_canvas.compile(RS::SHADER_CANVAS_ITEM, code, &actions, path, gen_code);
	ERR_FAIL_COND_MSG(err != OK, "Shader compilation failed.");

	if (version.is_null()) {
		version = MaterialStorage::get_singleton()->shaders.canvas_shader.version_create();
	}

#if 0
	print_line("**compiling shader:");
	print_line("**defines:\n");
	for (int i = 0; i < gen_code.defines.size(); i++) {
		print_line(gen_code.defines[i]);
	}
	print_line("\n**uniforms:\n" + gen_code.uniforms);
	print_line("\n**vertex_globals:\n" + gen_code.vertex_global);
	print_line("\n**vertex_code:\n" + gen_code.vertex);
	print_line("\n**fragment_globals:\n" + gen_code.fragment_global);
	print_line("\n**fragment_code:\n" + gen_code.fragment);
	print_line("\n**light_code:\n" + gen_code.light);
#endif

	Vector<StringName> texture_uniform_names;
	for (int i = 0; i < gen_code.texture_uniforms.size(); i++) {
		texture_uniform_names.push_back(gen_code.texture_uniforms[i].name);
	}

	MaterialStorage::get_singleton()->shaders.canvas_shader.version_set_code(version, gen_code.code, gen_code.uniforms, gen_code.stage_globals[ShaderCompiler::STAGE_VERTEX], gen_code.stage_globals[ShaderCompiler::STAGE_FRAGMENT], gen_code.defines, texture_uniform_names);
	ERR_FAIL_COND(!MaterialStorage::get_singleton()->shaders.canvas_shader.version_is_valid(version));

	ubo_size = gen_code.uniform_total_size;
	ubo_offsets = gen_code.uniform_offsets;
	texture_uniforms = gen_code.texture_uniforms;

	valid = true;
}

void CanvasShaderData::set_default_texture_param(const StringName &p_name, RID p_texture, int p_index) {
	if (!p_texture.is_valid()) {
		if (default_texture_params.has(p_name) && default_texture_params[p_name].has(p_index)) {
			default_texture_params[p_name].erase(p_index);

			if (default_texture_params[p_name].is_empty()) {
				default_texture_params.erase(p_name);
			}
		}
	} else {
		if (!default_texture_params.has(p_name)) {
			default_texture_params[p_name] = Map<int, RID>();
		}
		default_texture_params[p_name][p_index] = p_texture;
	}
}

void CanvasShaderData::get_param_list(List<PropertyInfo> *p_param_list) const {
	Map<int, StringName> order;

	for (const KeyValue<StringName, ShaderLanguage::ShaderNode::Uniform> &E : uniforms) {
		if (E.value.scope != ShaderLanguage::ShaderNode::Uniform::SCOPE_LOCAL) {
			continue;
		}
		if (E.value.texture_order >= 0) {
			order[E.value.texture_order + 100000] = E.key;
		} else {
			order[E.value.order] = E.key;
		}
	}

	for (const KeyValue<int, StringName> &E : order) {
		PropertyInfo pi = ShaderLanguage::uniform_to_property_info(uniforms[E.value]);
		pi.name = E.value;
		p_param_list->push_back(pi);
	}
}

void CanvasShaderData::get_instance_param_list(List<RendererMaterialStorage::InstanceShaderParam> *p_param_list) const {
	for (const KeyValue<StringName, ShaderLanguage::ShaderNode::Uniform> &E : uniforms) {
		if (E.value.scope != ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			continue;
		}

		RendererMaterialStorage::InstanceShaderParam p;
		p.info = ShaderLanguage::uniform_to_property_info(E.value);
		p.info.name = E.key; //supply name
		p.index = E.value.instance_index;
		p.default_value = ShaderLanguage::constant_value_to_variant(E.value.default_value, E.value.type, E.value.array_size, E.value.hint);
		p_param_list->push_back(p);
	}
}

bool CanvasShaderData::is_param_texture(const StringName &p_param) const {
	if (!uniforms.has(p_param)) {
		return false;
	}

	return uniforms[p_param].texture_order >= 0;
}

bool CanvasShaderData::is_animated() const {
	return false;
}

bool CanvasShaderData::casts_shadows() const {
	return false;
}

Variant CanvasShaderData::get_default_parameter(const StringName &p_parameter) const {
	if (uniforms.has(p_parameter)) {
		ShaderLanguage::ShaderNode::Uniform uniform = uniforms[p_parameter];
		Vector<ShaderLanguage::ConstantNode::Value> default_value = uniform.default_value;
		return ShaderLanguage::constant_value_to_variant(default_value, uniform.type, uniform.array_size, uniform.hint);
	}
	return Variant();
}

RS::ShaderNativeSourceCode CanvasShaderData::get_native_source_code() const {
	return MaterialStorage::get_singleton()->shaders.canvas_shader.version_get_native_source_code(version);
}

CanvasShaderData::CanvasShaderData() {
	valid = false;
	uses_screen_texture = false;
	uses_sdf = false;
}

CanvasShaderData::~CanvasShaderData() {
	if (version.is_valid()) {
		MaterialStorage::get_singleton()->shaders.canvas_shader.version_free(version);
	}
}

GLES3::ShaderData *GLES3::_create_canvas_shader_func() {
	CanvasShaderData *shader_data = memnew(CanvasShaderData);
	return shader_data;
}

void CanvasMaterialData::update_parameters(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty) {
	return update_parameters_internal(p_parameters, p_uniform_dirty, p_textures_dirty, shader_data->uniforms, shader_data->ubo_offsets.ptr(), shader_data->texture_uniforms, shader_data->default_texture_params, shader_data->ubo_size);
}

void CanvasMaterialData::bind_uniforms() {
	// Bind Material Uniforms
	glBindBufferBase(GL_UNIFORM_BUFFER, RasterizerCanvasGLES3::MATERIAL_UNIFORM_LOCATION, uniform_buffer);

	RID *textures = texture_cache.ptrw();
	ShaderCompiler::GeneratedCode::Texture *texture_uniforms = shader_data->texture_uniforms.ptrw();
	for (int ti = 0; ti < texture_cache.size(); ti++) {
		Texture *texture = TextureStorage::get_singleton()->get_texture(textures[ti]);
		glActiveTexture(GL_TEXTURE1 + ti); // Start at GL_TEXTURE1 because texture slot 0 is used by the base texture
		glBindTexture(target_from_type[texture_uniforms[ti].type], texture->tex_id);

		// Set sampler state here as the same texture can be used in multiple places with different flags
		// Need to convert sampler state from ShaderLanguage::Texture* to RS::CanvasItemTexture*
		RS::CanvasItemTextureFilter filter = RS::CanvasItemTextureFilter((int(texture_uniforms[ti].filter) + 1) % RS::CANVAS_ITEM_TEXTURE_FILTER_MAX);
		RS::CanvasItemTextureRepeat repeat = RS::CanvasItemTextureRepeat((int(texture_uniforms[ti].repeat) + 1) % RS::CANVAS_ITEM_TEXTURE_REPEAT_MIRROR);
		texture->gl_set_filter(filter);
		texture->gl_set_repeat(repeat);
	}
}

CanvasMaterialData::~CanvasMaterialData() {
}

GLES3::MaterialData *GLES3::_create_canvas_material_func(ShaderData *p_shader) {
	CanvasMaterialData *material_data = memnew(CanvasMaterialData);
	material_data->shader_data = static_cast<CanvasShaderData *>(p_shader);
	//update will happen later anyway so do nothing.
	return material_data;
}

////////////////////////////////////////////////////////////////////////////////
// SKY SHADER

void SkyShaderData::set_code(const String &p_code) {
	//compile

	code = p_code;
	valid = false;
	ubo_size = 0;
	uniforms.clear();

	if (code.is_empty()) {
		return; //just invalid, but no error
	}

	ShaderCompiler::GeneratedCode gen_code;
	ShaderCompiler::IdentifierActions actions;
	actions.entry_point_stages["sky"] = ShaderCompiler::STAGE_FRAGMENT;

	uses_time = false;
	uses_half_res = false;
	uses_quarter_res = false;
	uses_position = false;
	uses_light = false;

	actions.render_mode_flags["use_half_res_pass"] = &uses_half_res;
	actions.render_mode_flags["use_quarter_res_pass"] = &uses_quarter_res;

	actions.usage_flag_pointers["TIME"] = &uses_time;
	actions.usage_flag_pointers["POSITION"] = &uses_position;
	actions.usage_flag_pointers["LIGHT0_ENABLED"] = &uses_light;
	actions.usage_flag_pointers["LIGHT0_ENERGY"] = &uses_light;
	actions.usage_flag_pointers["LIGHT0_DIRECTION"] = &uses_light;
	actions.usage_flag_pointers["LIGHT0_COLOR"] = &uses_light;
	actions.usage_flag_pointers["LIGHT0_SIZE"] = &uses_light;
	actions.usage_flag_pointers["LIGHT1_ENABLED"] = &uses_light;
	actions.usage_flag_pointers["LIGHT1_ENERGY"] = &uses_light;
	actions.usage_flag_pointers["LIGHT1_DIRECTION"] = &uses_light;
	actions.usage_flag_pointers["LIGHT1_COLOR"] = &uses_light;
	actions.usage_flag_pointers["LIGHT1_SIZE"] = &uses_light;
	actions.usage_flag_pointers["LIGHT2_ENABLED"] = &uses_light;
	actions.usage_flag_pointers["LIGHT2_ENERGY"] = &uses_light;
	actions.usage_flag_pointers["LIGHT2_DIRECTION"] = &uses_light;
	actions.usage_flag_pointers["LIGHT2_COLOR"] = &uses_light;
	actions.usage_flag_pointers["LIGHT2_SIZE"] = &uses_light;
	actions.usage_flag_pointers["LIGHT3_ENABLED"] = &uses_light;
	actions.usage_flag_pointers["LIGHT3_ENERGY"] = &uses_light;
	actions.usage_flag_pointers["LIGHT3_DIRECTION"] = &uses_light;
	actions.usage_flag_pointers["LIGHT3_COLOR"] = &uses_light;
	actions.usage_flag_pointers["LIGHT3_SIZE"] = &uses_light;

	actions.uniforms = &uniforms;

	Error err = MaterialStorage::get_singleton()->shaders.compiler_sky.compile(RS::SHADER_SKY, code, &actions, path, gen_code);
	ERR_FAIL_COND_MSG(err != OK, "Shader compilation failed.");

	if (version.is_null()) {
		version = MaterialStorage::get_singleton()->shaders.sky_shader.version_create();
	}

#if 0
	print_line("**compiling shader:");
	print_line("**defines:\n");
	for (int i = 0; i < gen_code.defines.size(); i++) {
		print_line(gen_code.defines[i]);
	}
	print_line("\n**uniforms:\n" + gen_code.uniforms);
	//	print_line("\n**vertex_globals:\n" + gen_code.vertex_global);
	//	print_line("\n**vertex_code:\n" + gen_code.vertex);
	print_line("\n**fragment_globals:\n" + gen_code.fragment_global);
	print_line("\n**fragment_code:\n" + gen_code.fragment);
	print_line("\n**light_code:\n" + gen_code.light);
#endif

	Vector<StringName> texture_uniform_names;
	for (int i = 0; i < gen_code.texture_uniforms.size(); i++) {
		texture_uniform_names.push_back(gen_code.texture_uniforms[i].name);
	}

	MaterialStorage::get_singleton()->shaders.sky_shader.version_set_code(version, gen_code.code, gen_code.uniforms, gen_code.stage_globals[ShaderCompiler::STAGE_VERTEX], gen_code.stage_globals[ShaderCompiler::STAGE_FRAGMENT], gen_code.defines, texture_uniform_names);
	ERR_FAIL_COND(!MaterialStorage::get_singleton()->shaders.sky_shader.version_is_valid(version));

	ubo_size = gen_code.uniform_total_size;
	ubo_offsets = gen_code.uniform_offsets;
	texture_uniforms = gen_code.texture_uniforms;

	valid = true;
}

void SkyShaderData::set_default_texture_param(const StringName &p_name, RID p_texture, int p_index) {
	if (!p_texture.is_valid()) {
		if (default_texture_params.has(p_name) && default_texture_params[p_name].has(p_index)) {
			default_texture_params[p_name].erase(p_index);

			if (default_texture_params[p_name].is_empty()) {
				default_texture_params.erase(p_name);
			}
		}
	} else {
		if (!default_texture_params.has(p_name)) {
			default_texture_params[p_name] = Map<int, RID>();
		}
		default_texture_params[p_name][p_index] = p_texture;
	}
}

void SkyShaderData::get_param_list(List<PropertyInfo> *p_param_list) const {
	Map<int, StringName> order;

	for (const KeyValue<StringName, ShaderLanguage::ShaderNode::Uniform> &E : uniforms) {
		if (E.value.scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_GLOBAL || E.value.scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			continue;
		}

		if (E.value.texture_order >= 0) {
			order[E.value.texture_order + 100000] = E.key;
		} else {
			order[E.value.order] = E.key;
		}
	}

	for (const KeyValue<int, StringName> &E : order) {
		PropertyInfo pi = ShaderLanguage::uniform_to_property_info(uniforms[E.value]);
		pi.name = E.value;
		p_param_list->push_back(pi);
	}
}

void SkyShaderData::get_instance_param_list(List<RendererMaterialStorage::InstanceShaderParam> *p_param_list) const {
	for (const KeyValue<StringName, ShaderLanguage::ShaderNode::Uniform> &E : uniforms) {
		if (E.value.scope != ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			continue;
		}

		RendererMaterialStorage::InstanceShaderParam p;
		p.info = ShaderLanguage::uniform_to_property_info(E.value);
		p.info.name = E.key; //supply name
		p.index = E.value.instance_index;
		p.default_value = ShaderLanguage::constant_value_to_variant(E.value.default_value, E.value.type, E.value.array_size, E.value.hint);
		p_param_list->push_back(p);
	}
}

bool SkyShaderData::is_param_texture(const StringName &p_param) const {
	if (!uniforms.has(p_param)) {
		return false;
	}

	return uniforms[p_param].texture_order >= 0;
}

bool SkyShaderData::is_animated() const {
	return false;
}

bool SkyShaderData::casts_shadows() const {
	return false;
}

Variant SkyShaderData::get_default_parameter(const StringName &p_parameter) const {
	if (uniforms.has(p_parameter)) {
		ShaderLanguage::ShaderNode::Uniform uniform = uniforms[p_parameter];
		Vector<ShaderLanguage::ConstantNode::Value> default_value = uniform.default_value;
		return ShaderLanguage::constant_value_to_variant(default_value, uniform.type, uniform.array_size, uniform.hint);
	}
	return Variant();
}

RS::ShaderNativeSourceCode SkyShaderData::get_native_source_code() const {
	return MaterialStorage::get_singleton()->shaders.sky_shader.version_get_native_source_code(version);
}

SkyShaderData::SkyShaderData() {
	valid = false;
}

SkyShaderData::~SkyShaderData() {
	if (version.is_valid()) {
		MaterialStorage::get_singleton()->shaders.sky_shader.version_free(version);
	}
}

GLES3::ShaderData *GLES3::_create_sky_shader_func() {
	SkyShaderData *shader_data = memnew(SkyShaderData);
	return shader_data;
}

////////////////////////////////////////////////////////////////////////////////
// Sky material

void SkyMaterialData::update_parameters(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty) {
	return update_parameters_internal(p_parameters, p_uniform_dirty, p_textures_dirty, shader_data->uniforms, shader_data->ubo_offsets.ptr(), shader_data->texture_uniforms, shader_data->default_texture_params, shader_data->ubo_size);
}

SkyMaterialData::~SkyMaterialData() {
}
GLES3::MaterialData *GLES3::_create_sky_material_func(ShaderData *p_shader) {
	SkyMaterialData *material_data = memnew(SkyMaterialData);
	material_data->shader_data = static_cast<SkyShaderData *>(p_shader);
	//update will happen later anyway so do nothing.
	return material_data;
}

void SkyMaterialData::bind_uniforms() {
	// Bind Material Uniforms
	glBindBufferBase(GL_UNIFORM_BUFFER, SKY_MATERIAL_UNIFORM_LOCATION, uniform_buffer);

	RID *textures = texture_cache.ptrw();
	ShaderCompiler::GeneratedCode::Texture *texture_uniforms = shader_data->texture_uniforms.ptrw();
	for (int ti = 0; ti < texture_cache.size(); ti++) {
		Texture *texture = TextureStorage::get_singleton()->get_texture(textures[ti]);
		glActiveTexture(GL_TEXTURE0 + ti);
		glBindTexture(target_from_type[texture_uniforms[ti].type], texture->tex_id);

		// Set sampler state here as the same texture can be used in multiple places with different flags
		// Need to convert sampler state from ShaderLanguage::Texture* to RS::CanvasItemTexture*
		RS::CanvasItemTextureFilter filter = RS::CanvasItemTextureFilter((int(texture_uniforms[ti].filter) + 1) % RS::CANVAS_ITEM_TEXTURE_FILTER_MAX);
		RS::CanvasItemTextureRepeat repeat = RS::CanvasItemTextureRepeat((int(texture_uniforms[ti].repeat) + 1) % RS::CANVAS_ITEM_TEXTURE_REPEAT_MIRROR);
		texture->gl_set_filter(filter);
		texture->gl_set_repeat(repeat);
	}
}

////////////////////////////////////////////////////////////////////////////////
// Scene SHADER

void SceneShaderData::set_code(const String &p_code) {
	//compile

	code = p_code;
	valid = false;
	ubo_size = 0;
	uniforms.clear();
	uses_screen_texture = false;

	if (code.is_empty()) {
		return; //just invalid, but no error
	}

	ShaderCompiler::GeneratedCode gen_code;

	int blend_modei = BLEND_MODE_MIX;
	int depth_testi = DEPTH_TEST_ENABLED;
	int alpha_antialiasing_modei = ALPHA_ANTIALIASING_OFF;
	int cull_modei = CULL_BACK;
	int depth_drawi = DEPTH_DRAW_OPAQUE;

	uses_point_size = false;
	uses_alpha = false;
	uses_alpha_clip = false;
	uses_blend_alpha = false;
	uses_depth_pre_pass = false;
	uses_discard = false;
	uses_roughness = false;
	uses_normal = false;
	wireframe = false;

	unshaded = false;
	uses_vertex = false;
	uses_position = false;
	uses_sss = false;
	uses_transmittance = false;
	uses_screen_texture = false;
	uses_depth_texture = false;
	uses_normal_texture = false;
	uses_time = false;
	writes_modelview_or_projection = false;
	uses_world_coordinates = false;
	uses_particle_trails = false;

	ShaderCompiler::IdentifierActions actions;
	actions.entry_point_stages["vertex"] = ShaderCompiler::STAGE_VERTEX;
	actions.entry_point_stages["fragment"] = ShaderCompiler::STAGE_FRAGMENT;
	actions.entry_point_stages["light"] = ShaderCompiler::STAGE_FRAGMENT;

	actions.render_mode_values["blend_add"] = Pair<int *, int>(&blend_modei, BLEND_MODE_ADD);
	actions.render_mode_values["blend_mix"] = Pair<int *, int>(&blend_modei, BLEND_MODE_MIX);
	actions.render_mode_values["blend_sub"] = Pair<int *, int>(&blend_modei, BLEND_MODE_SUB);
	actions.render_mode_values["blend_mul"] = Pair<int *, int>(&blend_modei, BLEND_MODE_MUL);

	actions.render_mode_values["alpha_to_coverage"] = Pair<int *, int>(&alpha_antialiasing_modei, ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE);
	actions.render_mode_values["alpha_to_coverage_and_one"] = Pair<int *, int>(&alpha_antialiasing_modei, ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE_AND_TO_ONE);

	actions.render_mode_values["depth_draw_never"] = Pair<int *, int>(&depth_drawi, DEPTH_DRAW_DISABLED);
	actions.render_mode_values["depth_draw_opaque"] = Pair<int *, int>(&depth_drawi, DEPTH_DRAW_OPAQUE);
	actions.render_mode_values["depth_draw_always"] = Pair<int *, int>(&depth_drawi, DEPTH_DRAW_ALWAYS);

	actions.render_mode_values["depth_test_disabled"] = Pair<int *, int>(&depth_testi, DEPTH_TEST_DISABLED);

	actions.render_mode_values["cull_disabled"] = Pair<int *, int>(&cull_modei, CULL_DISABLED);
	actions.render_mode_values["cull_front"] = Pair<int *, int>(&cull_modei, CULL_FRONT);
	actions.render_mode_values["cull_back"] = Pair<int *, int>(&cull_modei, CULL_BACK);

	actions.render_mode_flags["unshaded"] = &unshaded;
	actions.render_mode_flags["wireframe"] = &wireframe;
	actions.render_mode_flags["particle_trails"] = &uses_particle_trails;

	actions.usage_flag_pointers["ALPHA"] = &uses_alpha;
	actions.usage_flag_pointers["ALPHA_SCISSOR_THRESHOLD"] = &uses_alpha_clip;
	actions.render_mode_flags["depth_prepass_alpha"] = &uses_depth_pre_pass;

	actions.usage_flag_pointers["SSS_STRENGTH"] = &uses_sss;
	actions.usage_flag_pointers["SSS_TRANSMITTANCE_DEPTH"] = &uses_transmittance;

	actions.usage_flag_pointers["SCREEN_TEXTURE"] = &uses_screen_texture;
	actions.usage_flag_pointers["DEPTH_TEXTURE"] = &uses_depth_texture;
	actions.usage_flag_pointers["NORMAL_TEXTURE"] = &uses_normal_texture;
	actions.usage_flag_pointers["DISCARD"] = &uses_discard;
	actions.usage_flag_pointers["TIME"] = &uses_time;
	actions.usage_flag_pointers["ROUGHNESS"] = &uses_roughness;
	actions.usage_flag_pointers["NORMAL"] = &uses_normal;
	actions.usage_flag_pointers["NORMAL_MAP"] = &uses_normal;

	actions.usage_flag_pointers["POINT_SIZE"] = &uses_point_size;
	actions.usage_flag_pointers["POINT_COORD"] = &uses_point_size;

	actions.write_flag_pointers["MODELVIEW_MATRIX"] = &writes_modelview_or_projection;
	actions.write_flag_pointers["PROJECTION_MATRIX"] = &writes_modelview_or_projection;
	actions.write_flag_pointers["VERTEX"] = &uses_vertex;
	actions.write_flag_pointers["POSITION"] = &uses_position;

	actions.usage_flag_pointers["TANGENT"] = &uses_tangent;
	actions.usage_flag_pointers["BINORMAL"] = &uses_tangent;
	actions.usage_flag_pointers["COLOR"] = &uses_color;
	actions.usage_flag_pointers["UV"] = &uses_uv;
	actions.usage_flag_pointers["UV2"] = &uses_uv2;
	actions.usage_flag_pointers["CUSTOM0"] = &uses_custom0;
	actions.usage_flag_pointers["CUSTOM1"] = &uses_custom1;
	actions.usage_flag_pointers["CUSTOM2"] = &uses_custom2;
	actions.usage_flag_pointers["CUSTOM3"] = &uses_custom3;
	actions.usage_flag_pointers["BONE_INDICES"] = &uses_bones;
	actions.usage_flag_pointers["BONE_WEIGHTS"] = &uses_weights;

	actions.uniforms = &uniforms;

	Error err = MaterialStorage::get_singleton()->shaders.compiler_scene.compile(RS::SHADER_SPATIAL, code, &actions, path, gen_code);
	ERR_FAIL_COND_MSG(err != OK, "Shader compilation failed.");

	if (version.is_null()) {
		version = MaterialStorage::get_singleton()->shaders.scene_shader.version_create();
	}

	depth_draw = DepthDraw(depth_drawi);
	depth_test = DepthTest(depth_testi);
	cull_mode = Cull(cull_modei);
	blend_mode = BlendMode(blend_modei);
	alpha_antialiasing_mode = AlphaAntiAliasing(alpha_antialiasing_modei);
	vertex_input_mask = uint32_t(uses_normal);
	vertex_input_mask |= uses_tangent << 1;
	vertex_input_mask |= uses_color << 2;
	vertex_input_mask |= uses_uv << 3;
	vertex_input_mask |= uses_uv2 << 4;
	vertex_input_mask |= uses_custom0 << 5;
	vertex_input_mask |= uses_custom1 << 6;
	vertex_input_mask |= uses_custom2 << 7;
	vertex_input_mask |= uses_custom3 << 8;
	vertex_input_mask |= uses_bones << 9;
	vertex_input_mask |= uses_weights << 10;

#if 0
	print_line("**compiling shader:");
	print_line("**defines:\n");
	for (int i = 0; i < gen_code.defines.size(); i++) {
		print_line(gen_code.defines[i]);
	}

	Map<String, String>::Element *el = gen_code.code.front();
	while (el) {
		print_line("\n**code " + el->key() + ":\n" + el->value());

		el = el->next();
	}

	print_line("\n**uniforms:\n" + gen_code.uniforms);
	print_line("\n**vertex_globals:\n" + gen_code.stage_globals[ShaderCompiler::STAGE_VERTEX]);
	print_line("\n**fragment_globals:\n" + gen_code.stage_globals[ShaderCompiler::STAGE_FRAGMENT]);
#endif

	Vector<StringName> texture_uniform_names;
	for (int i = 0; i < gen_code.texture_uniforms.size(); i++) {
		texture_uniform_names.push_back(gen_code.texture_uniforms[i].name);
	}

	MaterialStorage::get_singleton()->shaders.scene_shader.version_set_code(version, gen_code.code, gen_code.uniforms, gen_code.stage_globals[ShaderCompiler::STAGE_VERTEX], gen_code.stage_globals[ShaderCompiler::STAGE_FRAGMENT], gen_code.defines, texture_uniform_names);
	ERR_FAIL_COND(!MaterialStorage::get_singleton()->shaders.scene_shader.version_is_valid(version));

	ubo_size = gen_code.uniform_total_size;
	ubo_offsets = gen_code.uniform_offsets;
	texture_uniforms = gen_code.texture_uniforms;

	// if any form of Alpha Antialiasing is enabled, set the blend mode to alpha to coverage
	if (alpha_antialiasing_mode != ALPHA_ANTIALIASING_OFF) {
		blend_mode = BLEND_MODE_ALPHA_TO_COVERAGE;
	}

	valid = true;
}

void SceneShaderData::set_default_texture_param(const StringName &p_name, RID p_texture, int p_index) {
	if (!p_texture.is_valid()) {
		if (default_texture_params.has(p_name) && default_texture_params[p_name].has(p_index)) {
			default_texture_params[p_name].erase(p_index);

			if (default_texture_params[p_name].is_empty()) {
				default_texture_params.erase(p_name);
			}
		}
	} else {
		if (!default_texture_params.has(p_name)) {
			default_texture_params[p_name] = Map<int, RID>();
		}
		default_texture_params[p_name][p_index] = p_texture;
	}
}

void SceneShaderData::get_param_list(List<PropertyInfo> *p_param_list) const {
	Map<int, StringName> order;

	for (const KeyValue<StringName, ShaderLanguage::ShaderNode::Uniform> &E : uniforms) {
		if (E.value.scope != ShaderLanguage::ShaderNode::Uniform::SCOPE_LOCAL) {
			continue;
		}

		if (E.value.texture_order >= 0) {
			order[E.value.texture_order + 100000] = E.key;
		} else {
			order[E.value.order] = E.key;
		}
	}

	for (const KeyValue<int, StringName> &E : order) {
		PropertyInfo pi = ShaderLanguage::uniform_to_property_info(uniforms[E.value]);
		pi.name = E.value;
		p_param_list->push_back(pi);
	}
}

void SceneShaderData::get_instance_param_list(List<RendererMaterialStorage::InstanceShaderParam> *p_param_list) const {
	for (const KeyValue<StringName, ShaderLanguage::ShaderNode::Uniform> &E : uniforms) {
		if (E.value.scope != ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			continue;
		}

		RendererMaterialStorage::InstanceShaderParam p;
		p.info = ShaderLanguage::uniform_to_property_info(E.value);
		p.info.name = E.key; //supply name
		p.index = E.value.instance_index;
		p.default_value = ShaderLanguage::constant_value_to_variant(E.value.default_value, E.value.type, E.value.array_size, E.value.hint);
		p_param_list->push_back(p);
	}
}

bool SceneShaderData::is_param_texture(const StringName &p_param) const {
	if (!uniforms.has(p_param)) {
		return false;
	}

	return uniforms[p_param].texture_order >= 0;
}

bool SceneShaderData::is_animated() const {
	return false;
}

bool SceneShaderData::casts_shadows() const {
	return false;
}

Variant SceneShaderData::get_default_parameter(const StringName &p_parameter) const {
	if (uniforms.has(p_parameter)) {
		ShaderLanguage::ShaderNode::Uniform uniform = uniforms[p_parameter];
		Vector<ShaderLanguage::ConstantNode::Value> default_value = uniform.default_value;
		return ShaderLanguage::constant_value_to_variant(default_value, uniform.type, uniform.array_size, uniform.hint);
	}
	return Variant();
}

RS::ShaderNativeSourceCode SceneShaderData::get_native_source_code() const {
	return MaterialStorage::get_singleton()->shaders.scene_shader.version_get_native_source_code(version);
}

SceneShaderData::SceneShaderData() {
	valid = false;
	uses_screen_texture = false;
}

SceneShaderData::~SceneShaderData() {
	if (version.is_valid()) {
		MaterialStorage::get_singleton()->shaders.scene_shader.version_free(version);
	}
}

GLES3::ShaderData *GLES3::_create_scene_shader_func() {
	SceneShaderData *shader_data = memnew(SceneShaderData);
	return shader_data;
}

void SceneMaterialData::set_render_priority(int p_priority) {
	priority = p_priority - RS::MATERIAL_RENDER_PRIORITY_MIN; //8 bits
}

void SceneMaterialData::set_next_pass(RID p_pass) {
	next_pass = p_pass;
}

void SceneMaterialData::update_parameters(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty) {
	return update_parameters_internal(p_parameters, p_uniform_dirty, p_textures_dirty, shader_data->uniforms, shader_data->ubo_offsets.ptr(), shader_data->texture_uniforms, shader_data->default_texture_params, shader_data->ubo_size);
}

SceneMaterialData::~SceneMaterialData() {
}

GLES3::MaterialData *GLES3::_create_scene_material_func(ShaderData *p_shader) {
	SceneMaterialData *material_data = memnew(SceneMaterialData);
	material_data->shader_data = static_cast<SceneShaderData *>(p_shader);
	//update will happen later anyway so do nothing.
	return material_data;
}

void SceneMaterialData::bind_uniforms() {
	// Bind Material Uniforms
	glBindBufferBase(GL_UNIFORM_BUFFER, SCENE_MATERIAL_UNIFORM_LOCATION, uniform_buffer);

	RID *textures = texture_cache.ptrw();
	ShaderCompiler::GeneratedCode::Texture *texture_uniforms = shader_data->texture_uniforms.ptrw();
	for (int ti = 0; ti < texture_cache.size(); ti++) {
		Texture *texture = TextureStorage::get_singleton()->get_texture(textures[ti]);
		glActiveTexture(GL_TEXTURE0 + ti);
		glBindTexture(target_from_type[texture_uniforms[ti].type], texture->tex_id);

		// Set sampler state here as the same texture can be used in multiple places with different flags
		// Need to convert sampler state from ShaderLanguage::Texture* to RS::CanvasItemTexture*
		RS::CanvasItemTextureFilter filter = RS::CanvasItemTextureFilter((int(texture_uniforms[ti].filter) + 1) % RS::CANVAS_ITEM_TEXTURE_FILTER_MAX);
		RS::CanvasItemTextureRepeat repeat = RS::CanvasItemTextureRepeat((int(texture_uniforms[ti].repeat) + 1) % RS::CANVAS_ITEM_TEXTURE_REPEAT_MIRROR);
		texture->gl_set_filter(filter);
		texture->gl_set_repeat(repeat);
	}
}

#endif // !GLES3_ENABLED
