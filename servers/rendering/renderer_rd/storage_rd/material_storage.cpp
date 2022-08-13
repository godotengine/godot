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

#include "material_storage.h"
#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/io/resource_loader.h"
#include "texture_storage.h"

using namespace RendererRD;

///////////////////////////////////////////////////////////////////////////
// UBI helper functions

_FORCE_INLINE_ static void _fill_std140_variant_ubo_value(ShaderLanguage::DataType type, int p_array_size, const Variant &value, uint8_t *data, bool p_linear_color) {
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
			int32_t *gui = (int32_t *)data;
			if (p_array_size > 0) {
				Vector<int> iv = value;
				int s = iv.size();
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
			} else {
				Vector2i v = value;
				gui[0] = v.x;
				gui[1] = v.y;
			}
		} break;
		case ShaderLanguage::TYPE_IVEC3: {
			int32_t *gui = (int32_t *)data;
			if (p_array_size > 0) {
				Vector<int> iv = value;
				int s = iv.size();
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
			} else {
				Vector3i v = value;
				gui[0] = v.x;
				gui[1] = v.y;
				gui[2] = v.z;
			}
		} break;
		case ShaderLanguage::TYPE_IVEC4: {
			int32_t *gui = (int32_t *)data;
			if (p_array_size > 0) {
				Vector<int> iv = value;
				int s = iv.size();
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
			} else {
				Vector4i v = value;
				gui[0] = v.x;
				gui[1] = v.y;
				gui[2] = v.z;
				gui[3] = v.w;
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
			uint32_t *gui = (uint32_t *)data;
			if (p_array_size > 0) {
				Vector<int> iv = value;
				int s = iv.size();
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
			} else {
				Vector2i v = value;
				gui[0] = v.x;
				gui[1] = v.y;
			}
		} break;
		case ShaderLanguage::TYPE_UVEC3: {
			uint32_t *gui = (uint32_t *)data;
			if (p_array_size > 0) {
				Vector<int> iv = value;
				int s = iv.size();
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
			} else {
				Vector3i v = value;
				gui[0] = v.x;
				gui[1] = v.y;
				gui[2] = v.z;
			}
		} break;
		case ShaderLanguage::TYPE_UVEC4: {
			uint32_t *gui = (uint32_t *)data;
			if (p_array_size > 0) {
				Vector<int> iv = value;
				int s = iv.size();
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
			} else {
				Vector4i v = value;
				gui[0] = v.x;
				gui[1] = v.y;
				gui[2] = v.z;
				gui[3] = v.w;
			}
		} break;
		case ShaderLanguage::TYPE_FLOAT: {
			float *gui = reinterpret_cast<float *>(data);

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
			float *gui = reinterpret_cast<float *>(data);

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
			float *gui = reinterpret_cast<float *>(data);

			if (p_array_size > 0) {
				if (value.get_type() == Variant::PACKED_COLOR_ARRAY) {
					const PackedColorArray &a = value;
					int s = a.size();

					for (int i = 0, j = 0; i < p_array_size; i++, j += 4) {
						if (i < s) {
							Color color = a[i];
							if (p_linear_color) {
								color = color.srgb_to_linear();
							}
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

					if (p_linear_color) {
						v = v.srgb_to_linear();
					}

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
			float *gui = reinterpret_cast<float *>(data);

			if (p_array_size > 0) {
				if (value.get_type() == Variant::PACKED_COLOR_ARRAY) {
					const PackedColorArray &a = value;
					int s = a.size();

					for (int i = 0, j = 0; i < p_array_size; i++, j += 4) {
						if (i < s) {
							Color color = a[i];
							if (p_linear_color) {
								color = color.srgb_to_linear();
							}
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

					if (p_linear_color) {
						v = v.srgb_to_linear();
					}

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
				} else if (value.get_type() == Variant::PLANE) {
					Plane v = value;

					gui[0] = v.normal.x;
					gui[1] = v.normal.y;
					gui[2] = v.normal.z;
					gui[3] = v.d;
				} else {
					Vector4 v = value;

					gui[0] = v.x;
					gui[1] = v.y;
					gui[2] = v.z;
					gui[3] = v.w;
				}
			}
		} break;
		case ShaderLanguage::TYPE_MAT2: {
			float *gui = reinterpret_cast<float *>(data);

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
			float *gui = reinterpret_cast<float *>(data);

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
			float *gui = reinterpret_cast<float *>(data);

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
			} else if (value.get_type() == Variant::TRANSFORM3D) {
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
			} else {
				Projection v = value;
				for (int i = 0; i < 4; i++) {
					for (int j = 0; j < 4; j++) {
						gui[i * 4 + j] = v.matrix[i][j];
					}
				}
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
			float *gui = reinterpret_cast<float *>(data);
			gui[0] = value[0].real;

		} break;
		case ShaderLanguage::TYPE_VEC2: {
			float *gui = reinterpret_cast<float *>(data);

			for (int i = 0; i < 2; i++) {
				gui[i] = value[i].real;
			}

		} break;
		case ShaderLanguage::TYPE_VEC3: {
			float *gui = reinterpret_cast<float *>(data);

			for (int i = 0; i < 3; i++) {
				gui[i] = value[i].real;
			}

		} break;
		case ShaderLanguage::TYPE_VEC4: {
			float *gui = reinterpret_cast<float *>(data);

			for (int i = 0; i < 4; i++) {
				gui[i] = value[i].real;
			}
		} break;
		case ShaderLanguage::TYPE_MAT2: {
			float *gui = reinterpret_cast<float *>(data);

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
			float *gui = reinterpret_cast<float *>(data);

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
			float *gui = reinterpret_cast<float *>(data);

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
// MaterialStorage::MaterialData

void MaterialStorage::MaterialData::update_uniform_buffer(const HashMap<StringName, ShaderLanguage::ShaderNode::Uniform> &p_uniforms, const uint32_t *p_uniform_offsets, const HashMap<StringName, Variant> &p_parameters, uint8_t *p_buffer, uint32_t p_buffer_size, bool p_use_linear_color) {
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	bool uses_global_buffer = false;

	for (const KeyValue<StringName, ShaderLanguage::ShaderNode::Uniform> &E : p_uniforms) {
		if (E.value.order < 0) {
			continue; // texture, does not go here
		}

		if (E.value.scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			continue; //instance uniforms don't appear in the buffer
		}

		if (E.value.hint == ShaderLanguage::ShaderNode::Uniform::HINT_SCREEN_TEXTURE ||
				E.value.hint == ShaderLanguage::ShaderNode::Uniform::HINT_NORMAL_ROUGHNESS_TEXTURE ||
				E.value.hint == ShaderLanguage::ShaderNode::Uniform::HINT_DEPTH_TEXTURE) {
			continue;
		}

		if (E.value.scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_GLOBAL) {
			//this is a global variable, get the index to it
			GlobalShaderUniforms::Variable *gv = material_storage->global_shader_uniforms.variables.getptr(E.key);
			uint32_t index = 0;
			if (gv) {
				index = gv->buffer_index;
			} else {
				WARN_PRINT("Shader uses global parameter '" + E.key + "', but it was removed at some point. Material will not display correctly.");
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
		HashMap<StringName, Variant>::ConstIterator V = p_parameters.find(E.key);

		if (V) {
			//user provided
			_fill_std140_variant_ubo_value(E.value.type, E.value.array_size, V->value, data, p_use_linear_color);

		} else if (E.value.default_value.size()) {
			//default value
			_fill_std140_ubo_value(E.value.type, E.value.default_value, data);
			//value=E.value.default_value;
		} else {
			//zero because it was not provided
			if ((E.value.type == ShaderLanguage::TYPE_VEC3 || E.value.type == ShaderLanguage::TYPE_VEC4) && E.value.hint == ShaderLanguage::ShaderNode::Uniform::HINT_SOURCE_COLOR) {
				//colors must be set as black, with alpha as 1.0
				_fill_std140_variant_ubo_value(E.value.type, E.value.array_size, Color(0, 0, 0, 1), data, p_use_linear_color);
			} else {
				//else just zero it out
				_fill_std140_ubo_empty(E.value.type, E.value.array_size, data);
			}
		}
	}

	if (uses_global_buffer != (global_buffer_E != nullptr)) {
		if (uses_global_buffer) {
			global_buffer_E = material_storage->global_shader_uniforms.materials_using_buffer.push_back(self);
		} else {
			material_storage->global_shader_uniforms.materials_using_buffer.erase(global_buffer_E);
			global_buffer_E = nullptr;
		}
	}
}

MaterialStorage::MaterialData::~MaterialData() {
	MaterialStorage *material_storage = MaterialStorage::get_singleton();

	if (global_buffer_E) {
		//unregister global buffers
		material_storage->global_shader_uniforms.materials_using_buffer.erase(global_buffer_E);
	}

	if (global_texture_E) {
		//unregister global textures

		for (const KeyValue<StringName, uint64_t> &E : used_global_textures) {
			GlobalShaderUniforms::Variable *v = material_storage->global_shader_uniforms.variables.getptr(E.key);
			if (v) {
				v->texture_materials.erase(self);
			}
		}
		//unregister material from those using global textures
		material_storage->global_shader_uniforms.materials_using_texture.erase(global_texture_E);
	}

	if (uniform_buffer.is_valid()) {
		RD::get_singleton()->free(uniform_buffer);
	}
}

void MaterialStorage::MaterialData::update_textures(const HashMap<StringName, Variant> &p_parameters, const HashMap<StringName, HashMap<int, RID>> &p_default_textures, const Vector<ShaderCompiler::GeneratedCode::Texture> &p_texture_uniforms, RID *p_textures, bool p_use_linear_color) {
	TextureStorage *texture_storage = TextureStorage::get_singleton();
	MaterialStorage *material_storage = MaterialStorage::get_singleton();

#ifdef TOOLS_ENABLED
	TextureStorage::Texture *roughness_detect_texture = nullptr;
	RS::TextureDetectRoughnessChannel roughness_channel = RS::TEXTURE_DETECT_ROUGHNESS_R;
	TextureStorage::Texture *normal_detect_texture = nullptr;
#endif

	bool uses_global_textures = false;
	global_textures_pass++;

	for (int i = 0, k = 0; i < p_texture_uniforms.size(); i++) {
		const StringName &uniform_name = p_texture_uniforms[i].name;
		int uniform_array_size = p_texture_uniforms[i].array_size;

		Vector<RID> textures;

		if (p_texture_uniforms[i].hint == ShaderLanguage::ShaderNode::Uniform::HINT_SCREEN_TEXTURE ||
				p_texture_uniforms[i].hint == ShaderLanguage::ShaderNode::Uniform::HINT_NORMAL_ROUGHNESS_TEXTURE ||
				p_texture_uniforms[i].hint == ShaderLanguage::ShaderNode::Uniform::HINT_DEPTH_TEXTURE) {
			continue;
		}

		if (p_texture_uniforms[i].global) {
			uses_global_textures = true;

			GlobalShaderUniforms::Variable *v = material_storage->global_shader_uniforms.variables.getptr(uniform_name);
			if (v) {
				if (v->buffer_index >= 0) {
					WARN_PRINT("Shader uses global parameter texture '" + String(uniform_name) + "', but it changed type and is no longer a texture!.");

				} else {
					HashMap<StringName, uint64_t>::Iterator E = used_global_textures.find(uniform_name);
					if (!E) {
						E = used_global_textures.insert(uniform_name, global_textures_pass);
						v->texture_materials.insert(self);
					} else {
						E->value = global_textures_pass;
					}

					textures.push_back(v->override.get_type() != Variant::NIL ? v->override : v->value);
				}

			} else {
				WARN_PRINT("Shader uses global parameter texture '" + String(uniform_name) + "', but it was removed at some point. Material will not display correctly.");
			}
		} else {
			HashMap<StringName, Variant>::ConstIterator V = p_parameters.find(uniform_name);
			if (V) {
				if (V->value.is_array()) {
					Array array = (Array)V->value;
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
					textures.push_back(V->value);
				}
			}

			if (uniform_array_size > 0) {
				if (textures.size() < uniform_array_size) {
					HashMap<StringName, HashMap<int, RID>>::ConstIterator W = p_default_textures.find(uniform_name);
					for (int j = textures.size(); j < uniform_array_size; j++) {
						if (W && W->value.has(j)) {
							textures.push_back(W->value[j]);
						} else {
							textures.push_back(RID());
						}
					}
				}
			} else if (textures.is_empty()) {
				HashMap<StringName, HashMap<int, RID>>::ConstIterator W = p_default_textures.find(uniform_name);
				if (W && W->value.has(0)) {
					textures.push_back(W->value[0]);
				}
			}
		}

		RID rd_texture;

		if (textures.is_empty()) {
			//check default usage
			switch (p_texture_uniforms[i].type) {
				case ShaderLanguage::TYPE_ISAMPLER2D:
				case ShaderLanguage::TYPE_USAMPLER2D:
				case ShaderLanguage::TYPE_SAMPLER2D: {
					switch (p_texture_uniforms[i].hint) {
						case ShaderLanguage::ShaderNode::Uniform::HINT_DEFAULT_BLACK: {
							rd_texture = texture_storage->texture_rd_get_default(TextureStorage::DEFAULT_RD_TEXTURE_BLACK);
						} break;
						case ShaderLanguage::ShaderNode::Uniform::HINT_DEFAULT_TRANSPARENT: {
							rd_texture = texture_storage->texture_rd_get_default(TextureStorage::DEFAULT_RD_TEXTURE_TRANSPARENT);
						} break;
						case ShaderLanguage::ShaderNode::Uniform::HINT_ANISOTROPY: {
							rd_texture = texture_storage->texture_rd_get_default(TextureStorage::DEFAULT_RD_TEXTURE_ANISO);
						} break;
						case ShaderLanguage::ShaderNode::Uniform::HINT_NORMAL: {
							rd_texture = texture_storage->texture_rd_get_default(TextureStorage::DEFAULT_RD_TEXTURE_NORMAL);
						} break;
						case ShaderLanguage::ShaderNode::Uniform::HINT_ROUGHNESS_NORMAL: {
							rd_texture = texture_storage->texture_rd_get_default(TextureStorage::DEFAULT_RD_TEXTURE_NORMAL);
						} break;
						default: {
							rd_texture = texture_storage->texture_rd_get_default(TextureStorage::DEFAULT_RD_TEXTURE_WHITE);
						} break;
					}
				} break;

				case ShaderLanguage::TYPE_SAMPLERCUBE: {
					switch (p_texture_uniforms[i].hint) {
						case ShaderLanguage::ShaderNode::Uniform::HINT_DEFAULT_BLACK: {
							rd_texture = texture_storage->texture_rd_get_default(TextureStorage::DEFAULT_RD_TEXTURE_CUBEMAP_BLACK);
						} break;
						default: {
							rd_texture = texture_storage->texture_rd_get_default(TextureStorage::DEFAULT_RD_TEXTURE_CUBEMAP_WHITE);
						} break;
					}
				} break;
				case ShaderLanguage::TYPE_SAMPLERCUBEARRAY: {
					rd_texture = texture_storage->texture_rd_get_default(TextureStorage::DEFAULT_RD_TEXTURE_CUBEMAP_ARRAY_BLACK);
				} break;

				case ShaderLanguage::TYPE_ISAMPLER3D:
				case ShaderLanguage::TYPE_USAMPLER3D:
				case ShaderLanguage::TYPE_SAMPLER3D: {
					rd_texture = texture_storage->texture_rd_get_default(TextureStorage::DEFAULT_RD_TEXTURE_3D_WHITE);
				} break;

				case ShaderLanguage::TYPE_ISAMPLER2DARRAY:
				case ShaderLanguage::TYPE_USAMPLER2DARRAY:
				case ShaderLanguage::TYPE_SAMPLER2DARRAY: {
					rd_texture = texture_storage->texture_rd_get_default(TextureStorage::DEFAULT_RD_TEXTURE_2D_ARRAY_WHITE);
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
					p_textures[k++] = rd_texture;
				}
			} else {
				p_textures[k++] = rd_texture;
			}
		} else {
			bool srgb = p_use_linear_color && p_texture_uniforms[i].use_color;

			for (int j = 0; j < textures.size(); j++) {
				TextureStorage::Texture *tex = TextureStorage::get_singleton()->get_texture(textures[j]);

				if (tex) {
					rd_texture = (srgb && tex->rd_texture_srgb.is_valid()) ? tex->rd_texture_srgb : tex->rd_texture;
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
				if (rd_texture.is_null()) {
					rd_texture = texture_storage->texture_rd_get_default(TextureStorage::DEFAULT_RD_TEXTURE_WHITE);
				}
#ifdef TOOLS_ENABLED
				if (roughness_detect_texture && normal_detect_texture && !normal_detect_texture->path.is_empty()) {
					roughness_detect_texture->detect_roughness_callback(roughness_detect_texture->detect_roughness_callback_ud, normal_detect_texture->path, roughness_channel);
				}
#endif
				p_textures[k++] = rd_texture;
			}
		}
	}
	{
		//for textures no longer used, unregister them
		List<StringName> to_delete;
		for (KeyValue<StringName, uint64_t> &E : used_global_textures) {
			if (E.value != global_textures_pass) {
				to_delete.push_back(E.key);

				GlobalShaderUniforms::Variable *v = material_storage->global_shader_uniforms.variables.getptr(E.key);
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
				global_texture_E = material_storage->global_shader_uniforms.materials_using_texture.push_back(self);
			} else {
				material_storage->global_shader_uniforms.materials_using_texture.erase(global_texture_E);
				global_texture_E = nullptr;
			}
		}
	}
}

void MaterialStorage::MaterialData::free_parameters_uniform_set(RID p_uniform_set) {
	if (p_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(p_uniform_set)) {
		RD::get_singleton()->uniform_set_set_invalidation_callback(p_uniform_set, nullptr, nullptr);
		RD::get_singleton()->free(p_uniform_set);
	}
}

bool MaterialStorage::MaterialData::update_parameters_uniform_set(const HashMap<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty, const HashMap<StringName, ShaderLanguage::ShaderNode::Uniform> &p_uniforms, const uint32_t *p_uniform_offsets, const Vector<ShaderCompiler::GeneratedCode::Texture> &p_texture_uniforms, const HashMap<StringName, HashMap<int, RID>> &p_default_texture_params, uint32_t p_ubo_size, RID &uniform_set, RID p_shader, uint32_t p_shader_uniform_set, uint32_t p_barrier) {
	if ((uint32_t)ubo_data.size() != p_ubo_size) {
		p_uniform_dirty = true;
		if (uniform_buffer.is_valid()) {
			RD::get_singleton()->free(uniform_buffer);
			uniform_buffer = RID();
		}

		ubo_data.resize(p_ubo_size);
		if (ubo_data.size()) {
			uniform_buffer = RD::get_singleton()->uniform_buffer_create(ubo_data.size());
			memset(ubo_data.ptrw(), 0, ubo_data.size()); //clear
		}

		//clear previous uniform set
		if (uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			RD::get_singleton()->uniform_set_set_invalidation_callback(uniform_set, nullptr, nullptr);
			RD::get_singleton()->free(uniform_set);
			uniform_set = RID();
		}
	}

	//check whether buffer changed
	if (p_uniform_dirty && ubo_data.size()) {
		update_uniform_buffer(p_uniforms, p_uniform_offsets, p_parameters, ubo_data.ptrw(), ubo_data.size(), true);
		RD::get_singleton()->buffer_update(uniform_buffer, 0, ubo_data.size(), ubo_data.ptrw(), p_barrier);
	}

	uint32_t tex_uniform_count = 0U;
	for (int i = 0; i < p_texture_uniforms.size(); i++) {
		tex_uniform_count += uint32_t(p_texture_uniforms[i].array_size > 0 ? p_texture_uniforms[i].array_size : 1);
	}

	if ((uint32_t)texture_cache.size() != tex_uniform_count || p_textures_dirty) {
		texture_cache.resize(tex_uniform_count);
		p_textures_dirty = true;

		//clear previous uniform set
		if (uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			RD::get_singleton()->uniform_set_set_invalidation_callback(uniform_set, nullptr, nullptr);
			RD::get_singleton()->free(uniform_set);
			uniform_set = RID();
		}
	}

	if (p_textures_dirty && tex_uniform_count) {
		update_textures(p_parameters, p_default_texture_params, p_texture_uniforms, texture_cache.ptrw(), true);
	}

	if (p_ubo_size == 0 && (p_texture_uniforms.size() == 0)) {
		// This material does not require an uniform set, so don't create it.
		return false;
	}

	if (!p_textures_dirty && uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
		//no reason to update uniform set, only UBO (or nothing) was needed to update
		return false;
	}

	Vector<RD::Uniform> uniforms;

	{
		if (p_ubo_size) {
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.binding = 0;
			u.append_id(uniform_buffer);
			uniforms.push_back(u);
		}

		const RID *textures = texture_cache.ptrw();
		for (int i = 0, k = 0; i < p_texture_uniforms.size(); i++) {
			const int array_size = p_texture_uniforms[i].array_size;

			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 1 + k;
			if (array_size > 0) {
				for (int j = 0; j < array_size; j++) {
					u.append_id(textures[k++]);
				}
			} else {
				u.append_id(textures[k++]);
			}
			uniforms.push_back(u);
		}
	}

	uniform_set = RD::get_singleton()->uniform_set_create(uniforms, p_shader, p_shader_uniform_set);

	RD::get_singleton()->uniform_set_set_invalidation_callback(uniform_set, MaterialStorage::_material_uniform_set_erased, &self);

	return true;
}

///////////////////////////////////////////////////////////////////////////
// MaterialStorage

MaterialStorage *MaterialStorage::singleton = nullptr;

MaterialStorage *MaterialStorage::get_singleton() {
	return singleton;
}

MaterialStorage::MaterialStorage() {
	singleton = this;

	//default samplers
	for (int i = 1; i < RS::CANVAS_ITEM_TEXTURE_FILTER_MAX; i++) {
		for (int j = 1; j < RS::CANVAS_ITEM_TEXTURE_REPEAT_MAX; j++) {
			RD::SamplerState sampler_state;
			switch (i) {
				case RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.max_lod = 0;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.max_lod = 0;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}

				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}
					sampler_state.use_anisotropy = true;
					sampler_state.anisotropy_max = 1 << int(GLOBAL_GET("rendering/textures/default_filters/anisotropic_filtering_level"));
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}
					sampler_state.use_anisotropy = true;
					sampler_state.anisotropy_max = 1 << int(GLOBAL_GET("rendering/textures/default_filters/anisotropic_filtering_level"));

				} break;
				default: {
				}
			}
			switch (j) {
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED: {
					sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
					sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
					sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;

				} break;
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED: {
					sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_REPEAT;
					sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_REPEAT;
					sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_REPEAT;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_MIRROR: {
					sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
					sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
					sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
				} break;
				default: {
				}
			}

			default_rd_samplers[i][j] = RD::get_singleton()->sampler_create(sampler_state);
		}
	}

	//custom sampler
	sampler_rd_configure_custom(0.0f);

	// buffers
	{ //create index array for copy shaders
		Vector<uint8_t> pv;
		pv.resize(6 * 4);
		{
			uint8_t *w = pv.ptrw();
			int *p32 = (int *)w;
			p32[0] = 0;
			p32[1] = 1;
			p32[2] = 2;
			p32[3] = 0;
			p32[4] = 2;
			p32[5] = 3;
		}
		quad_index_buffer = RD::get_singleton()->index_buffer_create(6, RenderingDevice::INDEX_BUFFER_FORMAT_UINT32, pv);
		quad_index_array = RD::get_singleton()->index_array_create(quad_index_buffer, 0, 6);
	}

	// Shaders
	for (int i = 0; i < SHADER_TYPE_MAX; i++) {
		shader_data_request_func[i] = nullptr;
	}

	static_assert(sizeof(GlobalShaderUniforms::Value) == 16);

	global_shader_uniforms.buffer_size = MAX(4096, (int)GLOBAL_GET("rendering/limits/global_shader_variables/buffer_size"));
	global_shader_uniforms.buffer_values = memnew_arr(GlobalShaderUniforms::Value, global_shader_uniforms.buffer_size);
	memset(global_shader_uniforms.buffer_values, 0, sizeof(GlobalShaderUniforms::Value) * global_shader_uniforms.buffer_size);
	global_shader_uniforms.buffer_usage = memnew_arr(GlobalShaderUniforms::ValueUsage, global_shader_uniforms.buffer_size);
	global_shader_uniforms.buffer_dirty_regions = memnew_arr(bool, global_shader_uniforms.buffer_size / GlobalShaderUniforms::BUFFER_DIRTY_REGION_SIZE);
	memset(global_shader_uniforms.buffer_dirty_regions, 0, sizeof(bool) * global_shader_uniforms.buffer_size / GlobalShaderUniforms::BUFFER_DIRTY_REGION_SIZE);
	global_shader_uniforms.buffer = RD::get_singleton()->storage_buffer_create(sizeof(GlobalShaderUniforms::Value) * global_shader_uniforms.buffer_size);
}

MaterialStorage::~MaterialStorage() {
	memdelete_arr(global_shader_uniforms.buffer_values);
	memdelete_arr(global_shader_uniforms.buffer_usage);
	memdelete_arr(global_shader_uniforms.buffer_dirty_regions);
	RD::get_singleton()->free(global_shader_uniforms.buffer);

	// buffers

	RD::get_singleton()->free(quad_index_buffer); //array gets freed as dependency

	//def samplers
	for (int i = 1; i < RS::CANVAS_ITEM_TEXTURE_FILTER_MAX; i++) {
		for (int j = 1; j < RS::CANVAS_ITEM_TEXTURE_REPEAT_MAX; j++) {
			RD::get_singleton()->free(default_rd_samplers[i][j]);
		}
	}

	//custom samplers
	for (int i = 1; i < RS::CANVAS_ITEM_TEXTURE_FILTER_MAX; i++) {
		for (int j = 0; j < RS::CANVAS_ITEM_TEXTURE_REPEAT_MAX; j++) {
			if (custom_rd_samplers[i][j].is_valid()) {
				RD::get_singleton()->free(custom_rd_samplers[i][j]);
			}
		}
	}

	singleton = nullptr;
}

/* Samplers */

void MaterialStorage::sampler_rd_configure_custom(float p_mipmap_bias) {
	for (int i = 1; i < RS::CANVAS_ITEM_TEXTURE_FILTER_MAX; i++) {
		for (int j = 1; j < RS::CANVAS_ITEM_TEXTURE_REPEAT_MAX; j++) {
			RD::SamplerState sampler_state;
			switch (i) {
				case RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.max_lod = 0;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.max_lod = 0;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}
					sampler_state.lod_bias = p_mipmap_bias;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}
					sampler_state.lod_bias = p_mipmap_bias;

				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}
					sampler_state.lod_bias = p_mipmap_bias;
					sampler_state.use_anisotropy = true;
					sampler_state.anisotropy_max = 1 << int(GLOBAL_GET("rendering/textures/default_filters/anisotropic_filtering_level"));
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}
					sampler_state.lod_bias = p_mipmap_bias;
					sampler_state.use_anisotropy = true;
					sampler_state.anisotropy_max = 1 << int(GLOBAL_GET("rendering/textures/default_filters/anisotropic_filtering_level"));

				} break;
				default: {
				}
			}
			switch (j) {
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED: {
					sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
					sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
					sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;

				} break;
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED: {
					sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_REPEAT;
					sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_REPEAT;
					sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_REPEAT;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_MIRROR: {
					sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
					sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
					sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
				} break;
				default: {
				}
			}

			if (custom_rd_samplers[i][j].is_valid()) {
				RD::get_singleton()->free(custom_rd_samplers[i][j]);
			}

			custom_rd_samplers[i][j] = RD::get_singleton()->sampler_create(sampler_state);
		}
	}
}

/* GLOBAL SHADER UNIFORM API */

int32_t MaterialStorage::_global_shader_uniform_allocate(uint32_t p_elements) {
	int32_t idx = 0;
	while (idx + p_elements <= global_shader_uniforms.buffer_size) {
		if (global_shader_uniforms.buffer_usage[idx].elements == 0) {
			bool valid = true;
			for (uint32_t i = 1; i < p_elements; i++) {
				if (global_shader_uniforms.buffer_usage[idx + i].elements > 0) {
					valid = false;
					idx += i + global_shader_uniforms.buffer_usage[idx + i].elements;
					break;
				}
			}

			if (!valid) {
				continue; //if not valid, idx is in new position
			}

			return idx;
		} else {
			idx += global_shader_uniforms.buffer_usage[idx].elements;
		}
	}

	return -1;
}

void MaterialStorage::_global_shader_uniform_store_in_buffer(int32_t p_index, RS::GlobalShaderParameterType p_type, const Variant &p_value) {
	switch (p_type) {
		case RS::GLOBAL_VAR_TYPE_BOOL: {
			GlobalShaderUniforms::Value &bv = global_shader_uniforms.buffer_values[p_index];
			bool b = p_value;
			bv.x = b ? 1.0 : 0.0;
			bv.y = 0.0;
			bv.z = 0.0;
			bv.w = 0.0;

		} break;
		case RS::GLOBAL_VAR_TYPE_BVEC2: {
			GlobalShaderUniforms::Value &bv = global_shader_uniforms.buffer_values[p_index];
			uint32_t bvec = p_value;
			bv.x = (bvec & 1) ? 1.0 : 0.0;
			bv.y = (bvec & 2) ? 1.0 : 0.0;
			bv.z = 0.0;
			bv.w = 0.0;
		} break;
		case RS::GLOBAL_VAR_TYPE_BVEC3: {
			GlobalShaderUniforms::Value &bv = global_shader_uniforms.buffer_values[p_index];
			uint32_t bvec = p_value;
			bv.x = (bvec & 1) ? 1.0 : 0.0;
			bv.y = (bvec & 2) ? 1.0 : 0.0;
			bv.z = (bvec & 4) ? 1.0 : 0.0;
			bv.w = 0.0;
		} break;
		case RS::GLOBAL_VAR_TYPE_BVEC4: {
			GlobalShaderUniforms::Value &bv = global_shader_uniforms.buffer_values[p_index];
			uint32_t bvec = p_value;
			bv.x = (bvec & 1) ? 1.0 : 0.0;
			bv.y = (bvec & 2) ? 1.0 : 0.0;
			bv.z = (bvec & 4) ? 1.0 : 0.0;
			bv.w = (bvec & 8) ? 1.0 : 0.0;
		} break;
		case RS::GLOBAL_VAR_TYPE_INT: {
			GlobalShaderUniforms::ValueInt &bv = *(GlobalShaderUniforms::ValueInt *)&global_shader_uniforms.buffer_values[p_index];
			int32_t v = p_value;
			bv.x = v;
			bv.y = 0;
			bv.z = 0;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_IVEC2: {
			GlobalShaderUniforms::ValueInt &bv = *(GlobalShaderUniforms::ValueInt *)&global_shader_uniforms.buffer_values[p_index];
			Vector2i v = p_value;
			bv.x = v.x;
			bv.y = v.y;
			bv.z = 0;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_IVEC3: {
			GlobalShaderUniforms::ValueInt &bv = *(GlobalShaderUniforms::ValueInt *)&global_shader_uniforms.buffer_values[p_index];
			Vector3i v = p_value;
			bv.x = v.x;
			bv.y = v.y;
			bv.z = v.z;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_IVEC4: {
			GlobalShaderUniforms::ValueInt &bv = *(GlobalShaderUniforms::ValueInt *)&global_shader_uniforms.buffer_values[p_index];
			Vector<int32_t> v = p_value;
			bv.x = v.size() >= 1 ? v[0] : 0;
			bv.y = v.size() >= 2 ? v[1] : 0;
			bv.z = v.size() >= 3 ? v[2] : 0;
			bv.w = v.size() >= 4 ? v[3] : 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_RECT2I: {
			GlobalShaderUniforms::ValueInt &bv = *(GlobalShaderUniforms::ValueInt *)&global_shader_uniforms.buffer_values[p_index];
			Rect2i v = p_value;
			bv.x = v.position.x;
			bv.y = v.position.y;
			bv.z = v.size.x;
			bv.w = v.size.y;
		} break;
		case RS::GLOBAL_VAR_TYPE_UINT: {
			GlobalShaderUniforms::ValueUInt &bv = *(GlobalShaderUniforms::ValueUInt *)&global_shader_uniforms.buffer_values[p_index];
			uint32_t v = p_value;
			bv.x = v;
			bv.y = 0;
			bv.z = 0;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_UVEC2: {
			GlobalShaderUniforms::ValueUInt &bv = *(GlobalShaderUniforms::ValueUInt *)&global_shader_uniforms.buffer_values[p_index];
			Vector2i v = p_value;
			bv.x = v.x;
			bv.y = v.y;
			bv.z = 0;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_UVEC3: {
			GlobalShaderUniforms::ValueUInt &bv = *(GlobalShaderUniforms::ValueUInt *)&global_shader_uniforms.buffer_values[p_index];
			Vector3i v = p_value;
			bv.x = v.x;
			bv.y = v.y;
			bv.z = v.z;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_UVEC4: {
			GlobalShaderUniforms::ValueUInt &bv = *(GlobalShaderUniforms::ValueUInt *)&global_shader_uniforms.buffer_values[p_index];
			Vector<int32_t> v = p_value;
			bv.x = v.size() >= 1 ? v[0] : 0;
			bv.y = v.size() >= 2 ? v[1] : 0;
			bv.z = v.size() >= 3 ? v[2] : 0;
			bv.w = v.size() >= 4 ? v[3] : 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_FLOAT: {
			GlobalShaderUniforms::Value &bv = global_shader_uniforms.buffer_values[p_index];
			float v = p_value;
			bv.x = v;
			bv.y = 0;
			bv.z = 0;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_VEC2: {
			GlobalShaderUniforms::Value &bv = global_shader_uniforms.buffer_values[p_index];
			Vector2 v = p_value;
			bv.x = v.x;
			bv.y = v.y;
			bv.z = 0;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_VEC3: {
			GlobalShaderUniforms::Value &bv = global_shader_uniforms.buffer_values[p_index];
			Vector3 v = p_value;
			bv.x = v.x;
			bv.y = v.y;
			bv.z = v.z;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_VEC4: {
			GlobalShaderUniforms::Value &bv = global_shader_uniforms.buffer_values[p_index];
			Plane v = p_value;
			bv.x = v.normal.x;
			bv.y = v.normal.y;
			bv.z = v.normal.z;
			bv.w = v.d;
		} break;
		case RS::GLOBAL_VAR_TYPE_COLOR: {
			GlobalShaderUniforms::Value &bv = global_shader_uniforms.buffer_values[p_index];
			Color v = p_value;
			bv.x = v.r;
			bv.y = v.g;
			bv.z = v.b;
			bv.w = v.a;

			GlobalShaderUniforms::Value &bv_linear = global_shader_uniforms.buffer_values[p_index + 1];
			v = v.srgb_to_linear();
			bv_linear.x = v.r;
			bv_linear.y = v.g;
			bv_linear.z = v.b;
			bv_linear.w = v.a;

		} break;
		case RS::GLOBAL_VAR_TYPE_RECT2: {
			GlobalShaderUniforms::Value &bv = global_shader_uniforms.buffer_values[p_index];
			Rect2 v = p_value;
			bv.x = v.position.x;
			bv.y = v.position.y;
			bv.z = v.size.x;
			bv.w = v.size.y;
		} break;
		case RS::GLOBAL_VAR_TYPE_MAT2: {
			GlobalShaderUniforms::Value *bv = &global_shader_uniforms.buffer_values[p_index];
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
			GlobalShaderUniforms::Value *bv = &global_shader_uniforms.buffer_values[p_index];
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
			GlobalShaderUniforms::Value *bv = &global_shader_uniforms.buffer_values[p_index];

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
			GlobalShaderUniforms::Value *bv = &global_shader_uniforms.buffer_values[p_index];
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
			GlobalShaderUniforms::Value *bv = &global_shader_uniforms.buffer_values[p_index];
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

void MaterialStorage::_global_shader_uniform_mark_buffer_dirty(int32_t p_index, int32_t p_elements) {
	int32_t prev_chunk = -1;

	for (int32_t i = 0; i < p_elements; i++) {
		int32_t chunk = (p_index + i) / GlobalShaderUniforms::BUFFER_DIRTY_REGION_SIZE;
		if (chunk != prev_chunk) {
			if (!global_shader_uniforms.buffer_dirty_regions[chunk]) {
				global_shader_uniforms.buffer_dirty_regions[chunk] = true;
				global_shader_uniforms.buffer_dirty_region_count++;
			}
		}

		prev_chunk = chunk;
	}
}

void MaterialStorage::global_shader_parameter_add(const StringName &p_name, RS::GlobalShaderParameterType p_type, const Variant &p_value) {
	ERR_FAIL_COND(global_shader_uniforms.variables.has(p_name));
	GlobalShaderUniforms::Variable gv;
	gv.type = p_type;
	gv.value = p_value;
	gv.buffer_index = -1;

	if (p_type >= RS::GLOBAL_VAR_TYPE_SAMPLER2D) {
		//is texture
		global_shader_uniforms.must_update_texture_materials = true; //normally there are none
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
		gv.buffer_index = _global_shader_uniform_allocate(gv.buffer_elements);
		ERR_FAIL_COND_MSG(gv.buffer_index < 0, vformat("Failed allocating global variable '%s' out of buffer memory. Consider increasing it in the Project Settings.", String(p_name)));
		global_shader_uniforms.buffer_usage[gv.buffer_index].elements = gv.buffer_elements;
		_global_shader_uniform_store_in_buffer(gv.buffer_index, gv.type, gv.value);
		_global_shader_uniform_mark_buffer_dirty(gv.buffer_index, gv.buffer_elements);

		global_shader_uniforms.must_update_buffer_materials = true; //normally there are none
	}

	global_shader_uniforms.variables[p_name] = gv;
}

void MaterialStorage::global_shader_parameter_remove(const StringName &p_name) {
	if (!global_shader_uniforms.variables.has(p_name)) {
		return;
	}
	const GlobalShaderUniforms::Variable &gv = global_shader_uniforms.variables[p_name];

	if (gv.buffer_index >= 0) {
		global_shader_uniforms.buffer_usage[gv.buffer_index].elements = 0;
		global_shader_uniforms.must_update_buffer_materials = true;
	} else {
		global_shader_uniforms.must_update_texture_materials = true;
	}

	global_shader_uniforms.variables.erase(p_name);
}

Vector<StringName> MaterialStorage::global_shader_parameter_get_list() const {
	if (!Engine::get_singleton()->is_editor_hint()) {
		ERR_FAIL_V_MSG(Vector<StringName>(), "This function should never be used outside the editor, it can severely damage performance.");
	}

	Vector<StringName> names;
	for (const KeyValue<StringName, GlobalShaderUniforms::Variable> &E : global_shader_uniforms.variables) {
		names.push_back(E.key);
	}
	names.sort_custom<StringName::AlphCompare>();
	return names;
}

void MaterialStorage::global_shader_parameter_set(const StringName &p_name, const Variant &p_value) {
	ERR_FAIL_COND(!global_shader_uniforms.variables.has(p_name));
	GlobalShaderUniforms::Variable &gv = global_shader_uniforms.variables[p_name];
	gv.value = p_value;
	if (gv.override.get_type() == Variant::NIL) {
		if (gv.buffer_index >= 0) {
			//buffer
			_global_shader_uniform_store_in_buffer(gv.buffer_index, gv.type, gv.value);
			_global_shader_uniform_mark_buffer_dirty(gv.buffer_index, gv.buffer_elements);
		} else {
			//texture
			MaterialStorage *material_storage = MaterialStorage::get_singleton();
			for (const RID &E : gv.texture_materials) {
				Material *material = material_storage->get_material(E);
				ERR_CONTINUE(!material);
				material_storage->_material_queue_update(material, false, true);
			}
		}
	}
}

void MaterialStorage::global_shader_parameter_set_override(const StringName &p_name, const Variant &p_value) {
	if (!global_shader_uniforms.variables.has(p_name)) {
		return; //variable may not exist
	}

	ERR_FAIL_COND(p_value.get_type() == Variant::OBJECT);

	GlobalShaderUniforms::Variable &gv = global_shader_uniforms.variables[p_name];

	gv.override = p_value;

	if (gv.buffer_index >= 0) {
		//buffer
		if (gv.override.get_type() == Variant::NIL) {
			_global_shader_uniform_store_in_buffer(gv.buffer_index, gv.type, gv.value);
		} else {
			_global_shader_uniform_store_in_buffer(gv.buffer_index, gv.type, gv.override);
		}

		_global_shader_uniform_mark_buffer_dirty(gv.buffer_index, gv.buffer_elements);
	} else {
		//texture
		MaterialStorage *material_storage = MaterialStorage::get_singleton();
		for (const RID &E : gv.texture_materials) {
			Material *material = material_storage->get_material(E);
			ERR_CONTINUE(!material);
			material_storage->_material_queue_update(material, false, true);
		}
	}
}

Variant MaterialStorage::global_shader_parameter_get(const StringName &p_name) const {
	if (!Engine::get_singleton()->is_editor_hint()) {
		ERR_FAIL_V_MSG(Variant(), "This function should never be used outside the editor, it can severely damage performance.");
	}

	if (!global_shader_uniforms.variables.has(p_name)) {
		return Variant();
	}

	return global_shader_uniforms.variables[p_name].value;
}

RS::GlobalShaderParameterType MaterialStorage::global_shader_parameter_get_type_internal(const StringName &p_name) const {
	if (!global_shader_uniforms.variables.has(p_name)) {
		return RS::GLOBAL_VAR_TYPE_MAX;
	}

	return global_shader_uniforms.variables[p_name].type;
}

RS::GlobalShaderParameterType MaterialStorage::global_shader_parameter_get_type(const StringName &p_name) const {
	if (!Engine::get_singleton()->is_editor_hint()) {
		ERR_FAIL_V_MSG(RS::GLOBAL_VAR_TYPE_MAX, "This function should never be used outside the editor, it can severely damage performance.");
	}

	return global_shader_parameter_get_type_internal(p_name);
}

void MaterialStorage::global_shader_parameters_load_settings(bool p_load_textures) {
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

			RS::GlobalShaderParameterType gvtype = RS::GLOBAL_VAR_TYPE_MAX;

			for (int i = 0; i < RS::GLOBAL_VAR_TYPE_MAX; i++) {
				if (global_var_type_names[i] == type) {
					gvtype = RS::GlobalShaderParameterType(i);
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

			if (global_shader_uniforms.variables.has(name)) {
				//has it, update it
				global_shader_parameter_set(name, value);
			} else {
				global_shader_parameter_add(name, gvtype, value);
			}
		}
	}
}

void MaterialStorage::global_shader_parameters_clear() {
	global_shader_uniforms.variables.clear(); //not right but for now enough
}

RID MaterialStorage::global_shader_uniforms_get_storage_buffer() const {
	return global_shader_uniforms.buffer;
}

//An alternative way of allocating global shader uniforms.
//Allocates a single variable and returns its position in the global shader uniform buffer.
//Intended to be used for debug drawing purposes and should not be used for anything substantial.
int32_t MaterialStorage::global_shader_parameters_unit_variable_allocate() {
	int32_t pos = _global_shader_uniform_allocate(1);
	ERR_FAIL_COND_V_MSG(pos < 0, -1, "Too many instances using shader instance variables. Increase buffer size in Project Settings.");
	global_shader_uniforms.buffer_usage[pos].elements = 1;
	return pos;
}

void MaterialStorage::global_shader_parameters_unit_variable_free(int32_t p_pos) {
	ERR_FAIL_COND(p_pos < 0);
	global_shader_uniforms.buffer_usage[p_pos].elements = 0;
}

void MaterialStorage::global_shader_parameters_unit_variable_update(int32_t p_pos, const Variant &p_value) {
	ERR_FAIL_COND(p_pos < 0);
	ERR_FAIL_COND_MSG(p_value.get_type() > Variant::COLOR, "Unsupported variant type."); //anything greater not supported

	const ShaderLanguage::DataType datatype_from_value[Variant::COLOR + 1] = {
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
		ShaderLanguage::TYPE_VEC4, //vec4
		ShaderLanguage::TYPE_IVEC4, //vec4i
		ShaderLanguage::TYPE_VEC4, //plane
		ShaderLanguage::TYPE_VEC4, //quat
		ShaderLanguage::TYPE_MAX, //aabb not supported here
		ShaderLanguage::TYPE_MAX, //basis not supported here
		ShaderLanguage::TYPE_MAX, //xform not supported here
		ShaderLanguage::TYPE_MAX, //projection not supported here
		ShaderLanguage::TYPE_VEC4 //color
	};

	ShaderLanguage::DataType datatype = datatype_from_value[p_value.get_type()];

	ERR_FAIL_COND_MSG(datatype == ShaderLanguage::TYPE_MAX, "Unsupported variant type.");

	_fill_std140_variant_ubo_value(datatype, 0, p_value, (uint8_t *)&global_shader_uniforms.buffer_values[p_pos], true); //instances always use linear color in this renderer
	_global_shader_uniform_mark_buffer_dirty(p_pos, 1);
}

int32_t MaterialStorage::global_shader_parameters_instance_allocate(RID p_instance) {
	ERR_FAIL_COND_V(global_shader_uniforms.instance_buffer_pos.has(p_instance), -1);
	int32_t pos = _global_shader_uniform_allocate(ShaderLanguage::MAX_INSTANCE_UNIFORM_INDICES);
	global_shader_uniforms.instance_buffer_pos[p_instance] = pos; //save anyway
	ERR_FAIL_COND_V_MSG(pos < 0, -1, "Too many instances using shader instance variables. Increase buffer size in Project Settings.");
	global_shader_uniforms.buffer_usage[pos].elements = ShaderLanguage::MAX_INSTANCE_UNIFORM_INDICES;
	return pos;
}

void MaterialStorage::global_shader_parameters_instance_free(RID p_instance) {
	ERR_FAIL_COND(!global_shader_uniforms.instance_buffer_pos.has(p_instance));
	int32_t pos = global_shader_uniforms.instance_buffer_pos[p_instance];
	if (pos >= 0) {
		global_shader_uniforms.buffer_usage[pos].elements = 0;
	}
	global_shader_uniforms.instance_buffer_pos.erase(p_instance);
}

void MaterialStorage::global_shader_parameters_instance_update(RID p_instance, int p_index, const Variant &p_value) {
	if (!global_shader_uniforms.instance_buffer_pos.has(p_instance)) {
		return; //just not allocated, ignore
	}
	int32_t pos = global_shader_uniforms.instance_buffer_pos[p_instance];

	if (pos < 0) {
		return; //again, not allocated, ignore
	}
	ERR_FAIL_INDEX(p_index, ShaderLanguage::MAX_INSTANCE_UNIFORM_INDICES);
	ERR_FAIL_COND_MSG(p_value.get_type() > Variant::COLOR, "Unsupported variant type for instance parameter: " + Variant::get_type_name(p_value.get_type())); //anything greater not supported

	const ShaderLanguage::DataType datatype_from_value[Variant::COLOR + 1] = {
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
		ShaderLanguage::TYPE_VEC4, //vec4
		ShaderLanguage::TYPE_IVEC4, //vec4i
		ShaderLanguage::TYPE_VEC4, //plane
		ShaderLanguage::TYPE_VEC4, //quat
		ShaderLanguage::TYPE_MAX, //aabb not supported here
		ShaderLanguage::TYPE_MAX, //basis not supported here
		ShaderLanguage::TYPE_MAX, //xform not supported here
		ShaderLanguage::TYPE_MAX, //projection not supported here
		ShaderLanguage::TYPE_VEC4 //color
	};

	ShaderLanguage::DataType datatype = datatype_from_value[p_value.get_type()];

	ERR_FAIL_COND_MSG(datatype == ShaderLanguage::TYPE_MAX, "Unsupported variant type for instance parameter: " + Variant::get_type_name(p_value.get_type())); //anything greater not supported

	pos += p_index;

	_fill_std140_variant_ubo_value(datatype, 0, p_value, (uint8_t *)&global_shader_uniforms.buffer_values[pos], true); //instances always use linear color in this renderer
	_global_shader_uniform_mark_buffer_dirty(pos, 1);
}

void MaterialStorage::_update_global_shader_uniforms() {
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	if (global_shader_uniforms.buffer_dirty_region_count > 0) {
		uint32_t total_regions = global_shader_uniforms.buffer_size / GlobalShaderUniforms::BUFFER_DIRTY_REGION_SIZE;
		if (total_regions / global_shader_uniforms.buffer_dirty_region_count <= 4) {
			// 25% of regions dirty, just update all buffer
			RD::get_singleton()->buffer_update(global_shader_uniforms.buffer, 0, sizeof(GlobalShaderUniforms::Value) * global_shader_uniforms.buffer_size, global_shader_uniforms.buffer_values);
			memset(global_shader_uniforms.buffer_dirty_regions, 0, sizeof(bool) * total_regions);
		} else {
			uint32_t region_byte_size = sizeof(GlobalShaderUniforms::Value) * GlobalShaderUniforms::BUFFER_DIRTY_REGION_SIZE;

			for (uint32_t i = 0; i < total_regions; i++) {
				if (global_shader_uniforms.buffer_dirty_regions[i]) {
					RD::get_singleton()->buffer_update(global_shader_uniforms.buffer, i * region_byte_size, region_byte_size, &global_shader_uniforms.buffer_values[i * GlobalShaderUniforms::BUFFER_DIRTY_REGION_SIZE]);

					global_shader_uniforms.buffer_dirty_regions[i] = false;
				}
			}
		}

		global_shader_uniforms.buffer_dirty_region_count = 0;
	}

	if (global_shader_uniforms.must_update_buffer_materials) {
		// only happens in the case of a buffer variable added or removed,
		// so not often.
		for (const RID &E : global_shader_uniforms.materials_using_buffer) {
			Material *material = material_storage->get_material(E);
			ERR_CONTINUE(!material); //wtf

			material_storage->_material_queue_update(material, true, false);
		}

		global_shader_uniforms.must_update_buffer_materials = false;
	}

	if (global_shader_uniforms.must_update_texture_materials) {
		// only happens in the case of a buffer variable added or removed,
		// so not often.
		for (const RID &E : global_shader_uniforms.materials_using_texture) {
			Material *material = material_storage->get_material(E);
			ERR_CONTINUE(!material); //wtf

			material_storage->_material_queue_update(material, false, true);
		}

		global_shader_uniforms.must_update_texture_materials = false;
	}
}

/* SHADER API */

RID MaterialStorage::shader_allocate() {
	return shader_owner.allocate_rid();
}

void MaterialStorage::shader_initialize(RID p_rid) {
	Shader shader;
	shader.data = nullptr;
	shader.type = SHADER_TYPE_MAX;

	shader_owner.initialize_rid(p_rid, shader);
}

void MaterialStorage::shader_free(RID p_rid) {
	Shader *shader = shader_owner.get_or_null(p_rid);
	ERR_FAIL_COND(!shader);

	//make material unreference this
	while (shader->owners.size()) {
		material_set_shader((*shader->owners.begin())->self, RID());
	}

	//clear data if exists
	if (shader->data) {
		memdelete(shader->data);
	}
	shader_owner.free(p_rid);
}

void MaterialStorage::shader_set_code(RID p_shader, const String &p_code) {
	Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND(!shader);

	shader->code = p_code;
	String mode_string = ShaderLanguage::get_shader_type(p_code);

	ShaderType new_type;
	if (mode_string == "canvas_item") {
		new_type = SHADER_TYPE_2D;
	} else if (mode_string == "particles") {
		new_type = SHADER_TYPE_PARTICLES;
	} else if (mode_string == "spatial") {
		new_type = SHADER_TYPE_3D;
	} else if (mode_string == "sky") {
		new_type = SHADER_TYPE_SKY;
	} else if (mode_string == "fog") {
		new_type = SHADER_TYPE_FOG;
	} else {
		new_type = SHADER_TYPE_MAX;
	}

	if (new_type != shader->type) {
		if (shader->data) {
			memdelete(shader->data);
			shader->data = nullptr;
		}

		for (Material *E : shader->owners) {
			Material *material = E;
			material->shader_type = new_type;
			if (material->data) {
				memdelete(material->data);
				material->data = nullptr;
			}
		}

		shader->type = new_type;

		if (new_type < SHADER_TYPE_MAX && shader_data_request_func[new_type]) {
			shader->data = shader_data_request_func[new_type]();
		} else {
			shader->type = SHADER_TYPE_MAX; //invalid
		}

		for (Material *E : shader->owners) {
			Material *material = E;
			if (shader->data) {
				material->data = material_get_data_request_function(new_type)(shader->data);
				material->data->self = material->self;
				material->data->set_next_pass(material->next_pass);
				material->data->set_render_priority(material->priority);
			}
			material->shader_type = new_type;
		}

		if (shader->data) {
			for (const KeyValue<StringName, HashMap<int, RID>> &E : shader->default_texture_parameter) {
				for (const KeyValue<int, RID> &E2 : E.value) {
					shader->data->set_default_texture_parameter(E.key, E2.value, E2.key);
				}
			}
		}
	}

	if (shader->data) {
		shader->data->set_path_hint(shader->path_hint);
		shader->data->set_code(p_code);
	}

	for (Material *E : shader->owners) {
		Material *material = E;
		material->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MATERIAL);
		_material_queue_update(material, true, true);
	}
}

void MaterialStorage::shader_set_path_hint(RID p_shader, const String &p_path) {
	Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND(!shader);

	shader->path_hint = p_path;
	if (shader->data) {
		shader->data->set_path_hint(p_path);
	}
}

String MaterialStorage::shader_get_code(RID p_shader) const {
	Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND_V(!shader, String());
	return shader->code;
}

void MaterialStorage::get_shader_parameter_list(RID p_shader, List<PropertyInfo> *p_param_list) const {
	Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND(!shader);
	if (shader->data) {
		return shader->data->get_shader_uniform_list(p_param_list);
	}
}

void MaterialStorage::shader_set_default_texture_parameter(RID p_shader, const StringName &p_name, RID p_texture, int p_index) {
	Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND(!shader);

	if (p_texture.is_valid() && TextureStorage::get_singleton()->owns_texture(p_texture)) {
		if (!shader->default_texture_parameter.has(p_name)) {
			shader->default_texture_parameter[p_name] = HashMap<int, RID>();
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
		shader->data->set_default_texture_parameter(p_name, p_texture, p_index);
	}
	for (Material *E : shader->owners) {
		Material *material = E;
		_material_queue_update(material, false, true);
	}
}

RID MaterialStorage::shader_get_default_texture_parameter(RID p_shader, const StringName &p_name, int p_index) const {
	Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND_V(!shader, RID());
	if (shader->default_texture_parameter.has(p_name) && shader->default_texture_parameter[p_name].has(p_index)) {
		return shader->default_texture_parameter[p_name][p_index];
	}

	return RID();
}

Variant MaterialStorage::shader_get_parameter_default(RID p_shader, const StringName &p_param) const {
	Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND_V(!shader, Variant());
	if (shader->data) {
		return shader->data->get_default_parameter(p_param);
	}
	return Variant();
}

void MaterialStorage::shader_set_data_request_function(ShaderType p_shader_type, ShaderDataRequestFunction p_function) {
	ERR_FAIL_INDEX(p_shader_type, SHADER_TYPE_MAX);
	shader_data_request_func[p_shader_type] = p_function;
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

void MaterialStorage::_material_uniform_set_erased(void *p_material) {
	RID rid = *(RID *)p_material;
	Material *material = MaterialStorage::get_singleton()->get_material(rid);
	if (material) {
		if (material->data) {
			// Uniform set may be gone because a dependency was erased. This happens
			// if a texture is deleted, so re-create it.
			MaterialStorage::get_singleton()->_material_queue_update(material, false, true);
		}
		material->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MATERIAL);
	}
}

void MaterialStorage::_material_queue_update(Material *material, bool p_uniform, bool p_texture) {
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
		bool uniforms_changed = false;

		if (material->data) {
			uniforms_changed = material->data->update_parameters(material->params, material->uniform_dirty, material->texture_dirty);
		}
		material->texture_dirty = false;
		material->uniform_dirty = false;

		material_update_list.remove(&material->update_element);

		if (uniforms_changed) {
			//some implementations such as 3D renderer cache the matreial uniform set, so update is required
			material->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MATERIAL);
		}
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
	Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);

	if (material->data) {
		memdelete(material->data);
		material->data = nullptr;
	}

	if (material->shader) {
		material->shader->owners.erase(material);
		material->shader = nullptr;
		material->shader_type = SHADER_TYPE_MAX;
	}

	if (p_shader.is_null()) {
		material->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MATERIAL);
		material->shader_id = 0;
		return;
	}

	Shader *shader = get_shader(p_shader);
	ERR_FAIL_COND(!shader);
	material->shader = shader;
	material->shader_type = shader->type;
	material->shader_id = p_shader.get_local_index();
	shader->owners.insert(material);

	if (shader->type == SHADER_TYPE_MAX) {
		return;
	}

	ERR_FAIL_COND(shader->data == nullptr);

	material->data = material_data_request_func[shader->type](shader->data);
	material->data->self = p_material;
	material->data->set_next_pass(material->next_pass);
	material->data->set_render_priority(material->priority);
	//updating happens later
	material->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MATERIAL);
	_material_queue_update(material, true, true);
}

MaterialStorage::ShaderData *MaterialStorage::material_get_shader_data(RID p_material) {
	const MaterialStorage::Material *material = MaterialStorage::get_singleton()->get_material(p_material);
	if (material && material->shader && material->shader->data) {
		return material->shader->data;
	}

	return nullptr;
}

void MaterialStorage::material_set_param(RID p_material, const StringName &p_param, const Variant &p_value) {
	Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);

	if (p_value.get_type() == Variant::NIL) {
		material->params.erase(p_param);
	} else {
		ERR_FAIL_COND(p_value.get_type() == Variant::OBJECT); //object not allowed
		material->params[p_param] = p_value;
	}

	if (material->shader && material->shader->data) { //shader is valid
		bool is_texture = material->shader->data->is_parameter_texture(p_param);
		_material_queue_update(material, !is_texture, is_texture);
	} else {
		_material_queue_update(material, true, true);
	}
}

Variant MaterialStorage::material_get_param(RID p_material, const StringName &p_param) const {
	Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND_V(!material, Variant());
	if (material->params.has(p_param)) {
		return material->params[p_param];
	} else {
		return Variant();
	}
}

void MaterialStorage::material_set_next_pass(RID p_material, RID p_next_material) {
	Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);

	if (material->next_pass == p_next_material) {
		return;
	}

	material->next_pass = p_next_material;
	if (material->data) {
		material->data->set_next_pass(p_next_material);
	}

	material->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_MATERIAL);
}

void MaterialStorage::material_set_render_priority(RID p_material, int priority) {
	Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);
	material->priority = priority;
	if (material->data) {
		material->data->set_render_priority(priority);
	}
}

bool MaterialStorage::material_is_animated(RID p_material) {
	Material *material = material_owner.get_or_null(p_material);
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
	Material *material = material_owner.get_or_null(p_material);
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
	Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);
	if (material->shader && material->shader->data) {
		material->shader->data->get_instance_param_list(r_parameters);

		if (material->next_pass.is_valid()) {
			material_get_instance_shader_parameters(material->next_pass, r_parameters);
		}
	}
}

void MaterialStorage::material_update_dependency(RID p_material, DependencyTracker *p_instance) {
	Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);
	p_instance->update_dependency(&material->dependency);
	if (material->next_pass.is_valid()) {
		material_update_dependency(material->next_pass, p_instance);
	}
}

void MaterialStorage::material_set_data_request_function(ShaderType p_shader_type, MaterialStorage::MaterialDataRequestFunction p_function) {
	ERR_FAIL_INDEX(p_shader_type, SHADER_TYPE_MAX);
	material_data_request_func[p_shader_type] = p_function;
}

MaterialStorage::MaterialDataRequestFunction MaterialStorage::material_get_data_request_function(ShaderType p_shader_type) {
	ERR_FAIL_INDEX_V(p_shader_type, SHADER_TYPE_MAX, nullptr);
	return material_data_request_func[p_shader_type];
}
