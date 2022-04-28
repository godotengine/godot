/*************************************************************************/
/*  rasterizer.cpp                                                       */
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

#include "rasterizer.h"

#include "core/os/os.h"
#include "core/print_string.h"

#if defined(DEBUG_ENABLED) && defined(TOOLS_ENABLED)
#include "core/project_settings.h"
#endif

Rasterizer *(*Rasterizer::_create_func)() = nullptr;

Rasterizer *Rasterizer::create() {
	return _create_func();
}

RasterizerStorage *RasterizerStorage::base_singleton = nullptr;

RasterizerStorage::RasterizerStorage() {
	base_singleton = this;
}

bool RasterizerStorage::material_uses_tangents(RID p_material) {
	return false;
}

bool RasterizerStorage::material_uses_ensure_correct_normals(RID p_material) {
	return false;
}

void RasterizerStorage::InterpolationData::notify_free_multimesh(RID p_rid) {
	// print_line("free multimesh " + itos(p_rid.get_id()));

	// if the instance was on any of the lists, remove
	multimesh_interpolate_update_list.erase_multiple_unordered(p_rid);
	multimesh_transform_update_lists[0].erase_multiple_unordered(p_rid);
	multimesh_transform_update_lists[1].erase_multiple_unordered(p_rid);
}

void RasterizerStorage::update_interpolation_tick(bool p_process) {
	// detect any that were on the previous transform list that are no longer active,
	// we should remove them from the interpolate list

	for (unsigned int n = 0; n < _interpolation_data.multimesh_transform_update_list_prev->size(); n++) {
		const RID &rid = (*_interpolation_data.multimesh_transform_update_list_prev)[n];

		bool active = true;

		// no longer active? (either the instance deleted or no longer being transformed)

		MMInterpolator *mmi = _multimesh_get_interpolator(rid);
		if (mmi && !mmi->on_transform_update_list) {
			active = false;
			mmi->on_interpolate_update_list = false;

			// make sure the most recent transform is set
			// copy data rather than use Pool = function?
			mmi->_data_interpolated = mmi->_data_curr;

			// and that both prev and current are the same, just in case of any interpolations
			mmi->_data_prev = mmi->_data_curr;

			// make sure are updated one more time to ensure the AABBs are correct
			//_instance_queue_update(instance, true);
		}

		if (!mmi) {
			active = false;
		}

		if (!active) {
			_interpolation_data.multimesh_interpolate_update_list.erase(rid);
		}
	}

	if (p_process) {
		for (unsigned int i = 0; i < _interpolation_data.multimesh_transform_update_list_curr->size(); i++) {
			const RID &rid = (*_interpolation_data.multimesh_transform_update_list_curr)[i];

			MMInterpolator *mmi = _multimesh_get_interpolator(rid);
			if (mmi) {
				// reset for next tick
				mmi->on_transform_update_list = false;
				mmi->_data_prev = mmi->_data_curr;
			}
		} // for n
	}

	// if any have left the transform list, remove from the interpolate list

	// we maintain a mirror list for the transform updates, so we can detect when an instance
	// is no longer being transformed, and remove it from the interpolate list
	SWAP(_interpolation_data.multimesh_transform_update_list_curr, _interpolation_data.multimesh_transform_update_list_prev);

	// prepare for the next iteration
	_interpolation_data.multimesh_transform_update_list_curr->clear();
}

void RasterizerStorage::update_interpolation_frame(bool p_process) {
	if (p_process) {
		// Only need 32 bit for interpolation, don't use real_t
		float f = Engine::get_singleton()->get_physics_interpolation_fraction();

		for (unsigned int c = 0; c < _interpolation_data.multimesh_interpolate_update_list.size(); c++) {
			const RID &rid = _interpolation_data.multimesh_interpolate_update_list[c];

			// We could use the TransformInterpolator here to slerp transforms, but that might be too expensive,
			// so just using a Basis lerp for now.
			MMInterpolator *mmi = _multimesh_get_interpolator(rid);
			if (mmi) {
				// make sure arrays are correct size
				DEV_ASSERT(mmi->_data_prev.size() == mmi->_data_curr.size());

				if (mmi->_data_interpolated.size() < mmi->_data_curr.size()) {
					mmi->_data_interpolated.resize(mmi->_data_curr.size());
				}
				DEV_ASSERT(mmi->_data_interpolated.size() >= mmi->_data_curr.size());

				DEV_ASSERT((mmi->_data_curr.size() % mmi->_stride) == 0);
				int num = mmi->_data_curr.size() / mmi->_stride;

				PoolVector<float>::Read r_prev = mmi->_data_prev.read();
				PoolVector<float>::Read r_curr = mmi->_data_curr.read();
				PoolVector<float>::Write w = mmi->_data_interpolated.write();

				const float *pf_prev = r_prev.ptr();
				const float *pf_curr = r_curr.ptr();
				float *pf_int = w.ptr();

				bool use_lerp = mmi->quality == 0;

				// temporary transform (needed for swizzling)
				// (transform prev, curr and result)
				Transform tp, tc, tr;

				// Test for cache friendliness versus doing branchless
				for (int n = 0; n < num; n++) {
					// Transform
					if (use_lerp) {
						for (int i = 0; i < mmi->_vf_size_xform; i++) {
							float a = pf_prev[i];
							float b = pf_curr[i];
							pf_int[i] = (a + ((b - a) * f));
						}
					} else {
						// Silly swizzling, this will slow things down. no idea why it is using this format
						// .. maybe due to the shader.
						tp.basis.elements[0][0] = pf_prev[0];
						tp.basis.elements[0][1] = pf_prev[1];
						tp.basis.elements[0][2] = pf_prev[2];
						tp.basis.elements[1][0] = pf_prev[4];
						tp.basis.elements[1][1] = pf_prev[5];
						tp.basis.elements[1][2] = pf_prev[6];
						tp.basis.elements[2][0] = pf_prev[8];
						tp.basis.elements[2][1] = pf_prev[9];
						tp.basis.elements[2][2] = pf_prev[10];
						tp.origin.x = pf_prev[3];
						tp.origin.y = pf_prev[7];
						tp.origin.z = pf_prev[11];

						tc.basis.elements[0][0] = pf_curr[0];
						tc.basis.elements[0][1] = pf_curr[1];
						tc.basis.elements[0][2] = pf_curr[2];
						tc.basis.elements[1][0] = pf_curr[4];
						tc.basis.elements[1][1] = pf_curr[5];
						tc.basis.elements[1][2] = pf_curr[6];
						tc.basis.elements[2][0] = pf_curr[8];
						tc.basis.elements[2][1] = pf_curr[9];
						tc.basis.elements[2][2] = pf_curr[10];
						tc.origin.x = pf_curr[3];
						tc.origin.y = pf_curr[7];
						tc.origin.z = pf_curr[11];

						TransformInterpolator::interpolate_transform(tp, tc, tr, f);

						pf_int[0] = tr.basis.elements[0][0];
						pf_int[1] = tr.basis.elements[0][1];
						pf_int[2] = tr.basis.elements[0][2];
						pf_int[4] = tr.basis.elements[1][0];
						pf_int[5] = tr.basis.elements[1][1];
						pf_int[6] = tr.basis.elements[1][2];
						pf_int[8] = tr.basis.elements[2][0];
						pf_int[9] = tr.basis.elements[2][1];
						pf_int[10] = tr.basis.elements[2][2];
						pf_int[3] = tr.origin.x;
						pf_int[7] = tr.origin.y;
						pf_int[11] = tr.origin.z;
					}

					pf_prev += mmi->_vf_size_xform;
					pf_curr += mmi->_vf_size_xform;
					pf_int += mmi->_vf_size_xform;

					// Color
					if (mmi->_vf_size_color == 1) {
						const uint8_t *p8_prev = (const uint8_t *)pf_prev;
						const uint8_t *p8_curr = (const uint8_t *)pf_curr;
						uint8_t *p8_int = (uint8_t *)pf_int;
						_interpolate_RGBA8(p8_prev, p8_curr, p8_int, f);

						pf_prev += 1;
						pf_curr += 1;
						pf_int += 1;
					} else if (mmi->_vf_size_color == 4) {
						for (int i = 0; i < 4; i++) {
							pf_int[i] = pf_prev[i] + ((pf_curr[i] - pf_prev[i]) * f);
						}

						pf_prev += 4;
						pf_curr += 4;
						pf_int += 4;
					}

					// Custom Data
					if (mmi->_vf_size_data == 1) {
						const uint8_t *p8_prev = (const uint8_t *)pf_prev;
						const uint8_t *p8_curr = (const uint8_t *)pf_curr;
						uint8_t *p8_int = (uint8_t *)pf_int;
						_interpolate_RGBA8(p8_prev, p8_curr, p8_int, f);

						pf_prev += 1;
						pf_curr += 1;
						pf_int += 1;
					} else if (mmi->_vf_size_data == 4) {
						for (int i = 0; i < 4; i++) {
							pf_int[i] = pf_prev[i] + ((pf_curr[i] - pf_prev[i]) * f);
						}

						pf_prev += 4;
						pf_curr += 4;
						pf_int += 4;
					}
				}

				_multimesh_set_as_bulk_array(rid, mmi->_data_interpolated);

				// make sure AABBs are constantly up to date through the interpolation?
				// NYI
			}
		} // for n
	}
}

RID RasterizerStorage::multimesh_create() {
	return _multimesh_create();
}

void RasterizerStorage::multimesh_allocate(RID p_multimesh, int p_instances, VS::MultimeshTransformFormat p_transform_format, VS::MultimeshColorFormat p_color_format, VS::MultimeshCustomDataFormat p_data) {
	MMInterpolator *mmi = _multimesh_get_interpolator(p_multimesh);
	if (mmi) {
		mmi->_transform_format = p_transform_format;
		mmi->_color_format = p_color_format;
		mmi->_data_format = p_data;
		mmi->_num_instances = p_instances;

		mmi->_vf_size_xform = p_transform_format == VS::MULTIMESH_TRANSFORM_3D ? 12 : 8;
		switch (p_color_format) {
			default: {
				mmi->_vf_size_color = 0;
			} break;
			case VS::MULTIMESH_COLOR_8BIT: {
				mmi->_vf_size_color = 1;
			} break;
			case VS::MULTIMESH_COLOR_FLOAT: {
				mmi->_vf_size_color = 4;
			} break;
		}

		switch (p_data) {
			default: {
				mmi->_vf_size_data = 0;
			} break;
			case VS::MULTIMESH_CUSTOM_DATA_8BIT: {
				mmi->_vf_size_data = 1;
			} break;
			case VS::MULTIMESH_CUSTOM_DATA_FLOAT: {
				mmi->_vf_size_data = 4;
			} break;
		}

		mmi->_stride = mmi->_vf_size_xform + mmi->_vf_size_color + mmi->_vf_size_data;

		int size_in_floats = p_instances * mmi->_stride;
		mmi->_data_curr.resize(size_in_floats);
		mmi->_data_prev.resize(size_in_floats);
		mmi->_data_interpolated.resize(size_in_floats);
	}

	return _multimesh_allocate(p_multimesh, p_instances, p_transform_format, p_color_format, p_data);
}

int RasterizerStorage::multimesh_get_instance_count(RID p_multimesh) const {
	return _multimesh_get_instance_count(p_multimesh);
}

void RasterizerStorage::multimesh_set_mesh(RID p_multimesh, RID p_mesh) {
	_multimesh_set_mesh(p_multimesh, p_mesh);
}

void RasterizerStorage::multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform &p_transform) {
	MMInterpolator *mmi = _multimesh_get_interpolator(p_multimesh);
	if (mmi) {
		if (mmi->interpolated) {
			ERR_FAIL_COND(p_index >= mmi->_num_instances);
			ERR_FAIL_COND(mmi->_vf_size_xform != 12);

			PoolVector<float>::Write w = mmi->_data_curr.write();
			int start = p_index * mmi->_stride;

			float *ptr = w.ptr();
			ptr += start;

			const Transform &t = p_transform;
			ptr[0] = t.basis.elements[0][0];
			ptr[1] = t.basis.elements[0][1];
			ptr[2] = t.basis.elements[0][2];
			ptr[3] = t.origin.x;
			ptr[4] = t.basis.elements[1][0];
			ptr[5] = t.basis.elements[1][1];
			ptr[6] = t.basis.elements[1][2];
			ptr[7] = t.origin.y;
			ptr[8] = t.basis.elements[2][0];
			ptr[9] = t.basis.elements[2][1];
			ptr[10] = t.basis.elements[2][2];
			ptr[11] = t.origin.z;

			_multimesh_add_to_interpolation_lists(p_multimesh, *mmi);

#if defined(DEBUG_ENABLED) && defined(TOOLS_ENABLED)
			if (!Engine::get_singleton()->is_in_physics_frame()) {
				static int32_t warn_count = 0;
				warn_count++;
				if (((warn_count % 2048) == 0) && GLOBAL_GET("debug/settings/physics_interpolation/enable_warnings")) {
					WARN_PRINT("Interpolated MultiMesh transform should usually be set during physics process (possibly benign).");
				}
			}
#endif
			return;
		}
	}
	_multimesh_instance_set_transform(p_multimesh, p_index, p_transform);
}

void RasterizerStorage::multimesh_instance_set_transform_2d(RID p_multimesh, int p_index, const Transform2D &p_transform) {
	_multimesh_instance_set_transform_2d(p_multimesh, p_index, p_transform);
}

void RasterizerStorage::multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color) {
	MMInterpolator *mmi = _multimesh_get_interpolator(p_multimesh);
	if (mmi) {
		if (mmi->interpolated) {
			ERR_FAIL_COND(p_index >= mmi->_num_instances);
			ERR_FAIL_COND(mmi->_vf_size_color == 0);

			PoolVector<float>::Write w = mmi->_data_curr.write();
			int start = (p_index * mmi->_stride) + mmi->_vf_size_xform;

			float *ptr = w.ptr();
			ptr += start;

			if (mmi->_vf_size_color == 4) {
				for (int n = 0; n < 4; n++) {
					ptr[n] = p_color.components[n];
				}
			} else {
#ifdef DEV_ENABLED
				// The options are currently 4, 1, or zero, but just in case this changes in future...
				ERR_FAIL_COND(mmi->_vf_size_color != 1);
#endif
				uint32_t *pui = (uint32_t *)ptr;
				*pui = p_color.to_rgba32();
			}
			_multimesh_add_to_interpolation_lists(p_multimesh, *mmi);
			return;
		}
	}

	_multimesh_instance_set_color(p_multimesh, p_index, p_color);
}
void RasterizerStorage::multimesh_instance_set_custom_data(RID p_multimesh, int p_index, const Color &p_color) {
	MMInterpolator *mmi = _multimesh_get_interpolator(p_multimesh);
	if (mmi) {
		if (mmi->interpolated) {
			ERR_FAIL_COND(p_index >= mmi->_num_instances);
			ERR_FAIL_COND(mmi->_vf_size_data == 0);

			PoolVector<float>::Write w = mmi->_data_curr.write();
			int start = (p_index * mmi->_stride) + mmi->_vf_size_xform + mmi->_vf_size_color;

			float *ptr = w.ptr();
			ptr += start;

			if (mmi->_vf_size_data == 4) {
				for (int n = 0; n < 4; n++) {
					ptr[n] = p_color.components[n];
				}
			} else {
#ifdef DEV_ENABLED
				// The options are currently 4, 1, or zero, but just in case this changes in future...
				ERR_FAIL_COND(mmi->_vf_size_data != 1);
#endif
				uint32_t *pui = (uint32_t *)ptr;
				*pui = p_color.to_rgba32();
			}
			_multimesh_add_to_interpolation_lists(p_multimesh, *mmi);
			return;
		}
	}

	_multimesh_instance_set_custom_data(p_multimesh, p_index, p_color);
}

RID RasterizerStorage::multimesh_get_mesh(RID p_multimesh) const {
	return _multimesh_get_mesh(p_multimesh);
}

Transform RasterizerStorage::multimesh_instance_get_transform(RID p_multimesh, int p_index) const {
	return _multimesh_instance_get_transform(p_multimesh, p_index);
}

Transform2D RasterizerStorage::multimesh_instance_get_transform_2d(RID p_multimesh, int p_index) const {
	return _multimesh_instance_get_transform_2d(p_multimesh, p_index);
}

Color RasterizerStorage::multimesh_instance_get_color(RID p_multimesh, int p_index) const {
	return _multimesh_instance_get_color(p_multimesh, p_index);
}

Color RasterizerStorage::multimesh_instance_get_custom_data(RID p_multimesh, int p_index) const {
	return _multimesh_instance_get_custom_data(p_multimesh, p_index);
}

void RasterizerStorage::multimesh_set_physics_interpolated(RID p_multimesh, bool p_interpolated) {
	MMInterpolator *mmi = _multimesh_get_interpolator(p_multimesh);
	if (mmi) {
		mmi->interpolated = p_interpolated;
	}
}

void RasterizerStorage::multimesh_set_physics_interpolation_quality(RID p_multimesh, int p_quality) {
	ERR_FAIL_COND((p_quality < 0) || (p_quality > 1));
	MMInterpolator *mmi = _multimesh_get_interpolator(p_multimesh);
	if (mmi) {
		mmi->quality = p_quality;
	}
}

void RasterizerStorage::multimesh_instance_reset_physics_interpolation(RID p_multimesh, int p_index) {
	MMInterpolator *mmi = _multimesh_get_interpolator(p_multimesh);
	if (mmi) {
		ERR_FAIL_INDEX(p_index, mmi->_num_instances);

		PoolVector<float>::Write w = mmi->_data_prev.write();
		PoolVector<float>::Read r = mmi->_data_curr.read();

		int start = p_index * mmi->_stride;

		for (int n = 0; n < mmi->_stride; n++) {
			w[start + n] = r[start + n];
		}
	}
}

void RasterizerStorage::_multimesh_add_to_interpolation_lists(RID p_multimesh, MMInterpolator &r_mmi) {
	if (!r_mmi.on_interpolate_update_list) {
		r_mmi.on_interpolate_update_list = true;
		_interpolation_data.multimesh_interpolate_update_list.push_back(p_multimesh);
	}

	if (!r_mmi.on_transform_update_list) {
		r_mmi.on_transform_update_list = true;
		_interpolation_data.multimesh_transform_update_list_curr->push_back(p_multimesh);
	}
}

void RasterizerStorage::multimesh_set_as_bulk_array_interpolated(RID p_multimesh, const PoolVector<float> &p_array, const PoolVector<float> &p_array_prev) {
	MMInterpolator *mmi = _multimesh_get_interpolator(p_multimesh);
	if (mmi) {
		// We are assuming that mmi->interpolated is the case,
		// (can possibly assert this?)
		// even if this flag hasn't been set - just calling this function suggests
		// interpolation is desired.
		mmi->_data_prev = p_array_prev;
		mmi->_data_curr = p_array;
		_multimesh_add_to_interpolation_lists(p_multimesh, *mmi);
#if defined(DEBUG_ENABLED) && defined(TOOLS_ENABLED)
		if (!Engine::get_singleton()->is_in_physics_frame()) {
			static int32_t warn_count = 0;
			warn_count++;
			if (((warn_count % 2048) == 0) && GLOBAL_GET("debug/settings/physics_interpolation/enable_warnings")) {
				WARN_PRINT("Interpolated MultiMesh transform should usually be set during physics process (possibly benign).");
			}
		}
#endif
	}
}

void RasterizerStorage::multimesh_set_as_bulk_array(RID p_multimesh, const PoolVector<float> &p_array) {
	MMInterpolator *mmi = _multimesh_get_interpolator(p_multimesh);
	if (mmi) {
		if (mmi->interpolated) {
			mmi->_data_curr = p_array;
			_multimesh_add_to_interpolation_lists(p_multimesh, *mmi);
#if defined(DEBUG_ENABLED) && defined(TOOLS_ENABLED)
			if (!Engine::get_singleton()->is_in_physics_frame()) {
				static int32_t warn_count = 0;
				warn_count++;
				if (((warn_count % 2048) == 0) && GLOBAL_GET("debug/settings/physics_interpolation/enable_warnings")) {
					WARN_PRINT("Interpolated MultiMesh transform should usually be set during physics process (possibly benign).");
				}
			}
#endif
			return;
		}
	}
	_multimesh_set_as_bulk_array(p_multimesh, p_array);
}

void RasterizerStorage::multimesh_set_visible_instances(RID p_multimesh, int p_visible) {
	_multimesh_set_visible_instances(p_multimesh, p_visible);
}

int RasterizerStorage::multimesh_get_visible_instances(RID p_multimesh) const {
	return _multimesh_get_visible_instances(p_multimesh);
}

AABB RasterizerStorage::multimesh_get_aabb(RID p_multimesh) const {
	return _multimesh_get_aabb(p_multimesh);
}
