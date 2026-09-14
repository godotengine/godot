/**************************************************************************/
/*  soft_body_cap_validation.h                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "soft_body_cap_state.h"

#include "core/math/transform_3d.h"
#include "core/templates/hash_set.h"
#include "core/templates/local_vector.h"
#include "core/templates/span.h"
#include "core/variant/dictionary.h"

namespace SoftBodyCapValidation {

bool fail(String *r_error, const char *p_code, const String &p_field, const String &p_detail);
bool finite_float(double p_value, float &r_value, String *r_error, const String &p_field, bool p_positive = false);
bool determinant(double p_value);
bool normal_info(uint64_t p_start, uint64_t p_count, uint32_t &r_packed, String *r_error);
bool resource_size(uint64_t p_count, uint64_t p_min, uint64_t p_max, uint64_t p_stride, String *r_error, const String &p_field);
bool matrix(const Variant &p_value, String *r_error, const String &p_field);
bool skin_targets(const JoltSoftBodyCapState &p_state, const Array &p_pose, const Transform3D &p_frame, const String &p_stage, String *r_error);
bool volume_determinant_valid(double p_determinant, double p_max_edge_squared);
bool volume_topology(const Dictionary &p_config, PackedInt32Array &r_tetrahedra, PackedInt32Array &r_faces, String *r_error);
bool schema(const JoltSoftBodyCapState &p_state, const String &p_key, String *r_error);
bool column(const Dictionary &p_config, const String &p_field, double p_default, int p_source_count, const LocalVector<int> &p_map, LocalVector<float> &r_values, String *r_error);
bool new_path(const JoltSoftBodyCapState &p_state);
bool mesh_family(const JoltSoftBodyCapState &p_state);
bool typed(const JoltSoftBodyCapState &p_state, float p_mass, int p_precision, float p_stiffness, bool p_mesh, int p_vertices, float &r_inverse_mass, float &r_base, String *r_error);
Variant snapshot(const Variant &p_value);
bool equal(const Variant &p_left, const Variant &p_right);

} // namespace SoftBodyCapValidation
