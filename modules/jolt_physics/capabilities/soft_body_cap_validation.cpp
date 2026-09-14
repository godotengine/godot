/**************************************************************************/
/*  soft_body_cap_validation.cpp                                          */
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

#include "soft_body_cap_validation.h"

#include "core/config/engine.h"
#include "core/math/math_funcs.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <map>
#include <set>

namespace SoftBodyCapValidation {

bool fail(String *r_error, const char *p_code, const String &p_field, const String &p_detail) {
	if (r_error != nullptr) {
		*r_error = String(p_code) + " " + p_field + ": " + p_detail;
	}
	return false;
}

bool finite_float(double p_value, float &r_value, String *r_error, const String &p_field, bool p_positive) {
	if (!std::isfinite(p_value) || std::abs(p_value) > FLT_MAX) {
		return fail(r_error, "SBREM-VALUE", p_field, vformat("%s is not finite float32", p_value));
	}
	r_value = (float)p_value;
	if (p_positive && !(r_value > 0.0f)) {
		return fail(r_error, "SBREM-VALUE", p_field, vformat("%s must remain positive in float32", p_value));
	}
	return true;
}

bool determinant(double p_value) {
	return std::isfinite(p_value) && std::abs(p_value) > 1.0e-12;
}

bool normal_info(uint64_t p_start, uint64_t p_count, uint32_t &r_packed, String *r_error) {
	if (p_start >= (uint64_t(1) << 24) || p_count >= 256) {
		return fail(r_error, "SBREM-SKIN-NORMAL", "skin/config.normal_info", vformat("start %d, count %d exceed packed bounds", p_start, p_count));
	}
	r_packed = uint32_t(p_start) | (uint32_t(p_count) << 24);
	return true;
}

bool resource_size(uint64_t p_count, uint64_t p_min, uint64_t p_max, uint64_t p_stride, String *r_error, const String &p_field) {
	if (p_count < p_min || p_count > p_max || p_stride == 0 || p_count > uint64_t(INT32_MAX) / p_stride) {
		return fail(r_error, "SBREM-SIZE", p_field, vformat("count %d, allowed %d..%d, stride %d", p_count, p_min, p_max, p_stride));
	}
	return true;
}

bool matrix(const Variant &p_value, String *r_error, const String &p_field) {
	if (p_value.get_type() != Variant::TRANSFORM3D) {
		return fail(r_error, "SBREM-TYPE", p_field, "expected Transform3D, got " + Variant::get_type_name(p_value.get_type()));
	}
	const Transform3D transform = p_value;
	float converted;
	for (int row = 0; row < 3; ++row) {
		for (int col = 0; col < 3; ++col) {
			if (!finite_float(transform.basis[row][col], converted, r_error, p_field + vformat(".basis[%d][%d]", row, col))) {
				return false;
			}
		}
		if (!finite_float(transform.origin[row], converted, r_error, p_field + vformat(".origin[%d]", row))) {
			return false;
		}
	}
	const Basis &b = transform.basis;
	const double det = double(b[0][0]) * (double(b[1][1]) * b[2][2] - double(b[1][2]) * b[2][1]) - double(b[0][1]) * (double(b[1][0]) * b[2][2] - double(b[1][2]) * b[2][0]) + double(b[0][2]) * (double(b[1][0]) * b[2][1] - double(b[1][1]) * b[2][0]);
	return determinant(det) || fail(r_error, "SBREM-VALUE", p_field, vformat("determinant %s requires abs(det) > 1e-12", det));
}

Variant snapshot(const Variant &p_value) {
	return p_value.duplicate(true);
}

bool equal(const Variant &p_left, const Variant &p_right) {
	if (p_left.get_type() != p_right.get_type()) {
		return false;
	}
	if (p_left.get_type() == Variant::DICTIONARY) {
		const Dictionary left = p_left;
		const Dictionary right = p_right;
		if (left.size() != right.size()) {
			return false;
		}
		for (const Variant *key = left.next(nullptr); key != nullptr; key = left.next(key)) {
			if (!right.has(*key) || !equal(left[*key], right[*key])) {
				return false;
			}
		}
		return true;
	}
	if (p_left.get_type() == Variant::ARRAY) {
		const Array left = p_left;
		const Array right = p_right;
		if (left.size() != right.size()) {
			return false;
		}
		for (int i = 0; i < left.size(); ++i) {
			if (!equal(left[i], right[i])) {
				return false;
			}
		}
		return true;
	}
	return p_left == p_right;
}

namespace {
struct Field {
	const char *name;
	Variant::Type type;
	bool required = false;
	bool array_shape = false;
	double minimum = 0;
	double maximum = FLT_MAX;
	bool positive = false;
};

bool check_numeric(double p_value, const Field &p_field, String *r_error, const String &p_path) {
	float converted;
	if (!finite_float(p_value, converted, r_error, p_path, p_field.positive)) {
		return false;
	}
	if (p_value < p_field.minimum || p_value > p_field.maximum) {
		return fail(r_error, "SBREM-VALUE", p_path, vformat("%s outside [%s, %s]", p_value, p_field.minimum, p_field.maximum));
	}
	return true;
}

bool fields(const Dictionary &p_config, const String &p_key, Span<const Field> p_fields, String *r_error) {
	if (p_config.is_empty()) {
		return fail(r_error, "SBREM-KEY", p_key, "empty Dictionary");
	}
	for (const Variant *key = p_config.next(nullptr); key != nullptr; key = p_config.next(key)) {
		bool found = false;
		for (const Field &field : p_fields) {
			found |= (key->get_type() == Variant::STRING || key->get_type() == Variant::STRING_NAME) && String(*key) == field.name;
		}
		if (!found) {
			return fail(r_error, "SBREM-KEY", p_key + "." + String(*key), "unknown field");
		}
	}
	for (const Field &field : p_fields) {
		const String path = p_key + "." + field.name;
		if (!p_config.has(field.name)) {
			if (field.required) {
				return fail(r_error, "SBREM-KEY", path, "required field is missing");
			}
			continue;
		}
		const Variant value = p_config[field.name];
		const Variant::Type type = value.get_type();
		const bool float_array = field.array_shape && field.type == Variant::FLOAT && type == Variant::PACKED_FLOAT32_ARRAY;
		const bool int_array = field.array_shape && field.type == Variant::INT && type == Variant::PACKED_INT32_ARRAY;
		if (type != field.type && !float_array && !int_array) {
			return fail(r_error, "SBREM-TYPE", path, "expected " + Variant::get_type_name(field.type) + (field.array_shape ? " or packed array" : "") + ", got " + Variant::get_type_name(type));
		}
		if (type == Variant::FLOAT || type == Variant::INT) {
			if (!check_numeric(double(value), field, r_error, path)) {
				return false;
			}
		} else if (float_array || int_array) {
			const int count = float_array ? PackedFloat32Array(value).size() : PackedInt32Array(value).size();
			if (!resource_size(count, 1, 65536, 1, r_error, path)) {
				return false;
			}
			for (int i = 0; i < count; ++i) {
				const double number = float_array ? double(PackedFloat32Array(value)[i]) : double(PackedInt32Array(value)[i]);
				if (!check_numeric(number, field, r_error, path + vformat("[%d]", i))) {
					return false;
				}
			}
		}
	}
	return true;
}
} // namespace

bool schema(const JoltSoftBodyCapState &p_state, const String &p_key, String *r_error) {
	if (!p_state.has(p_key)) {
		return true;
	}
	const Dictionary config = p_state.get(p_key);
	if (p_key == "scalar/config") {
		const Field spec[] = { { "friction", Variant::FLOAT }, { "restitution", Variant::FLOAT, false, false, 0, 1 }, { "gravity_factor", Variant::FLOAT }, { "faces_double_sided", Variant::BOOL }, { "vertex_radius", Variant::FLOAT } };
		return fields(config, p_key, spec, r_error);
	}
	if (p_key == "bend/config") {
		const Field spec[] = { { "compliance", Variant::FLOAT, true, true }, { "type", Variant::INT, false, false, 0, 1 } };
		return fields(config, p_key, spec, r_error);
	}
	if (p_key == "lra/config") {
		const Field spec[] = { { "type", Variant::INT, true, true, 0, 2 }, { "max_distance_multiplier", Variant::FLOAT, false, true, 0, FLT_MAX, true } };
		if (!fields(config, p_key, spec, r_error)) {
			return false;
		}
		const Variant type = config["type"];
		bool enabled = false;
		if (type.get_type() == Variant::INT) {
			enabled = int(type) != 0;
		} else {
			for (int value : PackedInt32Array(type)) {
				enabled |= value != 0;
			}
		}
		return enabled || fail(r_error, "SBREM-VALUE", p_key + ".type", "at least one nonzero type is required");
	}
	if (p_key == "vertex/config") {
		const Field spec[] = { { "edge_scale", Variant::FLOAT, false, true, 0, FLT_MAX, true }, { "shear_scale", Variant::FLOAT, false, true, 0, FLT_MAX, true } };
		return fields(config, p_key, spec, r_error);
	}
	if (p_key == "volume/config") {
		const Field spec[] = { { "vertices", Variant::PACKED_VECTOR3_ARRAY, true }, { "tetrahedra", Variant::PACKED_INT32_ARRAY, true }, { "compliance", Variant::FLOAT }, { "fixed", Variant::PACKED_INT32_ARRAY } };
		if (!fields(config, p_key, spec, r_error)) {
			return false;
		}
		const PackedVector3Array vertices = config["vertices"];
		const PackedInt32Array tetrahedra = config["tetrahedra"];
		if (!resource_size(vertices.size(), 4, 65536, 1, r_error, p_key + ".vertices") || !resource_size(tetrahedra.size(), 4, 262144 * 4, 1, r_error, p_key + ".tetrahedra")) {
			return false;
		}
		if (tetrahedra.size() % 4 != 0) {
			return fail(r_error, "SBREM-SIZE", p_key + ".tetrahedra", vformat("count %d must be a multiple of 4", tetrahedra.size()));
		}
		float converted;
		for (int i = 0; i < vertices.size(); ++i) {
			for (int axis = 0; axis < 3; ++axis) {
				if (!finite_float(vertices[i][axis], converted, r_error, p_key + vformat(".vertices[%d][%d]", i, axis))) {
					return false;
				}
			}
		}
		return true;
	}
	if (p_key == "rod/config") {
		const Field spec[] = { { "joints", Variant::PACKED_VECTOR3_ARRAY, true }, { "compliance", Variant::PACKED_FLOAT32_ARRAY }, { "bend", Variant::PACKED_FLOAT32_ARRAY }, { "fixed", Variant::INT, false, false, -DBL_MAX, DBL_MAX } };
		if (!fields(config, p_key, spec, r_error)) {
			return false;
		}
		const PackedVector3Array joints = config["joints"];
		float converted;
		for (int i = 0; i < joints.size(); ++i) {
			for (int axis = 0; axis < 3; ++axis) {
				if (!finite_float(joints[i][axis], converted, r_error, p_key + vformat(".joints entry %d axis %d", i, axis))) {
					return false;
				}
			}
			if (i > 0 && !finite_float(joints[i].distance_squared_to(joints[i - 1]), converted, r_error, p_key + vformat(".Segment %d squared length", i - 1))) {
				return false;
			}
		}
		return true;
	}
	const Field spec[] = { { "vertices", Variant::PACKED_INT32_ARRAY, true }, { "joint_indices", Variant::PACKED_INT32_ARRAY, true }, { "joint_weights", Variant::PACKED_FLOAT32_ARRAY, true }, { "inv_bind", Variant::ARRAY, true }, { "initial_pose", Variant::ARRAY, true }, { "max_distance", Variant::FLOAT }, { "back_stop_distance", Variant::FLOAT }, { "back_stop_radius", Variant::FLOAT, false, false, 0, FLT_MAX, true } };
	if (!fields(config, p_key, spec, r_error)) {
		return false;
	}
	const PackedInt32Array vertices = config["vertices"];
	const PackedInt32Array indices = config["joint_indices"];
	const PackedFloat32Array weights = config["joint_weights"];
	const Array binds = config["inv_bind"];
	const Array poses = config["initial_pose"];
	if (!resource_size(vertices.size(), 1, 65536, 4, r_error, p_key + ".vertices") || !resource_size(binds.size(), 1, 1024, 1, r_error, p_key + ".inv_bind")) {
		return false;
	}
	if (indices.size() != vertices.size() * 4 || weights.size() != vertices.size() * 4 || poses.size() != binds.size()) {
		return fail(r_error, "SBREM-SIZE", p_key, vformat("vertices %d, joint_indices %d, joint_weights %d, inv_bind %d, initial_pose %d require 4K/4K/J", vertices.size(), indices.size(), weights.size(), binds.size(), poses.size()));
	}
	HashSet<int> selected;
	for (int i = 0; i < vertices.size(); ++i) {
		if (vertices[i] < 0 || selected.has(vertices[i])) {
			return fail(r_error, "SBREM-INDEX", p_key + vformat(".vertices[%d]", i), vformat("negative or duplicate vertex %d", vertices[i]));
		}
		selected.insert(vertices[i]);
		bool terminated = false;
		double sum = 0;
		for (int slot = 0; slot < 4; ++slot) {
			const int index = i * 4 + slot;
			if (indices[index] < 0 || indices[index] >= binds.size()) {
				return fail(r_error, "SBREM-INDEX", p_key + vformat(".joint_indices[%d]", index), vformat("joint %d outside [0, %d)", indices[index], binds.size()));
			}
			const float weight = weights[index];
			if (!Math::is_finite(weight) || weight < 0 || (terminated && weight != 0)) {
				return fail(r_error, "SBREM-VALUE", p_key + vformat(".joint_weights[%d]", index), vformat("invalid weight %s or nonzero after terminator", weight));
			}
			terminated |= weight == 0;
			sum += weight;
		}
		if (std::abs(sum - 1.0) > 1.0e-5) {
			return fail(r_error, "SBREM-VALUE", p_key + vformat(".joint_weights[%d..%d]", i * 4, i * 4 + 3), vformat("sum %s differs from 1 by more than 1e-5", sum));
		}
	}
	for (int i = 0; i < binds.size(); ++i) {
		if (!matrix(binds[i], r_error, p_key + vformat(".inv_bind[%d]", i)) || !matrix(poses[i], r_error, p_key + vformat(".initial_pose[%d]", i))) {
			return false;
		}
	}
	return true;
}

bool column(const Dictionary &p_config, const String &p_field, double p_default, int p_source_count, const LocalVector<int> &p_map, LocalVector<float> &r_values, String *r_error) {
	const Variant value = p_config.get(p_field, p_default);
	const bool floats = value.get_type() == Variant::PACKED_FLOAT32_ARRAY;
	const bool ints = value.get_type() == Variant::PACKED_INT32_ARRAY;
	const int count = floats ? PackedFloat32Array(value).size() : ints ? PackedInt32Array(value).size()
																	   : 1;
	if (count < 1 || count > p_source_count) {
		return fail(r_error, "SBREM-SIZE", p_field, vformat("count %d outside [1, %d]", count, p_source_count));
	}
	HashMap<int, int> owners;
	for (int source = 0; source < p_source_count; ++source) {
		const int tail = MIN(source, count - 1);
		const float number = floats ? PackedFloat32Array(value)[tail] : ints ? float(PackedInt32Array(value)[tail])
																			 : float(value);
		const int physics = p_map[source];
		if (physics < 0) {
			continue;
		}
		if (owners.has(physics)) {
			if (r_values[physics] != number) {
				return fail(r_error, "SBREM-ALIAS", p_field, vformat("source vertices %d and %d disagree (%s vs %s)", owners[physics], source, r_values[physics], number));
			}
		} else {
			owners[physics] = source;
			if ((int)r_values.size() <= physics) {
				r_values.resize(physics + 1);
			}
			r_values[physics] = number;
		}
	}
	return true;
}

namespace {
double tetra_determinant(const Vector3 &a, const Vector3 &b, const Vector3 &c, const Vector3 &d) {
	const double x[3] = { double(b.x) - a.x, double(b.y) - a.y, double(b.z) - a.z };
	const double y[3] = { double(c.x) - a.x, double(c.y) - a.y, double(c.z) - a.z };
	const double z[3] = { double(d.x) - a.x, double(d.y) - a.y, double(d.z) - a.z };
	return (x[1] * y[2] - x[2] * y[1]) * z[0] + (x[2] * y[0] - x[0] * y[2]) * z[1] + (x[0] * y[1] - x[1] * y[0]) * z[2];
}
} //namespace

bool volume_determinant_valid(double p_determinant, double p_max_edge_squared) {
	const double threshold = MAX(1e-18, 1e-9 * p_max_edge_squared * std::sqrt(p_max_edge_squared));
	return std::isfinite(p_determinant) && std::isfinite(threshold) && std::abs(p_determinant) > threshold;
}

bool volume_topology(const Dictionary &p_config, PackedInt32Array &r_tetrahedra, PackedInt32Array &r_faces, String *r_error) {
	const PackedVector3Array vertices = p_config["vertices"];
	const PackedInt32Array tetrahedra = p_config["tetrahedra"];
	const PackedInt32Array fixed = p_config.get("fixed", PackedInt32Array());
	HashSet<int> pinned;
	for (int i = 0; i < fixed.size(); ++i) {
		if (fixed[i] < 0 || fixed[i] >= vertices.size() || pinned.has(fixed[i])) {
			return fail(r_error, "SBREM-INDEX", vformat("volume/config.fixed[%d]", i), vformat("invalid or duplicate vertex %d for %d vertices", fixed[i], vertices.size()));
		}
		pinned.insert(fixed[i]);
	}
	struct Face {
		int uses = 1;
		double side;
		std::array<int, 3> outward;
	};
	std::map<std::array<int, 3>, Face> faces;
	std::set<std::array<int, 4>> unique;
	HashSet<int> referenced;
	r_tetrahedra = PackedInt32Array();
	r_faces = PackedInt32Array();
	for (int offset = 0; offset < tetrahedra.size(); offset += 4) {
		std::array<int, 4> indices;
		for (int slot = 0; slot < 4; ++slot) {
			const int index = tetrahedra[offset + slot];
			if (index < 0 || index >= vertices.size()) {
				return fail(r_error, "SBREM-INDEX", vformat("volume/config.tetrahedra[%d]", offset + slot), vformat("vertex %d outside [0,%d)", index, vertices.size()));
			}
			indices[slot] = index;
			referenced.insert(index);
		}
		auto canonical = indices;
		std::sort(canonical.begin(), canonical.end());
		for (int i = 1; i < 4; ++i) {
			if (canonical[i] == canonical[i - 1]) {
				return fail(r_error, "SBREM-INDEX", vformat("volume/config.tetrahedra[%d]", offset), vformat("repeated vertex %d", canonical[i]));
			}
		}
		if (!unique.insert(canonical).second) {
			return fail(r_error, "SBREM-TOPOLOGY", vformat("volume/config.tetrahedra[%d]", offset), "duplicate tetrahedron (including permutations)");
		}
		double max_edge_sq = 0;
		for (int a = 0; a < 4; ++a) {
			for (int b = a + 1; b < 4; ++b) {
				double length_sq = 0;
				for (int axis = 0; axis < 3; ++axis) {
					const double delta = double(vertices[indices[a]][axis]) - vertices[indices[b]][axis];
					length_sq += delta * delta;
				}
				max_edge_sq = MAX(max_edge_sq, length_sq);
			}
		}
		const double det = tetra_determinant(vertices[indices[0]], vertices[indices[1]], vertices[indices[2]], vertices[indices[3]]);
		const double threshold = MAX(1e-18, 1e-9 * max_edge_sq * std::sqrt(max_edge_sq));
		if (!std::isfinite(det) || !std::isfinite(threshold)) {
			return fail(r_error, "SBREM-VALUE", vformat("volume/config.tetrahedra[%d]", offset), "derived determinant or length is nonfinite");
		}
		if (!volume_determinant_valid(det, max_edge_sq)) {
			return fail(r_error, "SBREM-TOPOLOGY", vformat("volume/config.tetrahedra[%d]", offset), vformat("abs(det) %s must exceed %s", std::abs(det), threshold));
		}
		float converted;
		if (!finite_float(std::abs(det), converted, r_error, vformat("volume/config.tetrahedra[%d].determinant", offset))) {
			return false;
		}
		// Guard actual float32 cross/dot intermediates used by Jolt, not only
		// the widened determinant's final value.
		const Vector3 a = vertices[indices[0]], b = vertices[indices[1]], c = vertices[indices[2]], d = vertices[indices[3]];
		if (!(b - a).cross(c - a).is_finite() || !Math::is_finite((b - a).cross(c - a).dot(d - a))) {
			return fail(r_error, "SBREM-VALUE", vformat("volume/config.tetrahedra[%d]", offset), "float32 determinant intermediate overflows");
		}
		if (det < 0) {
			std::swap(indices[1], indices[2]);
		}
		for (int index : indices) {
			r_tetrahedra.push_back(index);
		}
		for (int opposite = 0; opposite < 4; ++opposite) {
			std::array<int, 3> key;
			int next = 0;
			for (int slot = 0; slot < 4; ++slot) {
				if (slot != opposite) {
					key[next++] = indices[slot];
				}
			}
			std::sort(key.begin(), key.end());
			const double side = tetra_determinant(vertices[key[0]], vertices[key[1]], vertices[key[2]], vertices[indices[opposite]]);
			auto found = faces.find(key);
			if (found != faces.end()) {
				if (++found->second.uses > 2 || ((found->second.side > 0) == (side > 0))) {
					return fail(r_error, "SBREM-TOPOLOGY", vformat("volume/config.tetrahedra[%d]", offset), vformat("face (%d,%d,%d) nonmanifold or same-side opposite vertices", key[0], key[1], key[2]));
				}
			} else {
				auto outward = key;
				if (side > 0) {
					std::swap(outward[1], outward[2]);
				}
				faces.emplace(key, Face{ 1, side, outward });
			}
		}
	}
	for (int i = 0; i < vertices.size(); ++i) {
		if (!referenced.has(i)) {
			return fail(r_error, "SBREM-INDEX", vformat("volume/config.vertices[%d]", i), "vertex is not referenced by any tetrahedron");
		}
	}
	// map order is canonical face-key order; orientation is independent of
	// author tetra order. Godot front faces use the reverse of Jolt winding.
	for (const auto &entry : faces) {
		if (entry.second.uses == 1) {
			r_faces.push_back(entry.second.outward[0]);
			r_faces.push_back(entry.second.outward[2]);
			r_faces.push_back(entry.second.outward[1]);
		}
	}
	return !r_faces.is_empty() || fail(r_error, "SBREM-TOPOLOGY", "volume/config.tetrahedra", "no boundary faces");
}

bool skin_targets(const JoltSoftBodyCapState &p_state, const Array &p_pose, const Transform3D &p_frame, const String &p_stage, String *r_error) {
	if (!p_state.has("skin/config")) {
		return fail(r_error, "SBREM-POSE", "skin/pose", "skin/config is required");
	}
	const Dictionary config = p_state.get("skin/config");
	const Array binds = config["inv_bind"];
	if (p_pose.size() != binds.size()) {
		return fail(r_error, "SBREM-POSE", "skin/pose", vformat("count %d requires %d", p_pose.size(), binds.size()));
	}
	// Without a mesh, static matrix checks already ran; targets depend on the
	// reference geometry and are checked again before any body can be built.
	if (p_state.skin_bind_points.is_empty()) {
		return true;
	}
	const char *code = p_stage == "initial" ? "SBREM-VALUE" : "SBREM-POSE";
	if (!p_frame.is_finite()) {
		return fail(r_error, code, "skin/pose." + p_stage, "body COM frame is nonfinite");
	}
	const Transform3D inverse_frame = p_frame.affine_inverse();
	LocalVector<Transform3D> transforms;
	for (int joint = 0; joint < p_pose.size(); ++joint) {
		const Transform3D local = inverse_frame * Transform3D(p_pose[joint]);
		const Transform3D combined = local * Transform3D(p_state.skin_inv_bind[joint]);
		if (!local.is_finite() || !combined.is_finite()) {
			return fail(r_error, code, vformat("skin/pose.%s.joint[%d]", p_stage, joint), "COM-local matrix or inverse-bind product is nonfinite");
		}
		transforms.push_back(combined);
	}
	const PackedInt32Array indices = config["joint_indices"];
	const PackedFloat32Array weights = config["joint_weights"];
	for (uint32_t vertex = 0; vertex < p_state.skin_tuples.size(); ++vertex) {
		const int tuple = p_state.skin_tuples[vertex];
		if (tuple < 0) {
			continue;
		}
		Vector3 target;
		for (int slot = 0; slot < 4; ++slot) {
			const int entry = 4 * tuple + slot;
			if (weights[entry] == 0) {
				break;
			}
			const Vector3 transformed = transforms[indices[entry]].xform(p_state.skin_bind_points[vertex]);
			target += transformed * weights[entry];
			if (!transformed.is_finite() || !target.is_finite()) {
				return fail(r_error, code, vformat("skin/pose.%s.vertex[%d].joint[%d]", p_stage, vertex, indices[entry]), "weighted target is nonfinite");
			}
		}
	}
	return true;
}

bool new_path(const JoltSoftBodyCapState &p_state) {
	for (const char *key : { "scalar/config", "bend/config", "lra/config", "vertex/config", "volume/config", "skin/config", "rod/config" }) {
		if (p_state.has(key)) {
			return true;
		}
	}
	return false;
}

bool mesh_family(const JoltSoftBodyCapState &p_state) {
	return p_state.has("bend/config") || p_state.has("lra/config") || p_state.has("vertex/config") || p_state.has("skin/config");
}

bool typed(const JoltSoftBodyCapState &p_state, float p_mass, int p_precision, float p_stiffness, bool p_mesh, int p_vertices, float &r_inverse_mass, float &r_base, String *r_error) {
	if (!Math::is_finite(p_mass) || p_mass <= 0 || p_precision < 1 || !Math::is_finite(p_stiffness) || p_stiffness < 0 || p_stiffness > 1 || (p_mesh && p_stiffness == 0)) {
		return fail(r_error, "SBREM-CONTEXT", "typed", vformat("mass %s, precision %d, stiffness %s, mesh %s", p_mass, p_precision, p_stiffness, p_mesh));
	}
	r_inverse_mass = 0;
	r_base = 0;
	if (p_vertices <= 0) {
		return true;
	}
	String error;
	if (!finite_float(double(p_vertices) / p_mass, r_inverse_mass, &error, "inverse_mass", true)) {
		return fail(r_error, "SBREM-CONTEXT", "typed.mass", error);
	}
	if (p_mesh) {
		const double hz = Engine::get_singleton()->get_user_physics_ticks_per_second();
		const double dt = 1.0 / hz / p_precision;
		const double base = dt * dt * (1.0 / p_stiffness - 1.0) * (2.0 * p_vertices / p_mass);
		if (hz <= 0 || base < 0 || !finite_float(base, r_base, &error, "compliance")) {
			return fail(r_error, "SBREM-CONTEXT", "typed.compliance", vformat("physics_hz %s: %s", hz, error));
		}
	}
	return true;
}

} // namespace SoftBodyCapValidation
