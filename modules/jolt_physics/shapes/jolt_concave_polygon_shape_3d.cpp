/**************************************************************************/
/*  jolt_concave_polygon_shape_3d.cpp                                     */
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

#include "jolt_concave_polygon_shape_3d.h"

#include "../jolt_project_settings.h"
#include "../misc/jolt_type_conversions.h"

#include "Jolt/Physics/Collision/Shape/MeshShape.h"
// #include "modules/voxel/util/profiling.h"

JPH::ShapeRefC JoltConcavePolygonShape3D::_build() const {
	// ZoneScoped;

	if (prebuilt_shape != nullptr) {
		// The shape was already built manually up-front
		return prebuilt_shape;
	}

	const int vertex_count = (int)faces.size();
	const int face_count = vertex_count / 3;
	const int excess_vertex_count = vertex_count % 3;

	if (unlikely(vertex_count == 0)) {
		return nullptr;
	}

	ERR_FAIL_COND_V_MSG(vertex_count < 3, nullptr, vformat("Failed to build Jolt Physics concave polygon shape with %s. It must have a vertex count of at least 3. This shape belongs to %s.", to_string(), _owners_to_string()));
	ERR_FAIL_COND_V_MSG(excess_vertex_count != 0, nullptr, vformat("Failed to build Jolt Physics concave polygon shape with %s. It must have a vertex count that is divisible by 3. This shape belongs to %s.", to_string(), _owners_to_string()));

	JPH::TriangleList jolt_faces;
	jolt_faces.reserve((size_t)face_count);

	const Vector3 *faces_begin = &faces[0];
	const Vector3 *faces_end = faces_begin + vertex_count;
	JPH::uint32 triangle_index = 0;

	for (const Vector3 *vertex = faces_begin; vertex != faces_end; vertex += 3) {
		const Vector3 *v0 = vertex + 0;
		const Vector3 *v1 = vertex + 1;
		const Vector3 *v2 = vertex + 2;

		// Jolt uses a different winding order, so we swizzle the vertices to account for that.
		jolt_faces.emplace_back(
				JPH::Float3((float)v2->x, (float)v2->y, (float)v2->z),
				JPH::Float3((float)v1->x, (float)v1->y, (float)v1->z),
				JPH::Float3((float)v0->x, (float)v0->y, (float)v0->z),
				0,
				triangle_index++);
	}

	JPH::MeshShapeSettings shape_settings(jolt_faces);
	shape_settings.mActiveEdgeCosThresholdAngle = JoltProjectSettings::get_active_edge_threshold();
	shape_settings.mPerTriangleUserData = JoltProjectSettings::enable_ray_cast_face_index();

	const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();
	ERR_FAIL_COND_V_MSG(shape_result.HasError(), nullptr, vformat("Failed to build Jolt Physics concave polygon shape with %s. It returned the following error: '%s'. This shape belongs to %s.", to_string(), to_godot(shape_result.GetError()), _owners_to_string()));

	return JoltShape3D::with_double_sided(shape_result.Get(), back_face_collision);
}

// TODO This should simply be a Span, see https://github.com/godotengine/godot/pull/100293
template <typename T>
struct ArrayView {
	const T *_data;
	int _size;

	ArrayView(T *p_ptr, const int p_size) :
			_data(p_ptr), _size(p_size) {
		CRASH_COND(p_size < 0);
	}

	inline const T &operator[](const int p_index) const {
#ifdef DEBUG_ENABLED
		CRASH_BAD_INDEX(p_index, _size);
#endif
		return _data[p_index];
	}

	inline int size() const {
		return _size;
	}
};

static bool copy_to_jolt_indexed_triangle_list(const ArrayView<const int32_t> indices, JPH::IndexedTriangleList &jolt_triangle_indices) {
	const int face_count = (int)indices.size() / 3;
	const int excess_index_count = indices.size() % 3;

	ERR_FAIL_COND_V(excess_index_count != 0, false);

	ERR_FAIL_COND_V(jolt_triangle_indices.size() != 0, false);
	jolt_triangle_indices.reserve((size_t)face_count);

	for (int tri_index = 0; tri_index < face_count; ++tri_index) {
		const unsigned int ii0 = tri_index * 3;
		const unsigned int ii1 = ii0 + 1;
		const unsigned int ii2 = ii0 + 2;

		const int32_t i0 = indices[ii0];
		const int32_t i1 = indices[ii1];
		const int32_t i2 = indices[ii2];

		// Jolt uses a different winding order, so we swizzle the vertices to account for that.
		jolt_triangle_indices.emplace_back(JPH::IndexedTriangle(i2, i1, i0));
	}

	return true;
}

// Alternate build function taking an indexed triangle list, which is closer to what Jolt natively expects,
// and faster since it doesn't have to use Indexify internally.
static JPH::ShapeRefC build_from_indexed_triangles(const ArrayView<const Vector3> vertices, const ArrayView<const int32_t> indices, const bool back_face_collision) {
	const int vertex_count = (int)vertices.size();
	const int excess_index_count = indices.size() % 3;

	if (unlikely(vertex_count == 0)) {
		return nullptr;
	}

	ERR_FAIL_COND_V_MSG(vertex_count < 3, nullptr,
			vformat("Failed to build Jolt Physics concave polygon shape. It must have a vertex count of at least 3."));
	ERR_FAIL_COND_V_MSG(excess_index_count != 0, nullptr,
			vformat("Failed to build Jolt Physics concave polygon shape. It must have an index count that is divisible by 3."));

	JPH::VertexList jolt_vertices;
	jolt_vertices.reserve((size_t)vertex_count);
	// Godot could be using doubles in Vector3, so can't memcpy
	for (int vertex_index = 0; vertex_index < vertex_count; ++vertex_index) {
		const Vector3 v = vertices[vertex_index];
		jolt_vertices.emplace_back(JPH::Float3(v.x, v.y, v.z));
	}

	JPH::IndexedTriangleList jolt_triangle_indices;
	ERR_FAIL_COND_V(!copy_to_jolt_indexed_triangle_list(indices, jolt_triangle_indices), nullptr);

	// TODO Why is this constructor taking arrays by value? Won't this cause unnecessary copies and allocations?
	// JPH::MeshShapeSettings shape_settings(jolt_vertices, jolt_triangles);
	JPH::MeshShapeSettings shape_settings;
	shape_settings.mTriangleVertices = std::move(jolt_vertices);
	shape_settings.mIndexedTriangles = std::move(jolt_triangle_indices);
	shape_settings.Sanitize();
	shape_settings.mActiveEdgeCosThresholdAngle = JoltProjectSettings::get_active_edge_threshold();
	shape_settings.mPerTriangleUserData = JoltProjectSettings::enable_ray_cast_face_index();

	const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();
	ERR_FAIL_COND_V_MSG(shape_result.HasError(), nullptr,
			vformat("Failed to build Jolt Physics concave polygon shape. It returned the following error: '%s'.",
					to_godot(shape_result.GetError())));

	// TODO Will this double the baking work? Need to investigate
	return JoltShape3D::with_double_sided(shape_result.Get(), back_face_collision);
}

static PackedVector3Array deindex_mesh(const ArrayView<const Vector3> vertices, const ArrayView<const int32_t> indices) {
	PackedVector3Array output_vertices;
	output_vertices.resize(indices.size());
	Vector3 *output_vertices_raw = output_vertices.ptrw();

	for (int ii = 0; ii < indices.size(); ++ii) {
		const int32_t i = indices[ii];
		output_vertices_raw[ii] = vertices[i];
	}

	return output_vertices;
}

static AABB get_aabb_from_vertices(const ArrayView<const Vector3> vertices) {
	const int vertex_count = (int)vertices.size();

	if (vertex_count == 0) {
		return AABB();
	}

	Vector3 min_p = vertices[0];
	Vector3 max_p = min_p;

	for (int i = 1; i < vertex_count; ++i) {
		const Vector3 p = vertices[i];
		min_p = min_p.min(p);
		max_p = max_p.max(p);
	}

	return AABB(min_p, max_p - min_p);
}

static bool read_and_validate_range(const Variant &v, Vector2i &out_range, const int size_limit) {
	ERR_FAIL_COND_V(v.get_type() != Variant::VECTOR2I, false);
	const Vector2i range = v;
	ERR_FAIL_COND_V(range.x > range.y, false); // Must be sorted or equal
	ERR_FAIL_COND_V(range.x < 0, false); // Must not be negative
	ERR_FAIL_COND_V(range.y > size_limit, false); // Must be in bounds
	out_range = range;
	return true;
}

JoltConcavePolygonShape3D *JoltConcavePolygonShape3D::create_with_data(const Variant &p_data) {
	JoltConcavePolygonShape3D *self = nullptr;

	// Not using a dictionary, creating strings and hashing keys sounds quite unnecessary.
	if (p_data.get_type() == Variant::ARRAY) {
		const Array data_array = p_data;
		const int expected_param_count = 5;
		ERR_FAIL_COND_V_MSG(data_array.size() != expected_param_count, nullptr, vformat("Expected %d parameters, got %d", expected_param_count, data_array.size()));

		const Variant vertex_data_v = data_array[0];
		const Variant indices_v = data_array[1];
		const Variant vertex_range_v = data_array[2];
		const Variant index_range_v = data_array[3];
		const Variant back_face_collision_v = data_array[4];

		// TODO In double-precision builds of Godot, expecting PackedVector3Array has extra overhead.
		// Because Vector3 will then use doubles, which Jolt doesn't need. On the side of the caller,
		// GDExtensions are also likely to use a different container type if they generate meshes procedurally,
		// and having to allocate a PackedVector3Array only to "talk" to the Godot API.
		// It would be nice if we could pass Spans around to avoid this overhead
		// (which we could do safely here, since the current function is meant to create the shape immediately,
		// instead of being deferred in MT mode), though it would be only a GDExtension thing.
		ERR_FAIL_COND_V(vertex_data_v.get_type() != Variant::PACKED_VECTOR3_ARRAY, nullptr);
		const PackedVector3Array vertices_array = vertex_data_v;

		ERR_FAIL_COND_V(indices_v.get_type() != Variant::PACKED_INT32_ARRAY, nullptr);
		const PackedInt32Array indices_array = indices_v;

		ERR_FAIL_COND_V(back_face_collision_v.get_type() != Variant::BOOL, nullptr);
		const bool back_face_collision = back_face_collision_v;

		// Allow to specify a sub-region of the mesh, to avoid having to allocate extra packed arrays
		// TODO If we had the ability to pass Spans, this wouldn't be necessary.
		// Maybe we can do that for GDExtensions, but not sure we can in script-land.
		Vector2i vertex_range(0, vertices_array.size());
		if (vertex_range_v.get_type() != Variant::NIL) {
			ERR_FAIL_COND_V(!read_and_validate_range(vertex_range_v, vertex_range, vertices_array.size()), nullptr);
		}
		const ArrayView<const Vector3> vertices(vertices_array.ptr() + vertex_range.x, vertex_range.y - vertex_range.x);

		Vector2i index_range(0, indices_array.size());
		if (index_range_v.get_type() != Variant::NIL) {
			ERR_FAIL_COND_V(!read_and_validate_range(index_range_v, index_range, indices_array.size()), nullptr);
		}
		const ArrayView<const int32_t> indices(indices_array.ptr() + index_range.x, index_range.y - index_range.x);

		self = memnew(JoltConcavePolygonShape3D);

		// TODO `faces` are almost useless after the Jolt object is created.
		// In a project that uses lots of mesh colliders, it wastes resources, especially with double-precision builds,
		// and the fact it is deindexed (vertices are not shared). May we reset them? What about debugging?
		// Couldn't we just get the data from Jolt if that's necessary?
		self->faces = deindex_mesh(vertices, indices);

		self->back_face_collision = back_face_collision;
		self->prebuilt_shape = build_from_indexed_triangles(vertices, indices, back_face_collision);
		// Don't use _get_aabb, so that we don't depend on `faces`, and will iterate less vertices when indexed
		self->aabb = ::get_aabb_from_vertices(vertices);

	} else {
		ERR_PRINT("Invalid parameter");
		return nullptr;
	}

	return self;
}

AABB JoltConcavePolygonShape3D::_calculate_aabb() const {
	return get_aabb_from_vertices(ArrayView<const Vector3>(faces.ptr(), faces.size()));
}

Variant JoltConcavePolygonShape3D::get_data() const {
	Dictionary data;
	data["faces"] = faces;
	data["backface_collision"] = back_face_collision;
	return data;
}

void JoltConcavePolygonShape3D::set_data(const Variant &p_data) {
	ERR_FAIL_COND(p_data.get_type() != Variant::DICTIONARY);

	const Dictionary data = p_data;

	const Variant maybe_faces = data.get("faces", Variant());
	ERR_FAIL_COND(maybe_faces.get_type() != Variant::PACKED_VECTOR3_ARRAY);

	const Variant maybe_back_face_collision = data.get("backface_collision", Variant());
	ERR_FAIL_COND(maybe_back_face_collision.get_type() != Variant::BOOL);

	faces = maybe_faces;
	back_face_collision = maybe_back_face_collision;

	aabb = _calculate_aabb();

	// When the physics server is set to "run in a separate thread",
	// it is wrapped in PhysicsServer3DWrapMT, which makes `set_data` a deferred function.
	// We can't guarantee whether `set_data` is called due to initializing the shape after creation,
	// or if it's the user changing properties later on.
	// `prebuilt_shape` is used when all data is specified at creation time, so we can't use it from now.
	prebuilt_shape = JPH::ShapeRefC();

	destroy();
}

String JoltConcavePolygonShape3D::to_string() const {
	return vformat("{vertex_count=%d}", faces.size());
}
