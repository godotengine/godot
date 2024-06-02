#include "jolt_height_map_shape_impl_3d.hpp"

#include "servers/jolt_project_settings.hpp"
#include "shapes/jolt_custom_double_sided_shape.hpp"

Variant JoltHeightMapShapeImpl3D::get_data() const {
	Dictionary data;
	data["width"] = width;
	data["depth"] = depth;
	data["heights"] = heights;
	return data;
}

void JoltHeightMapShapeImpl3D::set_data(const Variant& p_data) {
	ERR_FAIL_COND(p_data.get_type() != Variant::DICTIONARY);

	const Dictionary data = p_data;

	const Variant maybe_heights = data.get("heights", {});

#ifdef REAL_T_IS_DOUBLE
	ERR_FAIL_COND(maybe_heights.get_type() != Variant::PACKED_FLOAT64_ARRAY);
#else // REAL_T_IS_DOUBLE
	ERR_FAIL_COND(maybe_heights.get_type() != Variant::PACKED_FLOAT32_ARRAY);
#endif // REAL_T_IS_DOUBLE

	const Variant maybe_width = data.get("width", {});
	ERR_FAIL_COND(maybe_width.get_type() != Variant::INT);

	const Variant maybe_depth = data.get("depth", {});
	ERR_FAIL_COND(maybe_depth.get_type() != Variant::INT);

	heights = maybe_heights;
	width = maybe_width;
	depth = maybe_depth;

	destroy();
}

String JoltHeightMapShapeImpl3D::to_string() const {
	return vformat("{height_count=%d width=%d depth=%d}", heights.size(), width, depth);
}

JPH::ShapeRefC JoltHeightMapShapeImpl3D::_build() const {
	const auto height_count = (int32_t)heights.size();

	QUIET_FAIL_COND_D(height_count == 0);

	ERR_FAIL_COND_D_MSG(
		height_count != width * depth,
		vformat(
			"Godot Jolt failed to build height map shape with %s. "
			"Height count must be the product of width and depth. "
			"This shape belongs to %s.",
			to_string(),
			_owners_to_string()
		)
	);

	ERR_FAIL_COND_D_MSG(
		width < 2 || depth < 2,
		vformat(
			"Godot Jolt failed to build height map shape with %s. "
			"The height map must be at least 2x2. "
			"This shape belongs to %s.",
			to_string(),
			_owners_to_string()
		)
	);

	if (width != depth) {
		return _build_double_sided(_build_mesh());
	}

	const int32_t block_size = 2; // Default of JPH::HeightFieldShapeSettings::mBlockSize
	const int32_t block_count = width / block_size;

	if (block_count < 2) {
		return _build_double_sided(_build_mesh());
	}

	return _build_double_sided(_build_height_field());
}

JPH::ShapeRefC JoltHeightMapShapeImpl3D::_build_height_field() const {
	const int32_t quad_count_x = width - 1;
	const int32_t quad_count_y = depth - 1;

	const float offset_x = (float)-quad_count_x / 2.0f;
	const float offset_y = (float)-quad_count_y / 2.0f;

	// HACK(mihe): Jolt triangulates the height map differently from how Godot Physics does it, so
	// we mirror the shape along the Z-axis to get the desired triangulation and reverse the rows to
	// undo the mirroring.

	JLocalVector<float> heights_rev;
	heights_rev.resize((int32_t)heights.size());

	const real_t* heights_ptr = heights.ptr();
	float* heights_rev_ptr = heights_rev.ptr();

	for (int32_t z = 0; z < depth; ++z) {
		const int32_t z_rev = (depth - 1) - z;

		const real_t* row = heights_ptr + ptrdiff_t(z * width);
		float* row_rev = heights_rev_ptr + ptrdiff_t(z_rev * width);

		for (int32_t x = 0; x < width; ++x) {
			const real_t height = row[x];

			// HACK(mihe): Godot has undocumented (accidental?) support for holes by passing NaN as
			// the height value, whereas Jolt uses `FLT_MAX` instead, so we translate any NaN to
			// `FLT_MAX` in order to be drop-in compatible.
			row_rev[x] = Math::is_nan(height) ? FLT_MAX : (float)height;
		}
	}

	JPH::HeightFieldShapeSettings shape_settings(
		heights_rev.ptr(),
		JPH::Vec3(offset_x, 0, offset_y),
		JPH::Vec3::sReplicate(1.0f),
		(JPH::uint32)width
	);

	shape_settings.mBitsPerSample = shape_settings.CalculateBitsPerSampleForError(0.0f);
	shape_settings.mActiveEdgeCosThresholdAngle = JoltProjectSettings::get_active_edge_threshold();

	const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();

	ERR_FAIL_COND_D_MSG(
		shape_result.HasError(),
		vformat(
			"Godot Jolt failed to build height map shape with %s. "
			"It returned the following error: '%s'. "
			"This shape belongs to %s.",
			to_string(),
			to_godot(shape_result.GetError()),
			_owners_to_string()
		)
	);

	return with_scale(shape_result.Get(), Vector3(1, 1, -1));
}

JPH::ShapeRefC JoltHeightMapShapeImpl3D::_build_mesh() const {
	const auto height_count = (int32_t)heights.size();

	const int32_t quad_count_x = width - 1;
	const int32_t quad_count_z = depth - 1;

	const int32_t quad_count = quad_count_x * quad_count_z;
	const int32_t triangle_count = quad_count * 2;

	JPH::VertexList vertices;
	vertices.reserve((size_t)height_count);

	JPH::IndexedTriangleList indices;
	indices.reserve((size_t)triangle_count);

	const float offset_x = (float)-quad_count_x / 2.0f;
	const float offset_z = (float)-quad_count_z / 2.0f;

	for (int32_t z = 0; z < depth; ++z) {
		for (int32_t x = 0; x < width; ++x) {
			const float vertex_x = offset_x + (float)x;
			const auto vertex_y = (float)heights[z * width + x];
			const float vertex_z = offset_z + (float)z;

			vertices.emplace_back(vertex_x, vertex_y, vertex_z);
		}
	}

	auto to_index = [&](int32_t p_x, int32_t p_z) {
		return (p_z * width) + p_x;
	};

	auto is_vertex_hole = [&](int32_t p_index) {
		const float height = vertices[(size_t)p_index].y;
		return height == FLT_MAX || Math::is_nan(height);
	};

	auto is_triangle_hole = [&](int32_t p_index0, int32_t p_index1, int32_t p_index2) {
		return is_vertex_hole(p_index0) || is_vertex_hole(p_index1) || is_vertex_hole(p_index2);
	};

	for (int32_t z = 0; z < quad_count_z; ++z) {
		for (int32_t x = 0; x < quad_count_x; ++x) {
			const int32_t lr = to_index(x + 0, z + 0);
			const int32_t ll = to_index(x + 1, z + 0);
			const int32_t ur = to_index(x + 0, z + 1);
			const int32_t ul = to_index(x + 1, z + 1);

			if (!is_triangle_hole(lr, ur, ll)) {
				indices.emplace_back(lr, ur, ll);
			}

			if (!is_triangle_hole(ll, ur, ul)) {
				indices.emplace_back(ll, ur, ul);
			}
		}
	}

	JPH::MeshShapeSettings shape_settings(std::move(vertices), std::move(indices));
	shape_settings.mActiveEdgeCosThresholdAngle = JoltProjectSettings::get_active_edge_threshold();

	const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();

	ERR_FAIL_COND_D_MSG(
		shape_result.HasError(),
		vformat(
			"Godot Jolt failed to build height map shape (as polygon) with %s. "
			"It returned the following error: '%s'. "
			"This shape belongs to %s.",
			to_string(),
			to_godot(shape_result.GetError()),
			_owners_to_string()
		)
	);

	return shape_result.Get();
}

JPH::ShapeRefC JoltHeightMapShapeImpl3D::_build_double_sided(const JPH::Shape* p_shape) const {
	ERR_FAIL_NULL_D(p_shape);

	const JoltCustomDoubleSidedShapeSettings shape_settings(p_shape);
	const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();

	ERR_FAIL_COND_D_MSG(
		shape_result.HasError(),
		vformat(
			"Failed to make shape double-sided. "
			"It returned the following error: '%s'.",
			to_godot(shape_result.GetError())
		)
	);

	return shape_result.Get();
}
