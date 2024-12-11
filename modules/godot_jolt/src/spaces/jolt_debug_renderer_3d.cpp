#include "jolt_debug_renderer_3d.hpp"

#ifdef JPH_DEBUG_RENDERER

#include "jolt_space_3d.hpp"

namespace {

constexpr int64_t DEBUG_VERTEX_STRIDE = sizeof(float) * 3;
constexpr int64_t DEBUG_ATTRIBUTE_STRIDE = sizeof(uint32_t);

} // namespace

class JoltDebugTriangleBatch final : public JPH::RefTargetVirtual {
public:
	using Vertices = JPH::Array<JPH::Vec3>;

	explicit JoltDebugTriangleBatch(int p_capacity) { vertices.reserve((size_t)p_capacity * 3); }

	void AddRef() override { ++ref_count; }

	void Release() override {
		if (--ref_count == 0) {
			delete this;
		}
	}

	const Vertices& get_vertices() const { return vertices; }

	void add_triangle(JPH::Vec3 p_vertex1, JPH::Vec3 p_vertex2, JPH::Vec3 p_vertex3) {
		vertices.push_back(p_vertex1);
		vertices.push_back(p_vertex2);
		vertices.push_back(p_vertex3);
	}

private:
	int32_t ref_count = 0;

	Vertices vertices;
};

JoltDebugRenderer3D* JoltDebugRenderer3D::acquire() {
	if (ref_count++ == 0) {
		singleton = new JoltDebugRenderer3D();
	}

	return singleton;
}

void JoltDebugRenderer3D::release(JoltDebugRenderer3D*& p_ptr) {
	if (--ref_count == 0) {
		delete_safely(singleton);
	}

	p_ptr = nullptr;
}

void JoltDebugRenderer3D::draw(
	const JoltSpace3D& p_space,
	const Camera3D& p_camera,
	const DrawSettings& p_settings
) {
	camera_position = to_jolt(p_camera.get_camera_transform().origin);

	JPH::PhysicsSystem& physics_system = p_space.get_physics_system();

	if (p_settings.draw_bodies) {
		JPH::BodyManager::DrawSettings jolt_settings;
		jolt_settings.mDrawGetSupportFunction = false;
		jolt_settings.mDrawSupportDirection = false;
		jolt_settings.mDrawGetSupportingFace = false;
		jolt_settings.mDrawShape = p_settings.draw_shapes;
		jolt_settings.mDrawShapeWireframe = p_settings.draw_as_wireframe;
		jolt_settings.mDrawShapeColor = p_settings.color_scheme;
		jolt_settings.mDrawBoundingBox = p_settings.draw_bounding_boxes;
		jolt_settings.mDrawCenterOfMassTransform = p_settings.draw_centers_of_mass;
		jolt_settings.mDrawWorldTransform = p_settings.draw_transforms;
		jolt_settings.mDrawVelocity = p_settings.draw_velocities;
		jolt_settings.mDrawMassAndInertia = false;
		jolt_settings.mDrawSleepStats = false;
		jolt_settings.mDrawSoftBodyVertices = p_settings.draw_soft_body_vertices;
		jolt_settings.mDrawSoftBodyEdgeConstraints = p_settings.draw_soft_body_edge_constraints;
		jolt_settings.mDrawSoftBodyVolumeConstraints = p_settings.draw_soft_body_volume_constraints;
		jolt_settings.mDrawSoftBodyPredictedBounds = p_settings.draw_soft_body_predicted_bounds;

		physics_system.DrawBodies(jolt_settings, this);
	}

	if (p_settings.draw_constraints) {
		physics_system.DrawConstraints(this);
	}

	if (p_settings.draw_constraint_reference_frames) {
		physics_system.DrawConstraintReferenceFrame(this);
	}

	if (p_settings.draw_constraint_limits) {
		physics_system.DrawConstraintLimits(this);
	}
}

int32_t JoltDebugRenderer3D::submit(const RID& p_mesh) {
	RenderingServer* rendering_server = RenderingServer::get_singleton();

	rendering_server->mesh_clear(p_mesh);

	uint64_t vertex_format = 0;
	vertex_format |= (uint64_t)RenderingServer::ARRAY_FLAG_FORMAT_VERSION_2;
	vertex_format |= (uint64_t)RenderingServer::ARRAY_FORMAT_VERTEX;
	vertex_format |= (uint64_t)RenderingServer::ARRAY_FORMAT_COLOR;

	int32_t surface_count = 0;

	if (triangle_count > 0) {
		const int64_t vertex_count = (int64_t)triangle_count * 3;

		triangle_vertices.resize(vertex_count * DEBUG_VERTEX_STRIDE);
		triangle_attributes.resize(vertex_count * DEBUG_ATTRIBUTE_STRIDE);

		Dictionary triangles_surface_data;
		triangles_surface_data["format"] = vertex_format;
		triangles_surface_data["primitive"] = RenderingServer::PRIMITIVE_TRIANGLES;
		triangles_surface_data["vertex_data"] = triangle_vertices;
		triangles_surface_data["vertex_count"] = vertex_count;
		triangles_surface_data["aabb"] = triangles_aabb;
		triangles_surface_data["attribute_data"] = triangle_attributes;

		rendering_server->mesh_add_surface(p_mesh, triangles_surface_data);

		triangle_capacity = triangle_count;
		triangle_count = 0;
		triangles_aabb = AABB();

		surface_count++;
	}

	if (line_count > 0) {
		const int64_t vertex_count = (int64_t)line_count * 2;

		line_vertices.resize(vertex_count * DEBUG_VERTEX_STRIDE);
		line_attributes.resize(vertex_count * DEBUG_ATTRIBUTE_STRIDE);

		Dictionary lines_surface_data;
		lines_surface_data["format"] = vertex_format;
		lines_surface_data["primitive"] = RenderingServer::PRIMITIVE_LINES;
		lines_surface_data["vertex_data"] = line_vertices;
		lines_surface_data["vertex_count"] = vertex_count;
		lines_surface_data["aabb"] = lines_aabb;
		lines_surface_data["attribute_data"] = line_attributes;

		rendering_server->mesh_add_surface(p_mesh, lines_surface_data);

		line_capacity = line_count;
		line_count = 0;
		lines_aabb = AABB();

		surface_count++;
	}

	return surface_count;
}

void JoltDebugRenderer3D::DrawLine(JPH::RVec3Arg p_from, JPH::RVec3Arg p_to, JPH::Color p_color) {
	_reserve_lines(1);
	_add_line(to_godot(p_from), to_godot(p_to), to_godot(p_color).to_abgr32());
}

void JoltDebugRenderer3D::DrawTriangle(
	JPH::RVec3Arg p_vertex1,
	JPH::RVec3Arg p_vertex2,
	JPH::RVec3Arg p_vertex3,
	JPH::Color p_color,
	[[maybe_unused]] ECastShadow p_cast_shadow
) {
	_reserve_triangles(1);

	_add_triangle(
		to_godot(p_vertex3),
		to_godot(p_vertex2),
		to_godot(p_vertex1),
		to_godot(p_color).to_abgr32()
	);
}

JPH::DebugRenderer::Batch JoltDebugRenderer3D::CreateTriangleBatch(
	const JPH::DebugRenderer::Triangle* p_triangles,
	int p_triangle_count
) {
	auto* triangle_batch = new JoltDebugTriangleBatch(p_triangle_count);

	const JPH::DebugRenderer::Triangle* triangles_begin = p_triangles;
	const JPH::DebugRenderer::Triangle* triangles_end = triangles_begin + p_triangle_count;

	for (const auto* it = triangles_begin; it != triangles_end; ++it) {
		triangle_batch->add_triangle(
			JPH::Vec3(it->mV[0].mPosition),
			JPH::Vec3(it->mV[1].mPosition),
			JPH::Vec3(it->mV[2].mPosition)
		);
	}

	return triangle_batch;
}

JPH::DebugRenderer::Batch JoltDebugRenderer3D::CreateTriangleBatch(
	const JPH::DebugRenderer::Vertex* p_vertices,
	[[maybe_unused]] int p_vertex_count,
	const JPH::uint32* p_indices,
	int p_index_count
) {
	auto* triangle_batch = new JoltDebugTriangleBatch(p_index_count / 3);

	for (int i = 0; i < p_index_count; i += 3) {
		const JPH::uint32 i0 = *(p_indices + i + 0);
		const JPH::uint32 i1 = *(p_indices + i + 1);
		const JPH::uint32 i2 = *(p_indices + i + 2);

		triangle_batch->add_triangle(
			JPH::Vec3(p_vertices[i0].mPosition),
			JPH::Vec3(p_vertices[i1].mPosition),
			JPH::Vec3(p_vertices[i2].mPosition)
		);
	}

	return triangle_batch;
}

void JoltDebugRenderer3D::DrawGeometry(
	JPH::RMat44Arg p_model_matrix,
	const JPH::AABox& p_world_space_bounds,
	float p_lod_scale_sq,
	JPH::Color p_model_color,
	const JPH::DebugRenderer::GeometryRef& p_geometry,
	JPH::DebugRenderer::ECullMode p_cull_mode,
	[[maybe_unused]] JPH::DebugRenderer::ECastShadow p_cast_shadow,
	JPH::DebugRenderer::EDrawMode p_draw_mode
) {
	const float camera_distance_sq = p_world_space_bounds.GetSqDistanceTo(camera_position);

	const JPH::DebugRenderer::LOD* model_lod = nullptr;

	for (const JPH::DebugRenderer::LOD& lod : p_geometry->mLODs) {
		const float lod_distance_sq = lod.mDistance * lod.mDistance;
		if (camera_distance_sq <= lod_distance_sq * p_lod_scale_sq) {
			model_lod = &lod;
			break;
		}
	}

	const auto* triangle_batch = static_cast<const JoltDebugTriangleBatch*>(
		model_lod->mTriangleBatch.GetPtr()
	);

	const JoltDebugTriangleBatch::Vertices& model_vertices = triangle_batch->get_vertices();
	const JPH::Vec3* model_vertices_ptr = model_vertices.data();

	const auto model_vertex_count = (int32_t)model_vertices.size();
	const int32_t model_triangle_count = model_vertex_count / 3;

	const uint32_t model_color_abgr = to_godot(p_model_color).to_abgr32();

	if (p_draw_mode == JPH::DebugRenderer::EDrawMode::Solid) {
		if (p_cull_mode != JPH::DebugRenderer::ECullMode::Off) {
			_reserve_triangles(model_triangle_count);
		} else {
			_reserve_triangles(model_triangle_count * 2);
		}

		for (int32_t i = 0; i < model_triangle_count; ++i) {
			const int32_t vertex_offset = i * 3;

			const Vector3 v0 = to_godot(p_model_matrix * model_vertices_ptr[vertex_offset + 0]);
			const Vector3 v1 = to_godot(p_model_matrix * model_vertices_ptr[vertex_offset + 1]);
			const Vector3 v2 = to_godot(p_model_matrix * model_vertices_ptr[vertex_offset + 2]);

			switch (p_cull_mode) {
				case JPH::DebugRenderer::ECullMode::CullBackFace: {
					_add_triangle(v2, v1, v0, model_color_abgr);
				} break;
				case JPH::DebugRenderer::ECullMode::CullFrontFace: {
					_add_triangle(v0, v1, v2, model_color_abgr);
				} break;
				case JPH::DebugRenderer::ECullMode::Off: {
					_add_triangle(v0, v1, v2, model_color_abgr);
					_add_triangle(v2, v1, v0, model_color_abgr);
				} break;
			}
		}
	} else {
		_reserve_lines(model_triangle_count * 3);

		for (int32_t i = 0; i < model_triangle_count; ++i) {
			const int32_t vertex_offset = i * 3;

			const Vector3 v0 = to_godot(p_model_matrix * model_vertices_ptr[vertex_offset + 0]);
			const Vector3 v1 = to_godot(p_model_matrix * model_vertices_ptr[vertex_offset + 1]);
			const Vector3 v2 = to_godot(p_model_matrix * model_vertices_ptr[vertex_offset + 2]);

			_add_line(v0, v1, model_color_abgr);
			_add_line(v1, v2, model_color_abgr);
			_add_line(v2, v0, model_color_abgr);
		}
	}
}

void JoltDebugRenderer3D::DrawText3D(
	[[maybe_unused]] JPH::RVec3Arg p_position,
	[[maybe_unused]] const JPH::string_view& p_string,
	[[maybe_unused]] JPH::Color p_color,
	[[maybe_unused]] float p_height
) {
	ERR_FAIL_NOT_IMPL();
}

void JoltDebugRenderer3D::_reserve_triangles(int32_t p_extra_capacity) {
	if (triangle_count + p_extra_capacity <= triangle_capacity) {
		return;
	}

	triangle_capacity += p_extra_capacity;

	const int64_t vertex_count = (int64_t)triangle_capacity * 3;
	triangle_vertices.resize(vertex_count * DEBUG_VERTEX_STRIDE);
	triangle_attributes.resize(vertex_count * DEBUG_ATTRIBUTE_STRIDE);
}

void JoltDebugRenderer3D::_reserve_lines(int32_t p_extra_capacity) {
	if (line_count + p_extra_capacity <= line_capacity) {
		return;
	}

	line_capacity += p_extra_capacity;

	const int64_t vertex_count = (int64_t)line_capacity * 2;
	line_vertices.resize(vertex_count * DEBUG_VERTEX_STRIDE);
	line_attributes.resize(vertex_count * DEBUG_ATTRIBUTE_STRIDE);
}

void JoltDebugRenderer3D::_add_triangle(
	const Vector3& p_vertex1,
	const Vector3& p_vertex2,
	const Vector3& p_vertex3,
	uint32_t p_color_abgr
) {
	const int32_t vertex_count = triangle_count * 3;

	auto* vertices_ptr = reinterpret_cast<float*>(triangle_vertices.ptrw()) +
		ptrdiff_t(vertex_count * 3);

	auto* attributes_ptr = reinterpret_cast<uint32_t*>(triangle_attributes.ptrw()) +
		ptrdiff_t(vertex_count);

	*vertices_ptr++ = (float)p_vertex1.x;
	*vertices_ptr++ = (float)p_vertex1.y;
	*vertices_ptr++ = (float)p_vertex1.z;

	*vertices_ptr++ = (float)p_vertex2.x;
	*vertices_ptr++ = (float)p_vertex2.y;
	*vertices_ptr++ = (float)p_vertex2.z;

	*vertices_ptr++ = (float)p_vertex3.x;
	*vertices_ptr++ = (float)p_vertex3.y;
	*vertices_ptr++ = (float)p_vertex3.z;

	*attributes_ptr++ = p_color_abgr;
	*attributes_ptr++ = p_color_abgr;
	*attributes_ptr++ = p_color_abgr;

	triangles_aabb.expand_to(p_vertex1);
	triangles_aabb.expand_to(p_vertex2);
	triangles_aabb.expand_to(p_vertex3);

	triangle_count++;
}

void JoltDebugRenderer3D::_add_line(
	const Vector3& p_from,
	const Vector3& p_to,
	uint32_t p_color_abgr
) {
	const int32_t vertex_count = line_count * 2;

	auto* vertices_ptr = reinterpret_cast<float*>(line_vertices.ptrw()) +
		ptrdiff_t(vertex_count * 3);

	auto* attributes_ptr = reinterpret_cast<uint32_t*>(line_attributes.ptrw()) +
		ptrdiff_t(vertex_count);

	*vertices_ptr++ = (float)p_from.x;
	*vertices_ptr++ = (float)p_from.y;
	*vertices_ptr++ = (float)p_from.z;

	*vertices_ptr++ = (float)p_to.x;
	*vertices_ptr++ = (float)p_to.y;
	*vertices_ptr++ = (float)p_to.z;

	*attributes_ptr++ = p_color_abgr;
	*attributes_ptr++ = p_color_abgr;

	lines_aabb.expand_to(p_from);
	lines_aabb.expand_to(p_to);

	line_count++;
}

#endif // JPH_DEBUG_RENDERER
