#pragma once

#ifdef JPH_DEBUG_RENDERER

class JoltSpace3D;

class JoltDebugRenderer3D final : public JPH::DebugRenderer {
public:
	struct DrawSettings {
		JPH::BodyManager::EShapeColor color_scheme = JPH::BodyManager::EShapeColor::ShapeTypeColor;

		bool draw_bodies = true;

		bool draw_shapes = true;

		bool draw_constraints = true;

		bool draw_bounding_boxes = false;

		bool draw_centers_of_mass = false;

		bool draw_transforms = false;

		bool draw_velocities = false;

		bool draw_soft_body_vertices = false;

		bool draw_soft_body_edge_constraints = false;

		bool draw_soft_body_volume_constraints = false;

		bool draw_soft_body_predicted_bounds = false;

		bool draw_constraint_reference_frames = false;

		bool draw_constraint_limits = false;

		bool draw_as_wireframe = true;
	};

	static JoltDebugRenderer3D* acquire();

	static void release(JoltDebugRenderer3D*& p_ptr);

	void draw(const JoltSpace3D& p_space, const Camera3D& p_camera, const DrawSettings& p_settings);

	int32_t submit(const RID& p_mesh);

private:
	JoltDebugRenderer3D() { Initialize(); }

	void DrawLine(JPH::RVec3Arg p_from, JPH::RVec3Arg p_to, JPH::Color p_color) override;

	void DrawTriangle(
		JPH::RVec3Arg p_vertex1,
		JPH::RVec3Arg p_vertex2,
		JPH::RVec3Arg p_vertex3,
		JPH::Color p_color,
		ECastShadow p_cast_shadow
	) override;

	JPH::DebugRenderer::Batch CreateTriangleBatch(
		const JPH::DebugRenderer::Triangle* p_triangles,
		int p_triangle_count
	) override;

	JPH::DebugRenderer::Batch CreateTriangleBatch(
		const JPH::DebugRenderer::Vertex* p_vertices,
		int p_vertex_count,
		const JPH::uint32* p_indices,
		int p_index_count
	) override;

	void DrawGeometry(
		JPH::RMat44Arg p_model_matrix,
		const JPH::AABox& p_world_space_bounds,
		float p_lod_scale_sq,
		JPH::Color p_model_color,
		const JPH::DebugRenderer::GeometryRef& p_geometry,
		JPH::DebugRenderer::ECullMode p_cull_mode,
		JPH::DebugRenderer::ECastShadow p_cast_shadow,
		JPH::DebugRenderer::EDrawMode p_draw_mode
	) override;

	void DrawText3D(
		JPH::RVec3Arg p_position,
		const JPH::string_view& p_string,
		JPH::Color p_color = JPH::Color::sWhite,
		float p_height = 0.5f
	) override;

	void _reserve_triangles(int32_t p_extra_capacity);

	void _reserve_lines(int32_t p_extra_capacity);

	void _add_triangle(
		const Vector3& p_vertex1,
		const Vector3& p_vertex2,
		const Vector3& p_vertex3,
		uint32_t p_color_abgr
	);

	void _add_line(const Vector3& p_from, const Vector3& p_to, uint32_t p_color_abgr);

	inline static JoltDebugRenderer3D* singleton = nullptr;

	inline static int32_t ref_count = 0;

	AABB triangles_aabb;

	AABB lines_aabb;

	PackedByteArray triangle_vertices;

	PackedByteArray triangle_attributes;

	PackedByteArray line_vertices;

	PackedByteArray line_attributes;

	JPH::Vec3 camera_position = {0, 0, 0};

	int32_t triangle_capacity = 0;

	int32_t triangle_count = 0;

	int32_t line_capacity = 0;

	int32_t line_count = 0;
};

#endif // JPH_DEBUG_RENDERER
