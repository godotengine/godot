#include "jolt_debug_geometry_3d.hpp"

#include "servers/jolt_physics_server_3d.hpp"
#include "spaces/jolt_debug_renderer_3d.hpp"
#include "spaces/jolt_space_3d.hpp"

void JoltDebugGeometry3D::_bind_methods() {
	BIND_METHOD(JoltDebugGeometry3D, get_draw_bodies);
	BIND_METHOD(JoltDebugGeometry3D, set_draw_bodies, "enabled");

	BIND_METHOD(JoltDebugGeometry3D, get_draw_shapes);
	BIND_METHOD(JoltDebugGeometry3D, set_draw_shapes, "enabled");

	BIND_METHOD(JoltDebugGeometry3D, get_draw_constraints);
	BIND_METHOD(JoltDebugGeometry3D, set_draw_constraints, "enabled");

	BIND_METHOD(JoltDebugGeometry3D, get_draw_bounding_boxes);
	BIND_METHOD(JoltDebugGeometry3D, set_draw_bounding_boxes, "enabled");

	BIND_METHOD(JoltDebugGeometry3D, get_draw_centers_of_mass);
	BIND_METHOD(JoltDebugGeometry3D, set_draw_centers_of_mass, "enabled");

	BIND_METHOD(JoltDebugGeometry3D, get_draw_transforms);
	BIND_METHOD(JoltDebugGeometry3D, set_draw_transforms, "enabled");

	BIND_METHOD(JoltDebugGeometry3D, get_draw_velocities);
	BIND_METHOD(JoltDebugGeometry3D, set_draw_velocities, "enabled");

	BIND_METHOD(JoltDebugGeometry3D, get_draw_triangle_outlines);
	BIND_METHOD(JoltDebugGeometry3D, set_draw_triangle_outlines, "enabled");

	BIND_METHOD(JoltDebugGeometry3D, get_draw_soft_body_vertices);
	BIND_METHOD(JoltDebugGeometry3D, set_draw_soft_body_vertices, "enabled");

	BIND_METHOD(JoltDebugGeometry3D, get_draw_soft_body_edge_constraints);
	BIND_METHOD(JoltDebugGeometry3D, set_draw_soft_body_edge_constraints, "enabled");

	BIND_METHOD(JoltDebugGeometry3D, get_draw_soft_body_volume_constraints);
	BIND_METHOD(JoltDebugGeometry3D, set_draw_soft_body_volume_constraints, "enabled");

	BIND_METHOD(JoltDebugGeometry3D, get_draw_soft_body_predicted_bounds);
	BIND_METHOD(JoltDebugGeometry3D, set_draw_soft_body_predicted_bounds, "enabled");

	BIND_METHOD(JoltDebugGeometry3D, get_draw_constraint_reference_frames);
	BIND_METHOD(JoltDebugGeometry3D, set_draw_constraint_reference_frames, "enabled");

	BIND_METHOD(JoltDebugGeometry3D, get_draw_constraint_limits);
	BIND_METHOD(JoltDebugGeometry3D, set_draw_constraint_limits, "enabled");

	BIND_METHOD(JoltDebugGeometry3D, get_draw_as_wireframe);
	BIND_METHOD(JoltDebugGeometry3D, set_draw_as_wireframe, "enabled");

	BIND_METHOD(JoltDebugGeometry3D, get_draw_with_color_scheme);
	BIND_METHOD(JoltDebugGeometry3D, set_draw_with_color_scheme, "color_scheme");

	BIND_METHOD(JoltDebugGeometry3D, get_material_depth_test);
	BIND_METHOD(JoltDebugGeometry3D, set_material_depth_test, "enabled");

	ADD_GROUP("Draw", "draw_");

	BIND_PROPERTY("draw_bodies", Variant::BOOL);

	BIND_PROPERTY("draw_shapes", Variant::BOOL);

	BIND_PROPERTY("draw_constraints", Variant::BOOL);

	BIND_PROPERTY("draw_bounding_boxes", Variant::BOOL);

	BIND_PROPERTY("draw_centers_of_mass", Variant::BOOL);

	BIND_PROPERTY("draw_transforms", Variant::BOOL);

	BIND_PROPERTY("draw_velocities", Variant::BOOL);

	BIND_PROPERTY("draw_triangle_outlines", Variant::BOOL);

	BIND_PROPERTY("draw_soft_body_vertices", Variant::BOOL);

	BIND_PROPERTY("draw_soft_body_edge_constraints", Variant::BOOL);

	BIND_PROPERTY("draw_soft_body_volume_constraints", Variant::BOOL);

	BIND_PROPERTY("draw_soft_body_predicted_bounds", Variant::BOOL);

	BIND_PROPERTY("draw_constraint_reference_frames", Variant::BOOL);

	BIND_PROPERTY("draw_constraint_limits", Variant::BOOL);

	BIND_PROPERTY("draw_as_wireframe", Variant::BOOL);

	BIND_PROPERTY_ENUM(
		"draw_with_color_scheme",
		"Instance,Shape Type,Motion Type,Sleep State,Island"
	);

	ADD_GROUP("Material", "material_");

	BIND_PROPERTY("material_depth_test", Variant::BOOL);

	BIND_ENUM_CONSTANT(COLOR_SCHEME_INSTANCE);
	BIND_ENUM_CONSTANT(COLOR_SCHEME_SHAPE_TYPE);
	BIND_ENUM_CONSTANT(COLOR_SCHEME_MOTION_TYPE);
	BIND_ENUM_CONSTANT(COLOR_SCHEME_SLEEP_STATE);
	BIND_ENUM_CONSTANT(COLOR_SCHEME_ISLAND);
}

#ifdef JPH_DEBUG_RENDERER

JoltDebugGeometry3D::JoltDebugGeometry3D()
	: mesh(RenderingServer::get_singleton()->mesh_create())
	, debug_renderer(JoltDebugRenderer3D::acquire()) {
	set_base(mesh);

	set_cast_shadows_setting(GeometryInstance3D::SHADOW_CASTING_SETTING_OFF);

	default_material.instantiate();
	default_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	default_material->set_specular_mode(StandardMaterial3D::SPECULAR_DISABLED);
	default_material->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
}

JoltDebugGeometry3D::~JoltDebugGeometry3D() {
	if (mesh.is_valid()) {
		RenderingServer::get_singleton()->free_rid(mesh);
	}

	JoltDebugRenderer3D::release(debug_renderer);
}

#else // JPH_DEBUG_RENDERER

JoltDebugGeometry3D::JoltDebugGeometry3D() = default;

JoltDebugGeometry3D::~JoltDebugGeometry3D() = default;

#endif // JPH_DEBUG_RENDERER

void JoltDebugGeometry3D::process([[maybe_unused]] double p_delta) {
#ifdef JPH_DEBUG_RENDERER
	JoltPhysicsServer3D* physics_server = JoltPhysicsServer3D::get_singleton();

	if (unlikely(physics_server == nullptr)) {
		ERR_PRINT_ONCE(
			"JoltDebugGeometry3D was unable to retrieve the Jolt-based physics server. Make sure "
			"that you have set 'JoltPhysics3D' as the currently active physics server in your "
			"project settings."
		);

		return;
	}

	RenderingServer* rendering_server = RenderingServer::get_singleton();
	ERR_FAIL_NULL(rendering_server);

	const JoltSpace3D* space = physics_server->get_space(get_world_3d()->get_space());
	ERR_FAIL_NULL(space);

	const Viewport* viewport = get_viewport();
	ERR_FAIL_NULL(viewport);

	const Camera3D* camera = viewport->get_camera_3d();
	ERR_FAIL_NULL(camera);

	debug_renderer->draw(*space, *camera, draw_settings);
	const int32_t surface_count = debug_renderer->submit(mesh);

	const RID material_rid = default_material->get_rid();

	for (int32_t i = 0; i < surface_count; ++i) {
		rendering_server->mesh_surface_set_material(mesh, i, material_rid);
	}
#endif // JPH_DEBUG_RENDERER
}

bool JoltDebugGeometry3D::get_draw_bodies() const {
#ifdef JPH_DEBUG_RENDERER
	return draw_settings.draw_bodies;
#else // JPH_DEBUG_RENDERER
	return true;
#endif // JPH_DEBUG_RENDERER
}

void JoltDebugGeometry3D::set_draw_bodies([[maybe_unused]] bool p_enabled) {
#ifdef JPH_DEBUG_RENDERER
	draw_settings.draw_bodies = p_enabled;
#endif // JPH_DEBUG_RENDERER
}

bool JoltDebugGeometry3D::get_draw_shapes() const {
#ifdef JPH_DEBUG_RENDERER
	return draw_settings.draw_shapes;
#else // JPH_DEBUG_RENDERER
	return true;
#endif // JPH_DEBUG_RENDERER
}

void JoltDebugGeometry3D::set_draw_shapes([[maybe_unused]] bool p_enabled) {
#ifdef JPH_DEBUG_RENDERER
	draw_settings.draw_shapes = p_enabled;
#endif // JPH_DEBUG_RENDERER
}

bool JoltDebugGeometry3D::get_draw_constraints() const {
#ifdef JPH_DEBUG_RENDERER
	return draw_settings.draw_constraints;
#else // JPH_DEBUG_RENDERER
	return true;
#endif // JPH_DEBUG_RENDERER
}

void JoltDebugGeometry3D::set_draw_constraints([[maybe_unused]] bool p_enabled) {
#ifdef JPH_DEBUG_RENDERER
	draw_settings.draw_constraints = p_enabled;
#endif // JPH_DEBUG_RENDERER
}

bool JoltDebugGeometry3D::get_draw_bounding_boxes() const {
#ifdef JPH_DEBUG_RENDERER
	return draw_settings.draw_bounding_boxes;
#else // JPH_DEBUG_RENDERER
	return false;
#endif // JPH_DEBUG_RENDERER
}

void JoltDebugGeometry3D::set_draw_bounding_boxes([[maybe_unused]] bool p_enabled) {
#ifdef JPH_DEBUG_RENDERER
	draw_settings.draw_bounding_boxes = p_enabled;
#endif // JPH_DEBUG_RENDERER
}

bool JoltDebugGeometry3D::get_draw_centers_of_mass() const {
#ifdef JPH_DEBUG_RENDERER
	return draw_settings.draw_centers_of_mass;
#else // JPH_DEBUG_RENDERER
	return false;
#endif // JPH_DEBUG_RENDERER
}

void JoltDebugGeometry3D::set_draw_centers_of_mass([[maybe_unused]] bool p_enabled) {
#ifdef JPH_DEBUG_RENDERER
	draw_settings.draw_centers_of_mass = p_enabled;
#endif // JPH_DEBUG_RENDERER
}

bool JoltDebugGeometry3D::get_draw_transforms() const {
#ifdef JPH_DEBUG_RENDERER
	return draw_settings.draw_transforms;
#else // JPH_DEBUG_RENDERER
	return false;
#endif // JPH_DEBUG_RENDERER
}

void JoltDebugGeometry3D::set_draw_transforms([[maybe_unused]] bool p_enabled) {
#ifdef JPH_DEBUG_RENDERER
	draw_settings.draw_transforms = p_enabled;
#endif // JPH_DEBUG_RENDERER
}

bool JoltDebugGeometry3D::get_draw_velocities() const {
#ifdef JPH_DEBUG_RENDERER
	return draw_settings.draw_velocities;
#else // JPH_DEBUG_RENDERER
	return false;
#endif // JPH_DEBUG_RENDERER
}

void JoltDebugGeometry3D::set_draw_velocities([[maybe_unused]] bool p_enabled) {
#ifdef JPH_DEBUG_RENDERER
	draw_settings.draw_velocities = p_enabled;
#endif // JPH_DEBUG_RENDERER
}

bool JoltDebugGeometry3D::get_draw_triangle_outlines() const {
#ifdef JPH_DEBUG_RENDERER
	return JPH::MeshShape::sDrawTriangleOutlines;
#else // JPH_DEBUG_RENDERER
	return false;
#endif // JPH_DEBUG_RENDERER
}

void JoltDebugGeometry3D::set_draw_triangle_outlines([[maybe_unused]] bool p_enabled) {
#ifdef JPH_DEBUG_RENDERER
	JPH::MeshShape::sDrawTriangleOutlines = p_enabled;
	JPH::HeightFieldShape::sDrawTriangleOutlines = p_enabled;
#endif // JPH_DEBUG_RENDERER
}

bool JoltDebugGeometry3D::get_draw_soft_body_vertices() const {
#ifdef JPH_DEBUG_RENDERER
	return draw_settings.draw_soft_body_vertices;
#else // JPH_DEBUG_RENDERER
	return false;
#endif // JPH_DEBUG_RENDERER
}

void JoltDebugGeometry3D::set_draw_soft_body_vertices([[maybe_unused]] bool p_enabled) {
#ifdef JPH_DEBUG_RENDERER
	draw_settings.draw_soft_body_vertices = p_enabled;
#endif // JPH_DEBUG_RENDERER
}

bool JoltDebugGeometry3D::get_draw_soft_body_edge_constraints() const {
#ifdef JPH_DEBUG_RENDERER
	return draw_settings.draw_soft_body_edge_constraints;
#else // JPH_DEBUG_RENDERER
	return false;
#endif // JPH_DEBUG_RENDERER
}

void JoltDebugGeometry3D::set_draw_soft_body_edge_constraints([[maybe_unused]] bool p_enabled) {
#ifdef JPH_DEBUG_RENDERER
	draw_settings.draw_soft_body_edge_constraints = p_enabled;
#endif // JPH_DEBUG_RENDERER
}

bool JoltDebugGeometry3D::get_draw_soft_body_volume_constraints() const {
#ifdef JPH_DEBUG_RENDERER
	return draw_settings.draw_soft_body_volume_constraints;
#else // JPH_DEBUG_RENDERER
	return false;
#endif // JPH_DEBUG_RENDERER
}

void JoltDebugGeometry3D::set_draw_soft_body_volume_constraints([[maybe_unused]] bool p_enabled) {
#ifdef JPH_DEBUG_RENDERER
	draw_settings.draw_soft_body_volume_constraints = p_enabled;
#endif // JPH_DEBUG_RENDERER
}

bool JoltDebugGeometry3D::get_draw_soft_body_predicted_bounds() const {
#ifdef JPH_DEBUG_RENDERER
	return draw_settings.draw_soft_body_predicted_bounds;
#else // JPH_DEBUG_RENDERER
	return false;
#endif // JPH_DEBUG_RENDERER
}

void JoltDebugGeometry3D::set_draw_soft_body_predicted_bounds([[maybe_unused]] bool p_enabled) {
#ifdef JPH_DEBUG_RENDERER
	draw_settings.draw_soft_body_predicted_bounds = p_enabled;
#endif // JPH_DEBUG_RENDERER
}

bool JoltDebugGeometry3D::get_draw_constraint_reference_frames() const {
#ifdef JPH_DEBUG_RENDERER
	return draw_settings.draw_constraint_reference_frames;
#else // JPH_DEBUG_RENDERER
	return false;
#endif // JPH_DEBUG_RENDERER
}

void JoltDebugGeometry3D::set_draw_constraint_reference_frames([[maybe_unused]] bool p_enabled) {
#ifdef JPH_DEBUG_RENDERER
	draw_settings.draw_constraint_reference_frames = p_enabled;
#endif // JPH_DEBUG_RENDERER
}

bool JoltDebugGeometry3D::get_draw_constraint_limits() const {
#ifdef JPH_DEBUG_RENDERER
	return draw_settings.draw_constraint_limits;
#else // JPH_DEBUG_RENDERER
	return false;
#endif // JPH_DEBUG_RENDERER
}

void JoltDebugGeometry3D::set_draw_constraint_limits([[maybe_unused]] bool p_enabled) {
#ifdef JPH_DEBUG_RENDERER
	draw_settings.draw_constraint_limits = p_enabled;
#endif // JPH_DEBUG_RENDERER
}

bool JoltDebugGeometry3D::get_draw_as_wireframe() const {
#ifdef JPH_DEBUG_RENDERER
	return draw_settings.draw_as_wireframe;
#else // JPH_DEBUG_RENDERER
	return true;
#endif // JPH_DEBUG_RENDERER
}

void JoltDebugGeometry3D::set_draw_as_wireframe([[maybe_unused]] bool p_enabled) {
#ifdef JPH_DEBUG_RENDERER
	draw_settings.draw_as_wireframe = p_enabled;
#endif // JPH_DEBUG_RENDERER
}

JoltDebugGeometry3D::ColorScheme JoltDebugGeometry3D::get_draw_with_color_scheme() const {
#ifdef JPH_DEBUG_RENDERER
	return (ColorScheme)draw_settings.color_scheme;
#else // JPH_DEBUG_RENDERER
	return ColorScheme::COLOR_SCHEME_SHAPE_TYPE;
#endif // JPH_DEBUG_RENDERER
}

void JoltDebugGeometry3D::set_draw_with_color_scheme([[maybe_unused]] ColorScheme p_color_scheme) {
#ifdef JPH_DEBUG_RENDERER
	draw_settings.color_scheme = (JPH::BodyManager::EShapeColor)p_color_scheme;
#endif // JPH_DEBUG_RENDERER
}

bool JoltDebugGeometry3D::get_material_depth_test() const {
#ifdef JPH_DEBUG_RENDERER
	ERR_FAIL_NULL_V(default_material, true);
	return !default_material->get_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST);
#else // JPH_DEBUG_RENDERER
	return true;
#endif // JPH_DEBUG_RENDERER
}

void JoltDebugGeometry3D::set_material_depth_test([[maybe_unused]] bool p_enabled) {
#ifdef JPH_DEBUG_RENDERER
	ERR_FAIL_NULL(default_material);
	default_material->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, !p_enabled);
#endif // JPH_DEBUG_RENDERER
}

#ifdef JPH_DEBUG_RENDERER

static_assert(
	(int32_t)JoltDebugGeometry3D::COLOR_SCHEME_INSTANCE ==
	(int32_t)JPH::BodyManager::EShapeColor::InstanceColor
);

static_assert(
	(int32_t)JoltDebugGeometry3D::COLOR_SCHEME_SHAPE_TYPE ==
	(int32_t)JPH::BodyManager::EShapeColor::ShapeTypeColor
);

static_assert(
	(int32_t)JoltDebugGeometry3D::COLOR_SCHEME_MOTION_TYPE ==
	(int32_t)JPH::BodyManager::EShapeColor::MotionTypeColor
);

static_assert(
	(int32_t)JoltDebugGeometry3D::COLOR_SCHEME_SLEEP_STATE ==
	(int32_t)JPH::BodyManager::EShapeColor::SleepColor
);

static_assert(
	(int32_t)JoltDebugGeometry3D::COLOR_SCHEME_ISLAND ==
	(int32_t)JPH::BodyManager::EShapeColor::IslandColor
);

#endif // JPH_DEBUG_RENDERER
