#ifdef TOOLS_ENABLED

#include "gaussian_asset_preview_control.h"

#include <cfloat>

#include "core/input/input.h"
#include "core/input/input_event.h"
#include "core/math/math_funcs.h"
#include "core/string/translation.h"
#include "core/math/transform_3d.h"
#include "editor/themes/editor_scale.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/multimesh_instance_3d.h"
#include "scene/gui/label.h"
#include "scene/gui/subviewport_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/main/viewport.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/immediate_mesh.h"
#include "scene/resources/material.h"
#include "scene/resources/multimesh.h"
#include "scene/resources/3d/world_3d.h"

#include "servers/text_server.h"

#include "gaussian_thumbnail_generator.h"

namespace {

static float _clamp_or_default(float p_value, float p_min, float p_max, float p_default) {
	if (!Math::is_finite(p_value)) {
		return p_default;
	}
	return CLAMP(p_value, p_min, p_max);
}

} // namespace

void GaussianAssetPreviewControl::_bind_methods() {}

GaussianAssetPreviewControl::GaussianAssetPreviewControl() {
	set_h_size_flags(Control::SIZE_EXPAND_FILL);
	set_v_size_flags(Control::SIZE_EXPAND_FILL);
	set_mouse_filter(Control::MOUSE_FILTER_STOP);

	_build_ui();
}

void GaussianAssetPreviewControl::_build_ui() {
	viewport_container = memnew(SubViewportContainer);
	viewport_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	viewport_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	viewport_container->set_stretch(true);
	add_child(viewport_container);

	viewport = memnew(SubViewport);
	viewport->set_disable_input(true);
	viewport->set_update_mode(SubViewport::UPDATE_ALWAYS);
	viewport->set_clear_mode(SubViewport::CLEAR_MODE_ALWAYS);
	viewport_container->add_child(viewport);

	Ref<World3D> world;
	world.instantiate();
	viewport->set_world_3d(world);

	viewport_root = memnew(Node3D);
	viewport->add_child(viewport_root);

	orbit_root = memnew(Node3D);
	viewport_root->add_child(orbit_root);

	preview_root = memnew(Node3D);
	orbit_root->add_child(preview_root);

	camera = memnew(Camera3D);
	camera->set_current(true);
	camera->set_near(0.01);
	camera->set_far(1000.0);
	camera->set_perspective(45.0, 0.01, 1000.0);
	orbit_root->add_child(camera);

	key_light = memnew(DirectionalLight3D);
	key_light->set_transform(Transform3D().looking_at(Vector3(-1, -1, -1), Vector3(0, 1, 0)));
	viewport_root->add_child(key_light);

	fill_light = memnew(DirectionalLight3D);
	fill_light->set_transform(Transform3D().looking_at(Vector3(0.5, 0.8, 0.2), Vector3(0, 1, 0)));
	fill_light->set_color(Color(0.65, 0.7, 0.8));
	viewport_root->add_child(fill_light);

	fallback_container = memnew(VBoxContainer);
	fallback_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	fallback_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	add_child(fallback_container);

	fallback_texture = memnew(TextureRect);
	fallback_texture->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	fallback_texture->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	fallback_texture->set_custom_minimum_size(Size2(0, 180 * EDSCALE));
	fallback_texture->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	fallback_container->add_child(fallback_texture);

	fallback_label = memnew(Label);
	fallback_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	fallback_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	fallback_label->set_text(TTR("No Gaussian asset selected."));
	fallback_container->add_child(fallback_label);

	_update_preview_visibility();
}

void GaussianAssetPreviewControl::_clear_viewport_scene() {
	if (!viewport_root) {
		return;
	}
	if (bounds_instance) {
		if (bounds_instance->get_parent()) {
			bounds_instance->get_parent()->remove_child(bounds_instance);
		}
		bounds_instance->queue_free();
		bounds_instance = nullptr;
	}
	if (splat_instance) {
		if (splat_instance->get_parent()) {
			splat_instance->get_parent()->remove_child(splat_instance);
		}
		splat_instance->queue_free();
		splat_instance = nullptr;
	}
}

void GaussianAssetPreviewControl::_set_placeholder_text(const String &p_text) {
	if (fallback_label) {
		fallback_label->set_text(p_text);
	}
}

void GaussianAssetPreviewControl::_update_fallback_texture() {
	if (!fallback_texture) {
		return;
	}

	Ref<Texture2D> texture;
	if (asset.is_valid()) {
		texture = asset->get_thumbnail();
	}

	if (texture.is_null() && thumbnail_generator.is_valid() && asset.is_valid() && asset->get_splat_count() > 0) {
		texture = thumbnail_generator->generate_thumbnail(asset, 256, GaussianThumbnailGenerator::THUMBNAIL_STYLE_COLOR);
	}

	fallback_texture->set_texture(texture);
	if (texture.is_valid()) {
		_set_placeholder_text(TTR("Interactive preview unavailable. Showing stored thumbnail."));
	}
}

AABB GaussianAssetPreviewControl::_resolve_asset_bounds() const {
	if (asset.is_null()) {
		return AABB(Vector3(-0.5, -0.5, -0.5), Vector3(1.0, 1.0, 1.0));
	}

	Dictionary import_metadata = asset->get_import_metadata();
	if (import_metadata.has(StringName("bounds"))) {
		const Variant bounds_var = import_metadata[StringName("bounds")];
		if (bounds_var.get_type() == Variant::AABB) {
			AABB bounds = bounds_var;
			if (bounds.size.length() > CMP_EPSILON) {
				return bounds;
			}
		}
	}

	const PackedFloat32Array positions = asset->get_positions();
	const int count = asset->get_splat_count();
	if (positions.size() < 3 || count <= 0) {
		return AABB(Vector3(-0.5, -0.5, -0.5), Vector3(1.0, 1.0, 1.0));
	}

	Vector3 min_pos(FLT_MAX, FLT_MAX, FLT_MAX);
	Vector3 max_pos(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	const float *pos_ptr = positions.ptr();
	const int available = positions.size() / 3;
	for (int i = 0; i < available; i++) {
		const Vector3 pos(pos_ptr[i * 3 + 0], pos_ptr[i * 3 + 1], pos_ptr[i * 3 + 2]);
		min_pos = min_pos.min(pos);
		max_pos = max_pos.max(pos);
	}

	if (min_pos == max_pos) {
		const Vector3 padding(0.5, 0.5, 0.5);
		return AABB(min_pos - padding, padding * 2.0);
	}

	return AABB(min_pos, max_pos - min_pos);
}

void GaussianAssetPreviewControl::_update_camera_transform() {
	if (!orbit_root || !camera) {
		return;
	}

	orbit_root->set_rotation(Vector3(orbit_pitch, orbit_yaw, 0.0f));
	camera->set_position(Vector3(0.0f, 0.0f, orbit_distance));
	camera->set_current(true);
}

void GaussianAssetPreviewControl::_build_bounds_mesh(const AABB &p_bounds) {
	if (!viewport_root || !preview_root) {
		return;
	}

	Ref<ImmediateMesh> mesh;
	mesh.instantiate();

	const Vector3 min_pos = p_bounds.position;
	const Vector3 max_pos = p_bounds.position + p_bounds.size;
	const Vector3 center_offset = orbit_center;
	const Vector3 corners[8] = {
		Vector3(min_pos.x, min_pos.y, min_pos.z) - center_offset,
		Vector3(max_pos.x, min_pos.y, min_pos.z) - center_offset,
		Vector3(max_pos.x, min_pos.y, max_pos.z) - center_offset,
		Vector3(min_pos.x, min_pos.y, max_pos.z) - center_offset,
		Vector3(min_pos.x, max_pos.y, min_pos.z) - center_offset,
		Vector3(max_pos.x, max_pos.y, min_pos.z) - center_offset,
		Vector3(max_pos.x, max_pos.y, max_pos.z) - center_offset,
		Vector3(min_pos.x, max_pos.y, max_pos.z) - center_offset,
	};
	const int edges[][2] = {
		{ 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
		{ 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
		{ 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 },
	};

	mesh->surface_begin(Mesh::PRIMITIVE_LINES);
	for (const auto &edge : edges) {
		mesh->surface_add_vertex(corners[edge[0]]);
		mesh->surface_add_vertex(corners[edge[1]]);
	}
	mesh->surface_end();

	bounds_instance = memnew(MeshInstance3D);
	bounds_instance->set_mesh(mesh);
	preview_root->add_child(bounds_instance);
}

void GaussianAssetPreviewControl::_build_sample_cloud(const AABB &p_bounds) {
	if (!asset.is_valid() || !preview_root) {
		return;
	}

	const PackedFloat32Array positions = asset->get_positions();
	const PackedColorArray colors = asset->get_colors();
	const PackedFloat32Array scales = asset->get_scales();
	const int count = asset->get_splat_count();
	if (count <= 0 || positions.size() < 3) {
		return;
	}

	const int max_instances = 512;
	const int step = MAX(1, count / max_instances);
	const int instance_count = MAX(1, (count + step - 1) / step);

	Ref<BoxMesh> sample_mesh;
	sample_mesh.instantiate();
	sample_mesh->set_size(Vector3(1.0, 1.0, 1.0));

	Ref<StandardMaterial3D> preview_material;
	preview_material.instantiate();
	preview_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	preview_material->set_flag(BaseMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	preview_material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	preview_material->set_albedo(Color(0.8, 0.9, 1.0, 0.75));
	sample_mesh->surface_set_material(0, preview_material);

	Ref<MultiMesh> multi_mesh;
	multi_mesh.instantiate();
	multi_mesh->set_transform_format(MultiMesh::TRANSFORM_3D);
	multi_mesh->set_mesh(sample_mesh);
	multi_mesh->set_use_colors(true);
	multi_mesh->set_instance_count(instance_count);

	const float *position_ptr = positions.ptr();
	const Color *color_ptr = colors.ptr();
	const float *scale_ptr = scales.ptr();
	const float bounds_span = MAX(0.001f, p_bounds.size.length());
	const float point_size = CLAMP(bounds_span * 0.008f, 0.01f, 0.12f);

	int target_idx = 0;
	for (int src = 0; src < count && target_idx < instance_count; src += step, target_idx++) {
		const int pos_base = src * 3;
		if (pos_base + 2 >= positions.size()) {
			break;
		}

		Vector3 position(position_ptr[pos_base + 0], position_ptr[pos_base + 1], position_ptr[pos_base + 2]);
		position -= orbit_center;
		float uniform_scale = point_size;
		if (scale_ptr && scales.size() >= (src * 3 + 3)) {
			const float sx = Math::abs(scale_ptr[src * 3 + 0]);
			const float sy = Math::abs(scale_ptr[src * 3 + 1]);
			const float sz = Math::abs(scale_ptr[src * 3 + 2]);
			uniform_scale *= _clamp_or_default((sx + sy + sz) / 3.0f, 0.1f, 20.0f, 1.0f);
		}

		Transform3D xform;
		xform.basis = Basis().scaled(Vector3(uniform_scale, uniform_scale, uniform_scale));
		xform.origin = position;
		multi_mesh->set_instance_transform(target_idx, xform);

		Color instance_color = Color(0.8, 0.9, 1.0, 0.8);
		if (src < colors.size()) {
			instance_color = color_ptr[src];
		}
		multi_mesh->set_instance_color(target_idx, instance_color);
	}

	splat_instance = memnew(MultiMeshInstance3D);
	splat_instance->set_multimesh(multi_mesh);
	preview_root->add_child(splat_instance);
}

void GaussianAssetPreviewControl::_rebuild_scene_from_asset() {
	_clear_viewport_scene();
	interactive_preview_ready = false;

	if (asset.is_null() || asset->get_splat_count() == 0) {
		orbit_center = Vector3();
		if (preview_root) {
			preview_root->set_position(Vector3());
		}
		_set_placeholder_text(TTR("No Gaussian splat asset selected."));
		_update_fallback_texture();
		_update_preview_visibility();
		return;
	}

	const AABB bounds = _resolve_asset_bounds();
	orbit_center = bounds.get_center();
	orbit_distance = MAX(1.5f, bounds.get_longest_axis_size() * 2.4f);
	orbit_yaw = 0.7f;
	orbit_pitch = -0.35f;
	if (preview_root) {
		// Vertex and instance transforms are centered around `orbit_center` already.
		// Keep the preview root at origin to avoid applying the offset twice.
		preview_root->set_position(Vector3());
	}

	_build_bounds_mesh(bounds);
	_build_sample_cloud(bounds);
	_update_camera_transform();
	interactive_preview_ready = true;
	_set_placeholder_text(TTR("Drag to orbit. Scroll to zoom."));
	_update_fallback_texture();
	_update_preview_visibility();
}

void GaussianAssetPreviewControl::_update_preview_visibility() {
	const bool has_asset = asset.is_valid() && asset->get_splat_count() > 0 && interactive_preview_ready && preview_root != nullptr;
	if (viewport_container) {
		viewport_container->set_visible(has_asset);
	}
	if (fallback_container) {
		fallback_container->set_visible(!has_asset);
	}
}

void GaussianAssetPreviewControl::set_asset(const Ref<GaussianSplatAsset> &p_asset) {
	if (asset == p_asset) {
		return;
	}
	asset = p_asset;

	if (thumbnail_generator.is_null()) {
		thumbnail_generator.instantiate();
	}

	_rebuild_scene_from_asset();
}

void GaussianAssetPreviewControl::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (!viewport_container || !viewport_container->is_visible_in_tree()) {
		return;
	}

	const Rect2 viewport_rect = viewport_container->get_rect();
	const Vector2 local_mouse = get_local_mouse_position();
	if (!viewport_rect.has_point(local_mouse)) {
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::LEFT) {
			orbit_dragging = mb->is_pressed();
			orbit_drag_last = local_mouse;
			accept_event();
			return;
		}
		if (mb->get_button_index() == MouseButton::WHEEL_UP && mb->is_pressed()) {
			orbit_distance = MAX(0.2f, orbit_distance * 0.9f);
			_update_camera_transform();
			accept_event();
			return;
		}
		if (mb->get_button_index() == MouseButton::WHEEL_DOWN && mb->is_pressed()) {
			orbit_distance = orbit_distance * 1.1f;
			_update_camera_transform();
			accept_event();
			return;
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid() && orbit_dragging) {
		const Vector2 delta = local_mouse - orbit_drag_last;
		orbit_drag_last = local_mouse;
		orbit_yaw -= delta.x * 0.01f;
		orbit_pitch = CLAMP(orbit_pitch - delta.y * 0.01f, -Math::PI * 0.49f, Math::PI * 0.49f);
		_update_camera_transform();
		accept_event();
	}
}

#endif // TOOLS_ENABLED
