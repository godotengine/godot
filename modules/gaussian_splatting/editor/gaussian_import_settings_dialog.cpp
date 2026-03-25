#ifdef TOOLS_ENABLED

#include "gaussian_import_settings_dialog.h"

#include <cfloat>

#include "core/config/project_settings.h"
#include "core/input/input_event.h"
#include "core/io/resource_loader.h"
#include "core/math/math_funcs.h"
#include "core/string/translation.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/inspector/editor_inspector.h"
#include "editor/themes/editor_scale.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/separator.h"
#include "scene/gui/split_container.h"
#include "scene/gui/subviewport_container.h"
#include "scene/main/viewport.h"
#include "scene/resources/3d/world_3d.h"
#include "scene/resources/camera_attributes.h"
#include "scene/resources/environment.h"
#include "scene/resources/immediate_mesh.h"
#include "scene/resources/material.h"

#include "../core/gaussian_splat_asset.h"
#include "../io/gaussian_import_preset.h"
#include "../nodes/gaussian_splat_node_3d.h"

// ---------------------------------------------------------------------------
// GaussianImportSettingsData — Object that feeds properties to EditorInspector
// (same pattern as SceneImportSettingsData in scene_import_settings.cpp)
// ---------------------------------------------------------------------------

class GaussianImportSettingsData : public Object {
	GDCLASS(GaussianImportSettingsData, Object)
	friend class GaussianImportSettingsDialog;

	HashMap<StringName, Variant> current;
	HashMap<StringName, Variant> defaults;

	bool _set(const StringName &p_name, const Variant &p_value) {
		if (defaults.has(p_name)) {
			current[p_name] = p_value;
			return true;
		}
		return false;
	}

	bool _get(const StringName &p_name, Variant &r_ret) const {
		if (current.has(p_name)) {
			r_ret = current[p_name];
			return true;
		}
		if (defaults.has(p_name)) {
			r_ret = defaults[p_name];
			return true;
		}
		return false;
	}

	void _get_property_list(List<PropertyInfo> *r_list) const {
		// General
		r_list->push_back(PropertyInfo(Variant::NIL, "General", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_CATEGORY));
		r_list->push_back(PropertyInfo(Variant::INT, "general/asset_type", PROPERTY_HINT_ENUM, "Static,Dynamic"));

		// Quality
		r_list->push_back(PropertyInfo(Variant::NIL, "Quality", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_CATEGORY));
		r_list->push_back(PropertyInfo(Variant::STRING, "quality/preset", PROPERTY_HINT_ENUM, "mobile,desktop,high,ultra,development,custom"));
		r_list->push_back(PropertyInfo(Variant::INT, "quality/max_splats", PROPERTY_HINT_RANGE, "0,5000000,1000"));
		r_list->push_back(PropertyInfo(Variant::FLOAT, "quality/density_multiplier", PROPERTY_HINT_RANGE, "0.1,1.0,0.05"));
		r_list->push_back(PropertyInfo(Variant::BOOL, "quality/enable_lod"));
		r_list->push_back(PropertyInfo(Variant::BOOL, "quality/optimize_for_gpu"));

		// Processing
		r_list->push_back(PropertyInfo(Variant::NIL, "Processing", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_CATEGORY));
		r_list->push_back(PropertyInfo(Variant::BOOL, "processing/normalize_opacity"));
		r_list->push_back(PropertyInfo(Variant::BOOL, "processing/sort_by_opacity"));

		// Compression
		r_list->push_back(PropertyInfo(Variant::NIL, "Compression", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_CATEGORY));
		r_list->push_back(PropertyInfo(Variant::BOOL, "compression/quantize_positions"));
		r_list->push_back(PropertyInfo(Variant::BOOL, "compression/quantize_colors"));
		r_list->push_back(PropertyInfo(Variant::BOOL, "compression/quantize_scales"));
		r_list->push_back(PropertyInfo(Variant::BOOL, "compression/quantize_rotations"));
		r_list->push_back(PropertyInfo(Variant::BOOL, "compression/pack_opacity"));

		// Metadata
		r_list->push_back(PropertyInfo(Variant::NIL, "Metadata", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_CATEGORY));
		r_list->push_back(PropertyInfo(Variant::BOOL, "metadata/include_statistics"));
		r_list->push_back(PropertyInfo(Variant::BOOL, "metadata/include_memory_estimate"));
	}

protected:
	static void _bind_methods() {}
};

// ---------------------------------------------------------------------------
// GaussianImportSettingsDialog
// ---------------------------------------------------------------------------

GaussianImportSettingsDialog *GaussianImportSettingsDialog::singleton = nullptr;

GaussianImportSettingsDialog *GaussianImportSettingsDialog::get_singleton() {
	return singleton;
}

void GaussianImportSettingsDialog::_bind_methods() {}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

void GaussianImportSettingsDialog::_update_theme_item_cache() {
	ConfirmationDialog::_update_theme_item_cache();
	theme_cache.light_1_icon = get_editor_theme_icon(SNAME("MaterialPreviewLight1"));
	theme_cache.light_2_icon = get_editor_theme_icon(SNAME("MaterialPreviewLight2"));
	theme_cache.rotate_icon = get_editor_theme_icon(SNAME("PreviewRotate"));
}

// ---------------------------------------------------------------------------
// Notifications
// ---------------------------------------------------------------------------

void GaussianImportSettingsDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			connect(SceneStringName(confirmed), callable_mp(this, &GaussianImportSettingsDialog::_re_import));
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			if (light_1_switch) {
				light_1_switch->set_button_icon(theme_cache.light_1_icon);
			}
			if (light_2_switch) {
				light_2_switch->set_button_icon(theme_cache.light_2_icon);
			}
			if (light_rotate_switch) {
				light_rotate_switch->set_button_icon(theme_cache.rotate_icon);
			}
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible()) {
				_clear_viewport_scene();
				loaded_asset.unref();
				if (inspector) {
					inspector->edit(nullptr);
				}
			}
		} break;
	}
}

// ---------------------------------------------------------------------------
// Light toggle callbacks
// ---------------------------------------------------------------------------

void GaussianImportSettingsDialog::_on_light_1_switch_pressed() {
	light1->set_visible(light_1_switch->is_pressed());
}

void GaussianImportSettingsDialog::_on_light_2_switch_pressed() {
	light2->set_visible(light_2_switch->is_pressed());
}

void GaussianImportSettingsDialog::_on_light_rotate_switch_pressed() {
	bool light_top_level = !light_rotate_switch->is_pressed();
	light1->set_as_top_level_keep_local(light_top_level);
	light2->set_as_top_level_keep_local(light_top_level);
}

// ---------------------------------------------------------------------------
// Inspector property edited callback
// ---------------------------------------------------------------------------

void GaussianImportSettingsDialog::_on_inspector_property_edited(const String &p_name) {
	if (!settings_data) {
		return;
	}

	// When the quality preset changes, cascade dependent fields.
	if (p_name == "quality/preset") {
		String preset_id;
		if (settings_data->current.has(StringName("quality/preset"))) {
			preset_id = String(settings_data->current[StringName("quality/preset")]);
		}
		int idx = gaussian_find_import_preset_index(preset_id);
		if (idx >= 0) {
			const GaussianImportPresetDefinition &preset = gaussian_get_import_preset_by_index(idx);
			settings_data->current[StringName("general/asset_type")] = preset.default_asset_type;
			settings_data->current[StringName("quality/max_splats")] = preset.max_splats;
			settings_data->current[StringName("quality/density_multiplier")] = preset.density_multiplier;
			settings_data->current[StringName("quality/enable_lod")] = preset.enable_lod;
			settings_data->current[StringName("quality/optimize_for_gpu")] = preset.optimize_for_gpu;
			settings_data->current[StringName("compression/quantize_positions")] = preset.quantize_positions;
			settings_data->current[StringName("compression/quantize_colors")] = preset.quantize_colors;
			settings_data->current[StringName("compression/quantize_scales")] = preset.quantize_scales;
			settings_data->current[StringName("compression/quantize_rotations")] = preset.quantize_rotations;
			settings_data->current[StringName("compression/pack_opacity")] = preset.pack_opacity;
			settings_data->notify_property_list_changed();
		}
	}
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

GaussianImportSettingsDialog::GaussianImportSettingsDialog() {
	singleton = this;
	set_title(TTR("Advanced Import Settings for Gaussian Splat"));
	set_ok_button_text(TTR("Reimport"));
	set_cancel_button_text(TTR("Close"));

	settings_data = memnew(GaussianImportSettingsData);

	_build_ui();
}

// ---------------------------------------------------------------------------
// UI construction
// ---------------------------------------------------------------------------

void GaussianImportSettingsDialog::_build_ui() {
	VBoxContainer *main_vb = memnew(VBoxContainer);
	add_child(main_vb);

	HSplitContainer *split = memnew(HSplitContainer);
	split->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	main_vb->add_child(split);

	// ---- Left: 3D viewport ----
	VBoxContainer *vp_vb = memnew(VBoxContainer);
	vp_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	vp_vb->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vp_vb->set_anchors_and_offsets_preset(Control::LayoutPreset::PRESET_FULL_RECT);
	split->add_child(vp_vb);

	viewport_container = memnew(SubViewportContainer);
	viewport_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	viewport_container->set_custom_minimum_size(Size2(10, 10));
	viewport_container->set_stretch(true);
	viewport_container->connect(SceneStringName(gui_input), callable_mp(this, &GaussianImportSettingsDialog::_viewport_input));
	vp_vb->add_child(viewport_container);

	viewport = memnew(SubViewport);
	viewport->set_use_own_world_3d(true);
	viewport->set_update_mode(SubViewport::UPDATE_ALWAYS);
	viewport_container->add_child(viewport);

	// Light toggle buttons overlaid on the viewport (top-right corner).
	HBoxContainer *viewport_hbox = memnew(HBoxContainer);
	viewport_container->add_child(viewport_hbox);
	viewport_hbox->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT, Control::PRESET_MODE_MINSIZE, 2);

	viewport_hbox->add_spacer();

	VBoxContainer *vb_light = memnew(VBoxContainer);
	vb_light->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	viewport_hbox->add_child(vb_light);

	light_rotate_switch = memnew(Button);
	light_rotate_switch->set_theme_type_variation("PreviewLightButton");
	light_rotate_switch->set_toggle_mode(true);
	light_rotate_switch->set_pressed(true);
	light_rotate_switch->set_tooltip_text(TTR("Rotate Lights With Model"));
	light_rotate_switch->connect(SceneStringName(pressed), callable_mp(this, &GaussianImportSettingsDialog::_on_light_rotate_switch_pressed));
	vb_light->add_child(light_rotate_switch);

	light_1_switch = memnew(Button);
	light_1_switch->set_theme_type_variation("PreviewLightButton");
	light_1_switch->set_toggle_mode(true);
	light_1_switch->set_pressed(true);
	light_1_switch->set_tooltip_text(TTR("Primary Light"));
	light_1_switch->connect(SceneStringName(pressed), callable_mp(this, &GaussianImportSettingsDialog::_on_light_1_switch_pressed));
	vb_light->add_child(light_1_switch);

	light_2_switch = memnew(Button);
	light_2_switch->set_theme_type_variation("PreviewLightButton");
	light_2_switch->set_toggle_mode(true);
	light_2_switch->set_pressed(true);
	light_2_switch->set_tooltip_text(TTR("Secondary Light"));
	light_2_switch->connect(SceneStringName(pressed), callable_mp(this, &GaussianImportSettingsDialog::_on_light_2_switch_pressed));
	vb_light->add_child(light_2_switch);

	// Camera (orthogonal, matching FBX dialog).
	camera = memnew(Camera3D);
	viewport->add_child(camera);
	camera->make_current();

	if (GLOBAL_GET("rendering/lights_and_shadows/use_physical_light_units")) {
		camera_attributes.instantiate();
		camera->set_attributes(camera_attributes);
	}

	// Grayscale gradient sky matching the FBX SceneImportSettingsDialog.
	procedural_sky_material.instantiate();
	procedural_sky_material->set_sky_top_color(Color(1, 1, 1));
	procedural_sky_material->set_sky_horizon_color(Color(0.5, 0.5, 0.5));
	procedural_sky_material->set_ground_horizon_color(Color(0.5, 0.5, 0.5));
	procedural_sky_material->set_ground_bottom_color(Color(0, 0, 0));
	procedural_sky_material->set_sky_curve(2.0);
	procedural_sky_material->set_ground_curve(0.5);
	procedural_sky_material->set_sun_angle_max(0.0);
	sky.instantiate();
	sky->set_material(procedural_sky_material);
	environment.instantiate();
	environment->set_background(Environment::BG_SKY);
	environment->set_sky(sky);
	environment->set_sky_custom_fov(50.0);
	camera->set_environment(environment);

	// Directional lights (identical to FBX dialog).
	light1 = memnew(DirectionalLight3D);
	light1->set_transform(Transform3D(Basis::looking_at(Vector3(-1, -1, -1))));
	light1->set_shadow(true);
	camera->add_child(light1);

	light2 = memnew(DirectionalLight3D);
	light2->set_transform(Transform3D(Basis::looking_at(Vector3(0, 1, 0), Vector3(0, 0, 1))));
	light2->set_color(Color(0.5f, 0.5f, 0.5f));
	camera->add_child(light2);

	// ---- Right: info + import settings inspector ----
	VBoxContainer *right_vb = memnew(VBoxContainer);
	right_vb->set_custom_minimum_size(Size2(300 * EDSCALE, 0));
	right_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	split->add_child(right_vb);

	file_label = memnew(Label);
	file_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	right_vb->add_child(file_label);

	stats_label = memnew(Label);
	stats_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	right_vb->add_child(stats_label);

	right_vb->add_child(memnew(HSeparator));

	inspector = memnew(EditorInspector);
	inspector->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	inspector->set_custom_minimum_size(Size2(300 * EDSCALE, 0));
	inspector->connect(SNAME("property_edited"), callable_mp(this, &GaussianImportSettingsDialog::_on_inspector_property_edited));
	right_vb->add_child(inspector);
}

// ---------------------------------------------------------------------------
// Settings data population
// ---------------------------------------------------------------------------

void GaussianImportSettingsDialog::_populate_settings_data() {
	if (!settings_data) {
		return;
	}

	settings_data->current.clear();
	settings_data->defaults.clear();

	// Set defaults from "high" preset.
	const GaussianImportPresetDefinition &preset = gaussian_get_import_preset_by_name("high");

	settings_data->defaults[StringName("general/asset_type")] = 0; // Static
	settings_data->defaults[StringName("quality/preset")] = String(preset.id);
	settings_data->defaults[StringName("quality/max_splats")] = preset.max_splats;
	settings_data->defaults[StringName("quality/density_multiplier")] = preset.density_multiplier;
	settings_data->defaults[StringName("quality/enable_lod")] = preset.enable_lod;
	settings_data->defaults[StringName("quality/optimize_for_gpu")] = preset.optimize_for_gpu;
	settings_data->defaults[StringName("processing/normalize_opacity")] = true;
	settings_data->defaults[StringName("processing/sort_by_opacity")] = false;
	settings_data->defaults[StringName("compression/quantize_positions")] = preset.quantize_positions;
	settings_data->defaults[StringName("compression/quantize_colors")] = preset.quantize_colors;
	settings_data->defaults[StringName("compression/quantize_scales")] = preset.quantize_scales;
	settings_data->defaults[StringName("compression/quantize_rotations")] = preset.quantize_rotations;
	settings_data->defaults[StringName("compression/pack_opacity")] = preset.pack_opacity;
	settings_data->defaults[StringName("metadata/include_statistics")] = preset.include_statistics;
	settings_data->defaults[StringName("metadata/include_memory_estimate")] = preset.include_memory_estimate;

	// Copy defaults into current.
	for (const KeyValue<StringName, Variant> &kv : settings_data->defaults) {
		settings_data->current[kv.key] = kv.value;
	}

	// Override with values from the .import sidecar.
	if (!import_options.is_empty()) {
		Array keys = import_options.keys();
		for (int i = 0; i < keys.size(); i++) {
			StringName key = keys[i];
			if (settings_data->defaults.has(key)) {
				settings_data->current[key] = import_options[keys[i]];
			}
		}
	}

	inspector->edit(settings_data);
	settings_data->notify_property_list_changed();
}

// ---------------------------------------------------------------------------
// Bounds resolution
// ---------------------------------------------------------------------------

AABB GaussianImportSettingsDialog::_resolve_bounds() const {
	if (loaded_asset.is_null()) {
		return AABB(Vector3(-0.5, -0.5, -0.5), Vector3(1, 1, 1));
	}

	Dictionary meta = loaded_asset->get_import_metadata();
	if (meta.has(StringName("bounds"))) {
		Variant v = meta[StringName("bounds")];
		if (v.get_type() == Variant::AABB) {
			AABB b = v;
			if (b.size.length() > CMP_EPSILON) {
				return b;
			}
		}
	}

	const PackedFloat32Array positions = loaded_asset->get_positions();
	const int count = loaded_asset->get_splat_count();
	if (positions.size() < 3 || count <= 0) {
		return AABB(Vector3(-0.5, -0.5, -0.5), Vector3(1, 1, 1));
	}

	Vector3 min_pos(FLT_MAX, FLT_MAX, FLT_MAX);
	Vector3 max_pos(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	const float *ptr = positions.ptr();
	const int n = positions.size() / 3;
	for (int i = 0; i < n; i++) {
		Vector3 p(ptr[i * 3], ptr[i * 3 + 1], ptr[i * 3 + 2]);
		min_pos = min_pos.min(p);
		max_pos = max_pos.max(p);
	}

	if (min_pos == max_pos) {
		return AABB(min_pos - Vector3(0.5, 0.5, 0.5), Vector3(1, 1, 1));
	}
	return AABB(min_pos, max_pos - min_pos);
}

// ---------------------------------------------------------------------------
// Viewport scene management
// ---------------------------------------------------------------------------

void GaussianImportSettingsDialog::_clear_viewport_scene() {
	if (bounds_instance) {
		if (bounds_instance->get_parent()) {
			bounds_instance->get_parent()->remove_child(bounds_instance);
		}
		bounds_instance->queue_free();
		bounds_instance = nullptr;
	}
	if (splat_node) {
		if (splat_node->get_parent()) {
			splat_node->get_parent()->remove_child(splat_node);
		}
		splat_node->queue_free();
		splat_node = nullptr;
	}
}

void GaussianImportSettingsDialog::_build_viewport_scene() {
	_clear_viewport_scene();

	if (loaded_asset.is_null() || loaded_asset->get_splat_count() == 0) {
		return;
	}

	asset_bounds = _resolve_bounds();
	Vector3 center = asset_bounds.get_center();

	// -- Bounds wireframe --
	{
		Ref<ImmediateMesh> mesh;
		mesh.instantiate();

		Vector3 mn = asset_bounds.position - center;
		Vector3 mx = mn + asset_bounds.size;
		Vector3 corners[8] = {
			{ mn.x, mn.y, mn.z }, { mx.x, mn.y, mn.z },
			{ mx.x, mn.y, mx.z }, { mn.x, mn.y, mx.z },
			{ mn.x, mx.y, mn.z }, { mx.x, mx.y, mn.z },
			{ mx.x, mx.y, mx.z }, { mn.x, mx.y, mx.z },
		};
		int edges[][2] = {
			{ 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
			{ 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
			{ 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 },
		};

		Ref<StandardMaterial3D> wire_mat;
		wire_mat.instantiate();
		wire_mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
		wire_mat->set_albedo(Color(0.6, 0.8, 1.0, 0.5));
		wire_mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);

		mesh->surface_begin(Mesh::PRIMITIVE_LINES);
		for (const auto &e : edges) {
			mesh->surface_add_vertex(corners[e[0]]);
			mesh->surface_add_vertex(corners[e[1]]);
		}
		mesh->surface_end();
		mesh->surface_set_material(0, wire_mat);

		bounds_instance = memnew(MeshInstance3D);
		bounds_instance->set_mesh(mesh);
		viewport->add_child(bounds_instance);
	}

	// -- Real Gaussian Splat preview --
	{
		splat_node = memnew(GaussianSplatNode3D);
		splat_node->set_name("ImportPreviewSplat");
		splat_node->set_auto_load(false);
		splat_node->set_splat_asset(loaded_asset);
		splat_node->set_quality_preset(GaussianSplatNode3D::QUALITY_BALANCED);
		splat_node->set_update_mode(GaussianSplatNode3D::UPDATE_MODE_ALWAYS);
		// Center the splat at origin to match the bounds wireframe.
		splat_node->set_position(-center);
		viewport->add_child(splat_node);
	}
}

// ---------------------------------------------------------------------------
// Camera (orthogonal, matching FBX dialog)
// ---------------------------------------------------------------------------

void GaussianImportSettingsDialog::_update_camera() {
	if (!camera) {
		return;
	}

	float camera_size = asset_bounds.get_longest_axis_size();
	if (camera_size < CMP_EPSILON) {
		camera_size = 2.0f;
	}

	// Orthogonal projection matching SceneImportSettingsDialog.
	camera->set_orthogonal(camera_size * cam_zoom, 0.0001, camera_size * 2);

	Transform3D xf;
	xf.basis = Basis(Vector3(0, 1, 0), cam_rot_y) * Basis(Vector3(1, 0, 0), cam_rot_x);
	xf.origin = Vector3();
	xf.translate_local(0, 0, camera_size);

	camera->set_transform(xf);
}

// ---------------------------------------------------------------------------
// Stats display
// ---------------------------------------------------------------------------

void GaussianImportSettingsDialog::_update_stats() {
	if (!loaded_asset.is_valid()) {
		stats_label->set_text(TTR("No asset data available."));
		return;
	}

	String text;
	int count = loaded_asset->get_splat_count();
	text += vformat(TTR("Splats: %s"), String::num_int64(count)) + "\n";

	AABB b = _resolve_bounds();
	text += vformat(TTR("Bounds: %.2f x %.2f x %.2f"), b.size.x, b.size.y, b.size.z) + "\n";

	Dictionary meta = loaded_asset->get_import_metadata();
	if (meta.has(StringName("memory_estimate_bytes"))) {
		double mem_mb = double(int64_t(meta[StringName("memory_estimate_bytes")])) / (1024.0 * 1024.0);
		text += vformat(TTR("Est. Memory: %.1f MB"), mem_mb) + "\n";
	}

	uint32_t comp_flags = loaded_asset->get_compression_flags();
	if (comp_flags != 0) {
		text += TTR("Compressed: ");
		PackedStringArray parts;
		if (comp_flags & GaussianSplatAsset::COMPRESSION_POSITIONS) {
			parts.push_back("Pos");
		}
		if (comp_flags & GaussianSplatAsset::COMPRESSION_COLORS) {
			parts.push_back("Col");
		}
		if (comp_flags & GaussianSplatAsset::COMPRESSION_SCALES) {
			parts.push_back("Scl");
		}
		if (comp_flags & GaussianSplatAsset::COMPRESSION_ROTATIONS) {
			parts.push_back("Rot");
		}
		text += String(", ").join(parts) + "\n";
	}

	stats_label->set_text(text.strip_edges());
}

// ---------------------------------------------------------------------------
// Asset loading
// ---------------------------------------------------------------------------

void GaussianImportSettingsDialog::_load_source_asset() {
	loaded_asset.unref();
	asset_bounds = AABB(Vector3(-0.5, -0.5, -0.5), Vector3(1, 1, 1));

	if (source_path.is_empty()) {
		return;
	}

	loaded_asset = ResourceLoader::load(source_path, "GaussianSplatAsset");
}

// ---------------------------------------------------------------------------
// Input handling (orbit + zoom + magnify gesture)
// ---------------------------------------------------------------------------

void GaussianImportSettingsDialog::_viewport_input(const Ref<InputEvent> &p_input) {
	Ref<InputEventMouseMotion> mm = p_input;
	if (mm.is_valid() && mm->get_button_mask().has_flag(MouseButtonMask::LEFT)) {
		cam_rot_x -= mm->get_relative().y * 0.01 * EDSCALE;
		cam_rot_y -= mm->get_relative().x * 0.01 * EDSCALE;
		cam_rot_x = CLAMP(cam_rot_x, -Math::PI / 2, Math::PI / 2);
		_update_camera();
	}

	if (mm.is_valid() && DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_CURSOR_SHAPE)) {
		DisplayServer::get_singleton()->cursor_set_shape(DisplayServer::CursorShape::CURSOR_ARROW);
	}

	Ref<InputEventMouseButton> mb = p_input;
	if (mb.is_valid() && mb->get_button_index() == MouseButton::WHEEL_DOWN) {
		cam_zoom *= 1.1;
		if (cam_zoom > 10.0) {
			cam_zoom = 10.0;
		}
		_update_camera();
	}
	if (mb.is_valid() && mb->get_button_index() == MouseButton::WHEEL_UP) {
		cam_zoom /= 1.1;
		if (cam_zoom < 0.1) {
			cam_zoom = 0.1;
		}
		_update_camera();
	}

	Ref<InputEventMagnifyGesture> mg = p_input;
	if (mg.is_valid()) {
		real_t mg_factor = mg->get_factor();
		if (mg_factor == 0.0) {
			mg_factor = 1.0;
		}
		cam_zoom /= mg_factor;
		if (cam_zoom < 0.1) {
			cam_zoom = 0.1;
		} else if (cam_zoom > 10.0) {
			cam_zoom = 10.0;
		}
		_update_camera();
	}
}

// ---------------------------------------------------------------------------
// Reimport
// ---------------------------------------------------------------------------

void GaussianImportSettingsDialog::_re_import() {
	if (source_path.is_empty()) {
		return;
	}

	String ext = source_path.get_extension().to_lower();
	String importer_name;
	if (ext == "ply") {
		importer_name = "gaussian_splat_ply";
	} else if (ext == "spz") {
		importer_name = "gaussian_splat_spz";
	} else {
		importer_name = "";
	}

	// Gather all settings from the inspector data object.
	HashMap<StringName, Variant> params;
	if (settings_data) {
		for (const KeyValue<StringName, Variant> &kv : settings_data->current) {
			params[kv.key] = kv.value;
		}
	}

	// Carry over existing options from the .import file that we don't expose.
	if (!import_options.is_empty()) {
		Array keys = import_options.keys();
		for (int i = 0; i < keys.size(); i++) {
			StringName key = keys[i];
			if (!params.has(key)) {
				params[key] = import_options[keys[i]];
			}
		}
	}

	_clear_viewport_scene();
	loaded_asset.unref();

	if (!importer_name.is_empty()) {
		EditorFileSystem::get_singleton()->reimport_file_with_custom_parameters(source_path, importer_name, params);
	}
}

// ---------------------------------------------------------------------------
// open_settings — entry point
// ---------------------------------------------------------------------------

void GaussianImportSettingsDialog::open_settings(const String &p_path) {
	source_path = p_path;
	import_options.clear();

	// Read existing import params from .import sidecar.
	{
		Ref<ConfigFile> config;
		config.instantiate();
		Error err = config->load(p_path + ".import");
		if (err == OK) {
			if (config->has_section("params")) {
				Vector<String> keys = config->get_section_keys("params");
				for (const String &k : keys) {
					import_options[k] = config->get_value("params", k);
				}
			}
		}
	}

	file_label->set_text(vformat(TTR("Source: %s"), source_path.get_file()));

	// Load the imported asset for preview.
	_load_source_asset();

	// Populate the EditorInspector with import settings.
	_populate_settings_data();

	// Build 3D preview.
	cam_rot_x = -Math::PI / 4;
	cam_rot_y = -Math::PI / 4;
	cam_zoom = 1.0f;
	_build_viewport_scene();
	_update_camera();
	_update_stats();

	set_title(vformat(TTR("Advanced Import Settings for '%s'"), source_path.get_file()));
	popup_centered_ratio(0.7f);
}

#endif // TOOLS_ENABLED
