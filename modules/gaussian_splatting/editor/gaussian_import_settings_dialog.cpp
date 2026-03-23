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
#include "editor/themes/editor_scale.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/multimesh_instance_3d.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/check_box.h"
#include "scene/gui/label.h"
#include "scene/gui/option_button.h"
#include "scene/gui/separator.h"
#include "scene/gui/split_container.h"
#include "scene/gui/subviewport_container.h"
#include "scene/main/viewport.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/3d/world_3d.h"
#include "scene/resources/camera_attributes.h"
#include "scene/resources/environment.h"
#include "scene/resources/immediate_mesh.h"
#include "scene/resources/material.h"
#include "scene/resources/multimesh.h"

#include "../core/gaussian_splat_asset.h"
#include "../io/gaussian_import_preset.h"

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
// Constructor
// ---------------------------------------------------------------------------

GaussianImportSettingsDialog::GaussianImportSettingsDialog() {
	singleton = this;
	set_title(TTR("Advanced Import Settings for Gaussian Splat"));
	set_ok_button_text(TTR("Reimport"));
	set_cancel_button_text(TTR("Close"));
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

	// ---- Right: info + import settings ----
	VBoxContainer *right_vb = memnew(VBoxContainer);
	right_vb->set_custom_minimum_size(Size2(280 * EDSCALE, 0));
	right_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	split->add_child(right_vb);

	file_label = memnew(Label);
	file_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	right_vb->add_child(file_label);

	stats_label = memnew(Label);
	stats_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	stats_label->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	right_vb->add_child(stats_label);

	right_vb->add_child(memnew(HSeparator));

	Label *preset_label = memnew(Label);
	preset_label->set_text(TTR("Quality Preset"));
	right_vb->add_child(preset_label);

	quality_selector = memnew(OptionButton);
	const Vector<GaussianImportPresetDefinition> &presets = gaussian_get_import_presets();
	for (int i = 0; i < presets.size(); i++) {
		quality_selector->add_item(presets[i].display_name);
	}
	quality_selector->select(gaussian_find_import_preset_index("high"));
	right_vb->add_child(quality_selector);

	right_vb->add_child(memnew(HSeparator));

	Label *compression_label = memnew(Label);
	compression_label->set_text(TTR("Compression"));
	right_vb->add_child(compression_label);

	compress_positions = memnew(CheckBox);
	compress_positions->set_text(TTR("Quantize Positions"));
	right_vb->add_child(compress_positions);

	compress_colors = memnew(CheckBox);
	compress_colors->set_text(TTR("Quantize Colors"));
	right_vb->add_child(compress_colors);

	compress_scales = memnew(CheckBox);
	compress_scales->set_text(TTR("Quantize Scales"));
	right_vb->add_child(compress_scales);

	compress_rotations = memnew(CheckBox);
	compress_rotations->set_text(TTR("Quantize Rotations"));
	right_vb->add_child(compress_rotations);
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
	if (splat_instance) {
		if (splat_instance->get_parent()) {
			splat_instance->get_parent()->remove_child(splat_instance);
		}
		splat_instance->queue_free();
		splat_instance = nullptr;
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

	// -- Sampled point cloud --
	{
		const PackedFloat32Array positions = loaded_asset->get_positions();
		const PackedColorArray colors = loaded_asset->get_colors();
		const int count = loaded_asset->get_splat_count();
		if (count <= 0 || positions.size() < 3) {
			return;
		}

		const int max_pts = 2048;
		const int step = MAX(1, count / max_pts);
		const int instance_count = MAX(1, (count + step - 1) / step);

		Ref<BoxMesh> box;
		box.instantiate();
		box->set_size(Vector3(1, 1, 1));

		Ref<StandardMaterial3D> pt_mat;
		pt_mat.instantiate();
		pt_mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
		pt_mat->set_flag(BaseMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
		pt_mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
		pt_mat->set_albedo(Color(0.9, 0.95, 1.0, 0.8));
		box->surface_set_material(0, pt_mat);

		Ref<MultiMesh> mm;
		mm.instantiate();
		mm->set_transform_format(MultiMesh::TRANSFORM_3D);
		mm->set_mesh(box);
		mm->set_use_colors(true);
		mm->set_instance_count(instance_count);

		const float *pos_ptr = positions.ptr();
		const Color *col_ptr = colors.size() > 0 ? colors.ptr() : nullptr;
		const float span = MAX(0.001f, asset_bounds.size.length());
		const float pt_size = CLAMP(span * 0.005f, 0.005f, 0.1f);

		int dst = 0;
		for (int src = 0; src < count && dst < instance_count; src += step, dst++) {
			int base = src * 3;
			if (base + 2 >= positions.size()) {
				break;
			}
			Vector3 pos(pos_ptr[base], pos_ptr[base + 1], pos_ptr[base + 2]);
			pos -= center;

			Transform3D xf;
			xf.basis = Basis().scaled(Vector3(pt_size, pt_size, pt_size));
			xf.origin = pos;
			mm->set_instance_transform(dst, xf);

			Color c(0.8, 0.9, 1.0, 0.85);
			if (col_ptr && src < colors.size()) {
				c = col_ptr[src];
			}
			mm->set_instance_color(dst, c);
		}

		splat_instance = memnew(MultiMeshInstance3D);
		splat_instance->set_multimesh(mm);
		viewport->add_child(splat_instance);
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

	Vector3 center = asset_bounds.get_center();
	// Place scene content at origin; the bounds wireframe and point cloud are
	// already centred, so the camera orbits the origin.
	Transform3D xf;
	xf.basis = Basis(Vector3(0, 1, 0), cam_rot_y) * Basis(Vector3(1, 0, 0), cam_rot_x);
	xf.origin = Vector3(); // content is centred at origin
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

	// Load the imported asset (not the raw source file).
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

	// Trackpad magnify gesture (pinch to zoom).
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

	// Determine importer name from extension.
	String ext = source_path.get_extension().to_lower();
	String importer_name;
	if (ext == "ply") {
		importer_name = "gaussian_splat_ply";
	} else if (ext == "spz") {
		importer_name = "gaussian_splat_spz";
	} else {
		importer_name = "";
	}

	// Gather settings from UI.
	HashMap<StringName, Variant> params;

	int preset_idx = quality_selector->get_selected();
	const GaussianImportPresetDefinition &preset = gaussian_get_import_preset_by_index(preset_idx);
	params[StringName("quality/preset")] = preset.id;
	params[StringName("quality/max_splats")] = preset.max_splats;
	params[StringName("quality/density_multiplier")] = preset.density_multiplier;
	params[StringName("quality/enable_lod")] = preset.enable_lod;
	params[StringName("quality/optimize_for_gpu")] = preset.optimize_for_gpu;

	params[StringName("compression/quantize_positions")] = compress_positions->is_pressed();
	params[StringName("compression/quantize_colors")] = compress_colors->is_pressed();
	params[StringName("compression/quantize_scales")] = compress_scales->is_pressed();
	params[StringName("compression/quantize_rotations")] = compress_rotations->is_pressed();

	// Carry over existing options from the .import file that we don't expose here.
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

	// Populate UI from existing import options / preset.
	{
		String preset_id = "high";
		if (import_options.has(StringName("quality/preset"))) {
			preset_id = String(import_options[StringName("quality/preset")]);
		}
		int idx = gaussian_find_import_preset_index(preset_id);
		quality_selector->select(idx >= 0 ? idx : gaussian_find_import_preset_index("high"));

		auto get_bool = [&](const StringName &key, bool def) -> bool {
			if (import_options.has(key)) {
				return bool(import_options[key]);
			}
			return def;
		};

		compress_positions->set_pressed(get_bool(StringName("compression/quantize_positions"), false));
		compress_colors->set_pressed(get_bool(StringName("compression/quantize_colors"), false));
		compress_scales->set_pressed(get_bool(StringName("compression/quantize_scales"), false));
		compress_rotations->set_pressed(get_bool(StringName("compression/quantize_rotations"), false));
	}

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
