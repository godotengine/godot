/**************************************************************************/
/*  editor_node.cpp                                                       */
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

#include "editor_node.h"

#include "core/config/project_settings.h"
#include "core/extension/gdextension_manager.h"
#include "core/input/input.h"
#include "core/io/config_file.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/object/class_db.h"
#include "core/object/message_queue.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/os/time.h"
#include "core/string/print_string.h"
#include "core/string/translation.h"
#include "core/version.h"
#include "editor/editor_string_names.h"
#include "main/main.h"
#include "scene/3d/bone_attachment_3d.h"
#include "scene/gui/color_picker.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/link_button.h"
#include "scene/gui/menu_bar.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/panel.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/popup.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tab_container.h"
#include "scene/main/window.h"
#include "scene/property_utils.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/packed_scene.h"
#include "scene/resources/portable_compressed_texture.h"
#include "scene/theme/theme_db.h"
#include "servers/display_server.h"
#include "servers/navigation_server_3d.h"
#include "servers/physics_server_2d.h"
#include "servers/rendering_server.h"

#include "editor/audio_stream_preview.h"
#include "editor/debugger/editor_debugger_node.h"
#include "editor/debugger/script_editor_debugger.h"
#include "editor/dependency_editor.h"
#include "editor/editor_about.h"
#include "editor/editor_audio_buses.h"
#include "editor/editor_build_profile.h"
#include "editor/editor_command_palette.h"
#include "editor/editor_data.h"
#include "editor/editor_feature_profile.h"
#include "editor/editor_folding.h"
#include "editor/editor_help.h"
#include "editor/editor_inspector.h"
#include "editor/editor_interface.h"
#include "editor/editor_layouts_dialog.h"
#include "editor/editor_log.h"
#include "editor/editor_native_shader_source_visualizer.h"
#include "editor/editor_paths.h"
#include "editor/editor_plugin.h"
#include "editor/editor_properties.h"
#include "editor/editor_property_name_processor.h"
#include "editor/editor_quick_open.h"
#include "editor/editor_resource_preview.h"
#include "editor/editor_run.h"
#include "editor/editor_run_native.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/editor_settings_dialog.h"
#include "editor/editor_themes.h"
#include "editor/editor_translation_parser.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/export/editor_export.h"
#include "editor/export/export_template_manager.h"
#include "editor/export/project_export.h"
#include "editor/fbx_importer_manager.h"
#include "editor/filesystem_dock.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/gui/editor_run_bar.h"
#include "editor/gui/editor_scene_tabs.h"
#include "editor/gui/editor_title_bar.h"
#include "editor/gui/editor_toaster.h"
#include "editor/history_dock.h"
#include "editor/import/audio_stream_import_settings.h"
#include "editor/import/dynamic_font_import_settings.h"
#include "editor/import/editor_import_collada.h"
#include "editor/import/resource_importer_bitmask.h"
#include "editor/import/resource_importer_bmfont.h"
#include "editor/import/resource_importer_csv_translation.h"
#include "editor/import/resource_importer_dynamic_font.h"
#include "editor/import/resource_importer_image.h"
#include "editor/import/resource_importer_imagefont.h"
#include "editor/import/resource_importer_layered_texture.h"
#include "editor/import/resource_importer_obj.h"
#include "editor/import/resource_importer_shader_file.h"
#include "editor/import/resource_importer_texture.h"
#include "editor/import/resource_importer_texture_atlas.h"
#include "editor/import/resource_importer_wav.h"
#include "editor/import/scene_import_settings.h"
#include "editor/import_dock.h"
#include "editor/inspector_dock.h"
#include "editor/multi_node_edit.h"
#include "editor/node_dock.h"
#include "editor/plugin_config_dialog.h"
#include "editor/plugins/animation_player_editor_plugin.h"
#include "editor/plugins/asset_library_editor_plugin.h"
#include "editor/plugins/canvas_item_editor_plugin.h"
#include "editor/plugins/debugger_editor_plugin.h"
#include "editor/plugins/dedicated_server_export_plugin.h"
#include "editor/plugins/editor_preview_plugins.h"
#include "editor/plugins/editor_resource_conversion_plugin.h"
#include "editor/plugins/gdextension_export_plugin.h"
#include "editor/plugins/material_editor_plugin.h"
#include "editor/plugins/mesh_library_editor_plugin.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "editor/plugins/packed_scene_translation_parser_plugin.h"
#include "editor/plugins/root_motion_editor_plugin.h"
#include "editor/plugins/script_text_editor.h"
#include "editor/plugins/text_editor.h"
#include "editor/plugins/version_control_editor_plugin.h"
#include "editor/plugins/visual_shader_editor_plugin.h"
#include "editor/progress_dialog.h"
#include "editor/project_settings_editor.h"
#include "editor/register_exporters.h"
#include "editor/scene_tree_dock.h"
#include "editor/surface_upgrade_tool.h"
#include "editor/window_wrapper.h"

#include <stdio.h>
#include <stdlib.h>

EditorNode *EditorNode::singleton = nullptr;

// The metadata key used to store and retrieve the version text to copy to the clipboard.
static const String META_TEXT_TO_COPY = "text_to_copy";

static const String EDITOR_NODE_CONFIG_SECTION = "EditorNode";

void EditorNode::disambiguate_filenames(const Vector<String> p_full_paths, Vector<String> &r_filenames) {
	ERR_FAIL_COND_MSG(p_full_paths.size() != r_filenames.size(), vformat("disambiguate_filenames requires two string vectors of same length (%d != %d).", p_full_paths.size(), r_filenames.size()));

	// Keep track of a list of "index sets," i.e. sets of indices
	// within disambiguated_scene_names which contain the same name.
	Vector<RBSet<int>> index_sets;
	HashMap<String, int> scene_name_to_set_index;
	for (int i = 0; i < r_filenames.size(); i++) {
		String scene_name = r_filenames[i];
		if (!scene_name_to_set_index.has(scene_name)) {
			index_sets.append(RBSet<int>());
			scene_name_to_set_index.insert(r_filenames[i], index_sets.size() - 1);
		}
		index_sets.write[scene_name_to_set_index[scene_name]].insert(i);
	}

	// For each index set with a size > 1, we need to disambiguate.
	for (int i = 0; i < index_sets.size(); i++) {
		RBSet<int> iset = index_sets[i];
		while (iset.size() > 1) {
			// Append the parent folder to each scene name.
			for (const int &E : iset) {
				int set_idx = E;
				String scene_name = r_filenames[set_idx];
				String full_path = p_full_paths[set_idx];

				// Get rid of file extensions and res:// prefixes.
				scene_name = scene_name.get_basename();
				if (full_path.begins_with("res://")) {
					full_path = full_path.substr(6);
				}
				full_path = full_path.get_basename();

				// Normalize trailing slashes when normalizing directory names.
				scene_name = scene_name.trim_suffix("/");
				full_path = full_path.trim_suffix("/");

				int scene_name_size = scene_name.size();
				int full_path_size = full_path.size();
				int difference = full_path_size - scene_name_size;

				// Find just the parent folder of the current path and append it.
				// If the current name is foo.tscn, and the full path is /some/folder/foo.tscn
				// then slash_idx is the second '/', so that we select just "folder", and
				// append that to yield "folder/foo.tscn".
				if (difference > 0) {
					String parent = full_path.substr(0, difference);
					int slash_idx = parent.rfind("/");
					slash_idx = parent.rfind("/", slash_idx - 1);
					parent = (slash_idx >= 0 && parent.length() > 1) ? parent.substr(slash_idx + 1) : parent;
					r_filenames.write[set_idx] = parent + r_filenames[set_idx];
				}
			}

			// Loop back through scene names and remove non-ambiguous names.
			bool can_proceed = false;
			RBSet<int>::Element *E = iset.front();
			while (E) {
				String scene_name = r_filenames[E->get()];
				bool duplicate_found = false;
				for (const int &F : iset) {
					if (E->get() == F) {
						continue;
					}
					String other_scene_name = r_filenames[F];
					if (other_scene_name == scene_name) {
						duplicate_found = true;
						break;
					}
				}

				RBSet<int>::Element *to_erase = duplicate_found ? nullptr : E;

				// We need to check that we could actually append anymore names
				// if we wanted to for disambiguation. If we can't, then we have
				// to abort even with ambiguous names. We clean the full path
				// and the scene name first to remove extensions so that this
				// comparison actually works.
				String path = p_full_paths[E->get()];

				// Get rid of file extensions and res:// prefixes.
				scene_name = scene_name.get_basename();
				if (path.begins_with("res://")) {
					path = path.substr(6);
				}
				path = path.get_basename();

				// Normalize trailing slashes when normalizing directory names.
				scene_name = scene_name.trim_suffix("/");
				path = path.trim_suffix("/");

				// We can proceed if the full path is longer than the scene name,
				// meaning that there is at least one more parent folder we can
				// tack onto the name.
				can_proceed = can_proceed || (path.size() - scene_name.size()) >= 1;

				E = E->next();
				if (to_erase) {
					iset.erase(to_erase);
				}
			}

			if (!can_proceed) {
				break;
			}
		}
	}
}

void EditorNode::_version_control_menu_option(int p_idx) {
	switch (vcs_actions_menu->get_item_id(p_idx)) {
		case RUN_VCS_METADATA: {
			VersionControlEditorPlugin::get_singleton()->popup_vcs_metadata_dialog();
		} break;
		case RUN_VCS_SETTINGS: {
			VersionControlEditorPlugin::get_singleton()->popup_vcs_set_up_dialog(gui_base);
		} break;
	}
}

void EditorNode::_update_title() {
	const String appname = GLOBAL_GET("application/config/name");
	String title = (appname.is_empty() ? TTR("Unnamed Project") : appname);
	const String edited = editor_data.get_edited_scene_root() ? editor_data.get_edited_scene_root()->get_scene_file_path() : String();
	if (!edited.is_empty()) {
		// Display the edited scene name before the program name so that it can be seen in the OS task bar.
		title = vformat("%s - %s", edited.get_file(), title);
	}
	if (unsaved_cache) {
		// Display the "modified" mark before anything else so that it can always be seen in the OS task bar.
		title = vformat("(*) %s", title);
	}
	DisplayServer::get_singleton()->window_set_title(title + String(" - ") + VERSION_NAME);
	if (project_title) {
		project_title->set_text(title);
	}
}

void EditorNode::shortcut_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventKey> k = p_event;
	if ((k.is_valid() && k->is_pressed() && !k->is_echo()) || Object::cast_to<InputEventShortcut>(*p_event)) {
		EditorPlugin *old_editor = editor_plugin_screen;

		if (ED_IS_SHORTCUT("editor/filter_files", p_event)) {
			FileSystemDock::get_singleton()->focus_on_filter();
			get_tree()->get_root()->set_input_as_handled();
		}

		if (ED_IS_SHORTCUT("editor/editor_2d", p_event)) {
			editor_select(EDITOR_2D);
		} else if (ED_IS_SHORTCUT("editor/editor_3d", p_event)) {
			editor_select(EDITOR_3D);
		} else if (ED_IS_SHORTCUT("editor/editor_script", p_event)) {
			editor_select(EDITOR_SCRIPT);
		} else if (ED_IS_SHORTCUT("editor/editor_help", p_event)) {
			emit_signal(SNAME("request_help_search"), "");
		} else if (ED_IS_SHORTCUT("editor/editor_assetlib", p_event) && AssetLibraryEditorPlugin::is_available()) {
			editor_select(EDITOR_ASSETLIB);
		} else if (ED_IS_SHORTCUT("editor/editor_next", p_event)) {
			_editor_select_next();
		} else if (ED_IS_SHORTCUT("editor/editor_prev", p_event)) {
			_editor_select_prev();
		} else if (ED_IS_SHORTCUT("editor/command_palette", p_event)) {
			_open_command_palette();
		} else {
		}

		if (old_editor != editor_plugin_screen) {
			get_tree()->get_root()->set_input_as_handled();
		}
	}
}

void EditorNode::_update_from_settings() {
	_update_title();

	int current_filter = GLOBAL_GET("rendering/textures/canvas_textures/default_texture_filter");
	if (current_filter != scene_root->get_default_canvas_item_texture_filter()) {
		Viewport::DefaultCanvasItemTextureFilter tf = (Viewport::DefaultCanvasItemTextureFilter)current_filter;
		scene_root->set_default_canvas_item_texture_filter(tf);
	}
	int current_repeat = GLOBAL_GET("rendering/textures/canvas_textures/default_texture_repeat");
	if (current_repeat != scene_root->get_default_canvas_item_texture_repeat()) {
		Viewport::DefaultCanvasItemTextureRepeat tr = (Viewport::DefaultCanvasItemTextureRepeat)current_repeat;
		scene_root->set_default_canvas_item_texture_repeat(tr);
	}

	RS::DOFBokehShape dof_shape = RS::DOFBokehShape(int(GLOBAL_GET("rendering/camera/depth_of_field/depth_of_field_bokeh_shape")));
	RS::get_singleton()->camera_attributes_set_dof_blur_bokeh_shape(dof_shape);
	RS::DOFBlurQuality dof_quality = RS::DOFBlurQuality(int(GLOBAL_GET("rendering/camera/depth_of_field/depth_of_field_bokeh_quality")));
	bool dof_jitter = GLOBAL_GET("rendering/camera/depth_of_field/depth_of_field_use_jitter");
	RS::get_singleton()->camera_attributes_set_dof_blur_quality(dof_quality, dof_jitter);
	RS::get_singleton()->environment_set_ssao_quality(RS::EnvironmentSSAOQuality(int(GLOBAL_GET("rendering/environment/ssao/quality"))), GLOBAL_GET("rendering/environment/ssao/half_size"), GLOBAL_GET("rendering/environment/ssao/adaptive_target"), GLOBAL_GET("rendering/environment/ssao/blur_passes"), GLOBAL_GET("rendering/environment/ssao/fadeout_from"), GLOBAL_GET("rendering/environment/ssao/fadeout_to"));
	RS::get_singleton()->screen_space_roughness_limiter_set_active(GLOBAL_GET("rendering/anti_aliasing/screen_space_roughness_limiter/enabled"), GLOBAL_GET("rendering/anti_aliasing/screen_space_roughness_limiter/amount"), GLOBAL_GET("rendering/anti_aliasing/screen_space_roughness_limiter/limit"));
	bool glow_bicubic = int(GLOBAL_GET("rendering/environment/glow/upscale_mode")) > 0;
	RS::get_singleton()->environment_set_ssil_quality(RS::EnvironmentSSILQuality(int(GLOBAL_GET("rendering/environment/ssil/quality"))), GLOBAL_GET("rendering/environment/ssil/half_size"), GLOBAL_GET("rendering/environment/ssil/adaptive_target"), GLOBAL_GET("rendering/environment/ssil/blur_passes"), GLOBAL_GET("rendering/environment/ssil/fadeout_from"), GLOBAL_GET("rendering/environment/ssil/fadeout_to"));
	RS::get_singleton()->environment_glow_set_use_bicubic_upscale(glow_bicubic);
	RS::EnvironmentSSRRoughnessQuality ssr_roughness_quality = RS::EnvironmentSSRRoughnessQuality(int(GLOBAL_GET("rendering/environment/screen_space_reflection/roughness_quality")));
	RS::get_singleton()->environment_set_ssr_roughness_quality(ssr_roughness_quality);
	RS::SubSurfaceScatteringQuality sss_quality = RS::SubSurfaceScatteringQuality(int(GLOBAL_GET("rendering/environment/subsurface_scattering/subsurface_scattering_quality")));
	RS::get_singleton()->sub_surface_scattering_set_quality(sss_quality);
	float sss_scale = GLOBAL_GET("rendering/environment/subsurface_scattering/subsurface_scattering_scale");
	float sss_depth_scale = GLOBAL_GET("rendering/environment/subsurface_scattering/subsurface_scattering_depth_scale");
	RS::get_singleton()->sub_surface_scattering_set_scale(sss_scale, sss_depth_scale);

	uint32_t directional_shadow_size = GLOBAL_GET("rendering/lights_and_shadows/directional_shadow/size");
	uint32_t directional_shadow_16_bits = GLOBAL_GET("rendering/lights_and_shadows/directional_shadow/16_bits");
	RS::get_singleton()->directional_shadow_atlas_set_size(directional_shadow_size, directional_shadow_16_bits);

	RS::ShadowQuality shadows_quality = RS::ShadowQuality(int(GLOBAL_GET("rendering/lights_and_shadows/positional_shadow/soft_shadow_filter_quality")));
	RS::get_singleton()->positional_soft_shadow_filter_set_quality(shadows_quality);
	RS::ShadowQuality directional_shadow_quality = RS::ShadowQuality(int(GLOBAL_GET("rendering/lights_and_shadows/directional_shadow/soft_shadow_filter_quality")));
	RS::get_singleton()->directional_soft_shadow_filter_set_quality(directional_shadow_quality);
	float probe_update_speed = GLOBAL_GET("rendering/lightmapping/probe_capture/update_speed");
	RS::get_singleton()->lightmap_set_probe_capture_update_speed(probe_update_speed);
	RS::EnvironmentSDFGIFramesToConverge frames_to_converge = RS::EnvironmentSDFGIFramesToConverge(int(GLOBAL_GET("rendering/global_illumination/sdfgi/frames_to_converge")));
	RS::get_singleton()->environment_set_sdfgi_frames_to_converge(frames_to_converge);
	RS::EnvironmentSDFGIRayCount ray_count = RS::EnvironmentSDFGIRayCount(int(GLOBAL_GET("rendering/global_illumination/sdfgi/probe_ray_count")));
	RS::get_singleton()->environment_set_sdfgi_ray_count(ray_count);
	RS::VoxelGIQuality voxel_gi_quality = RS::VoxelGIQuality(int(GLOBAL_GET("rendering/global_illumination/voxel_gi/quality")));
	RS::get_singleton()->voxel_gi_set_quality(voxel_gi_quality);
	RS::get_singleton()->environment_set_volumetric_fog_volume_size(GLOBAL_GET("rendering/environment/volumetric_fog/volume_size"), GLOBAL_GET("rendering/environment/volumetric_fog/volume_depth"));
	RS::get_singleton()->environment_set_volumetric_fog_filter_active(bool(GLOBAL_GET("rendering/environment/volumetric_fog/use_filter")));
	RS::get_singleton()->canvas_set_shadow_texture_size(GLOBAL_GET("rendering/2d/shadow_atlas/size"));

	bool use_half_res_gi = GLOBAL_GET("rendering/global_illumination/gi/use_half_resolution");
	RS::get_singleton()->gi_set_use_half_resolution(use_half_res_gi);

	bool snap_2d_transforms = GLOBAL_GET("rendering/2d/snap/snap_2d_transforms_to_pixel");
	scene_root->set_snap_2d_transforms_to_pixel(snap_2d_transforms);
	bool snap_2d_vertices = GLOBAL_GET("rendering/2d/snap/snap_2d_vertices_to_pixel");
	scene_root->set_snap_2d_vertices_to_pixel(snap_2d_vertices);

	Viewport::SDFOversize sdf_oversize = Viewport::SDFOversize(int(GLOBAL_GET("rendering/2d/sdf/oversize")));
	scene_root->set_sdf_oversize(sdf_oversize);
	Viewport::SDFScale sdf_scale = Viewport::SDFScale(int(GLOBAL_GET("rendering/2d/sdf/scale")));
	scene_root->set_sdf_scale(sdf_scale);

	Viewport::MSAA msaa = Viewport::MSAA(int(GLOBAL_GET("rendering/anti_aliasing/quality/msaa_2d")));
	scene_root->set_msaa_2d(msaa);

	bool use_hdr_2d = GLOBAL_GET("rendering/viewport/hdr_2d");
	scene_root->set_use_hdr_2d(use_hdr_2d);

	float mesh_lod_threshold = GLOBAL_GET("rendering/mesh_lod/lod_change/threshold_pixels");
	scene_root->set_mesh_lod_threshold(mesh_lod_threshold);

	RS::get_singleton()->decals_set_filter(RS::DecalFilter(int(GLOBAL_GET("rendering/textures/decals/filter"))));
	RS::get_singleton()->light_projectors_set_filter(RS::LightProjectorFilter(int(GLOBAL_GET("rendering/textures/light_projectors/filter"))));

	SceneTree *tree = get_tree();
	tree->set_debug_collisions_color(GLOBAL_GET("debug/shapes/collision/shape_color"));
	tree->set_debug_collision_contact_color(GLOBAL_GET("debug/shapes/collision/contact_color"));

	ResourceImporterTexture::get_singleton()->update_imports();

#ifdef DEBUG_ENABLED
	NavigationServer3D::get_singleton()->set_debug_navigation_edge_connection_color(GLOBAL_GET("debug/shapes/navigation/edge_connection_color"));
	NavigationServer3D::get_singleton()->set_debug_navigation_geometry_edge_color(GLOBAL_GET("debug/shapes/navigation/geometry_edge_color"));
	NavigationServer3D::get_singleton()->set_debug_navigation_geometry_face_color(GLOBAL_GET("debug/shapes/navigation/geometry_face_color"));
	NavigationServer3D::get_singleton()->set_debug_navigation_geometry_edge_disabled_color(GLOBAL_GET("debug/shapes/navigation/geometry_edge_disabled_color"));
	NavigationServer3D::get_singleton()->set_debug_navigation_geometry_face_disabled_color(GLOBAL_GET("debug/shapes/navigation/geometry_face_disabled_color"));
	NavigationServer3D::get_singleton()->set_debug_navigation_enable_edge_connections(GLOBAL_GET("debug/shapes/navigation/enable_edge_connections"));
	NavigationServer3D::get_singleton()->set_debug_navigation_enable_edge_connections_xray(GLOBAL_GET("debug/shapes/navigation/enable_edge_connections_xray"));
	NavigationServer3D::get_singleton()->set_debug_navigation_enable_edge_lines(GLOBAL_GET("debug/shapes/navigation/enable_edge_lines"));
	NavigationServer3D::get_singleton()->set_debug_navigation_enable_edge_lines_xray(GLOBAL_GET("debug/shapes/navigation/enable_edge_lines_xray"));
	NavigationServer3D::get_singleton()->set_debug_navigation_enable_geometry_face_random_color(GLOBAL_GET("debug/shapes/navigation/enable_geometry_face_random_color"));
#endif // DEBUG_ENABLED
}

void EditorNode::_gdextensions_reloaded() {
	// In case the developer is inspecting an object that will be changed by the reload.
	InspectorDock::get_inspector_singleton()->update_tree();
}

void EditorNode::_select_default_main_screen_plugin() {
	if (EDITOR_3D < main_editor_buttons.size() && main_editor_buttons[EDITOR_3D]->is_visible()) {
		// If the 3D editor is enabled, use this as the default.
		editor_select(EDITOR_3D);
		return;
	}

	// Switch to the first main screen plugin that is enabled. Usually this is
	// 2D, but may be subsequent ones if 2D is disabled in the feature profile.
	for (int i = 0; i < main_editor_buttons.size(); i++) {
		Button *editor_button = main_editor_buttons[i];
		if (editor_button->is_visible()) {
			editor_select(i);
			return;
		}
	}

	editor_select(-1);
}

void EditorNode::_update_theme(bool p_skip_creation) {
	if (!p_skip_creation) {
		theme = create_custom_theme(theme);
		DisplayServer::set_early_window_clear_color_override(true, theme->get_color(SNAME("background"), EditorStringName(Editor)));
	}

	List<Ref<Theme>> editor_themes;
	editor_themes.push_back(theme);
	editor_themes.push_back(ThemeDB::get_singleton()->get_default_theme());

	ThemeContext *node_tc = ThemeDB::get_singleton()->get_theme_context(this);
	if (node_tc) {
		node_tc->set_themes(editor_themes);
	} else {
		ThemeDB::get_singleton()->create_theme_context(this, editor_themes);
	}

	Window *window = get_window();
	if (window) {
		ThemeContext *window_tc = ThemeDB::get_singleton()->get_theme_context(window);
		if (window_tc) {
			window_tc->set_themes(editor_themes);
		} else {
			ThemeDB::get_singleton()->create_theme_context(window, editor_themes);
		}
	}

	if (CanvasItemEditor::get_singleton()->get_theme_preview() == CanvasItemEditor::THEME_PREVIEW_EDITOR) {
		update_preview_themes(CanvasItemEditor::THEME_PREVIEW_EDITOR);
	}

	gui_base->add_theme_style_override("panel", theme->get_stylebox(SNAME("Background"), EditorStringName(EditorStyles)));
	main_vbox->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT, Control::PRESET_MODE_MINSIZE, theme->get_constant(SNAME("window_border_margin"), EditorStringName(Editor)));
	main_vbox->add_theme_constant_override("separation", theme->get_constant(SNAME("top_bar_separation"), EditorStringName(Editor)));

	scene_root_parent->add_theme_style_override("panel", theme->get_stylebox(SNAME("Content"), EditorStringName(EditorStyles)));
	bottom_panel->add_theme_style_override("panel", theme->get_stylebox(SNAME("BottomPanel"), EditorStringName(EditorStyles)));
	main_menu->add_theme_style_override("hover", theme->get_stylebox(SNAME("MenuHover"), EditorStringName(EditorStyles)));
	distraction_free->set_icon(theme->get_icon(SNAME("DistractionFree"), EditorStringName(EditorIcons)));
	bottom_panel_raise->set_icon(theme->get_icon(SNAME("ExpandBottomDock"), EditorStringName(EditorIcons)));

	if (gui_base->is_layout_rtl()) {
		dock_tab_move_left->set_icon(theme->get_icon(SNAME("Forward"), EditorStringName(EditorIcons)));
		dock_tab_move_right->set_icon(theme->get_icon(SNAME("Back"), EditorStringName(EditorIcons)));
	} else {
		dock_tab_move_left->set_icon(theme->get_icon(SNAME("Back"), EditorStringName(EditorIcons)));
		dock_tab_move_right->set_icon(theme->get_icon(SNAME("Forward"), EditorStringName(EditorIcons)));
	}

	help_menu->set_item_icon(help_menu->get_item_index(HELP_SEARCH), theme->get_icon(SNAME("HelpSearch"), EditorStringName(EditorIcons)));
	help_menu->set_item_icon(help_menu->get_item_index(HELP_DOCS), theme->get_icon(SNAME("ExternalLink"), EditorStringName(EditorIcons)));
	help_menu->set_item_icon(help_menu->get_item_index(HELP_QA), theme->get_icon(SNAME("ExternalLink"), EditorStringName(EditorIcons)));
	help_menu->set_item_icon(help_menu->get_item_index(HELP_REPORT_A_BUG), theme->get_icon(SNAME("ExternalLink"), EditorStringName(EditorIcons)));
	help_menu->set_item_icon(help_menu->get_item_index(HELP_COPY_SYSTEM_INFO), theme->get_icon(SNAME("ActionCopy"), EditorStringName(EditorIcons)));
	help_menu->set_item_icon(help_menu->get_item_index(HELP_SUGGEST_A_FEATURE), theme->get_icon(SNAME("ExternalLink"), EditorStringName(EditorIcons)));
	help_menu->set_item_icon(help_menu->get_item_index(HELP_SEND_DOCS_FEEDBACK), theme->get_icon(SNAME("ExternalLink"), EditorStringName(EditorIcons)));
	help_menu->set_item_icon(help_menu->get_item_index(HELP_COMMUNITY), theme->get_icon(SNAME("ExternalLink"), EditorStringName(EditorIcons)));
	help_menu->set_item_icon(help_menu->get_item_index(HELP_ABOUT), theme->get_icon(SNAME("Godot"), EditorStringName(EditorIcons)));
	help_menu->set_item_icon(help_menu->get_item_index(HELP_SUPPORT_GODOT_DEVELOPMENT), theme->get_icon(SNAME("Heart"), EditorStringName(EditorIcons)));

	for (int i = 0; i < main_editor_buttons.size(); i++) {
		main_editor_buttons.write[i]->add_theme_font_override("font", theme->get_font(SNAME("main_button_font"), EditorStringName(EditorFonts)));
		main_editor_buttons.write[i]->add_theme_font_size_override("font_size", theme->get_font_size(SNAME("main_button_font_size"), EditorStringName(EditorFonts)));
	}

	if (EditorDebuggerNode::get_singleton()->is_visible()) {
		bottom_panel->add_theme_style_override("panel", theme->get_stylebox(SNAME("BottomPanelDebuggerOverride"), EditorStringName(EditorStyles)));
	}

	for (int i = 0; i < main_editor_buttons.size(); i++) {
		Button *tb = main_editor_buttons[i];
		EditorPlugin *p_editor = editor_table[i];
		Ref<Texture2D> icon = p_editor->get_icon();

		if (icon.is_valid()) {
			tb->set_icon(icon);
		} else if (theme->has_icon(p_editor->get_name(), EditorStringName(EditorIcons))) {
			tb->set_icon(theme->get_icon(p_editor->get_name(), EditorStringName(EditorIcons)));
		}
	}
}

void EditorNode::update_preview_themes(int p_mode) {
	if (!scene_root->is_inside_tree()) {
		return; // Too early.
	}

	List<Ref<Theme>> preview_themes;

	switch (p_mode) {
		case CanvasItemEditor::THEME_PREVIEW_PROJECT:
			preview_themes.push_back(ThemeDB::get_singleton()->get_project_theme());
			break;

		case CanvasItemEditor::THEME_PREVIEW_EDITOR:
			preview_themes.push_back(get_editor_theme());
			break;

		default:
			break;
	}

	preview_themes.push_back(ThemeDB::get_singleton()->get_default_theme());

	ThemeContext *preview_context = ThemeDB::get_singleton()->get_theme_context(scene_root);
	if (preview_context) {
		preview_context->set_themes(preview_themes);
	} else {
		ThemeDB::get_singleton()->create_theme_context(scene_root, preview_themes);
	}
}

void EditorNode::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POSTINITIALIZE: {
			EditorHelp::generate_doc();
		} break;

		case NOTIFICATION_PROCESS: {
			if (opening_prev && !confirmation->is_visible()) {
				opening_prev = false;
			}

			bool global_unsaved = EditorUndoRedoManager::get_singleton()->is_history_unsaved(EditorUndoRedoManager::GLOBAL_HISTORY);
			bool scene_or_global_unsaved = global_unsaved || EditorUndoRedoManager::get_singleton()->is_history_unsaved(editor_data.get_current_edited_scene_history_id());
			if (unsaved_cache != scene_or_global_unsaved) {
				unsaved_cache = scene_or_global_unsaved;
				_update_title();
			}

			if (editor_data.is_scene_changed(-1)) {
				scene_tabs->update_scene_tabs();
			}

			// Update the animation frame of the update spinner.
			uint64_t frame = Engine::get_singleton()->get_frames_drawn();
			uint64_t tick = OS::get_singleton()->get_ticks_msec();

			if (frame != update_spinner_step_frame && (tick - update_spinner_step_msec) > (1000 / 8)) {
				update_spinner_step++;
				if (update_spinner_step >= 8) {
					update_spinner_step = 0;
				}

				update_spinner_step_msec = tick;
				update_spinner_step_frame = frame + 1;

				// Update the icon itself only when the spinner is visible.
				if (EDITOR_GET("interface/editor/show_update_spinner")) {
					update_spinner->set_icon(theme->get_icon("Progress" + itos(update_spinner_step + 1), EditorStringName(EditorIcons)));
				}
			}

			editor_selection->update();

			ResourceImporterTexture::get_singleton()->update_imports();

			bottom_panel_updating = false;

			if (requested_first_scan) {
				requested_first_scan = false;

				OS::get_singleton()->benchmark_begin_measure("editor_scan_and_import");

				if (run_surface_upgrade_tool) {
					run_surface_upgrade_tool = false;
					SurfaceUpgradeTool::get_singleton()->connect("upgrade_finished", callable_mp(EditorFileSystem::get_singleton(), &EditorFileSystem::scan), CONNECT_ONE_SHOT);
					SurfaceUpgradeTool::get_singleton()->finish_upgrade();
				} else {
					EditorFileSystem::get_singleton()->scan();
				}
			}
		} break;

		case NOTIFICATION_ENTER_TREE: {
			get_tree()->set_disable_node_threading(true); // No node threading while running editor.

			Engine::get_singleton()->set_editor_hint(true);

			Window *window = get_window();
			if (window) {
				// Handle macOS fullscreen and extend-to-title changes.
				window->connect("titlebar_changed", callable_mp(this, &EditorNode::_titlebar_resized));
			}

			// Theme has already been created in the constructor, so we can skip that step.
			_update_theme(true);

			OS::get_singleton()->set_low_processor_usage_mode_sleep_usec(int(EDITOR_GET("interface/editor/low_processor_mode_sleep_usec")));
			get_tree()->get_root()->set_as_audio_listener_3d(false);
			get_tree()->get_root()->set_as_audio_listener_2d(false);
			get_tree()->get_root()->set_snap_2d_transforms_to_pixel(false);
			get_tree()->get_root()->set_snap_2d_vertices_to_pixel(false);
			get_tree()->set_auto_accept_quit(false);
#ifdef ANDROID_ENABLED
			get_tree()->set_quit_on_go_back(false);
#endif
			get_tree()->get_root()->connect("files_dropped", callable_mp(this, &EditorNode::_dropped_files));

			command_palette->register_shortcuts_as_command();

			MessageQueue::get_singleton()->push_callable(callable_mp(this, &EditorNode::_begin_first_scan));

			/* DO NOT LOAD SCENES HERE, WAIT FOR FILE SCANNING AND REIMPORT TO COMPLETE */
		} break;

		case NOTIFICATION_EXIT_TREE: {
			if (progress_dialog) {
				progress_dialog->queue_free();
			}
			if (load_error_dialog) {
				load_error_dialog->queue_free();
			}
			if (execute_output_dialog) {
				execute_output_dialog->queue_free();
			}
			if (warning) {
				warning->queue_free();
			}
			if (accept) {
				accept->queue_free();
			}
			if (save_accept) {
				save_accept->queue_free();
			}
			editor_data.save_editor_external_data();
			FileAccess::set_file_close_fail_notify_callback(nullptr);
			log->deinit(); // Do not get messages anymore.
			editor_data.clear_edited_scenes();
		} break;

		case NOTIFICATION_READY: {
			{
				started_timestamp = Time::get_singleton()->get_unix_time_from_system();
				_initializing_plugins = true;
				Vector<String> addons;
				if (ProjectSettings::get_singleton()->has_setting("editor_plugins/enabled")) {
					addons = GLOBAL_GET("editor_plugins/enabled");
				}

				for (int i = 0; i < addons.size(); i++) {
					set_addon_plugin_enabled(addons[i], true);
				}
				_initializing_plugins = false;

				if (!pending_addons.is_empty()) {
					EditorFileSystem::get_singleton()->connect("script_classes_updated", callable_mp(this, &EditorNode::_enable_pending_addons));
				}
			}

			RenderingServer::get_singleton()->viewport_set_disable_2d(get_scene_root()->get_viewport_rid(), true);
			RenderingServer::get_singleton()->viewport_set_environment_mode(get_viewport()->get_viewport_rid(), RenderingServer::VIEWPORT_ENVIRONMENT_DISABLED);

			feature_profile_manager->notify_changed();

			_select_default_main_screen_plugin();

			// Save the project after opening to mark it as last modified, except in headless mode.
			if (DisplayServer::get_singleton()->window_can_draw()) {
				ProjectSettings::get_singleton()->save();
			}

			_titlebar_resized();

			// Set up a theme context for the 2D preview viewport using the stored preview theme.
			CanvasItemEditor::ThemePreviewMode theme_preview_mode = (CanvasItemEditor::ThemePreviewMode)(int)EditorSettings::get_singleton()->get_project_metadata("2d_editor", "theme_preview", CanvasItemEditor::THEME_PREVIEW_PROJECT);
			update_preview_themes(theme_preview_mode);

			/* DO NOT LOAD SCENES HERE, WAIT FOR FILE SCANNING AND REIMPORT TO COMPLETE */
		} break;

		case NOTIFICATION_APPLICATION_FOCUS_IN: {
			// Restore the original FPS cap after focusing back on the editor.
			OS::get_singleton()->set_low_processor_usage_mode_sleep_usec(int(EDITOR_GET("interface/editor/low_processor_mode_sleep_usec")));

			EditorFileSystem::get_singleton()->scan_changes();
			_scan_external_changes();

			GDExtensionManager *gdextension_manager = GDExtensionManager::get_singleton();
			callable_mp(gdextension_manager, &GDExtensionManager::reload_extensions).call_deferred();
		} break;

		case NOTIFICATION_APPLICATION_FOCUS_OUT: {
			// Save on focus loss before applying the FPS limit to avoid slowing down the saving process.
			if (EDITOR_GET("interface/editor/save_on_focus_loss")) {
				_menu_option_confirm(FILE_SAVE_SCENE, false);
			}

			// Set a low FPS cap to decrease CPU/GPU usage while the editor is unfocused.
			OS::get_singleton()->set_low_processor_usage_mode_sleep_usec(int(EDITOR_GET("interface/editor/unfocused_low_processor_mode_sleep_usec")));
		} break;

		case NOTIFICATION_WM_ABOUT: {
			show_about();
		} break;

		case NOTIFICATION_WM_CLOSE_REQUEST: {
			_menu_option_confirm(FILE_QUIT, false);
		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			FileDialog::set_default_show_hidden_files(EDITOR_GET("filesystem/file_dialog/show_hidden_files"));
			EditorFileDialog::set_default_show_hidden_files(EDITOR_GET("filesystem/file_dialog/show_hidden_files"));
			EditorFileDialog::set_default_display_mode((EditorFileDialog::DisplayMode)EDITOR_GET("filesystem/file_dialog/display_mode").operator int());

			bool theme_changed =
					EditorSettings::get_singleton()->check_changed_settings_in_group("interface/theme") ||
					EditorSettings::get_singleton()->check_changed_settings_in_group("interface/editor/font") ||
					EditorSettings::get_singleton()->check_changed_settings_in_group("interface/editor/main_font") ||
					EditorSettings::get_singleton()->check_changed_settings_in_group("interface/editor/code_font") ||
					EditorSettings::get_singleton()->check_changed_settings_in_group("text_editor/theme") ||
					EditorSettings::get_singleton()->check_changed_settings_in_group("text_editor/help/help") ||
					EditorSettings::get_singleton()->check_changed_settings_in_group("filesystem/file_dialog/thumbnail_size") ||
					EditorSettings::get_singleton()->check_changed_settings_in_group("run/output/font_size") ||
					EditorSettings::get_singleton()->check_changed_settings_in_group("interface/touchscreen/increase_scrollbar_touch_area") ||
					EditorSettings::get_singleton()->check_changed_settings_in_group("interface/touchscreen/scale_gizmo_handles");

			if (theme_changed) {
				_update_theme();
			}

			scene_tabs->update_scene_tabs();
			recent_scenes->reset_size();

			_build_icon_type_cache();

			HashSet<String> updated_textfile_extensions;
			bool extensions_match = true;
			const Vector<String> textfile_ext = ((String)(EDITOR_GET("docks/filesystem/textfile_extensions"))).split(",", false);
			for (const String &E : textfile_ext) {
				updated_textfile_extensions.insert(E);
				if (extensions_match && !textfile_extensions.has(E)) {
					extensions_match = false;
				}
			}

			if (!extensions_match || updated_textfile_extensions.size() < textfile_extensions.size()) {
				textfile_extensions = updated_textfile_extensions;
				EditorFileSystem::get_singleton()->scan();
			}

			_update_update_spinner();
		} break;
	}
}

void EditorNode::_update_update_spinner() {
	update_spinner->set_visible(!RenderingServer::get_singleton()->canvas_item_get_debug_redraw() && EDITOR_GET("interface/editor/show_update_spinner"));

	const bool update_continuously = EDITOR_GET("interface/editor/update_continuously");
	PopupMenu *update_popup = update_spinner->get_popup();
	update_popup->set_item_checked(update_popup->get_item_index(SETTINGS_UPDATE_CONTINUOUSLY), update_continuously);
	update_popup->set_item_checked(update_popup->get_item_index(SETTINGS_UPDATE_WHEN_CHANGED), !update_continuously);

	if (update_continuously) {
		update_spinner->set_tooltip_text(TTR("Spins when the editor window redraws.\nUpdate Continuously is enabled, which can increase power usage. Click to disable it."));

		// Use a different color for the update spinner when Update Continuously is enabled,
		// as this feature should only be enabled for troubleshooting purposes.
		// Make the icon modulate color overbright because icons are not completely white on a dark theme.
		// On a light theme, icons are dark, so we need to modulate them with an even brighter color.
		const bool dark_theme = EditorSettings::get_singleton()->is_dark_theme();
		update_spinner->set_self_modulate(theme->get_color(SNAME("error_color"), EditorStringName(Editor)) * (dark_theme ? Color(1.1, 1.1, 1.1) : Color(4.25, 4.25, 4.25)));
	} else {
		update_spinner->set_tooltip_text(TTR("Spins when the editor window redraws."));
		update_spinner->set_self_modulate(Color(1, 1, 1));
	}

	OS::get_singleton()->set_low_processor_usage_mode(!update_continuously);
}

void EditorNode::_on_plugin_ready(Object *p_script, const String &p_activate_name) {
	Ref<Script> scr = Object::cast_to<Script>(p_script);
	if (scr.is_null()) {
		return;
	}
	project_settings_editor->update_plugins();
	project_settings_editor->hide();
	push_item(scr.operator->());
	if (p_activate_name.length()) {
		set_addon_plugin_enabled(p_activate_name, true);
	}
}

void EditorNode::_remove_plugin_from_enabled(const String &p_name) {
	ProjectSettings *ps = ProjectSettings::get_singleton();
	PackedStringArray enabled_plugins = ps->get("editor_plugins/enabled");
	for (int i = 0; i < enabled_plugins.size(); ++i) {
		if (enabled_plugins.get(i) == p_name) {
			enabled_plugins.remove_at(i);
			break;
		}
	}
	ps->set("editor_plugins/enabled", enabled_plugins);
}

void EditorNode::_plugin_over_edit(EditorPlugin *p_plugin, Object *p_object) {
	if (p_object) {
		editor_plugins_over->add_plugin(p_plugin);
		p_plugin->make_visible(true);
		p_plugin->edit(p_object);
	} else {
		editor_plugins_over->remove_plugin(p_plugin);
		p_plugin->make_visible(false);
		p_plugin->edit(nullptr);
	}
}

void EditorNode::_plugin_over_self_own(EditorPlugin *p_plugin) {
	active_plugins[p_plugin->get_instance_id()].insert(p_plugin);
}

void EditorNode::_resources_changed(const Vector<String> &p_resources) {
	List<Ref<Resource>> changed;

	int rc = p_resources.size();
	for (int i = 0; i < rc; i++) {
		Ref<Resource> res = ResourceCache::get_ref(p_resources.get(i));
		if (res.is_null()) {
			continue;
		}

		if (!res->editor_can_reload_from_file()) {
			continue;
		}
		if (!res->get_path().is_resource_file() && !res->get_path().is_absolute_path()) {
			continue;
		}
		if (!FileAccess::exists(res->get_path())) {
			continue;
		}

		if (!res->get_import_path().is_empty()) {
			// This is an imported resource, will be reloaded if reimported via the _resources_reimported() callback.
			continue;
		}

		changed.push_back(res);
	}

	if (changed.size()) {
		for (Ref<Resource> &res : changed) {
			res->reload_from_file();
		}
	}
}

void EditorNode::_fs_changed() {
	for (FileDialog *E : file_dialogs) {
		E->invalidate();
	}

	for (EditorFileDialog *E : editor_file_dialogs) {
		E->invalidate();
	}

	_mark_unsaved_scenes();

	// FIXME: Move this to a cleaner location, it's hacky to do this in _fs_changed.
	String export_error;
	Error err = OK;
	if (!export_defer.preset.is_empty() && !EditorFileSystem::get_singleton()->is_scanning()) {
		String preset_name = export_defer.preset;
		// Ensures export_project does not loop infinitely, because notifications may
		// come during the export.
		export_defer.preset = "";
		Ref<EditorExportPreset> export_preset;
		for (int i = 0; i < EditorExport::get_singleton()->get_export_preset_count(); ++i) {
			export_preset = EditorExport::get_singleton()->get_export_preset(i);
			if (export_preset->get_name() == preset_name) {
				break;
			}
			export_preset.unref();
		}

		if (export_preset.is_null()) {
			Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
			if (da->file_exists("res://export_presets.cfg")) {
				err = FAILED;
				export_error = vformat(
						"Invalid export preset name: %s.\nThe following presets were detected in this project's `export_presets.cfg`:\n\n",
						preset_name);
				for (int i = 0; i < EditorExport::get_singleton()->get_export_preset_count(); ++i) {
					// Write the preset name between double quotes since it needs to be written between quotes on the command line if it contains spaces.
					export_error += vformat("        \"%s\"\n", EditorExport::get_singleton()->get_export_preset(i)->get_name());
				}
			} else {
				err = FAILED;
				export_error = "This project doesn't have an `export_presets.cfg` file at its root.\nCreate an export preset from the \"Project > Export\" dialog and try again.";
			}
		} else {
			Ref<EditorExportPlatform> platform = export_preset->get_platform();
			const String export_path = export_defer.path.is_empty() ? export_preset->get_export_path() : export_defer.path;
			if (export_path.is_empty()) {
				err = FAILED;
				export_error = vformat("Export preset \"%s\" doesn't have a default export path, and none was specified.", preset_name);
			} else if (platform.is_null()) {
				err = FAILED;
				export_error = vformat("Export preset \"%s\" doesn't have a matching platform.", preset_name);
			} else {
				if (export_defer.pack_only) { // Only export .pck or .zip data pack.
					if (export_path.ends_with(".zip")) {
						err = platform->export_zip(export_preset, export_defer.debug, export_path);
					} else if (export_path.ends_with(".pck")) {
						err = platform->export_pack(export_preset, export_defer.debug, export_path);
					}
				} else { // Normal project export.
					String config_error;
					bool missing_templates;
					if (!platform->can_export(export_preset, config_error, missing_templates, export_defer.debug)) {
						ERR_PRINT(vformat("Cannot export project with preset \"%s\" due to configuration errors:\n%s", preset_name, config_error));
						err = missing_templates ? ERR_FILE_NOT_FOUND : ERR_UNCONFIGURED;
					} else {
						platform->clear_messages();
						err = platform->export_project(export_preset, export_defer.debug, export_path);
					}
				}
				if (err != OK) {
					export_error = vformat("Project export for preset \"%s\" failed.", preset_name);
				} else if (platform->get_worst_message_type() >= EditorExportPlatform::EXPORT_MESSAGE_WARNING) {
					export_error = vformat("Project export for preset \"%s\" completed with warnings.", preset_name);
				}
			}
		}

		if (err != OK) {
			ERR_PRINT(export_error);
			_exit_editor(EXIT_FAILURE);
			return;
		}
		if (!export_error.is_empty()) {
			WARN_PRINT(export_error);
		}
		_exit_editor(EXIT_SUCCESS);
	}
}

void EditorNode::_resources_reimported(const Vector<String> &p_resources) {
	List<String> scenes;
	int current_tab = scene_tabs->get_current_tab();

	for (int i = 0; i < p_resources.size(); i++) {
		String file_type = ResourceLoader::get_resource_type(p_resources[i]);
		if (file_type == "PackedScene") {
			scenes.push_back(p_resources[i]);
			// Reload later if needed, first go with normal resources.
			continue;
		}

		if (!ResourceCache::has(p_resources[i])) {
			// Not loaded, no need to reload.
			continue;
		}
		// Reload normally.
		Ref<Resource> resource = ResourceCache::get_ref(p_resources[i]);
		if (resource.is_valid()) {
			resource->reload_from_file();
		}
	}

	// Editor may crash when related animation is playing while re-importing GLTF scene, stop it in advance.
	AnimationPlayer *ap = AnimationPlayerEditor::get_singleton()->get_player();
	if (ap && scenes.size() > 0) {
		ap->stop(true);
	}

	for (const String &E : scenes) {
		reload_scene(E);
		reload_instances_with_path_in_edited_scenes(E);
	}

	scene_tabs->set_current_tab(current_tab);
}

void EditorNode::_sources_changed(bool p_exist) {
	if (waiting_for_first_scan) {
		waiting_for_first_scan = false;

		OS::get_singleton()->benchmark_end_measure("editor_scan_and_import");

		// Reload the global shader variables, but this time
		// loading textures, as they are now properly imported.
		RenderingServer::get_singleton()->global_shader_parameters_load_settings(true);

		_load_editor_layout();

		if (!defer_load_scene.is_empty()) {
			OS::get_singleton()->benchmark_begin_measure("editor_load_scene");
			load_scene(defer_load_scene);
			defer_load_scene = "";
			OS::get_singleton()->benchmark_end_measure("editor_load_scene");

			OS::get_singleton()->benchmark_dump();
		}

		if (SurfaceUpgradeTool::get_singleton()->is_show_requested()) {
			SurfaceUpgradeTool::get_singleton()->show_popup();
		}

		// Start preview thread now that it's safe.
		if (!singleton->cmdline_export_mode) {
			EditorResourcePreview::get_singleton()->start();
		}
	}
}

void EditorNode::_scan_external_changes() {
	disk_changed_list->clear();
	TreeItem *r = disk_changed_list->create_item();
	disk_changed_list->set_hide_root(true);
	bool need_reload = false;

	// Check if any edited scene has changed.

	for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {
		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		if (editor_data.get_scene_path(i) == "" || !da->file_exists(editor_data.get_scene_path(i))) {
			continue;
		}

		uint64_t last_date = editor_data.get_scene_modified_time(i);
		uint64_t date = FileAccess::get_modified_time(editor_data.get_scene_path(i));

		if (date > last_date) {
			TreeItem *ti = disk_changed_list->create_item(r);
			ti->set_text(0, editor_data.get_scene_path(i).get_file());
			need_reload = true;
		}
	}

	String project_settings_path = ProjectSettings::get_singleton()->get_resource_path().path_join("project.godot");
	if (FileAccess::get_modified_time(project_settings_path) > ProjectSettings::get_singleton()->get_last_saved_time()) {
		TreeItem *ti = disk_changed_list->create_item(r);
		ti->set_text(0, "project.godot");
		need_reload = true;
	}

	if (need_reload) {
		disk_changed->call_deferred(SNAME("popup_centered_ratio"), 0.3);
	}
}

void EditorNode::_resave_scenes(String p_str) {
	save_all_scenes();
	ProjectSettings::get_singleton()->save();
	disk_changed->hide();
}

void EditorNode::_reload_modified_scenes() {
	int current_idx = editor_data.get_edited_scene();

	for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {
		if (editor_data.get_scene_path(i) == "") {
			continue;
		}

		uint64_t last_date = editor_data.get_scene_modified_time(i);
		uint64_t date = FileAccess::get_modified_time(editor_data.get_scene_path(i));

		if (date > last_date) {
			String filename = editor_data.get_scene_path(i);
			editor_data.set_edited_scene(i);
			_remove_edited_scene(false);

			Error err = load_scene(filename, false, false, true, false, true);
			if (err != OK) {
				ERR_PRINT(vformat("Failed to load scene: %s", filename));
			}
			editor_data.move_edited_scene_to_index(i);
		}
	}

	_set_current_scene(current_idx);
	scene_tabs->update_scene_tabs();
	disk_changed->hide();
}

void EditorNode::_reload_project_settings() {
	ProjectSettings::get_singleton()->setup(ProjectSettings::get_singleton()->get_resource_path(), String(), true);
}

void EditorNode::_vp_resized() {
}

void EditorNode::_titlebar_resized() {
	DisplayServer::get_singleton()->window_set_window_buttons_offset(Vector2i(title_bar->get_global_position().y + title_bar->get_size().y / 2, title_bar->get_global_position().y + title_bar->get_size().y / 2), DisplayServer::MAIN_WINDOW_ID);
	const Vector3i &margin = DisplayServer::get_singleton()->window_get_safe_title_margins(DisplayServer::MAIN_WINDOW_ID);
	if (left_menu_spacer) {
		int w = (gui_base->is_layout_rtl()) ? margin.y : margin.x;
		left_menu_spacer->set_custom_minimum_size(Size2(w, 0));
	}
	if (right_menu_spacer) {
		int w = (gui_base->is_layout_rtl()) ? margin.x : margin.y;
		right_menu_spacer->set_custom_minimum_size(Size2(w, 0));
	}
	if (title_bar) {
		title_bar->set_custom_minimum_size(Size2(0, margin.z - title_bar->get_global_position().y));
	}
}

void EditorNode::_version_button_pressed() {
	DisplayServer::get_singleton()->clipboard_set(version_btn->get_meta(META_TEXT_TO_COPY));
}

void EditorNode::_update_undo_redo_allowed() {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	file_menu->set_item_disabled(file_menu->get_item_index(EDIT_UNDO), !undo_redo->has_undo());
	file_menu->set_item_disabled(file_menu->get_item_index(EDIT_REDO), !undo_redo->has_redo());
}

void EditorNode::_node_renamed() {
	if (InspectorDock::get_inspector_singleton()) {
		InspectorDock::get_inspector_singleton()->update_tree();
	}
}

void EditorNode::_editor_select_next() {
	int editor = _get_current_main_editor();

	do {
		if (editor == editor_table.size() - 1) {
			editor = 0;
		} else {
			editor++;
		}
	} while (!main_editor_buttons[editor]->is_visible());

	editor_select(editor);
}

void EditorNode::_open_command_palette() {
	command_palette->open_popup();
}

void EditorNode::_editor_select_prev() {
	int editor = _get_current_main_editor();

	do {
		if (editor == 0) {
			editor = editor_table.size() - 1;
		} else {
			editor--;
		}
	} while (!main_editor_buttons[editor]->is_visible());

	editor_select(editor);
}

Error EditorNode::load_resource(const String &p_resource, bool p_ignore_broken_deps) {
	dependency_errors.clear();

	Error err;

	Ref<Resource> res;
	if (ResourceLoader::exists(p_resource, "")) {
		res = ResourceLoader::load(p_resource, "", ResourceFormatLoader::CACHE_MODE_REUSE, &err);
	} else if (textfile_extensions.has(p_resource.get_extension())) {
		res = ScriptEditor::get_singleton()->open_file(p_resource);
	}
	ERR_FAIL_COND_V(!res.is_valid(), ERR_CANT_OPEN);

	if (!p_ignore_broken_deps && dependency_errors.has(p_resource)) {
		Vector<String> errors;
		for (const String &E : dependency_errors[p_resource]) {
			errors.push_back(E);
		}
		dependency_error->show(DependencyErrorDialog::MODE_RESOURCE, p_resource, errors);
		dependency_errors.erase(p_resource);

		return ERR_FILE_MISSING_DEPENDENCIES;
	}

	InspectorDock::get_singleton()->edit_resource(res);
	return OK;
}

void EditorNode::edit_node(Node *p_node) {
	push_item(p_node);
}

void EditorNode::edit_resource(const Ref<Resource> &p_resource) {
	InspectorDock::get_singleton()->edit_resource(p_resource);
}

void EditorNode::save_resource_in_path(const Ref<Resource> &p_resource, const String &p_path) {
	editor_data.apply_changes_in_editors();

	if (saving_resources_in_path.has(p_resource)) {
		return;
	}
	saving_resources_in_path.insert(p_resource);

	int flg = 0;
	if (EDITOR_GET("filesystem/on_save/compress_binary_resources")) {
		flg |= ResourceSaver::FLAG_COMPRESS;
	}

	String path = ProjectSettings::get_singleton()->localize_path(p_path);
	Error err = ResourceSaver::save(p_resource, path, flg | ResourceSaver::FLAG_REPLACE_SUBRESOURCE_PATHS);

	if (err != OK) {
		if (ResourceLoader::is_imported(p_resource->get_path())) {
			show_accept(TTR("Imported resources can't be saved."), TTR("OK"));
		} else {
			show_accept(TTR("Error saving resource!"), TTR("OK"));
		}

		saving_resources_in_path.erase(p_resource);
		return;
	}

	((Resource *)p_resource.ptr())->set_path(path);
	saving_resources_in_path.erase(p_resource);

	_resource_saved(p_resource, path);

	emit_signal(SNAME("resource_saved"), p_resource);
	editor_data.notify_resource_saved(p_resource);
}

void EditorNode::save_resource(const Ref<Resource> &p_resource) {
	// If built-in resource, save the scene instead.
	if (p_resource->is_built_in()) {
		const String scene_path = p_resource->get_path().get_slice("::", 0);
		if (!scene_path.is_empty()) {
			save_scene_if_open(scene_path);
			return;
		}
	}

	// If the resource has been imported, ask the user to use a different path in order to save it.
	String path = p_resource->get_path();
	if (path.is_resource_file() && !FileAccess::exists(path + ".import")) {
		save_resource_in_path(p_resource, p_resource->get_path());
	} else {
		save_resource_as(p_resource);
	}
}

void EditorNode::save_resource_as(const Ref<Resource> &p_resource, const String &p_at_path) {
	{
		String path = p_resource->get_path();
		if (!path.is_resource_file()) {
			int srpos = path.find("::");
			if (srpos != -1) {
				String base = path.substr(0, srpos);
				if (!get_edited_scene() || get_edited_scene()->get_scene_file_path() != base) {
					show_warning(TTR("This resource can't be saved because it does not belong to the edited scene. Make it unique first."));
					return;
				}
			}
		} else if (FileAccess::exists(path + ".import")) {
			show_warning(TTR("This resource can't be saved because it was imported from another file. Make it unique first."));
			return;
		}
	}

	file->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	saving_resource = p_resource;

	current_menu_option = RESOURCE_SAVE_AS;
	List<String> extensions;
	Ref<PackedScene> sd = memnew(PackedScene);
	ResourceSaver::get_recognized_extensions(p_resource, &extensions);
	file->clear_filters();

	List<String> preferred;
	for (const String &E : extensions) {
		if (p_resource->is_class("Script") && (E == "tres" || E == "res")) {
			// This serves no purpose and confused people.
			continue;
		}
		file->add_filter("*." + E, E.to_upper());
		preferred.push_back(E);
	}
	// Lowest priority extension.
	List<String>::Element *res_element = preferred.find("res");
	if (res_element) {
		preferred.move_to_back(res_element);
	}
	// Highest priority extension.
	List<String>::Element *tres_element = preferred.find("tres");
	if (tres_element) {
		preferred.move_to_front(tres_element);
	}

	if (!p_at_path.is_empty()) {
		file->set_current_dir(p_at_path);
		if (p_resource->get_path().is_resource_file()) {
			file->set_current_file(p_resource->get_path().get_file());
		} else {
			if (extensions.size()) {
				String resource_name_snake_case = p_resource->get_class().to_snake_case();
				file->set_current_file("new_" + resource_name_snake_case + "." + preferred.front()->get().to_lower());
			} else {
				file->set_current_file(String());
			}
		}
	} else if (!p_resource->get_path().is_empty()) {
		file->set_current_path(p_resource->get_path());
		if (extensions.size()) {
			String ext = p_resource->get_path().get_extension().to_lower();
			if (extensions.find(ext) == nullptr) {
				file->set_current_path(p_resource->get_path().replacen("." + ext, "." + extensions.front()->get()));
			}
		}
	} else if (preferred.size()) {
		String existing;
		if (extensions.size()) {
			String resource_name_snake_case = p_resource->get_class().to_snake_case();
			existing = "new_" + resource_name_snake_case + "." + preferred.front()->get().to_lower();
		}
		file->set_current_path(existing);
	}
	file->set_title(TTR("Save Resource As..."));
	file->popup_file_dialog();
}

void EditorNode::_menu_option(int p_option) {
	_menu_option_confirm(p_option, false);
}

void EditorNode::_menu_confirm_current() {
	_menu_option_confirm(current_menu_option, true);
}

void EditorNode::trigger_menu_option(int p_option, bool p_confirmed) {
	_menu_option_confirm(p_option, p_confirmed);
}

void EditorNode::_dialog_display_save_error(String p_file, Error p_error) {
	if (p_error) {
		switch (p_error) {
			case ERR_FILE_CANT_WRITE: {
				show_accept(TTR("Can't open file for writing:") + " " + p_file.get_extension(), TTR("OK"));
			} break;
			case ERR_FILE_UNRECOGNIZED: {
				show_accept(TTR("Requested file format unknown:") + " " + p_file.get_extension(), TTR("OK"));
			} break;
			default: {
				show_accept(TTR("Error while saving."), TTR("OK"));
			} break;
		}
	}
}

void EditorNode::_dialog_display_load_error(String p_file, Error p_error) {
	if (p_error) {
		switch (p_error) {
			case ERR_CANT_OPEN: {
				show_accept(vformat(TTR("Can't open file '%s'. The file could have been moved or deleted."), p_file.get_file()), TTR("OK"));
			} break;
			case ERR_PARSE_ERROR: {
				show_accept(vformat(TTR("Error while parsing file '%s'."), p_file.get_file()), TTR("OK"));
			} break;
			case ERR_FILE_CORRUPT: {
				show_accept(vformat(TTR("Scene file '%s' appears to be invalid/corrupt."), p_file.get_file()), TTR("OK"));
			} break;
			case ERR_FILE_NOT_FOUND: {
				show_accept(vformat(TTR("Missing file '%s' or one of its dependencies."), p_file.get_file()), TTR("OK"));
			} break;
			default: {
				show_accept(vformat(TTR("Error while loading file '%s'."), p_file.get_file()), TTR("OK"));
			} break;
		}
	}
}

void EditorNode::_load_editor_plugin_states_from_config(const Ref<ConfigFile> &p_config_file) {
	Node *scene = editor_data.get_edited_scene_root();

	if (!scene) {
		return;
	}

	List<String> esl;
	p_config_file->get_section_keys("editor_states", &esl);

	Dictionary md;
	for (const String &E : esl) {
		Variant st = p_config_file->get_value("editor_states", E);
		if (st.get_type() != Variant::NIL) {
			md[E] = st;
		}
	}

	editor_data.set_editor_plugin_states(md);
}

void EditorNode::_save_editor_states(const String &p_file, int p_idx) {
	Node *scene = editor_data.get_edited_scene_root(p_idx);

	if (!scene) {
		return;
	}

	String path = EditorPaths::get_singleton()->get_project_settings_dir().path_join(p_file.get_file() + "-editstate-" + p_file.md5_text() + ".cfg");

	Ref<ConfigFile> cf;
	cf.instantiate();

	Dictionary md;
	if (p_idx < 0 || editor_data.get_edited_scene() == p_idx) {
		md = editor_data.get_editor_plugin_states();
	} else {
		md = editor_data.get_scene_editor_states(p_idx);
	}

	List<Variant> keys;
	md.get_key_list(&keys);
	for (const Variant &E : keys) {
		cf->set_value("editor_states", E, md[E]);
	}

	// Save the currently selected nodes.

	List<Node *> selection = editor_selection->get_full_selected_node_list();
	TypedArray<NodePath> selection_paths;
	for (Node *selected_node : selection) {
		selection_paths.push_back(selected_node->get_path());
	}
	cf->set_value("editor_states", "selected_nodes", selection_paths);

	Error err = cf->save(path);
	ERR_FAIL_COND_MSG(err != OK, "Cannot save config file to '" + path + "'.");
}

bool EditorNode::_find_and_save_resource(Ref<Resource> p_res, HashMap<Ref<Resource>, bool> &processed, int32_t flags) {
	if (p_res.is_null()) {
		return false;
	}

	if (processed.has(p_res)) {
		return processed[p_res];
	}

	bool changed = p_res->is_edited();
	p_res->set_edited(false);

	bool subchanged = _find_and_save_edited_subresources(p_res.ptr(), processed, flags);

	if (p_res->get_path().is_resource_file()) {
		if (changed || subchanged) {
			ResourceSaver::save(p_res, p_res->get_path(), flags);
		}
		processed[p_res] = false; // Because it's a file.
		return false;
	} else {
		processed[p_res] = changed;
		return changed;
	}
}

bool EditorNode::_find_and_save_edited_subresources(Object *obj, HashMap<Ref<Resource>, bool> &processed, int32_t flags) {
	bool ret_changed = false;
	List<PropertyInfo> pi;
	obj->get_property_list(&pi);
	for (const PropertyInfo &E : pi) {
		if (!(E.usage & PROPERTY_USAGE_STORAGE)) {
			continue;
		}

		switch (E.type) {
			case Variant::OBJECT: {
				Ref<Resource> res = obj->get(E.name);

				if (_find_and_save_resource(res, processed, flags)) {
					ret_changed = true;
				}

			} break;
			case Variant::ARRAY: {
				Array varray = obj->get(E.name);
				int len = varray.size();
				for (int i = 0; i < len; i++) {
					const Variant &v = varray.get(i);
					Ref<Resource> res = v;
					if (_find_and_save_resource(res, processed, flags)) {
						ret_changed = true;
					}
				}

			} break;
			case Variant::DICTIONARY: {
				Dictionary d = obj->get(E.name);
				List<Variant> keys;
				d.get_key_list(&keys);
				for (const Variant &F : keys) {
					Variant v = d[F];
					Ref<Resource> res = v;
					if (_find_and_save_resource(res, processed, flags)) {
						ret_changed = true;
					}
				}
			} break;
			default: {
			}
		}
	}

	return ret_changed;
}

void EditorNode::_save_edited_subresources(Node *scene, HashMap<Ref<Resource>, bool> &processed, int32_t flags) {
	_find_and_save_edited_subresources(scene, processed, flags);

	for (int i = 0; i < scene->get_child_count(); i++) {
		Node *n = scene->get_child(i);
		if (n->get_owner() != editor_data.get_edited_scene_root()) {
			continue;
		}
		_save_edited_subresources(n, processed, flags);
	}
}

void EditorNode::_find_node_types(Node *p_node, int &count_2d, int &count_3d) {
	if (p_node->is_class("Viewport") || (p_node != editor_data.get_edited_scene_root() && p_node->get_owner() != editor_data.get_edited_scene_root())) {
		return;
	}

	if (p_node->is_class("CanvasItem")) {
		count_2d++;
	} else if (p_node->is_class("Node3D")) {
		count_3d++;
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_find_node_types(p_node->get_child(i), count_2d, count_3d);
	}
}

void EditorNode::_save_scene_with_preview(String p_file, int p_idx) {
	EditorProgress save("save", TTR("Saving Scene"), 4);

	if (editor_data.get_edited_scene_root() != nullptr) {
		save.step(TTR("Analyzing"), 0);

		int c2d = 0;
		int c3d = 0;

		_find_node_types(editor_data.get_edited_scene_root(), c2d, c3d);

		save.step(TTR("Creating Thumbnail"), 1);
		// Current view?

		Ref<Image> img;
		// If neither 3D or 2D nodes are present, make a 1x1 black texture.
		// We cannot fallback on the 2D editor, because it may not have been used yet,
		// which would result in an invalid texture.
		if (c3d == 0 && c2d == 0) {
			img.instantiate();
			img->initialize_data(1, 1, false, Image::FORMAT_RGB8);
		} else if (c3d < c2d) {
			Ref<ViewportTexture> viewport_texture = scene_root->get_texture();
			if (viewport_texture->get_width() > 0 && viewport_texture->get_height() > 0) {
				img = viewport_texture->get_image();
			}
		} else {
			// The 3D editor may be disabled as a feature, but scenes can still be opened.
			// This check prevents the preview from regenerating in case those scenes are then saved.
			// The preview will be generated if no feature profile is set (as the 3D editor is enabled by default).
			Ref<EditorFeatureProfile> profile = feature_profile_manager->get_current_profile();
			if (!profile.is_valid() || !profile->is_feature_disabled(EditorFeatureProfile::FEATURE_3D)) {
				img = Node3DEditor::get_singleton()->get_editor_viewport(0)->get_viewport_node()->get_texture()->get_image();
			}
		}

		if (img.is_valid() && img->get_width() > 0 && img->get_height() > 0) {
			img = img->duplicate();

			save.step(TTR("Creating Thumbnail"), 2);
			save.step(TTR("Creating Thumbnail"), 3);

			int preview_size = EDITOR_GET("filesystem/file_dialog/thumbnail_size");
			preview_size *= EDSCALE;

			// Consider a square region.
			int vp_size = MIN(img->get_width(), img->get_height());
			int x = (img->get_width() - vp_size) / 2;
			int y = (img->get_height() - vp_size) / 2;

			if (vp_size < preview_size) {
				// Just square it.
				img->crop_from_point(x, y, vp_size, vp_size);
			} else {
				int ratio = vp_size / preview_size;
				int size = preview_size * MAX(1, ratio / 2);

				x = (img->get_width() - size) / 2;
				y = (img->get_height() - size) / 2;

				img->crop_from_point(x, y, size, size);
				img->resize(preview_size, preview_size, Image::INTERPOLATE_LANCZOS);
			}
			img->convert(Image::FORMAT_RGB8);

			// Save thumbnail directly, as thumbnailer may not update due to actual scene not changing md5.
			String temp_path = EditorPaths::get_singleton()->get_cache_dir();
			String cache_base = ProjectSettings::get_singleton()->globalize_path(p_file).md5_text();
			cache_base = temp_path.path_join("resthumb-" + cache_base);

			// Does not have it, try to load a cached thumbnail.
			post_process_preview(img);
			img->save_png(cache_base + ".png");
		}
	}

	save.step(TTR("Saving Scene"), 4);
	_save_scene(p_file, p_idx);

	if (!singleton->cmdline_export_mode) {
		EditorResourcePreview::get_singleton()->check_for_invalidation(p_file);
	}
}

bool EditorNode::_validate_scene_recursive(const String &p_filename, Node *p_node) {
	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *child = p_node->get_child(i);
		if (child->get_scene_file_path() == p_filename) {
			return true;
		}

		if (_validate_scene_recursive(p_filename, child)) {
			return true;
		}
	}

	return false;
}

int EditorNode::_save_external_resources() {
	// Save external resources and its subresources if any was modified.

	int flg = 0;
	if (EDITOR_GET("filesystem/on_save/compress_binary_resources")) {
		flg |= ResourceSaver::FLAG_COMPRESS;
	}
	flg |= ResourceSaver::FLAG_REPLACE_SUBRESOURCE_PATHS;

	HashSet<String> edited_resources;
	int saved = 0;
	List<Ref<Resource>> cached;
	ResourceCache::get_cached_resources(&cached);

	for (Ref<Resource> res : cached) {
		if (!res->is_edited()) {
			continue;
		}

		String path = res->get_path();
		if (path.begins_with("res://")) {
			int subres_pos = path.find("::");
			if (subres_pos == -1) {
				// Actual resource.
				edited_resources.insert(path);
			} else {
				edited_resources.insert(path.substr(0, subres_pos));
			}
		}

		res->set_edited(false);
	}

	for (const String &E : edited_resources) {
		Ref<Resource> res = ResourceCache::get_ref(E);
		if (!res.is_valid()) {
			continue; // Maybe it was erased in a thread, who knows.
		}
		Ref<PackedScene> ps = res;
		if (ps.is_valid()) {
			continue; // Do not save PackedScenes, this will mess up the editor.
		}
		ResourceSaver::save(res, res->get_path(), flg);
		saved++;
	}

	EditorUndoRedoManager::get_singleton()->set_history_as_saved(EditorUndoRedoManager::GLOBAL_HISTORY);

	return saved;
}

static void _reset_animation_mixers(Node *p_node, List<Pair<AnimationMixer *, Ref<AnimatedValuesBackup>>> *r_anim_backups) {
	for (int i = 0; i < p_node->get_child_count(); i++) {
		AnimationMixer *mixer = Object::cast_to<AnimationMixer>(p_node->get_child(i));
		if (mixer && mixer->is_reset_on_save_enabled() && mixer->can_apply_reset()) {
			Ref<AnimatedValuesBackup> backup = mixer->apply_reset();
			if (backup.is_valid()) {
				Pair<AnimationMixer *, Ref<AnimatedValuesBackup>> pair;
				pair.first = mixer;
				pair.second = backup;
				r_anim_backups->push_back(pair);
			}
		}
		_reset_animation_mixers(p_node->get_child(i), r_anim_backups);
	}
}

void EditorNode::_save_scene(String p_file, int idx) {
	if (!saving_scene.is_empty() && saving_scene == p_file) {
		return;
	}

	Node *scene = editor_data.get_edited_scene_root(idx);

	if (!scene) {
		show_accept(TTR("This operation can't be done without a tree root."), TTR("OK"));
		return;
	}

	if (!scene->get_scene_file_path().is_empty() && _validate_scene_recursive(scene->get_scene_file_path(), scene)) {
		show_accept(TTR("This scene can't be saved because there is a cyclic instance inclusion.\nPlease resolve it and then attempt to save again."), TTR("OK"));
		return;
	}

	scene->propagate_notification(NOTIFICATION_EDITOR_PRE_SAVE);

	editor_data.apply_changes_in_editors();
	List<Pair<AnimationMixer *, Ref<AnimatedValuesBackup>>> anim_backups;
	_reset_animation_mixers(scene, &anim_backups);
	save_default_environment();

	_save_editor_states(p_file, idx);

	Ref<PackedScene> sdata;

	if (ResourceCache::has(p_file)) {
		// Something may be referencing this resource and we are good with that.
		// We must update it, but also let the previous scene state go, as
		// old version still work for referencing changes in instantiated or inherited scenes.

		sdata = ResourceCache::get_ref(p_file);
		if (sdata.is_valid()) {
			sdata->recreate_state();
		} else {
			sdata.instantiate();
		}
	} else {
		sdata.instantiate();
	}
	Error err = sdata->pack(scene);

	if (err != OK) {
		show_accept(TTR("Couldn't save scene. Likely dependencies (instances or inheritance) couldn't be satisfied."), TTR("OK"));
		return;
	}

	int flg = 0;
	if (EDITOR_GET("filesystem/on_save/compress_binary_resources")) {
		flg |= ResourceSaver::FLAG_COMPRESS;
	}
	flg |= ResourceSaver::FLAG_REPLACE_SUBRESOURCE_PATHS;

	err = ResourceSaver::save(sdata, p_file, flg);

	// This needs to be emitted before saving external resources.
	emit_signal(SNAME("scene_saved"), p_file);

	_save_external_resources();
	saving_scene = p_file; // Some editors may save scenes of built-in resources as external data, so avoid saving this scene again.
	editor_data.save_editor_external_data();
	saving_scene = "";

	for (Pair<AnimationMixer *, Ref<AnimatedValuesBackup>> &E : anim_backups) {
		E.first->restore(E.second);
	}

	if (err == OK) {
		scene->set_scene_file_path(ProjectSettings::get_singleton()->localize_path(p_file));
		editor_data.set_scene_as_saved(idx);
		editor_data.set_scene_modified_time(idx, FileAccess::get_modified_time(p_file));

		editor_folding.save_scene_folding(scene, p_file);

		_update_title();
		scene_tabs->update_scene_tabs();
	} else {
		_dialog_display_save_error(p_file, err);
	}

	scene->propagate_notification(NOTIFICATION_EDITOR_POST_SAVE);
}

void EditorNode::save_all_scenes() {
	project_run_bar->stop_playing();
	_save_all_scenes();
}

void EditorNode::save_scene_if_open(const String &p_scene_path) {
	int idx = editor_data.get_edited_scene_from_path(p_scene_path);
	if (idx >= 0) {
		_save_scene(p_scene_path, idx);
	}
}

void EditorNode::save_scene_list(const HashSet<String> &p_scene_paths) {
	for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {
		Node *scene = editor_data.get_edited_scene_root(i);

		if (scene && p_scene_paths.has(scene->get_scene_file_path())) {
			_save_scene(scene->get_scene_file_path(), i);
		}
	}
}

void EditorNode::save_before_run() {
	current_menu_option = FILE_SAVE_AND_RUN;
	_menu_option_confirm(FILE_SAVE_AS_SCENE, true);
	file->set_title(TTR("Save scene before running..."));
}

void EditorNode::try_autosave() {
	if (!bool(EDITOR_GET("run/auto_save/save_before_running"))) {
		return;
	}

	if (unsaved_cache) {
		Node *scene = editor_data.get_edited_scene_root();

		if (scene && !scene->get_scene_file_path().is_empty()) { // Only autosave if there is a scene and if it has a path.
			_save_scene_with_preview(scene->get_scene_file_path());
		}
	}
	_menu_option(FILE_SAVE_ALL_SCENES);
	editor_data.save_editor_external_data();
}

void EditorNode::restart_editor() {
	exiting = true;

	if (project_run_bar->is_playing()) {
		project_run_bar->stop_playing();
	}

	String to_reopen;
	if (get_tree()->get_edited_scene_root()) {
		to_reopen = get_tree()->get_edited_scene_root()->get_scene_file_path();
	}

	_exit_editor(EXIT_SUCCESS);

	List<String> args;

	for (const String &a : Main::get_forwardable_cli_arguments(Main::CLI_SCOPE_TOOL)) {
		args.push_back(a);
	}

	args.push_back("--path");
	args.push_back(ProjectSettings::get_singleton()->get_resource_path());

	args.push_back("-e");

	if (!to_reopen.is_empty()) {
		args.push_back(to_reopen);
	}

	OS::get_singleton()->set_restart_on_exit(true, args);
}

void EditorNode::_save_all_scenes() {
	bool all_saved = true;
	for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {
		Node *scene = editor_data.get_edited_scene_root(i);
		if (scene) {
			if (!scene->get_scene_file_path().is_empty() && DirAccess::exists(scene->get_scene_file_path().get_base_dir())) {
				if (i != editor_data.get_edited_scene()) {
					_save_scene(scene->get_scene_file_path(), i);
				} else {
					_save_scene_with_preview(scene->get_scene_file_path());
				}
			} else if (!scene->get_scene_file_path().is_empty()) {
				all_saved = false;
			}
		}
	}

	if (!all_saved) {
		show_warning(TTR("Could not save one or more scenes!"), TTR("Save All Scenes"));
	}
	save_default_environment();
}

void EditorNode::_mark_unsaved_scenes() {
	for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {
		Node *node = editor_data.get_edited_scene_root(i);
		if (!node) {
			continue;
		}

		String path = node->get_scene_file_path();
		if (!path.is_empty() && !FileAccess::exists(path)) {
			// Mark scene tab as unsaved if the file is gone.
			EditorUndoRedoManager::get_singleton()->set_history_as_unsaved(editor_data.get_scene_history_id(i));
		}
	}

	_update_title();
	scene_tabs->update_scene_tabs();
}

void EditorNode::_dialog_action(String p_file) {
	switch (current_menu_option) {
		case FILE_NEW_INHERITED_SCENE: {
			Node *scene = editor_data.get_edited_scene_root();
			// If the previous scene is rootless, just close it in favor of the new one.
			if (!scene) {
				_menu_option_confirm(FILE_CLOSE, true);
			}

			load_scene(p_file, false, true);
		} break;
		case FILE_OPEN_SCENE: {
			load_scene(p_file);
		} break;
		case SETTINGS_PICK_MAIN_SCENE: {
			ProjectSettings::get_singleton()->set("application/run/main_scene", p_file);
			ProjectSettings::get_singleton()->save();
			// TODO: Would be nice to show the project manager opened with the highlighted field.

			project_run_bar->play_main_scene((bool)pick_main_scene->get_meta("from_native", false));
		} break;
		case FILE_CLOSE:
		case SCENE_TAB_CLOSE:
		case FILE_SAVE_SCENE:
		case FILE_SAVE_AS_SCENE: {
			int scene_idx = (current_menu_option == FILE_SAVE_SCENE || current_menu_option == FILE_SAVE_AS_SCENE) ? -1 : tab_closing_idx;

			if (file->get_file_mode() == EditorFileDialog::FILE_MODE_SAVE_FILE) {
				bool same_open_scene = false;
				for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {
					if (editor_data.get_scene_path(i) == p_file && i != scene_idx) {
						same_open_scene = true;
					}
				}

				if (same_open_scene) {
					show_warning(TTR("Can't overwrite scene that is still open!"));
					return;
				}

				save_default_environment();
				_save_scene_with_preview(p_file, scene_idx);
				_add_to_recent_scenes(p_file);
				save_editor_layout_delayed();

				if (scene_idx != -1) {
					_discard_changes();
				} else {
					// Update the path of the edited scene to ensure later do/undo action history matches.
					editor_data.set_scene_path(editor_data.get_edited_scene(), p_file);
				}
			}

		} break;

		case FILE_SAVE_AND_RUN: {
			if (file->get_file_mode() == EditorFileDialog::FILE_MODE_SAVE_FILE) {
				save_default_environment();
				_save_scene_with_preview(p_file);
				project_run_bar->play_custom_scene(p_file);
			}
		} break;

		case FILE_SAVE_AND_RUN_MAIN_SCENE: {
			ProjectSettings::get_singleton()->set("application/run/main_scene", p_file);
			ProjectSettings::get_singleton()->save();

			if (file->get_file_mode() == EditorFileDialog::FILE_MODE_SAVE_FILE) {
				save_default_environment();
				_save_scene_with_preview(p_file);
				project_run_bar->play_main_scene((bool)pick_main_scene->get_meta("from_native", false));
			}
		} break;

		case FILE_EXPORT_MESH_LIBRARY: {
			Ref<MeshLibrary> ml;
			if (file_export_lib_merge->is_pressed() && FileAccess::exists(p_file)) {
				ml = ResourceLoader::load(p_file, "MeshLibrary");

				if (ml.is_null()) {
					show_accept(TTR("Can't load MeshLibrary for merging!"), TTR("OK"));
					return;
				}
			}

			if (ml.is_null()) {
				ml = Ref<MeshLibrary>(memnew(MeshLibrary));
			}

			MeshLibraryEditor::update_library_file(editor_data.get_edited_scene_root(), ml, true, file_export_lib_apply_xforms->is_pressed());

			Error err = ResourceSaver::save(ml, p_file);
			if (err) {
				show_accept(TTR("Error saving MeshLibrary!"), TTR("OK"));
				return;
			} else if (ResourceCache::has(p_file)) {
				// Make sure MeshLibrary is updated in the editor.
				ResourceLoader::load(p_file)->reload_from_file();
			}

		} break;

		case RESOURCE_SAVE:
		case RESOURCE_SAVE_AS: {
			ERR_FAIL_COND(saving_resource.is_null());
			save_resource_in_path(saving_resource, p_file);
			saving_resource = Ref<Resource>();
			ObjectID current_id = editor_history.get_current();
			Object *current_obj = current_id.is_valid() ? ObjectDB::get_instance(current_id) : nullptr;
			ERR_FAIL_NULL(current_obj);
			current_obj->notify_property_list_changed();
		} break;
		case SETTINGS_LAYOUT_SAVE: {
			if (p_file.is_empty()) {
				return;
			}

			Ref<ConfigFile> config;
			config.instantiate();
			Error err = config->load(EditorSettings::get_singleton()->get_editor_layouts_config());

			if (err == ERR_FILE_CANT_OPEN || err == ERR_FILE_NOT_FOUND) {
				config.instantiate();
			} else if (err != OK) {
				show_warning(TTR("An error occurred while trying to save the editor layout.\nMake sure the editor's user data path is writable."));
				return;
			}

			_save_docks_to_config(config, p_file);

			config->save(EditorSettings::get_singleton()->get_editor_layouts_config());

			layout_dialog->hide();
			_update_layouts_menu();

			if (p_file == "Default") {
				show_warning(TTR("Default editor layout overridden.\nTo restore the Default layout to its base settings, use the Delete Layout option and delete the Default layout."));
			}

		} break;
		case SETTINGS_LAYOUT_DELETE: {
			if (p_file.is_empty()) {
				return;
			}

			Ref<ConfigFile> config;
			config.instantiate();
			Error err = config->load(EditorSettings::get_singleton()->get_editor_layouts_config());

			if (err != OK || !config->has_section(p_file)) {
				show_warning(TTR("Layout name not found!"));
				return;
			}

			// Erase key values.
			List<String> keys;
			config->get_section_keys(p_file, &keys);
			for (const String &key : keys) {
				config->set_value(p_file, key, Variant());
			}

			config->save(EditorSettings::get_singleton()->get_editor_layouts_config());

			layout_dialog->hide();
			_update_layouts_menu();

			if (p_file == "Default") {
				show_warning(TTR("Restored the Default layout to its base settings."));
			}

		} break;
		default: {
			// Save scene?
			if (file->get_file_mode() == EditorFileDialog::FILE_MODE_SAVE_FILE) {
				_save_scene_with_preview(p_file);
			}

		} break;
	}
}

bool EditorNode::_is_class_editor_disabled_by_feature_profile(const StringName &p_class) {
	Ref<EditorFeatureProfile> profile = EditorFeatureProfileManager::get_singleton()->get_current_profile();
	if (profile.is_null()) {
		return false;
	}

	StringName class_name = p_class;

	while (class_name != StringName()) {
		if (profile->is_class_disabled(class_name)) {
			return true;
		}
		if (profile->is_class_editor_disabled(class_name)) {
			return true;
		}
		class_name = ClassDB::get_parent_class(class_name);
	}

	return false;
}

void EditorNode::edit_item(Object *p_object, Object *p_editing_owner) {
	ERR_FAIL_NULL(p_editing_owner);

	// Editing for this type of object may be disabled by user's feature profile.
	if (!p_object || _is_class_editor_disabled_by_feature_profile(p_object->get_class())) {
		// Nothing to edit, clean up the owner context and return.
		hide_unused_editors(p_editing_owner);
		return;
	}

	// Get a list of editor plugins that can handle this type of object.
	Vector<EditorPlugin *> available_plugins = editor_data.get_handling_sub_editors(p_object);
	if (available_plugins.is_empty()) {
		// None, clean up the owner context and return.
		hide_unused_editors(p_editing_owner);
		return;
	}

	ObjectID owner_id = p_editing_owner->get_instance_id();

	// Remove editor plugins no longer used by this editing owner. Keep the ones that can
	// still be reused by the new edited object.

	List<EditorPlugin *> to_remove;
	for (EditorPlugin *plugin : active_plugins[owner_id]) {
		if (!available_plugins.has(plugin)) {
			to_remove.push_back(plugin);
			if (plugin->can_auto_hide()) {
				_plugin_over_edit(plugin, nullptr);
			} else {
				// If plugin can't be hidden, make it own itself and become responsible for closing.
				_plugin_over_self_own(plugin);
			}
		}
	}

	for (EditorPlugin *plugin : to_remove) {
		active_plugins[owner_id].erase(plugin);
	}

	// Send the edited object to the plugins.
	for (EditorPlugin *plugin : available_plugins) {
		if (active_plugins[owner_id].has(plugin)) {
			// Plugin was already active, just change the object.
			plugin->edit(p_object);
			continue;
		}

		if (active_plugins.has(plugin->get_instance_id())) {
			// Plugin is already active, but as self-owning, so it needs a separate check.
			plugin->edit(p_object);
			continue;
		}

		// If plugin is already associated with another owner, remove it from there first.
		for (KeyValue<ObjectID, HashSet<EditorPlugin *>> &kv : active_plugins) {
			if (kv.key != owner_id) {
				EditorPropertyResource *epres = Object::cast_to<EditorPropertyResource>(ObjectDB::get_instance(kv.key));
				if (epres && kv.value.has(plugin)) {
					// If it's resource property editing the same resource type, fold it.
					epres->fold_resource();
				}
				kv.value.erase(plugin);
			}
		}

		// Activate previously inactive plugin and edit the object.
		active_plugins[owner_id].insert(plugin);
		_plugin_over_edit(plugin, p_object);
	}
}

void EditorNode::push_node_item(Node *p_node) {
	if (p_node || Object::cast_to<Node>(InspectorDock::get_inspector_singleton()->get_edited_object()) || Object::cast_to<MultiNodeEdit>(InspectorDock::get_inspector_singleton()->get_edited_object())) {
		// Don't push null if the currently edited object is not a Node.
		push_item(p_node);
	}
}

void EditorNode::push_item(Object *p_object, const String &p_property, bool p_inspector_only) {
	if (!p_object) {
		InspectorDock::get_inspector_singleton()->edit(nullptr);
		NodeDock::get_singleton()->set_node(nullptr);
		SceneTreeDock::get_singleton()->set_selected(nullptr);
		InspectorDock::get_singleton()->update(nullptr);
		hide_unused_editors();
		return;
	}

	ObjectID id = p_object->get_instance_id();
	if (id != editor_history.get_current()) {
		if (p_inspector_only) {
			editor_history.add_object(id, String(), true);
		} else if (p_property.is_empty()) {
			editor_history.add_object(id);
		} else {
			editor_history.add_object(id, p_property);
		}
	}

	_edit_current();
}

void EditorNode::save_default_environment() {
	Ref<Environment> fallback = get_tree()->get_root()->get_world_3d()->get_fallback_environment();

	if (fallback.is_valid() && fallback->get_path().is_resource_file()) {
		HashMap<Ref<Resource>, bool> processed;
		_find_and_save_edited_subresources(fallback.ptr(), processed, 0);
		save_resource_in_path(fallback, fallback->get_path());
	}
}

void EditorNode::hide_unused_editors(const Object *p_editing_owner) {
	if (p_editing_owner) {
		const ObjectID id = p_editing_owner->get_instance_id();
		for (EditorPlugin *plugin : active_plugins[id]) {
			if (plugin->can_auto_hide()) {
				_plugin_over_edit(plugin, nullptr);
			} else {
				_plugin_over_self_own(plugin);
			}
		}
		active_plugins.erase(id);
	} else {
		// If no editing owner is provided, this method will go over all owners and check if they are valid.
		// This is to sweep properties that were removed from the inspector.
		List<ObjectID> to_remove;
		for (KeyValue<ObjectID, HashSet<EditorPlugin *>> &kv : active_plugins) {
			const Object *context = ObjectDB::get_instance(kv.key);
			if (context) {
				// In case of self-owning plugins, they are disabled here if they can auto hide.
				const EditorPlugin *self_owning = Object::cast_to<EditorPlugin>(context);
				if (self_owning && self_owning->can_auto_hide()) {
					context = nullptr;
				}
			}

			if (!context) {
				to_remove.push_back(kv.key);
				for (EditorPlugin *plugin : kv.value) {
					if (plugin->can_auto_hide()) {
						_plugin_over_edit(plugin, nullptr);
					} else {
						_plugin_over_self_own(plugin);
					}
				}
			}
		}

		for (const ObjectID &id : to_remove) {
			active_plugins.erase(id);
		}
	}
}

static bool overrides_external_editor(Object *p_object) {
	Script *script = Object::cast_to<Script>(p_object);

	if (!script) {
		return false;
	}

	return script->get_language()->overrides_external_editor();
}

void EditorNode::_edit_current(bool p_skip_foreign) {
	ObjectID current_id = editor_history.get_current();
	Object *current_obj = current_id.is_valid() ? ObjectDB::get_instance(current_id) : nullptr;

	Ref<Resource> res = Object::cast_to<Resource>(current_obj);
	if (p_skip_foreign && res.is_valid()) {
		const int current_tab = scene_tabs->get_current_tab();
		if (res->get_path().find("::") > -1 && res->get_path().get_slice("::", 0) != editor_data.get_scene_path(current_tab)) {
			// Trying to edit resource that belongs to another scene; abort.
			current_obj = nullptr;
		}
	}

	bool inspector_only = editor_history.is_current_inspector_only();
	this->current = current_obj;

	if (!current_obj) {
		SceneTreeDock::get_singleton()->set_selected(nullptr);
		InspectorDock::get_inspector_singleton()->edit(nullptr);
		NodeDock::get_singleton()->set_node(nullptr);
		InspectorDock::get_singleton()->update(nullptr);
		hide_unused_editors();
		return;
	}

	// Update the use folding setting and state.
	bool disable_folding = bool(EDITOR_GET("interface/inspector/disable_folding")) || current_obj->is_class("EditorDebuggerRemoteObject");
	if (InspectorDock::get_inspector_singleton()->is_using_folding() == disable_folding) {
		InspectorDock::get_inspector_singleton()->set_use_folding(!disable_folding, false);
	}

	bool is_resource = current_obj->is_class("Resource");
	bool is_node = current_obj->is_class("Node");
	bool stay_in_script_editor_on_node_selected = bool(EDITOR_GET("text_editor/behavior/navigation/stay_in_script_editor_on_node_selected"));
	bool skip_main_plugin = false;

	String editable_info; // None by default.
	bool info_is_warning = false;

	if (current_obj->has_method("_is_read_only")) {
		if (current_obj->call("_is_read_only")) {
			editable_info = TTR("This object is marked as read-only, so it's not editable.");
		}
	}

	if (is_resource) {
		Resource *current_res = Object::cast_to<Resource>(current_obj);
		ERR_FAIL_NULL(current_res);

		InspectorDock::get_inspector_singleton()->edit(current_res);
		SceneTreeDock::get_singleton()->set_selected(nullptr);
		NodeDock::get_singleton()->set_node(nullptr);
		InspectorDock::get_singleton()->update(nullptr);
		ImportDock::get_singleton()->set_edit_path(current_res->get_path());

		int subr_idx = current_res->get_path().find("::");
		if (subr_idx != -1) {
			String base_path = current_res->get_path().substr(0, subr_idx);
			if (FileAccess::exists(base_path + ".import")) {
				if (!base_path.is_resource_file()) {
					if (get_edited_scene() && get_edited_scene()->get_scene_file_path() == base_path) {
						info_is_warning = true;
					}
				}
				editable_info = TTR("This resource belongs to a scene that was imported, so it's not editable.\nPlease read the documentation relevant to importing scenes to better understand this workflow.");
			} else if ((!get_edited_scene() || get_edited_scene()->get_scene_file_path() != base_path) && ResourceLoader::get_resource_type(base_path) == "PackedScene") {
				editable_info = TTR("This resource belongs to a scene that was instantiated or inherited.\nChanges to it must be made inside the original scene.");
			}
		} else if (current_res->get_path().is_resource_file()) {
			if (FileAccess::exists(current_res->get_path() + ".import")) {
				editable_info = TTR("This resource was imported, so it's not editable. Change its settings in the import panel and then re-import.");
			}
		}
	} else if (is_node) {
		Node *current_node = Object::cast_to<Node>(current_obj);
		ERR_FAIL_NULL(current_node);

		InspectorDock::get_inspector_singleton()->edit(current_node);
		if (current_node->is_inside_tree()) {
			NodeDock::get_singleton()->set_node(current_node);
			SceneTreeDock::get_singleton()->set_selected(current_node);
			InspectorDock::get_singleton()->update(current_node);
			if (!inspector_only && !skip_main_plugin) {
				skip_main_plugin = stay_in_script_editor_on_node_selected && !ScriptEditor::get_singleton()->is_editor_floating() && ScriptEditor::get_singleton()->is_visible_in_tree();
			}
		} else {
			NodeDock::get_singleton()->set_node(nullptr);
			SceneTreeDock::get_singleton()->set_selected(nullptr);
			InspectorDock::get_singleton()->update(nullptr);
		}

		if (get_edited_scene() && !get_edited_scene()->get_scene_file_path().is_empty()) {
			String source_scene = get_edited_scene()->get_scene_file_path();
			if (FileAccess::exists(source_scene + ".import")) {
				editable_info = TTR("This scene was imported, so changes to it won't be kept.\nInstantiating or inheriting it will allow you to make changes to it.\nPlease read the documentation relevant to importing scenes to better understand this workflow.");
				info_is_warning = true;
			}
		}

	} else {
		Node *selected_node = nullptr;

		if (current_obj->is_class("MultiNodeEdit")) {
			Node *scene = get_edited_scene();
			if (scene) {
				MultiNodeEdit *multi_node_edit = Object::cast_to<MultiNodeEdit>(current_obj);
				int node_count = multi_node_edit->get_node_count();
				if (node_count > 0) {
					List<Node *> multi_nodes;
					for (int node_index = 0; node_index < node_count; ++node_index) {
						Node *node = scene->get_node(multi_node_edit->get_node(node_index));
						if (node) {
							multi_nodes.push_back(node);
						}
					}
					if (!multi_nodes.is_empty()) {
						// Pick the top-most node.
						multi_nodes.sort_custom<Node::Comparator>();
						selected_node = multi_nodes.front()->get();
					}
				}
			}
		}

		InspectorDock::get_inspector_singleton()->edit(current_obj);
		NodeDock::get_singleton()->set_node(nullptr);
		SceneTreeDock::get_singleton()->set_selected(selected_node);
		InspectorDock::get_singleton()->update(nullptr);
	}

	InspectorDock::get_singleton()->set_info(
			info_is_warning ? TTR("Changes may be lost!") : TTR("This object is read-only."),
			editable_info,
			info_is_warning);

	Object *editor_owner = is_node ? (Object *)SceneTreeDock::get_singleton() : is_resource ? (Object *)InspectorDock::get_inspector_singleton()
																							: (Object *)this;

	// Take care of the main editor plugin.

	if (!inspector_only) {
		EditorPlugin *main_plugin = editor_data.get_handling_main_editor(current_obj);

		int plugin_index = 0;
		for (; plugin_index < editor_table.size(); plugin_index++) {
			if (editor_table[plugin_index] == main_plugin) {
				if (!main_editor_buttons[plugin_index]->is_visible()) {
					main_plugin = nullptr; // If button is not visible, then no plugin is active.
				}

				break;
			}
		}

		ObjectID editor_owner_id = editor_owner->get_instance_id();
		if (main_plugin && !skip_main_plugin) {
			// Special case if use of external editor is true.
			Resource *current_res = Object::cast_to<Resource>(current_obj);
			if (main_plugin->get_name() == "Script" && current_res && !current_res->is_built_in() && (bool(EDITOR_GET("text_editor/external/use_external_editor")) || overrides_external_editor(current_obj))) {
				if (!changing_scene) {
					main_plugin->edit(current_obj);
				}
			}

			else if (main_plugin != editor_plugin_screen && (!ScriptEditor::get_singleton() || !ScriptEditor::get_singleton()->is_visible_in_tree() || ScriptEditor::get_singleton()->can_take_away_focus())) {
				// Unedit previous plugin.
				editor_plugin_screen->edit(nullptr);
				active_plugins[editor_owner_id].erase(editor_plugin_screen);
				// Update screen main_plugin.
				editor_select(plugin_index);
				main_plugin->edit(current_obj);
			} else {
				editor_plugin_screen->edit(current_obj);
			}
			is_main_screen_editing = true;
		} else if (!main_plugin && editor_plugin_screen && is_main_screen_editing) {
			editor_plugin_screen->edit(nullptr);
			is_main_screen_editing = false;
		}

		edit_item(current_obj, editor_owner);
	}

	InspectorDock::get_singleton()->update(current_obj);
}

void EditorNode::_android_build_source_selected(const String &p_file) {
	export_template_manager->install_android_template_from_file(p_file);
}

void EditorNode::_menu_option_confirm(int p_option, bool p_confirmed) {
	if (!p_confirmed) { // FIXME: this may be a hack.
		current_menu_option = (MenuOptions)p_option;
	}

	switch (p_option) {
		case FILE_NEW_SCENE: {
			new_scene();

		} break;
		case FILE_NEW_INHERITED_SCENE:
		case FILE_OPEN_SCENE: {
			file->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
			List<String> extensions;
			ResourceLoader::get_recognized_extensions_for_type("PackedScene", &extensions);
			file->clear_filters();
			for (int i = 0; i < extensions.size(); i++) {
				file->add_filter("*." + extensions[i], extensions[i].to_upper());
			}

			Node *scene = editor_data.get_edited_scene_root();
			if (scene) {
				file->set_current_path(scene->get_scene_file_path());
			};
			file->set_title(p_option == FILE_OPEN_SCENE ? TTR("Open Scene") : TTR("Open Base Scene"));
			file->popup_file_dialog();

		} break;
		case FILE_QUICK_OPEN: {
			quick_open->popup_dialog("Resource", true);
			quick_open->set_title(TTR("Quick Open..."));

		} break;
		case FILE_QUICK_OPEN_SCENE: {
			quick_open->popup_dialog("PackedScene", true);
			quick_open->set_title(TTR("Quick Open Scene..."));

		} break;
		case FILE_QUICK_OPEN_SCRIPT: {
			quick_open->popup_dialog("Script", true);
			quick_open->set_title(TTR("Quick Open Script..."));

		} break;
		case FILE_OPEN_PREV: {
			if (previous_scenes.is_empty()) {
				break;
			}
			opening_prev = true;
			open_request(previous_scenes.back()->get());
			previous_scenes.pop_back();

		} break;
		case FILE_CLOSE_OTHERS: {
			tab_closing_menu_option = -1;
			for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {
				if (i == editor_data.get_edited_scene()) {
					continue;
				}
				tabs_to_close.push_back(editor_data.get_scene_path(i));
			}
			_proceed_closing_scene_tabs();
		} break;
		case FILE_CLOSE_RIGHT: {
			tab_closing_menu_option = -1;
			for (int i = editor_data.get_edited_scene() + 1; i < editor_data.get_edited_scene_count(); i++) {
				tabs_to_close.push_back(editor_data.get_scene_path(i));
			}
			_proceed_closing_scene_tabs();
		} break;
		case FILE_CLOSE_ALL: {
			tab_closing_menu_option = -1;
			for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {
				tabs_to_close.push_back(editor_data.get_scene_path(i));
			}
			_proceed_closing_scene_tabs();
		} break;
		case FILE_CLOSE: {
			_scene_tab_closed(editor_data.get_edited_scene());
		} break;
		case SCENE_TAB_CLOSE:
		case FILE_SAVE_SCENE: {
			int scene_idx = (p_option == FILE_SAVE_SCENE) ? -1 : tab_closing_idx;
			Node *scene = editor_data.get_edited_scene_root(scene_idx);
			if (scene && !scene->get_scene_file_path().is_empty()) {
				if (DirAccess::exists(scene->get_scene_file_path().get_base_dir())) {
					if (scene_idx != editor_data.get_edited_scene()) {
						_save_scene_with_preview(scene->get_scene_file_path(), scene_idx);
					} else {
						_save_scene_with_preview(scene->get_scene_file_path());
					}

					if (scene_idx != -1) {
						_discard_changes();
					}
					save_editor_layout_delayed();
				} else {
					show_save_accept(vformat(TTR("%s no longer exists! Please specify a new save location."), scene->get_scene_file_path().get_base_dir()), TTR("OK"));
				}
				break;
			}
			[[fallthrough]];
		}
		case FILE_SAVE_AS_SCENE: {
			int scene_idx = (p_option == FILE_SAVE_SCENE || p_option == FILE_SAVE_AS_SCENE) ? -1 : tab_closing_idx;

			Node *scene = editor_data.get_edited_scene_root(scene_idx);

			if (!scene) {
				if (p_option == FILE_SAVE_SCENE) {
					// Pressing Ctrl + S saves the current script if a scene is currently open, but it won't if the scene has no root node.
					// Work around this by explicitly saving the script in this case (similar to pressing Ctrl + Alt + S).
					ScriptEditor::get_singleton()->save_current_script();
				}

				const int saved = _save_external_resources();
				if (saved > 0) {
					show_accept(
							vformat(TTR("The current scene has no root node, but %d modified external resource(s) were saved anyway."), saved),
							TTR("OK"));
				} else if (p_option == FILE_SAVE_AS_SCENE) {
					// Don't show this dialog when pressing Ctrl + S to avoid interfering with script saving.
					show_accept(
							TTR("A root node is required to save the scene. You can add a root node using the Scene tree dock."),
							TTR("OK"));
				}

				break;
			}

			file->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);

			List<String> extensions;
			Ref<PackedScene> sd = memnew(PackedScene);
			ResourceSaver::get_recognized_extensions(sd, &extensions);
			file->clear_filters();
			for (int i = 0; i < extensions.size(); i++) {
				file->add_filter("*." + extensions[i], extensions[i].to_upper());
			}

			if (!scene->get_scene_file_path().is_empty()) {
				String path = scene->get_scene_file_path();
				file->set_current_path(path);
				if (extensions.size()) {
					String ext = path.get_extension().to_lower();
					if (extensions.find(ext) == nullptr) {
						file->set_current_path(path.replacen("." + ext, "." + extensions.front()->get()));
					}
				}
			} else if (extensions.size()) {
				String root_name = scene->get_name();
				root_name = EditorNode::adjust_scene_name_casing(root_name);
				file->set_current_path(root_name + "." + extensions.front()->get().to_lower());
			}
			file->popup_file_dialog();
			file->set_title(TTR("Save Scene As..."));

		} break;

		case FILE_SAVE_ALL_SCENES: {
			_save_all_scenes();
		} break;

		case FILE_RUN_SCENE: {
			project_run_bar->play_current_scene();
		} break;

		case FILE_EXPORT_PROJECT: {
			project_export->popup_export();
		} break;

		case FILE_EXTERNAL_OPEN_SCENE: {
			if (unsaved_cache && !p_confirmed) {
				confirmation->set_ok_button_text(TTR("Open"));
				confirmation->set_text(TTR("Current scene not saved. Open anyway?"));
				confirmation->popup_centered();
				break;
			}

			bool oprev = opening_prev;
			Error err = load_scene(external_file);
			if (err == OK && oprev) {
				previous_scenes.pop_back();
				opening_prev = false;
			}

		} break;

		case EDIT_UNDO: {
			if ((int)Input::get_singleton()->get_mouse_button_mask() & 0x7) {
				log->add_message(TTR("Can't undo while mouse buttons are pressed."), EditorLog::MSG_TYPE_EDITOR);
			} else {
				EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
				String action = undo_redo->get_current_action_name();
				int id = undo_redo->get_current_action_history_id();
				if (!undo_redo->undo()) {
					log->add_message(TTR("Nothing to undo."), EditorLog::MSG_TYPE_EDITOR);
				} else if (!action.is_empty()) {
					switch (id) {
						case EditorUndoRedoManager::GLOBAL_HISTORY:
							log->add_message(vformat(TTR("Global Undo: %s"), action), EditorLog::MSG_TYPE_EDITOR);
							break;
						case EditorUndoRedoManager::REMOTE_HISTORY:
							log->add_message(vformat(TTR("Remote Undo: %s"), action), EditorLog::MSG_TYPE_EDITOR);
							break;
						default:
							log->add_message(vformat(TTR("Scene Undo: %s"), action), EditorLog::MSG_TYPE_EDITOR);
					}
				}
			}
		} break;
		case EDIT_REDO: {
			EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
			if ((int)Input::get_singleton()->get_mouse_button_mask() & 0x7) {
				log->add_message(TTR("Can't redo while mouse buttons are pressed."), EditorLog::MSG_TYPE_EDITOR);
			} else {
				if (!undo_redo->redo()) {
					log->add_message(TTR("Nothing to redo."), EditorLog::MSG_TYPE_EDITOR);
				} else {
					String action = undo_redo->get_current_action_name();
					if (action.is_empty()) {
						break;
					}

					switch (undo_redo->get_current_action_history_id()) {
						case EditorUndoRedoManager::GLOBAL_HISTORY:
							log->add_message(vformat(TTR("Global Redo: %s"), action), EditorLog::MSG_TYPE_EDITOR);
							break;
						case EditorUndoRedoManager::REMOTE_HISTORY:
							log->add_message(vformat(TTR("Remote Redo: %s"), action), EditorLog::MSG_TYPE_EDITOR);
							break;
						default:
							log->add_message(vformat(TTR("Scene Redo: %s"), action), EditorLog::MSG_TYPE_EDITOR);
					}
				}
			}
		} break;

		case EDIT_RELOAD_SAVED_SCENE: {
			Node *scene = get_edited_scene();

			if (!scene) {
				break;
			}

			String filename = scene->get_scene_file_path();

			if (filename.is_empty()) {
				show_warning(TTR("Can't reload a scene that was never saved."));
				break;
			}

			if (unsaved_cache && !p_confirmed) {
				confirmation->set_ok_button_text(TTR("Reload Saved Scene"));
				confirmation->set_text(
						TTR("The current scene has unsaved changes.\nReload the saved scene anyway? This action cannot be undone."));
				confirmation->popup_centered();
				break;
			}

			int cur_idx = editor_data.get_edited_scene();
			_remove_edited_scene();
			Error err = load_scene(filename);
			if (err != OK) {
				ERR_PRINT("Failed to load scene");
			}
			editor_data.move_edited_scene_to_index(cur_idx);
			EditorUndoRedoManager::get_singleton()->clear_history(false, editor_data.get_current_edited_scene_history_id());
			scene_tabs->set_current_tab(cur_idx);

		} break;

		case FILE_SHOW_IN_FILESYSTEM: {
			String path = editor_data.get_scene_path(editor_data.get_edited_scene());
			if (!path.is_empty()) {
				FileSystemDock::get_singleton()->navigate_to_path(path);
			}
		} break;

		case RUN_SETTINGS: {
			project_settings_editor->popup_project_settings();
		} break;
		case FILE_INSTALL_ANDROID_SOURCE: {
			if (p_confirmed) {
				export_template_manager->install_android_template();
			} else {
				if (DirAccess::exists("res://android/build")) {
					remove_android_build_template->popup_centered();
				} else if (export_template_manager->can_install_android_template()) {
					install_android_build_template->popup_centered();
				} else {
					gradle_build_manage_templates->popup_centered();
				}
			}
		} break;
		case TOOLS_BUILD_PROFILE_MANAGER: {
			build_profile_manager->popup_centered_clamped(Size2(700, 800) * EDSCALE, 0.8);
		} break;
		case RUN_USER_DATA_FOLDER: {
			// Ensure_user_data_dir() to prevent the edge case: "Open User Data Folder" won't work after the project was renamed in ProjectSettingsEditor unless the project is saved.
			OS::get_singleton()->ensure_user_data_dir();
			OS::get_singleton()->shell_show_in_file_manager(OS::get_singleton()->get_user_data_dir(), true);
		} break;
		case FILE_EXPLORE_ANDROID_BUILD_TEMPLATES: {
			OS::get_singleton()->shell_show_in_file_manager(ProjectSettings::get_singleton()->get_resource_path().path_join("android"), true);
		} break;
		case FILE_QUIT:
		case RUN_PROJECT_MANAGER:
		case RELOAD_CURRENT_PROJECT: {
			if (p_confirmed && plugin_to_save) {
				plugin_to_save->save_external_data();
				p_confirmed = false;
			}

			if (!p_confirmed) {
				bool save_each = EDITOR_GET("interface/editor/save_each_scene_on_quit");
				if (_next_unsaved_scene(!save_each) == -1) {
					if (EditorUndoRedoManager::get_singleton()->is_history_unsaved(EditorUndoRedoManager::GLOBAL_HISTORY)) {
						if (p_option == RELOAD_CURRENT_PROJECT) {
							save_confirmation->set_ok_button_text(TTR("Save & Reload"));
							save_confirmation->set_text(TTR("Save modified resources before reloading?"));
						} else {
							save_confirmation->set_ok_button_text(TTR("Save & Quit"));
							save_confirmation->set_text(TTR("Save modified resources before closing?"));
						}
						save_confirmation->reset_size();
						save_confirmation->popup_centered();
						break;
					}

					plugin_to_save = nullptr;
					for (int i = 0; i < editor_data.get_editor_plugin_count(); i++) {
						const String unsaved_status = editor_data.get_editor_plugin(i)->get_unsaved_status();
						if (!unsaved_status.is_empty()) {
							if (p_option == RELOAD_CURRENT_PROJECT) {
								save_confirmation->set_ok_button_text(TTR("Save & Reload"));
								save_confirmation->set_text(unsaved_status);
							} else {
								save_confirmation->set_ok_button_text(TTR("Save & Quit"));
								save_confirmation->set_text(unsaved_status);
							}
							save_confirmation->reset_size();
							save_confirmation->popup_centered();
							plugin_to_save = editor_data.get_editor_plugin(i);
							break;
						}
					}

					if (plugin_to_save) {
						break;
					}

					_discard_changes();
					break;
				}

				if (save_each) {
					tab_closing_menu_option = current_menu_option;
					for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {
						tabs_to_close.push_back(editor_data.get_scene_path(i));
					}
					_proceed_closing_scene_tabs();
				} else {
					String unsaved_scenes;
					int i = _next_unsaved_scene(true, 0);
					while (i != -1) {
						unsaved_scenes += "\n            " + editor_data.get_edited_scene_root(i)->get_scene_file_path();
						i = _next_unsaved_scene(true, ++i);
					}
					if (p_option == RELOAD_CURRENT_PROJECT) {
						save_confirmation->set_ok_button_text(TTR("Save & Reload"));
						save_confirmation->set_text(TTR("Save changes to the following scene(s) before reloading?") + unsaved_scenes);
					} else {
						save_confirmation->set_ok_button_text(TTR("Save & Quit"));
						save_confirmation->set_text((p_option == FILE_QUIT ? TTR("Save changes to the following scene(s) before quitting?") : TTR("Save changes to the following scene(s) before opening Project Manager?")) + unsaved_scenes);
					}
					save_confirmation->reset_size();
					save_confirmation->popup_centered();
				}

				DisplayServer::get_singleton()->window_request_attention();
				break;
			}
			_save_external_resources();
			_discard_changes();
		} break;
		case SETTINGS_UPDATE_CONTINUOUSLY: {
			EditorSettings::get_singleton()->set("interface/editor/update_continuously", true);
			_update_update_spinner();
			show_accept(TTR("This option is deprecated. Situations where refresh must be forced are now considered a bug. Please report."), TTR("OK"));
		} break;
		case SETTINGS_UPDATE_WHEN_CHANGED: {
			EditorSettings::get_singleton()->set("interface/editor/update_continuously", false);
			_update_update_spinner();
		} break;
		case SETTINGS_UPDATE_SPINNER_HIDE: {
			EditorSettings::get_singleton()->set("interface/editor/show_update_spinner", false);
			_update_update_spinner();
		} break;
		case SETTINGS_PREFERENCES: {
			editor_settings_dialog->popup_edit_settings();
		} break;
		case SETTINGS_EDITOR_DATA_FOLDER: {
			OS::get_singleton()->shell_show_in_file_manager(EditorPaths::get_singleton()->get_data_dir(), true);
		} break;
		case SETTINGS_EDITOR_CONFIG_FOLDER: {
			OS::get_singleton()->shell_show_in_file_manager(EditorPaths::get_singleton()->get_config_dir(), true);
		} break;
		case SETTINGS_MANAGE_EXPORT_TEMPLATES: {
			export_template_manager->popup_manager();
		} break;
		case SETTINGS_MANAGE_FBX_IMPORTER: {
#if !defined(ANDROID_ENABLED) && !defined(WEB_ENABLED)
			fbx_importer_manager->show_dialog();
#endif
		} break;
		case SETTINGS_INSTALL_ANDROID_BUILD_TEMPLATE: {
			gradle_build_manage_templates->hide();
			file_android_build_source->popup_centered_ratio();
		} break;
		case SETTINGS_MANAGE_FEATURE_PROFILES: {
			feature_profile_manager->popup_centered_clamped(Size2(900, 800) * EDSCALE, 0.8);
		} break;
		case SETTINGS_TOGGLE_FULLSCREEN: {
			DisplayServer::WindowMode mode = DisplayServer::get_singleton()->window_get_mode();
			if (mode == DisplayServer::WINDOW_MODE_FULLSCREEN || mode == DisplayServer::WINDOW_MODE_EXCLUSIVE_FULLSCREEN) {
				DisplayServer::get_singleton()->window_set_mode(prev_mode);
			} else {
				prev_mode = mode;
				DisplayServer::get_singleton()->window_set_mode(DisplayServer::WINDOW_MODE_FULLSCREEN);
			}
		} break;
		case EDITOR_SCREENSHOT: {
			screenshot_timer->start();
		} break;
		case SETTINGS_PICK_MAIN_SCENE: {
			file->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
			List<String> extensions;
			ResourceLoader::get_recognized_extensions_for_type("PackedScene", &extensions);
			file->clear_filters();
			for (int i = 0; i < extensions.size(); i++) {
				file->add_filter("*." + extensions[i], extensions[i].to_upper());
			}

			Node *scene = editor_data.get_edited_scene_root();
			if (scene) {
				file->set_current_path(scene->get_scene_file_path());
			};
			file->set_title(TTR("Pick a Main Scene"));
			file->popup_file_dialog();

		} break;
		case HELP_SEARCH: {
			emit_signal(SNAME("request_help_search"), "");
		} break;
		case HELP_COMMAND_PALETTE: {
			command_palette->open_popup();
		} break;
		case HELP_DOCS: {
			OS::get_singleton()->shell_open(VERSION_DOCS_URL "/");
		} break;
		case HELP_QA: {
			OS::get_singleton()->shell_open("https://godotengine.org/qa/");
		} break;
		case HELP_REPORT_A_BUG: {
			OS::get_singleton()->shell_open("https://github.com/godotengine/godot/issues");
		} break;
		case HELP_COPY_SYSTEM_INFO: {
			String info = _get_system_info();
			DisplayServer::get_singleton()->clipboard_set(info);
		} break;
		case HELP_SUGGEST_A_FEATURE: {
			OS::get_singleton()->shell_open("https://github.com/godotengine/godot-proposals#readme");
		} break;
		case HELP_SEND_DOCS_FEEDBACK: {
			OS::get_singleton()->shell_open("https://github.com/godotengine/godot-docs/issues");
		} break;
		case HELP_COMMUNITY: {
			OS::get_singleton()->shell_open("https://godotengine.org/community");
		} break;
		case HELP_ABOUT: {
			about->popup_centered(Size2(780, 500) * EDSCALE);
		} break;
		case HELP_SUPPORT_GODOT_DEVELOPMENT: {
			OS::get_singleton()->shell_open("https://godotengine.org/donate");
		} break;
		case SET_RENDERER_NAME_SAVE_AND_RESTART: {
			ProjectSettings::get_singleton()->set("rendering/renderer/rendering_method", renderer_request);
			if (renderer_request == "mobile" || renderer_request == "gl_compatibility") {
				// Also change the mobile override if changing to a compatible rendering method.
				// This prevents visual discrepancies between desktop and mobile platforms.
				ProjectSettings::get_singleton()->set("rendering/renderer/rendering_method.mobile", renderer_request);
			} else if (renderer_request == "forward_plus") {
				// Use the equivalent mobile rendering method. This prevents the rendering method from staying
				// on its old choice if moving from `gl_compatibility` to `forward_plus`.
				ProjectSettings::get_singleton()->set("rendering/renderer/rendering_method.mobile", "mobile");
			}

			ProjectSettings::get_singleton()->save();

			save_all_scenes();
			restart_editor();
		} break;
	}
}

String EditorNode::adjust_scene_name_casing(const String &root_name) {
	switch (GLOBAL_GET("editor/naming/scene_name_casing").operator int()) {
		case SCENE_NAME_CASING_AUTO:
			// Use casing of the root node.
			break;
		case SCENE_NAME_CASING_PASCAL_CASE:
			return root_name.to_pascal_case();
		case SCENE_NAME_CASING_SNAKE_CASE:
			return root_name.replace("-", "_").to_snake_case();
	}
	return root_name;
}

void EditorNode::_request_screenshot() {
	_screenshot();
}

void EditorNode::_screenshot(bool p_use_utc) {
	String name = "editor_screenshot_" + Time::get_singleton()->get_datetime_string_from_system(p_use_utc).replace(":", "") + ".png";
	NodePath path = String("user://") + name;
	_save_screenshot(path);
	if (EDITOR_GET("interface/editor/automatically_open_screenshots")) {
		OS::get_singleton()->shell_show_in_file_manager(ProjectSettings::get_singleton()->globalize_path(path), true);
	}
}

void EditorNode::_save_screenshot(NodePath p_path) {
	Control *editor_main_screen = EditorInterface::get_singleton()->get_editor_main_screen();
	ERR_FAIL_NULL_MSG(editor_main_screen, "Cannot get the editor main screen control.");
	Viewport *viewport = editor_main_screen->get_viewport();
	ERR_FAIL_NULL_MSG(viewport, "Cannot get a viewport from the editor main screen.");
	Ref<ViewportTexture> texture = viewport->get_texture();
	ERR_FAIL_COND_MSG(texture.is_null(), "Cannot get a viewport texture from the editor main screen.");
	Ref<Image> img = texture->get_image();
	ERR_FAIL_COND_MSG(img.is_null(), "Cannot get an image from a viewport texture of the editor main screen.");
	Error error = img->save_png(p_path);
	ERR_FAIL_COND_MSG(error != OK, "Cannot save screenshot to file '" + p_path + "'.");
}

void EditorNode::_tool_menu_option(int p_idx) {
	switch (tool_menu->get_item_id(p_idx)) {
		case TOOLS_ORPHAN_RESOURCES: {
			orphan_resources->show();
		} break;
		case TOOLS_SURFACE_UPGRADE: {
			surface_upgrade_dialog->popup_on_demand();
		} break;
		case TOOLS_CUSTOM: {
			if (tool_menu->get_item_submenu(p_idx) == "") {
				Callable callback = tool_menu->get_item_metadata(p_idx);
				Callable::CallError ce;
				Variant result;
				callback.callp(nullptr, 0, result, ce);

				if (ce.error != Callable::CallError::CALL_OK) {
					String err = Variant::get_callable_error_text(callback, nullptr, 0, ce);
					ERR_PRINT("Error calling function from tool menu: " + err);
				}
			} // Else it's a submenu so don't do anything.
		} break;
	}
}

void EditorNode::_export_as_menu_option(int p_idx) {
	if (p_idx == 0) { // MeshLibrary
		current_menu_option = FILE_EXPORT_MESH_LIBRARY;

		if (!editor_data.get_edited_scene_root()) {
			show_accept(TTR("This operation can't be done without a scene."), TTR("OK"));
			return;
		}

		List<String> extensions;
		Ref<MeshLibrary> ml(memnew(MeshLibrary));
		ResourceSaver::get_recognized_extensions(ml, &extensions);
		file_export_lib->clear_filters();
		for (const String &E : extensions) {
			file_export_lib->add_filter("*." + E);
		}

		file_export_lib->popup_file_dialog();
		file_export_lib->set_title(TTR("Export Mesh Library"));
	} else { // Custom menu options added by plugins
		if (export_as_menu->get_item_submenu(p_idx).is_empty()) { // If not a submenu
			Callable callback = export_as_menu->get_item_metadata(p_idx);
			Callable::CallError ce;
			Variant result;
			callback.callp(nullptr, 0, result, ce);

			if (ce.error != Callable::CallError::CALL_OK) {
				String err = Variant::get_callable_error_text(callback, nullptr, 0, ce);
				ERR_PRINT("Error calling function from export_as menu: " + err);
			}
		}
	}
}

int EditorNode::_next_unsaved_scene(bool p_valid_filename, int p_start) {
	for (int i = p_start; i < editor_data.get_edited_scene_count(); i++) {
		if (!editor_data.get_edited_scene_root(i)) {
			continue;
		}

		String scene_filename = editor_data.get_edited_scene_root(i)->get_scene_file_path();
		if (p_valid_filename && scene_filename.is_empty()) {
			continue;
		}

		bool unsaved = EditorUndoRedoManager::get_singleton()->is_history_unsaved(editor_data.get_scene_history_id(i));
		if (unsaved) {
			return i;
		} else {
			for (int j = 0; j < editor_data.get_editor_plugin_count(); j++) {
				if (!editor_data.get_editor_plugin(j)->get_unsaved_status(scene_filename).is_empty()) {
					return i;
				}
			}
		}
	}
	return -1;
}

void EditorNode::_exit_editor(int p_exit_code) {
	exiting = true;
	resource_preview->stop(); // Stop early to avoid crashes.
	_save_editor_layout();

	// Dim the editor window while it's quitting to make it clearer that it's busy.
	dim_editor(true);

	get_tree()->quit(p_exit_code);
}

void EditorNode::_discard_changes(const String &p_str) {
	switch (current_menu_option) {
		case FILE_CLOSE:
		case SCENE_TAB_CLOSE: {
			Node *scene = editor_data.get_edited_scene_root(tab_closing_idx);
			if (scene != nullptr) {
				String scene_filename = scene->get_scene_file_path();
				if (!scene_filename.is_empty()) {
					previous_scenes.push_back(scene_filename);
				}
			}

			// Don't close tabs when exiting the editor (required for "restore_scenes_on_load" setting).
			if (!_is_closing_editor()) {
				_remove_scene(tab_closing_idx);
				scene_tabs->update_scene_tabs();
			}
			_proceed_closing_scene_tabs();
		} break;
		case FILE_QUIT: {
			project_run_bar->stop_playing();
			_exit_editor(EXIT_SUCCESS);

		} break;
		case RUN_PROJECT_MANAGER: {
			project_run_bar->stop_playing();
			_exit_editor(EXIT_SUCCESS);
			String exec = OS::get_singleton()->get_executable_path();

			List<String> args;
			for (const String &a : Main::get_forwardable_cli_arguments(Main::CLI_SCOPE_TOOL)) {
				args.push_back(a);
			}

			String exec_base_dir = exec.get_base_dir();
			if (!exec_base_dir.is_empty()) {
				args.push_back("--path");
				args.push_back(exec_base_dir);
			}
			args.push_back("--project-manager");

			Error err = OS::get_singleton()->create_instance(args);
			ERR_FAIL_COND(err);
		} break;
		case RELOAD_CURRENT_PROJECT: {
			restart_editor();
		} break;
	}
}

void EditorNode::_update_file_menu_opened() {
	file_menu->set_item_disabled(file_menu->get_item_index(FILE_OPEN_PREV), previous_scenes.is_empty());
	_update_undo_redo_allowed();
}

void EditorNode::_update_file_menu_closed() {
	file_menu->set_item_disabled(file_menu->get_item_index(FILE_OPEN_PREV), false);
}

VBoxContainer *EditorNode::get_main_screen_control() {
	return main_screen_vbox;
}

void EditorNode::editor_select(int p_which) {
	static bool selecting = false;
	if (selecting || changing_scene) {
		return;
	}

	ERR_FAIL_INDEX(p_which, editor_table.size());

	if (!main_editor_buttons[p_which]->is_visible()) { // Button hidden, no editor.
		return;
	}

	selecting = true;

	for (int i = 0; i < main_editor_buttons.size(); i++) {
		main_editor_buttons[i]->set_pressed(i == p_which);
	}

	selecting = false;

	EditorPlugin *new_editor = editor_table[p_which];
	ERR_FAIL_NULL(new_editor);

	if (editor_plugin_screen == new_editor) {
		return;
	}

	if (editor_plugin_screen) {
		editor_plugin_screen->make_visible(false);
	}

	editor_plugin_screen = new_editor;
	editor_plugin_screen->make_visible(true);
	editor_plugin_screen->selected_notify();

	int plugin_count = editor_data.get_editor_plugin_count();
	for (int i = 0; i < plugin_count; i++) {
		editor_data.get_editor_plugin(i)->notify_main_screen_changed(editor_plugin_screen->get_name());
	}

	if (EDITOR_GET("interface/editor/separate_distraction_mode")) {
		if (p_which == EDITOR_SCRIPT) {
			set_distraction_free_mode(script_distraction_free);
		} else {
			set_distraction_free_mode(scene_distraction_free);
		}
	}
}

void EditorNode::select_editor_by_name(const String &p_name) {
	ERR_FAIL_COND(p_name.is_empty());

	for (int i = 0; i < main_editor_buttons.size(); i++) {
		if (main_editor_buttons[i]->get_text() == p_name) {
			editor_select(i);
			return;
		}
	}

	ERR_FAIL_MSG("The editor name '" + p_name + "' was not found.");
}

void EditorNode::add_editor_plugin(EditorPlugin *p_editor, bool p_config_changed) {
	if (p_editor->has_main_screen()) {
		Button *tb = memnew(Button);
		tb->set_flat(true);
		tb->set_toggle_mode(true);
		tb->connect("pressed", callable_mp(singleton, &EditorNode::editor_select).bind(singleton->main_editor_buttons.size()));
		tb->set_name(p_editor->get_name());
		tb->set_text(p_editor->get_name());

		Ref<Texture2D> icon = p_editor->get_icon();
		if (icon.is_valid()) {
			tb->set_icon(icon);
			// Make sure the control is updated if the icon is reimported.
			icon->connect_changed(callable_mp((Control *)tb, &Control::update_minimum_size));
		} else if (singleton->theme->has_icon(p_editor->get_name(), EditorStringName(EditorIcons))) {
			tb->set_icon(singleton->theme->get_icon(p_editor->get_name(), EditorStringName(EditorIcons)));
		}

		tb->add_theme_font_override("font", singleton->theme->get_font(SNAME("main_button_font"), EditorStringName(EditorFonts)));
		tb->add_theme_font_size_override("font_size", singleton->theme->get_font_size(SNAME("main_button_font_size"), EditorStringName(EditorFonts)));

		singleton->main_editor_buttons.push_back(tb);
		singleton->main_editor_button_hb->add_child(tb);
		singleton->editor_table.push_back(p_editor);

		singleton->distraction_free->move_to_front();
	}
	singleton->editor_data.add_editor_plugin(p_editor);
	singleton->add_child(p_editor);
	if (p_config_changed) {
		p_editor->enable_plugin();
	}
}

void EditorNode::remove_editor_plugin(EditorPlugin *p_editor, bool p_config_changed) {
	if (p_editor->has_main_screen()) {
		// Remove the main editor button and update the bindings of
		// all buttons behind it to point to the correct main window.
		for (int i = singleton->main_editor_buttons.size() - 1; i >= 0; i--) {
			if (p_editor->get_name() == singleton->main_editor_buttons[i]->get_text()) {
				if (singleton->main_editor_buttons[i]->is_pressed()) {
					singleton->editor_select(EDITOR_SCRIPT);
				}

				memdelete(singleton->main_editor_buttons[i]);
				singleton->main_editor_buttons.remove_at(i);

				break;
			} else {
				singleton->main_editor_buttons[i]->disconnect("pressed", callable_mp(singleton, &EditorNode::editor_select));
				singleton->main_editor_buttons[i]->connect("pressed", callable_mp(singleton, &EditorNode::editor_select).bind(i - 1));
			}
		}

		singleton->editor_table.erase(p_editor);
	}
	p_editor->make_visible(false);
	p_editor->clear();
	if (p_config_changed) {
		p_editor->disable_plugin();
	}
	singleton->editor_plugins_over->remove_plugin(p_editor);
	singleton->editor_plugins_force_over->remove_plugin(p_editor);
	singleton->editor_plugins_force_input_forwarding->remove_plugin(p_editor);
	singleton->remove_child(p_editor);
	singleton->editor_data.remove_editor_plugin(p_editor);

	for (KeyValue<ObjectID, HashSet<EditorPlugin *>> &kv : singleton->active_plugins) {
		kv.value.erase(p_editor);
	}
}

void EditorNode::add_extension_editor_plugin(const StringName &p_class_name) {
	ERR_FAIL_COND_MSG(!ClassDB::class_exists(p_class_name), vformat("No such editor plugin registered: %s", p_class_name));
	ERR_FAIL_COND_MSG(!ClassDB::is_parent_class(p_class_name, SNAME("EditorPlugin")), vformat("Class is not an editor plugin: %s", p_class_name));
	ERR_FAIL_COND_MSG(singleton->editor_data.has_extension_editor_plugin(p_class_name), vformat("Editor plugin already added for class: %s", p_class_name));

	EditorPlugin *plugin = Object::cast_to<EditorPlugin>(ClassDB::instantiate(p_class_name));
	singleton->editor_data.add_extension_editor_plugin(p_class_name, plugin);
	add_editor_plugin(plugin);
}

void EditorNode::remove_extension_editor_plugin(const StringName &p_class_name) {
	// If we're exiting, the editor plugins will get cleaned up anyway, so don't do anything.
	if (singleton->exiting) {
		return;
	}

	ERR_FAIL_COND_MSG(!singleton->editor_data.has_extension_editor_plugin(p_class_name), vformat("No editor plugin added for class: %s", p_class_name));

	EditorPlugin *plugin = singleton->editor_data.get_extension_editor_plugin(p_class_name);
	remove_editor_plugin(plugin);
	memdelete(plugin);
	singleton->editor_data.remove_extension_editor_plugin(p_class_name);
}

void EditorNode::_update_addon_config() {
	if (_initializing_plugins) {
		return;
	}

	Vector<String> enabled_addons;

	for (const KeyValue<String, EditorPlugin *> &E : addon_name_to_plugin) {
		enabled_addons.push_back(E.key);
	}

	if (enabled_addons.size() == 0) {
		ProjectSettings::get_singleton()->set("editor_plugins/enabled", Variant());
	} else {
		enabled_addons.sort();
		ProjectSettings::get_singleton()->set("editor_plugins/enabled", enabled_addons);
	}

	project_settings_editor->queue_save();
}

void EditorNode::set_addon_plugin_enabled(const String &p_addon, bool p_enabled, bool p_config_changed) {
	String addon_path = p_addon;

	if (!addon_path.begins_with("res://")) {
		addon_path = "res://addons/" + addon_path + "/plugin.cfg";
	}

	ERR_FAIL_COND(p_enabled && addon_name_to_plugin.has(addon_path));
	ERR_FAIL_COND(!p_enabled && !addon_name_to_plugin.has(addon_path));

	if (!p_enabled) {
		EditorPlugin *addon = addon_name_to_plugin[addon_path];
		remove_editor_plugin(addon, p_config_changed);
		memdelete(addon);
		addon_name_to_plugin.erase(addon_path);
		_update_addon_config();
		return;
	}

	Ref<ConfigFile> cf;
	cf.instantiate();
	if (!DirAccess::exists(addon_path.get_base_dir())) {
		_remove_plugin_from_enabled(addon_path);
		WARN_PRINT("Addon '" + addon_path + "' failed to load. No directory found. Removing from enabled plugins.");
		return;
	}
	Error err = cf->load(addon_path);
	if (err != OK) {
		show_warning(vformat(TTR("Unable to enable addon plugin at: '%s' parsing of config failed."), addon_path));
		return;
	}

	String plugin_version;
	if (cf->has_section_key("plugin", "version")) {
		plugin_version = cf->get_value("plugin", "version");
	}

	if (!cf->has_section_key("plugin", "script")) {
		show_warning(vformat(TTR("Unable to find script field for addon plugin at: '%s'."), addon_path));
		return;
	}

	String script_path = cf->get_value("plugin", "script");
	Ref<Script> scr; // We need to save it for creating "ep" below.

	// Only try to load the script if it has a name. Else, the plugin has no init script.
	if (script_path.length() > 0) {
		script_path = addon_path.get_base_dir().path_join(script_path);
		scr = ResourceLoader::load(script_path, "Script", ResourceFormatLoader::CACHE_MODE_IGNORE);

		if (scr.is_null()) {
			show_warning(vformat(TTR("Unable to load addon script from path: '%s'."), script_path));
			return;
		}

		// Errors in the script cause the base_type to be an empty StringName.
		if (scr->get_instance_base_type() == StringName()) {
			if (_initializing_plugins) {
				// However, if it happens during initialization, waiting for file scan might help.
				pending_addons.push_back(p_addon);
				return;
			}

			show_warning(vformat(TTR("Unable to load addon script from path: '%s'. This might be due to a code error in that script.\nDisabling the addon at '%s' to prevent further errors."), script_path, addon_path));
			_remove_plugin_from_enabled(addon_path);
			return;
		}

		// Plugin init scripts must inherit from EditorPlugin and be tools.
		if (!ClassDB::is_parent_class(scr->get_instance_base_type(), "EditorPlugin")) {
			show_warning(vformat(TTR("Unable to load addon script from path: '%s'. Base type is not 'EditorPlugin'."), script_path));
			return;
		}

		if (!scr->is_tool()) {
			show_warning(vformat(TTR("Unable to load addon script from path: '%s'. Script is not in tool mode."), script_path));
			return;
		}
	}

	EditorPlugin *ep = memnew(EditorPlugin);
	ep->set_script(scr);
	ep->set_plugin_version(plugin_version);
	addon_name_to_plugin[addon_path] = ep;
	add_editor_plugin(ep, p_config_changed);

	_update_addon_config();
}

bool EditorNode::is_addon_plugin_enabled(const String &p_addon) const {
	if (p_addon.begins_with("res://")) {
		return addon_name_to_plugin.has(p_addon);
	}

	return addon_name_to_plugin.has("res://addons/" + p_addon + "/plugin.cfg");
}

void EditorNode::_remove_edited_scene(bool p_change_tab) {
	// When scene gets closed no node is edited anymore, so make sure the editors are notified before nodes are freed.
	hide_unused_editors(SceneTreeDock::get_singleton());

	int new_index = editor_data.get_edited_scene();
	int old_index = new_index;

	if (new_index > 0) {
		new_index = new_index - 1;
	} else if (editor_data.get_edited_scene_count() > 1) {
		new_index = 1;
	} else {
		editor_data.add_edited_scene(-1);
		new_index = 1;
	}

	if (p_change_tab) {
		_set_current_scene(new_index);
	}
	editor_data.remove_scene(old_index);
	_update_title();
	scene_tabs->update_scene_tabs();
}

void EditorNode::_remove_scene(int index, bool p_change_tab) {
	// Clear icon cache in case some scripts are no longer needed.
	// FIXME: Ideally the cache should never be cleared and only updated on per-script basis, when an icon changes.
	editor_data.clear_script_icon_cache();

	if (editor_data.get_edited_scene() == index) {
		// Scene to remove is current scene.
		_remove_edited_scene(p_change_tab);
	} else {
		// Scene to remove is not active scene.
		editor_data.remove_scene(index);
	}
}

void EditorNode::set_edited_scene(Node *p_scene) {
	Node *old_edited_scene_root = get_editor_data().get_edited_scene_root();
	if (old_edited_scene_root) {
		if (old_edited_scene_root->get_parent() == scene_root) {
			scene_root->remove_child(old_edited_scene_root);
		}
		old_edited_scene_root->disconnect(SNAME("replacing_by"), callable_mp(this, &EditorNode::set_edited_scene));
	}
	get_editor_data().set_edited_scene_root(p_scene);

	if (Object::cast_to<Popup>(p_scene)) {
		Object::cast_to<Popup>(p_scene)->show();
	}
	SceneTreeDock::get_singleton()->set_edited_scene(p_scene);
	if (get_tree()) {
		get_tree()->set_edited_scene_root(p_scene);
	}

	if (p_scene) {
		if (p_scene->get_parent() != scene_root) {
			scene_root->add_child(p_scene, true);
		}
		p_scene->connect(SNAME("replacing_by"), callable_mp(this, &EditorNode::set_edited_scene));
	}
}

int EditorNode::_get_current_main_editor() {
	for (int i = 0; i < editor_table.size(); i++) {
		if (editor_table[i] == editor_plugin_screen) {
			return i;
		}
	}

	return 0;
}

Dictionary EditorNode::_get_main_scene_state() {
	Dictionary state;
	state["main_tab"] = _get_current_main_editor();
	state["scene_tree_offset"] = SceneTreeDock::get_singleton()->get_tree_editor()->get_scene_tree()->get_vscroll_bar()->get_value();
	state["property_edit_offset"] = InspectorDock::get_inspector_singleton()->get_scroll_offset();
	state["node_filter"] = SceneTreeDock::get_singleton()->get_filter();
	return state;
}

void EditorNode::_set_main_scene_state(Dictionary p_state, Node *p_for_scene) {
	if (get_edited_scene() != p_for_scene && p_for_scene != nullptr) {
		return; // Not for this scene.
	}

	changing_scene = false;

	int current_tab = -1;
	for (int i = 0; i < editor_table.size(); i++) {
		if (editor_plugin_screen == editor_table[i]) {
			current_tab = i;
			break;
		}
	}

	if (p_state.has("editor_index")) {
		int index = p_state["editor_index"];
		if (current_tab < 2) { // If currently in spatial/2d, only switch to spatial/2d. If currently in script, stay there.
			if (index < 2 || !get_edited_scene()) {
				editor_select(index);
			}
		}
	}

	if (get_edited_scene()) {
		if (current_tab < 2) {
			Node *editor_node = SceneTreeDock::get_singleton()->get_tree_editor()->get_selected();
			editor_node = editor_node == nullptr ? get_edited_scene() : editor_node;

			if (Object::cast_to<CanvasItem>(editor_node)) {
				editor_select(EDITOR_2D);
			} else if (Object::cast_to<Node3D>(editor_node)) {
				editor_select(EDITOR_3D);
			}
		}
	}

	if (p_state.has("scene_tree_offset")) {
		SceneTreeDock::get_singleton()->get_tree_editor()->get_scene_tree()->get_vscroll_bar()->set_value(p_state["scene_tree_offset"]);
	}
	if (p_state.has("property_edit_offset")) {
		InspectorDock::get_inspector_singleton()->set_scroll_offset(p_state["property_edit_offset"]);
	}

	if (p_state.has("node_filter")) {
		SceneTreeDock::get_singleton()->set_filter(p_state["node_filter"]);
	}

	// This should only happen at the very end.

	EditorDebuggerNode::get_singleton()->update_live_edit_root();
	ScriptEditor::get_singleton()->set_scene_root_script(editor_data.get_scene_root_script(editor_data.get_edited_scene()));
	editor_data.notify_edited_scene_changed();
	emit_signal(SNAME("scene_changed"));

	// Reset SDFGI after everything else so that any last-second scene modifications will be processed.
	RenderingServer::get_singleton()->sdfgi_reset();
}

bool EditorNode::is_changing_scene() const {
	return changing_scene;
}

void EditorNode::_set_current_scene(int p_idx) {
	if (p_idx == editor_data.get_edited_scene()) {
		return; // Pointless.
	}

	_set_current_scene_nocheck(p_idx);
}

void EditorNode::_set_current_scene_nocheck(int p_idx) {
	// Save the folding in case the scene gets reloaded.
	if (editor_data.get_scene_path(p_idx) != "" && editor_data.get_edited_scene_root(p_idx)) {
		editor_folding.save_scene_folding(editor_data.get_edited_scene_root(p_idx), editor_data.get_scene_path(p_idx));
	}

	if (editor_data.check_and_update_scene(p_idx)) {
		if (editor_data.get_scene_path(p_idx) != "") {
			editor_folding.load_scene_folding(editor_data.get_edited_scene_root(p_idx), editor_data.get_scene_path(p_idx));
		}

		EditorUndoRedoManager::get_singleton()->clear_history(false, editor_data.get_scene_history_id(p_idx));
	}

	changing_scene = true;
	editor_data.save_edited_scene_state(editor_selection, &editor_history, _get_main_scene_state());

	if (get_editor_data().get_edited_scene_root()) {
		if (get_editor_data().get_edited_scene_root()->get_parent() == scene_root) {
			scene_root->remove_child(get_editor_data().get_edited_scene_root());
		}
	}

	editor_selection->clear();
	editor_data.set_edited_scene(p_idx);

	Node *new_scene = editor_data.get_edited_scene_root();

	if (Popup *p = Object::cast_to<Popup>(new_scene)) {
		p->show();
	}

	SceneTreeDock::get_singleton()->set_edited_scene(new_scene);
	if (get_tree()) {
		get_tree()->set_edited_scene_root(new_scene);
	}

	if (new_scene) {
		if (new_scene->get_parent() != scene_root) {
			scene_root->add_child(new_scene, true);
		}
	}

	Dictionary state = editor_data.restore_edited_scene_state(editor_selection, &editor_history);
	_edit_current(true);

	_update_title();
	scene_tabs->update_scene_tabs();

	if (tabs_to_close.is_empty()) {
		call_deferred(SNAME("_set_main_scene_state"), state, get_edited_scene()); // Do after everything else is done setting up.
	}
}

void EditorNode::setup_color_picker(ColorPicker *p_picker) {
	p_picker->set_editor_settings(EditorSettings::get_singleton());
	int default_color_mode = EDITOR_GET("interface/inspector/default_color_picker_mode");
	int picker_shape = EDITOR_GET("interface/inspector/default_color_picker_shape");
	p_picker->set_color_mode((ColorPicker::ColorModeType)default_color_mode);
	p_picker->set_picker_shape((ColorPicker::PickerShapeType)picker_shape);
}

bool EditorNode::is_scene_open(const String &p_path) {
	for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {
		if (editor_data.get_scene_path(i) == p_path) {
			return true;
		}
	}

	return false;
}

void EditorNode::fix_dependencies(const String &p_for_file) {
	dependency_fixer->edit(p_for_file);
}

int EditorNode::new_scene() {
	int idx = editor_data.add_edited_scene(-1);
	_set_current_scene(idx); // Before trying to remove an empty scene, set the current tab index to the newly added tab index.

	// Remove placeholder empty scene.
	if (editor_data.get_edited_scene_count() > 1) {
		for (int i = 0; i < editor_data.get_edited_scene_count() - 1; i++) {
			bool unsaved = EditorUndoRedoManager::get_singleton()->is_history_unsaved(editor_data.get_scene_history_id(i));
			if (!unsaved && editor_data.get_scene_path(i).is_empty() && editor_data.get_edited_scene_root(i) == nullptr) {
				editor_data.remove_scene(i);
				idx--;
			}
		}
	}

	editor_data.clear_editor_states();
	scene_tabs->update_scene_tabs();
	return idx;
}

Error EditorNode::load_scene(const String &p_scene, bool p_ignore_broken_deps, bool p_set_inherited, bool p_clear_errors, bool p_force_open_imported, bool p_silent_change_tab) {
	if (!is_inside_tree()) {
		defer_load_scene = p_scene;
		return OK;
	}

	if (!p_set_inherited) {
		for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {
			if (editor_data.get_scene_path(i) == p_scene) {
				_set_current_scene(i);
				return OK;
			}
		}

		if (!p_force_open_imported && FileAccess::exists(p_scene + ".import")) {
			open_imported->set_text(vformat(TTR("Scene '%s' was automatically imported, so it can't be modified.\nTo make changes to it, a new inherited scene can be created."), p_scene.get_file()));
			open_imported->popup_centered();
			new_inherited_button->grab_focus();
			open_import_request = p_scene;
			return OK;
		}
	}

	if (p_clear_errors) {
		load_errors->clear();
	}

	String lpath = ProjectSettings::get_singleton()->localize_path(p_scene);

	if (!lpath.begins_with("res://")) {
		show_accept(TTR("Error loading scene, it must be inside the project path. Use 'Import' to open the scene, then save it inside the project path."), TTR("OK"));
		opening_prev = false;
		return ERR_FILE_NOT_FOUND;
	}

	int prev = editor_data.get_edited_scene();
	int idx = editor_data.add_edited_scene(-1);

	if (!editor_data.get_edited_scene_root() && editor_data.get_edited_scene_count() == 2) {
		_remove_edited_scene();
	} else if (p_silent_change_tab) {
		_set_current_scene_nocheck(idx);
	} else {
		_set_current_scene(idx);
	}

	dependency_errors.clear();

	Error err;
	Ref<PackedScene> sdata = ResourceLoader::load(lpath, "", ResourceFormatLoader::CACHE_MODE_REPLACE, &err);

	if (!p_ignore_broken_deps && dependency_errors.has(lpath)) {
		current_menu_option = -1;
		Vector<String> errors;
		for (const String &E : dependency_errors[lpath]) {
			errors.push_back(E);
		}
		dependency_error->show(DependencyErrorDialog::MODE_SCENE, lpath, errors);
		opening_prev = false;

		if (prev != -1) {
			_set_current_scene(prev);
			editor_data.remove_scene(idx);
		}
		return ERR_FILE_MISSING_DEPENDENCIES;
	}

	if (!sdata.is_valid()) {
		_dialog_display_load_error(lpath, err);
		opening_prev = false;

		if (prev != -1) {
			_set_current_scene(prev);
			editor_data.remove_scene(idx);
		}
		return ERR_FILE_NOT_FOUND;
	}

	dependency_errors.erase(lpath); // At least not self path.

	for (KeyValue<String, HashSet<String>> &E : dependency_errors) {
		String txt = vformat(TTR("Scene '%s' has broken dependencies:"), E.key) + "\n";
		for (const String &F : E.value) {
			txt += "\t" + F + "\n";
		}
		add_io_error(txt);
	}

	if (ResourceCache::has(lpath)) {
		// Used from somewhere else? No problem! Update state and replace sdata.
		Ref<PackedScene> ps = ResourceCache::get_ref(lpath);
		if (ps.is_valid()) {
			ps->replace_state(sdata->get_state());
			ps->set_last_modified_time(sdata->get_last_modified_time());
			sdata = ps;
		}

	} else {
		sdata->set_path(lpath, true); // Take over path.
	}

	Node *new_scene = sdata->instantiate(p_set_inherited ? PackedScene::GEN_EDIT_STATE_MAIN_INHERITED : PackedScene::GEN_EDIT_STATE_MAIN);

	if (!new_scene) {
		sdata.unref();
		_dialog_display_load_error(lpath, ERR_FILE_CORRUPT);
		opening_prev = false;
		if (prev != -1) {
			_set_current_scene(prev);
			editor_data.remove_scene(idx);
		}
		return ERR_FILE_CORRUPT;
	}

	if (p_set_inherited) {
		Ref<SceneState> state = sdata->get_state();
		state->set_path(lpath);
		new_scene->set_scene_inherited_state(state);
		new_scene->set_scene_file_path(String());
	}

	new_scene->set_scene_instance_state(Ref<SceneState>());

	set_edited_scene(new_scene);

	String config_file_path = EditorPaths::get_singleton()->get_project_settings_dir().path_join(p_scene.get_file() + "-editstate-" + p_scene.md5_text() + ".cfg");
	Ref<ConfigFile> editor_state_cf;
	editor_state_cf.instantiate();
	Error editor_state_cf_err = editor_state_cf->load(config_file_path);
	if (editor_state_cf_err == OK || editor_state_cf->has_section("editor_states")) {
		_load_editor_plugin_states_from_config(editor_state_cf);
	}

	_update_title();
	scene_tabs->update_scene_tabs();
	_add_to_recent_scenes(lpath);

	if (editor_folding.has_folding_data(lpath)) {
		editor_folding.load_scene_folding(new_scene, lpath);
	} else if (EDITOR_GET("interface/inspector/auto_unfold_foreign_scenes")) {
		editor_folding.unfold_scene(new_scene);
		editor_folding.save_scene_folding(new_scene, lpath);
	}

	opening_prev = false;

	EditorDebuggerNode::get_singleton()->update_live_edit_root();

	// Tell everything to edit this object, unless we're in the process of restoring scenes.
	// If we are, we'll edit it after the restoration is done.
	if (!restoring_scenes) {
		push_item(new_scene);
	} else {
		// Initialize history for restored scenes.
		ObjectID id = new_scene->get_instance_id();
		if (id != editor_history.get_current()) {
			editor_history.add_object(id);
		}
	}

	// Load the selected nodes.
	if (editor_state_cf->has_section_key("editor_states", "selected_nodes")) {
		TypedArray<NodePath> selected_node_list = editor_state_cf->get_value("editor_states", "selected_nodes", TypedArray<String>());

		for (int i = 0; i < selected_node_list.size(); i++) {
			Node *selected_node = new_scene->get_node_or_null(selected_node_list[i]);
			if (selected_node) {
				editor_selection->add_node(selected_node);
			}
		}
	}

	if (!restoring_scenes) {
		save_editor_layout_delayed();
	}

	return OK;
}

HashMap<StringName, Variant> EditorNode::get_modified_properties_for_node(Node *p_node) {
	HashMap<StringName, Variant> modified_property_map;

	List<PropertyInfo> pinfo;
	p_node->get_property_list(&pinfo);
	for (const PropertyInfo &E : pinfo) {
		if (E.usage & PROPERTY_USAGE_STORAGE) {
			bool is_valid_revert = false;
			Variant revert_value = EditorPropertyRevert::get_property_revert_value(p_node, E.name, &is_valid_revert);
			Variant current_value = p_node->get(E.name);
			if (is_valid_revert) {
				if (PropertyUtils::is_property_value_different(current_value, revert_value)) {
					modified_property_map[E.name] = current_value;
				}
			}
		}
	}

	return modified_property_map;
}

void EditorNode::update_ownership_table_for_addition_node_ancestors(Node *p_current_node, HashMap<Node *, Node *> &p_ownership_table) {
	p_ownership_table.insert(p_current_node, p_current_node->get_owner());

	for (int i = 0; i < p_current_node->get_child_count(); i++) {
		Node *child = p_current_node->get_child(i);
		update_ownership_table_for_addition_node_ancestors(child, p_ownership_table);
	}
}

void EditorNode::update_diff_data_for_node(
		Node *p_edited_scene,
		Node *p_root,
		Node *p_node,
		HashMap<NodePath, ModificationNodeEntry> &p_modification_table,
		List<AdditiveNodeEntry> &p_addition_list) {
	bool node_part_of_subscene = p_node != p_edited_scene &&
			p_edited_scene->get_scene_inherited_state().is_valid() &&
			p_edited_scene->get_scene_inherited_state()->find_node_by_path(p_edited_scene->get_path_to(p_node)) >= 0;

	// Loop through the owners until either we reach the root node or nullptr
	Node *valid_node_owner = p_node->get_owner();
	while (valid_node_owner) {
		if (valid_node_owner == p_root) {
			break;
		}
		valid_node_owner = valid_node_owner->get_owner();
	}

	if ((valid_node_owner == p_root && (p_root != p_edited_scene || !p_edited_scene->get_scene_file_path().is_empty())) || node_part_of_subscene || p_node == p_root) {
		HashMap<StringName, Variant> modified_properties = get_modified_properties_for_node(p_node);

		// Find all valid connections to other nodes.
		List<Connection> connections_to;
		p_node->get_all_signal_connections(&connections_to);

		List<ConnectionWithNodePath> valid_connections_to;
		for (const Connection &c : connections_to) {
			Node *connection_target_node = Object::cast_to<Node>(c.callable.get_object());
			if (connection_target_node) {
				// TODO: add support for reinstating custom callables
				if (!c.callable.is_custom()) {
					ConnectionWithNodePath connection_to;
					connection_to.connection = c;
					connection_to.node_path = p_node->get_path_to(connection_target_node);
					valid_connections_to.push_back(connection_to);
				}
			}
		}

		// Find all valid connections from other nodes.
		List<Connection> connections_from;
		p_node->get_signals_connected_to_this(&connections_from);

		List<Connection> valid_connections_from;
		for (const Connection &c : connections_from) {
			Node *source_node = Object::cast_to<Node>(c.signal.get_object());

			Node *valid_source_owner = nullptr;
			if (source_node) {
				valid_source_owner = source_node->get_owner();
				while (valid_source_owner) {
					if (valid_source_owner == p_root) {
						break;
					}
					valid_source_owner = valid_source_owner->get_owner();
				}
			}

			if (!source_node || valid_source_owner == nullptr) {
				// TODO: add support for reinstating custom callables
				if (!c.callable.is_custom()) {
					valid_connections_from.push_back(c);
				}
			}
		}

		// Find all node groups.
		List<Node::GroupInfo> groups;
		p_node->get_groups(&groups);

		if (!modified_properties.is_empty() || !valid_connections_to.is_empty() || !valid_connections_from.is_empty() || !groups.is_empty()) {
			ModificationNodeEntry modification_node_entry;
			modification_node_entry.property_table = modified_properties;
			modification_node_entry.connections_to = valid_connections_to;
			modification_node_entry.connections_from = valid_connections_from;
			modification_node_entry.groups = groups;

			p_modification_table[p_root->get_path_to(p_node)] = modification_node_entry;
		}
	} else {
		AdditiveNodeEntry new_additive_node_entry;
		new_additive_node_entry.node = p_node;
		new_additive_node_entry.parent = p_root->get_path_to(p_node->get_parent());
		new_additive_node_entry.owner = p_node->get_owner();
		new_additive_node_entry.index = p_node->get_index();

		Node2D *node_2d = Object::cast_to<Node2D>(p_node);
		if (node_2d) {
			new_additive_node_entry.transform_2d = node_2d->get_relative_transform_to_parent(node_2d->get_parent());
		}
		Node3D *node_3d = Object::cast_to<Node3D>(p_node);
		if (node_3d) {
			new_additive_node_entry.transform_3d = node_3d->get_relative_transform(node_3d->get_parent());
		}

		// Gathers the ownership of all ancestor nodes for later use.
		HashMap<Node *, Node *> ownership_table;
		for (int i = 0; i < p_node->get_child_count(); i++) {
			Node *child = p_node->get_child(i);
			update_ownership_table_for_addition_node_ancestors(child, ownership_table);
		}

		new_additive_node_entry.ownership_table = ownership_table;

		p_addition_list.push_back(new_additive_node_entry);

		return;
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *child = p_node->get_child(i);
		update_diff_data_for_node(p_edited_scene, p_root, child, p_modification_table, p_addition_list);
	}
}
//

void EditorNode::open_request(const String &p_path) {
	if (!opening_prev) {
		List<String>::Element *prev_scene_item = previous_scenes.find(p_path);
		if (prev_scene_item != nullptr) {
			prev_scene_item->erase();
		}
	}

	load_scene(p_path); // As it will be opened in separate tab.
}

bool EditorNode::has_previous_scenes() const {
	return !previous_scenes.is_empty();
}

void EditorNode::edit_foreign_resource(Ref<Resource> p_resource) {
	load_scene(p_resource->get_path().get_slice("::", 0));
	InspectorDock::get_singleton()->call_deferred("edit_resource", p_resource);
}

bool EditorNode::is_resource_read_only(Ref<Resource> p_resource, bool p_foreign_resources_are_writable) {
	ERR_FAIL_COND_V(p_resource.is_null(), false);

	String path = p_resource->get_path();
	if (!path.is_resource_file()) {
		// If the resource name contains '::', that means it is a subresource embedded in another resource.
		int srpos = path.find("::");
		if (srpos != -1) {
			String base = path.substr(0, srpos);
			// If the base resource is a packed scene, we treat it as read-only if it is not the currently edited scene.
			if (ResourceLoader::get_resource_type(base) == "PackedScene") {
				if (!get_tree()->get_edited_scene_root() || get_tree()->get_edited_scene_root()->get_scene_file_path() != base) {
					// If we have not flagged foreign resources as writable or the base scene the resource is
					// part was imported, it can be considered read-only.
					if (!p_foreign_resources_are_writable || FileAccess::exists(base + ".import")) {
						return true;
					}
				}
			} else {
				// If a corresponding .import file exists for the base file, we assume it to be imported and should therefore treated as read-only.
				if (FileAccess::exists(base + ".import")) {
					return true;
				}
			}
		}
	} else if (FileAccess::exists(path + ".import")) {
		// The resource is not a subresource, but if it has an .import file, it's imported so treat it as read only.
		return true;
	}

	return false;
}

void EditorNode::request_instantiate_scene(const String &p_path) {
	SceneTreeDock::get_singleton()->instantiate(p_path);
}

void EditorNode::request_instantiate_scenes(const Vector<String> &p_files) {
	SceneTreeDock::get_singleton()->instantiate_scenes(p_files);
}

void EditorNode::_inherit_request(String p_file) {
	current_menu_option = FILE_NEW_INHERITED_SCENE;
	_dialog_action(p_file);
}

void EditorNode::_instantiate_request(const Vector<String> &p_files) {
	request_instantiate_scenes(p_files);
}

void EditorNode::_close_messages() {
	old_split_ofs = center_split->get_split_offset();
	center_split->set_split_offset(0);
}

void EditorNode::_show_messages() {
	center_split->set_split_offset(old_split_ofs);
}

void EditorNode::_add_to_recent_scenes(const String &p_scene) {
	Array rc = EditorSettings::get_singleton()->get_project_metadata("recent_files", "scenes", Array());
	if (rc.has(p_scene)) {
		rc.erase(p_scene);
	}
	rc.push_front(p_scene);
	if (rc.size() > 10) {
		rc.resize(10);
	}

	EditorSettings::get_singleton()->set_project_metadata("recent_files", "scenes", rc);
	_update_recent_scenes();
}

void EditorNode::_open_recent_scene(int p_idx) {
	if (p_idx == recent_scenes->get_item_count() - 1) {
		EditorSettings::get_singleton()->set_project_metadata("recent_files", "scenes", Array());
		call_deferred(SNAME("_update_recent_scenes"));
	} else {
		Array rc = EditorSettings::get_singleton()->get_project_metadata("recent_files", "scenes", Array());
		ERR_FAIL_INDEX(p_idx, rc.size());

		if (load_scene(rc[p_idx]) != OK) {
			rc.remove_at(p_idx);
			EditorSettings::get_singleton()->set_project_metadata("recent_files", "scenes", rc);
			_update_recent_scenes();
		}
	}
}

void EditorNode::_update_recent_scenes() {
	Array rc = EditorSettings::get_singleton()->get_project_metadata("recent_files", "scenes", Array());
	recent_scenes->clear();

	String path;
	for (int i = 0; i < rc.size(); i++) {
		path = rc[i];
		recent_scenes->add_item(path.replace("res://", ""), i);
	}

	recent_scenes->add_separator();
	recent_scenes->add_shortcut(ED_SHORTCUT("editor/clear_recent", TTR("Clear Recent Scenes")));
	recent_scenes->reset_size();
}

void EditorNode::_quick_opened() {
	Vector<String> files = quick_open->get_selected_files();

	bool open_scene_dialog = quick_open->get_base_type() == "PackedScene";
	for (int i = 0; i < files.size(); i++) {
		String res_path = files[i];
		if (open_scene_dialog || ClassDB::is_parent_class(ResourceLoader::get_resource_type(res_path), "PackedScene")) {
			open_request(res_path);
		} else {
			load_resource(res_path);
		}
	}
}

void EditorNode::_project_run_started() {
	if (bool(EDITOR_GET("run/output/always_clear_output_on_play"))) {
		log->clear();
	}

	if (bool(EDITOR_GET("run/output/always_open_output_on_play"))) {
		make_bottom_panel_item_visible(log);
	}
}

void EditorNode::_project_run_stopped() {
	if (!bool(EDITOR_GET("run/output/always_close_output_on_stop"))) {
		return;
	}

	for (int i = 0; i < bottom_panel_items.size(); i++) {
		if (bottom_panel_items[i].control == log) {
			_bottom_panel_switch(false, i);
			break;
		}
	}
}

void EditorNode::notify_all_debug_sessions_exited() {
	project_run_bar->stop_playing();
}

void EditorNode::add_io_error(const String &p_error) {
	DEV_ASSERT(Thread::get_caller_id() == Thread::get_main_id());
	singleton->load_errors->add_image(singleton->theme->get_icon(SNAME("Error"), EditorStringName(EditorIcons)));
	singleton->load_errors->add_text(p_error + "\n");
	EditorInterface::get_singleton()->popup_dialog_centered_ratio(singleton->load_error_dialog, 0.5);
}

void EditorNode::add_io_warning(const String &p_warning) {
	DEV_ASSERT(Thread::get_caller_id() == Thread::get_main_id());
	singleton->load_errors->add_image(singleton->theme->get_icon(SNAME("Warning"), EditorStringName(EditorIcons)));
	singleton->load_errors->add_text(p_warning + "\n");
	EditorInterface::get_singleton()->popup_dialog_centered_ratio(singleton->load_error_dialog, 0.5);
}

bool EditorNode::_find_scene_in_use(Node *p_node, const String &p_path) const {
	if (p_node->get_scene_file_path() == p_path) {
		return true;
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		if (_find_scene_in_use(p_node->get_child(i), p_path)) {
			return true;
		}
	}

	return false;
}

bool EditorNode::is_scene_in_use(const String &p_path) {
	Node *es = get_edited_scene();
	if (es) {
		return _find_scene_in_use(es, p_path);
	}
	return false;
}

OS::ProcessID EditorNode::has_child_process(OS::ProcessID p_pid) const {
	return project_run_bar->has_child_process(p_pid);
}

void EditorNode::stop_child_process(OS::ProcessID p_pid) {
	project_run_bar->stop_child_process(p_pid);
}

Ref<Script> EditorNode::get_object_custom_type_base(const Object *p_object) const {
	ERR_FAIL_NULL_V(p_object, nullptr);

	Ref<Script> scr = p_object->get_script();

	if (scr.is_valid()) {
		// Uncommenting would break things! Consider adding a parameter if you need it.
		// StringName name = EditorNode::get_editor_data().script_class_get_name(base_script->get_path());
		// if (name != StringName()) {
		// 	return name;
		// }

		// TODO: Should probably be deprecated in 4.x
		StringName base = scr->get_instance_base_type();
		if (base != StringName() && EditorNode::get_editor_data().get_custom_types().has(base)) {
			const Vector<EditorData::CustomType> &types = EditorNode::get_editor_data().get_custom_types()[base];

			Ref<Script> base_scr = scr;
			while (base_scr.is_valid()) {
				for (int i = 0; i < types.size(); ++i) {
					if (types[i].script == base_scr) {
						return types[i].script;
					}
				}
				base_scr = base_scr->get_base_script();
			}
		}
	}

	return nullptr;
}

StringName EditorNode::get_object_custom_type_name(const Object *p_object) const {
	ERR_FAIL_NULL_V(p_object, StringName());

	Ref<Script> scr = p_object->get_script();
	if (scr.is_null() && Object::cast_to<Script>(p_object)) {
		scr = p_object;
	}

	if (scr.is_valid()) {
		Ref<Script> base_scr = scr;
		while (base_scr.is_valid()) {
			StringName name = EditorNode::get_editor_data().script_class_get_name(base_scr->get_path());
			if (name != StringName()) {
				return name;
			}

			// TODO: Should probably be deprecated in 4.x.
			StringName base = base_scr->get_instance_base_type();
			if (base != StringName() && EditorNode::get_editor_data().get_custom_types().has(base)) {
				const Vector<EditorData::CustomType> &types = EditorNode::get_editor_data().get_custom_types()[base];
				for (int i = 0; i < types.size(); ++i) {
					if (types[i].script == base_scr) {
						return types[i].name;
					}
				}
			}
			base_scr = base_scr->get_base_script();
		}
	}

	return StringName();
}

void EditorNode::_pick_main_scene_custom_action(const String &p_custom_action_name) {
	if (p_custom_action_name == "select_current") {
		Node *scene = editor_data.get_edited_scene_root();

		if (!scene) {
			show_accept(TTR("There is no defined scene to run."), TTR("OK"));
			return;
		}

		pick_main_scene->hide();

		if (!FileAccess::exists(scene->get_scene_file_path())) {
			current_menu_option = FILE_SAVE_AND_RUN_MAIN_SCENE;
			_menu_option_confirm(FILE_SAVE_AS_SCENE, true);
			file->set_title(TTR("Save scene before running..."));
		} else {
			current_menu_option = SETTINGS_PICK_MAIN_SCENE;
			_dialog_action(scene->get_scene_file_path());
		}
	}
}

Ref<Texture2D> EditorNode::_get_class_or_script_icon(const String &p_class, const Ref<Script> &p_script, const String &p_fallback, bool p_fallback_script_to_theme) {
	ERR_FAIL_COND_V_MSG(p_class.is_empty(), nullptr, "Class name cannot be empty.");
	EditorData &ed = EditorNode::get_editor_data();

	// Check for a script icon first.
	if (p_script.is_valid()) {
		Ref<Texture2D> script_icon = ed.get_script_icon(p_script);
		if (script_icon.is_valid()) {
			return script_icon;
		}

		if (p_fallback_script_to_theme) {
			// Look for the native base type in the editor theme. This is relevant for
			// scripts extending other scripts and for built-in classes.
			String script_class_name = p_script->get_language()->get_global_class_name(p_script->get_path());
			String base_type = ScriptServer::get_global_class_native_base(script_class_name);
			if (theme.is_valid() && theme->has_icon(base_type, EditorStringName(EditorIcons))) {
				return theme->get_icon(base_type, EditorStringName(EditorIcons));
			}
		}
	}

	// Script was not valid or didn't yield any useful values, try the class name
	// directly.

	// Check if the class name is an extension-defined type.
	Ref<Texture2D> ext_icon = ed.extension_class_get_icon(p_class);
	if (ext_icon.is_valid()) {
		return ext_icon;
	}

	// Check if the class name is a custom type.
	// TODO: Should probably be deprecated in 4.x
	const EditorData::CustomType *ctype = ed.get_custom_type_by_name(p_class);
	if (ctype && ctype->icon.is_valid()) {
		return ctype->icon;
	}

	// Look up the class name or the fallback name in the editor theme.
	// This is only relevant for built-in classes.
	if (theme.is_valid()) {
		if (theme->has_icon(p_class, EditorStringName(EditorIcons))) {
			return theme->get_icon(p_class, EditorStringName(EditorIcons));
		}

		if (!p_fallback.is_empty() && theme->has_icon(p_fallback, EditorStringName(EditorIcons))) {
			return theme->get_icon(p_fallback, EditorStringName(EditorIcons));
		}

		// If the fallback is empty or wasn't found, use the default fallback.
		if (ClassDB::class_exists(p_class)) {
			bool instantiable = !ClassDB::is_virtual(p_class) && ClassDB::can_instantiate(p_class);
			if (ClassDB::is_parent_class(p_class, SNAME("Node"))) {
				return theme->get_icon(instantiable ? "Node" : "NodeDisabled", EditorStringName(EditorIcons));
			} else {
				return theme->get_icon(instantiable ? "Object" : "ObjectDisabled", EditorStringName(EditorIcons));
			}
		}
	}

	return nullptr;
}

Ref<Texture2D> EditorNode::get_object_icon(const Object *p_object, const String &p_fallback) {
	ERR_FAIL_NULL_V_MSG(p_object, nullptr, "Object cannot be null.");

	Ref<Script> scr = p_object->get_script();
	if (scr.is_null() && p_object->is_class("Script")) {
		scr = p_object;
	}

	return _get_class_or_script_icon(p_object->get_class(), scr, p_fallback);
}

Ref<Texture2D> EditorNode::get_class_icon(const String &p_class, const String &p_fallback) {
	ERR_FAIL_COND_V_MSG(p_class.is_empty(), nullptr, "Class name cannot be empty.");

	Ref<Script> scr;
	if (ScriptServer::is_global_class(p_class)) {
		scr = EditorNode::get_editor_data().script_class_load_script(p_class);
	}

	return _get_class_or_script_icon(p_class, scr, p_fallback, true);
}

bool EditorNode::is_object_of_custom_type(const Object *p_object, const StringName &p_class) {
	ERR_FAIL_NULL_V(p_object, false);

	Ref<Script> scr = p_object->get_script();
	if (scr.is_null() && Object::cast_to<Script>(p_object)) {
		scr = p_object;
	}

	if (scr.is_valid()) {
		Ref<Script> base_script = scr;
		while (base_script.is_valid()) {
			StringName name = EditorNode::get_editor_data().script_class_get_name(base_script->get_path());
			if (name == p_class) {
				return true;
			}
			base_script = base_script->get_base_script();
		}
	}
	return false;
}

void EditorNode::progress_add_task(const String &p_task, const String &p_label, int p_steps, bool p_can_cancel) {
	if (singleton->cmdline_export_mode) {
		print_line(p_task + ": begin: " + p_label + " steps: " + itos(p_steps));
	} else if (singleton->progress_dialog) {
		singleton->progress_dialog->add_task(p_task, p_label, p_steps, p_can_cancel);
	}
}

bool EditorNode::progress_task_step(const String &p_task, const String &p_state, int p_step, bool p_force_refresh) {
	if (singleton->cmdline_export_mode) {
		print_line("\t" + p_task + ": step " + itos(p_step) + ": " + p_state);
		return false;
	} else if (singleton->progress_dialog) {
		return singleton->progress_dialog->task_step(p_task, p_state, p_step, p_force_refresh);
	} else {
		return false;
	}
}

void EditorNode::progress_end_task(const String &p_task) {
	if (singleton->cmdline_export_mode) {
		print_line(p_task + ": end");
	} else if (singleton->progress_dialog) {
		singleton->progress_dialog->end_task(p_task);
	}
}

void EditorNode::progress_add_task_bg(const String &p_task, const String &p_label, int p_steps) {
	singleton->progress_hb->add_task(p_task, p_label, p_steps);
}

void EditorNode::progress_task_step_bg(const String &p_task, int p_step) {
	singleton->progress_hb->task_step(p_task, p_step);
}

void EditorNode::progress_end_task_bg(const String &p_task) {
	singleton->progress_hb->end_task(p_task);
}

String EditorNode::_get_system_info() const {
	String distribution_name = OS::get_singleton()->get_distribution_name();
	if (distribution_name.is_empty()) {
		distribution_name = OS::get_singleton()->get_name();
	}
	if (distribution_name.is_empty()) {
		distribution_name = "Other";
	}
	const String distribution_version = OS::get_singleton()->get_version();

	String godot_version = "Godot v" + String(VERSION_FULL_CONFIG);
	if (String(VERSION_BUILD) != "official") {
		String hash = String(VERSION_HASH);
		hash = hash.is_empty() ? String("unknown") : vformat("(%s)", hash.left(9));
		godot_version += " " + hash;
	}

#ifdef LINUXBSD_ENABLED
	const String display_server = OS::get_singleton()->get_environment("XDG_SESSION_TYPE").capitalize().replace(" ", ""); // `replace` is necessary, because `capitalize` introduces a whitespace between "x" and "11".
#endif // LINUXBSD_ENABLED
	String driver_name = GLOBAL_GET("rendering/rendering_device/driver");
	String rendering_method = GLOBAL_GET("rendering/renderer/rendering_method");

	const String rendering_device_name = RenderingServer::get_singleton()->get_video_adapter_name();

	RenderingDevice::DeviceType device_type = RenderingServer::get_singleton()->get_video_adapter_type();
	String device_type_string;
	switch (device_type) {
		case RenderingDevice::DeviceType::DEVICE_TYPE_INTEGRATED_GPU:
			device_type_string = "integrated";
			break;
		case RenderingDevice::DeviceType::DEVICE_TYPE_DISCRETE_GPU:
			device_type_string = "dedicated";
			break;
		case RenderingDevice::DeviceType::DEVICE_TYPE_VIRTUAL_GPU:
			device_type_string = "virtual";
			break;
		case RenderingDevice::DeviceType::DEVICE_TYPE_CPU:
			device_type_string = "(software emulation on CPU)";
			break;
		case RenderingDevice::DeviceType::DEVICE_TYPE_OTHER:
		case RenderingDevice::DeviceType::DEVICE_TYPE_MAX:
			break; // Can't happen, but silences warning for DEVICE_TYPE_MAX
	}

	const Vector<String> video_adapter_driver_info = OS::get_singleton()->get_video_adapter_driver_info();

	const String processor_name = OS::get_singleton()->get_processor_name();
	const int processor_count = OS::get_singleton()->get_processor_count();

	// Prettify
	if (rendering_method == "forward_plus") {
		rendering_method = "Forward+";
	} else if (rendering_method == "mobile") {
		rendering_method = "Mobile";
	} else if (rendering_method == "gl_compatibility") {
		rendering_method = "Compatibility";
		driver_name = GLOBAL_GET("rendering/gl_compatibility/driver");
	}
	if (driver_name == "vulkan") {
		driver_name = "Vulkan";
	} else if (driver_name.begins_with("opengl3")) {
		driver_name = "GLES3";
	}

	// Join info.
	Vector<String> info;
	info.push_back(godot_version);
	if (!distribution_version.is_empty()) {
		info.push_back(distribution_name + " " + distribution_version);
	} else {
		info.push_back(distribution_name);
	}
#ifdef LINUXBSD_ENABLED
	if (!display_server.is_empty()) {
		info.push_back(display_server);
	}
#endif // LINUXBSD_ENABLED
	info.push_back(vformat("%s (%s)", driver_name, rendering_method));

	String graphics;
	if (!device_type_string.is_empty()) {
		graphics = device_type_string + " ";
	}
	graphics += rendering_device_name;
	if (video_adapter_driver_info.size() == 2) { // This vector is always either of length 0 or 2.
		String vad_name = video_adapter_driver_info[0];
		String vad_version = video_adapter_driver_info[1]; // Version could be potentially empty on Linux/BSD.
		if (!vad_version.is_empty()) {
			graphics += vformat(" (%s; %s)", vad_name, vad_version);
		} else {
			graphics += vformat(" (%s)", vad_name);
		}
	}
	info.push_back(graphics);

	info.push_back(vformat("%s (%d Threads)", processor_name, processor_count));

	return String(" - ").join(info);
}

Ref<Texture2D> EditorNode::_file_dialog_get_icon(const String &p_path) {
	EditorFileSystemDirectory *efsd = EditorFileSystem::get_singleton()->get_filesystem_path(p_path.get_base_dir());
	if (efsd) {
		String file = p_path.get_file();
		for (int i = 0; i < efsd->get_file_count(); i++) {
			if (efsd->get_file(i) == file) {
				String type = efsd->get_file_type(i);

				if (singleton->icon_type_cache.has(type)) {
					return singleton->icon_type_cache[type];
				} else {
					return singleton->icon_type_cache["Object"];
				}
			}
		}
	}

	return singleton->icon_type_cache["Object"];
}

void EditorNode::_build_icon_type_cache() {
	List<StringName> tl;
	theme->get_icon_list(EditorStringName(EditorIcons), &tl);
	for (const StringName &E : tl) {
		if (!ClassDB::class_exists(E)) {
			continue;
		}
		icon_type_cache[E] = theme->get_icon(E, EditorStringName(EditorIcons));
	}
}

void EditorNode::_enable_pending_addons() {
	for (uint32_t i = 0; i < pending_addons.size(); i++) {
		set_addon_plugin_enabled(pending_addons[i], true);
	}
	pending_addons.clear();
}

void EditorNode::_file_dialog_register(FileDialog *p_dialog) {
	singleton->file_dialogs.insert(p_dialog);
}

void EditorNode::_file_dialog_unregister(FileDialog *p_dialog) {
	singleton->file_dialogs.erase(p_dialog);
}

void EditorNode::_editor_file_dialog_register(EditorFileDialog *p_dialog) {
	singleton->editor_file_dialogs.insert(p_dialog);
}

void EditorNode::_editor_file_dialog_unregister(EditorFileDialog *p_dialog) {
	singleton->editor_file_dialogs.erase(p_dialog);
}

Vector<EditorNodeInitCallback> EditorNode::_init_callbacks;

void EditorNode::_begin_first_scan() {
	// In headless mode, scan right away.
	// This allows users to continue using `godot --headless --editor --quit` to prepare a project.
	if (!DisplayServer::get_singleton()->window_can_draw()) {
		OS::get_singleton()->benchmark_begin_measure("editor_scan_and_import");
		EditorFileSystem::get_singleton()->scan();
		return;
	}

	if (!waiting_for_first_scan) {
		return;
	}
	requested_first_scan = true;
}

Error EditorNode::export_preset(const String &p_preset, const String &p_path, bool p_debug, bool p_pack_only) {
	export_defer.preset = p_preset;
	export_defer.path = p_path;
	export_defer.debug = p_debug;
	export_defer.pack_only = p_pack_only;
	cmdline_export_mode = true;
	return OK;
}

bool EditorNode::is_project_exporting() const {
	return project_export && project_export->is_exporting();
}

void EditorNode::show_accept(const String &p_text, const String &p_title) {
	current_menu_option = -1;
	if (accept) {
		accept->set_ok_button_text(p_title);
		accept->set_text(p_text);
		EditorInterface::get_singleton()->popup_dialog_centered(accept);
	}
}

void EditorNode::show_save_accept(const String &p_text, const String &p_title) {
	current_menu_option = -1;
	if (save_accept) {
		save_accept->set_ok_button_text(p_title);
		save_accept->set_text(p_text);
		EditorInterface::get_singleton()->popup_dialog_centered(save_accept);
	}
}

void EditorNode::show_warning(const String &p_text, const String &p_title) {
	if (warning) {
		warning->set_text(p_text);
		warning->set_title(p_title);
		EditorInterface::get_singleton()->popup_dialog_centered(warning);
	} else {
		WARN_PRINT(p_title + " " + p_text);
	}
}

void EditorNode::_copy_warning(const String &p_str) {
	DisplayServer::get_singleton()->clipboard_set(warning->get_text());
}

void EditorNode::_dock_floating_close_request(WindowWrapper *p_wrapper) {
	int dock_slot_num = p_wrapper->get_meta("dock_slot");
	int dock_slot_index = p_wrapper->get_meta("dock_index");

	// Give back the dock to the original owner.
	Control *dock = p_wrapper->release_wrapped_control();

	int target_index = MIN(dock_slot_index, dock_slot[dock_slot_num]->get_tab_count());
	dock_slot[dock_slot_num]->add_child(dock);
	dock_slot[dock_slot_num]->move_child(dock, target_index);
	dock_slot[dock_slot_num]->set_current_tab(target_index);

	floating_docks.erase(p_wrapper);
	p_wrapper->queue_free();

	_update_dock_slots_visibility(true);

	_edit_current();
}

void EditorNode::_dock_make_selected_float() {
	Control *dock = dock_slot[dock_popup_selected_idx]->get_current_tab_control();
	_dock_make_float(dock, dock_popup_selected_idx);

	dock_select_popup->hide();
	_edit_current();
}

void EditorNode::_dock_make_float(Control *p_dock, int p_slot_index, bool p_show_window) {
	ERR_FAIL_NULL(p_dock);

	Size2 borders = Size2(4, 4) * EDSCALE;
	// Remember size and position before removing it from the main window.
	Size2 dock_size = p_dock->get_size() + borders * 2;
	Point2 dock_screen_pos = p_dock->get_screen_position();

	int dock_index = p_dock->get_index() - 1;
	dock_slot[p_slot_index]->remove_child(p_dock);

	WindowWrapper *wrapper = memnew(WindowWrapper);
	wrapper->set_window_title(vformat(TTR("%s - Godot Engine"), p_dock->get_name()));
	wrapper->set_margins_enabled(true);

	gui_base->add_child(wrapper);

	wrapper->set_wrapped_control(p_dock);
	wrapper->set_meta("dock_slot", p_slot_index);
	wrapper->set_meta("dock_index", dock_index);
	wrapper->set_meta("dock_name", p_dock->get_name().operator String());
	p_dock->show();

	wrapper->connect("window_close_requested", callable_mp(this, &EditorNode::_dock_floating_close_request).bind(wrapper));

	dock_select_popup->hide();

	if (p_show_window) {
		wrapper->restore_window(Rect2i(dock_screen_pos, dock_size), get_window()->get_current_screen());
	}

	_update_dock_slots_visibility(true);

	floating_docks.push_back(wrapper);

	_edit_current();
}

void EditorNode::_dock_select_input(const Ref<InputEvent> &p_input) {
	Ref<InputEventMouse> me = p_input;

	if (me.is_valid()) {
		Vector2 point = me->get_position();

		int nrect = -1;
		for (int i = 0; i < DOCK_SLOT_MAX; i++) {
			if (dock_select_rect[i].has_point(point)) {
				nrect = i;
				break;
			}
		}

		if (nrect != dock_select_rect_over_idx) {
			dock_select->queue_redraw();
			dock_select_rect_over_idx = nrect;
		}

		if (nrect == -1) {
			return;
		}

		Ref<InputEventMouseButton> mb = me;

		if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT && mb->is_pressed() && dock_popup_selected_idx != nrect) {
			dock_slot[nrect]->move_tab_from_tab_container(dock_slot[dock_popup_selected_idx], dock_slot[dock_popup_selected_idx]->get_current_tab(), dock_slot[nrect]->get_tab_count());

			if (dock_slot[dock_popup_selected_idx]->get_tab_count() == 0) {
				dock_slot[dock_popup_selected_idx]->hide();
			} else {
				dock_slot[dock_popup_selected_idx]->set_current_tab(0);
			}

			dock_popup_selected_idx = nrect;
			dock_slot[nrect]->show();
			dock_select->queue_redraw();

			_update_dock_slots_visibility(true);

			_edit_current();
			_save_editor_layout();
		}
	}
}

void EditorNode::_dock_popup_exit() {
	dock_select_rect_over_idx = -1;
	dock_select->queue_redraw();
}

void EditorNode::_dock_pre_popup(int p_which) {
	dock_popup_selected_idx = p_which;
}

void EditorNode::_dock_move_left() {
	if (dock_popup_selected_idx < 0 || dock_popup_selected_idx >= DOCK_SLOT_MAX) {
		return;
	}
	Control *current_ctl = dock_slot[dock_popup_selected_idx]->get_tab_control(dock_slot[dock_popup_selected_idx]->get_current_tab());
	Control *prev_ctl = dock_slot[dock_popup_selected_idx]->get_tab_control(dock_slot[dock_popup_selected_idx]->get_current_tab() - 1);
	if (!current_ctl || !prev_ctl) {
		return;
	}
	dock_slot[dock_popup_selected_idx]->move_child(current_ctl, prev_ctl->get_index(false));
	dock_select->queue_redraw();
	_edit_current();
	_save_editor_layout();
}

void EditorNode::_dock_move_right() {
	Control *current_ctl = dock_slot[dock_popup_selected_idx]->get_tab_control(dock_slot[dock_popup_selected_idx]->get_current_tab());
	Control *next_ctl = dock_slot[dock_popup_selected_idx]->get_tab_control(dock_slot[dock_popup_selected_idx]->get_current_tab() + 1);
	if (!current_ctl || !next_ctl) {
		return;
	}
	dock_slot[dock_popup_selected_idx]->move_child(next_ctl, current_ctl->get_index(false));
	dock_select->queue_redraw();
	_edit_current();
	_save_editor_layout();
}

void EditorNode::_dock_select_draw() {
	Size2 s = dock_select->get_size();
	s.y /= 2.0;
	s.x /= 6.0;

	Color used = Color(0.6, 0.6, 0.6, 0.8);
	Color used_selected = Color(0.8, 0.8, 0.8, 0.8);
	Color tab_selected = theme->get_color(SNAME("mono_color"), EditorStringName(Editor));
	Color unused = used;
	unused.a = 0.4;
	Color unusable = unused;
	unusable.a = 0.1;

	Rect2 unr(s.x * 2, 0, s.x * 2, s.y * 2);
	unr.position += Vector2(2, 5);
	unr.size -= Vector2(4, 7);

	dock_select->draw_rect(unr, unusable);

	dock_tab_move_left->set_disabled(true);
	dock_tab_move_right->set_disabled(true);

	if (dock_popup_selected_idx != -1 && dock_slot[dock_popup_selected_idx]->get_tab_count()) {
		dock_tab_move_left->set_disabled(dock_slot[dock_popup_selected_idx]->get_current_tab() == 0);
		dock_tab_move_right->set_disabled(dock_slot[dock_popup_selected_idx]->get_current_tab() >= dock_slot[dock_popup_selected_idx]->get_tab_count() - 1);
	}

	for (int i = 0; i < DOCK_SLOT_MAX; i++) {
		Vector2 ofs;

		switch (i) {
			case DOCK_SLOT_LEFT_UL: {
			} break;
			case DOCK_SLOT_LEFT_BL: {
				ofs.y += s.y;
			} break;
			case DOCK_SLOT_LEFT_UR: {
				ofs.x += s.x;
			} break;
			case DOCK_SLOT_LEFT_BR: {
				ofs += s;
			} break;
			case DOCK_SLOT_RIGHT_UL: {
				ofs.x += s.x * 4;
			} break;
			case DOCK_SLOT_RIGHT_BL: {
				ofs.x += s.x * 4;
				ofs.y += s.y;

			} break;
			case DOCK_SLOT_RIGHT_UR: {
				ofs.x += s.x * 4;
				ofs.x += s.x;

			} break;
			case DOCK_SLOT_RIGHT_BR: {
				ofs.x += s.x * 4;
				ofs += s;

			} break;
		}

		Rect2 r(ofs, s);
		dock_select_rect[i] = r;
		r.position += Vector2(2, 5);
		r.size -= Vector2(4, 7);

		if (i == dock_select_rect_over_idx) {
			dock_select->draw_rect(r, used_selected);
		} else if (dock_slot[i]->get_tab_count() == 0) {
			dock_select->draw_rect(r, unused);
		} else {
			dock_select->draw_rect(r, used);
		}

		for (int j = 0; j < MIN(3, dock_slot[i]->get_tab_count()); j++) {
			int xofs = (r.size.width / 3) * j;
			Color c = used;
			if (i == dock_popup_selected_idx && (dock_slot[i]->get_current_tab() > 3 || dock_slot[i]->get_current_tab() == j)) {
				c = tab_selected;
			}
			dock_select->draw_rect(Rect2(2 + ofs.x + xofs, ofs.y, r.size.width / 3 - 1, 3), c);
		}
	}
}

void EditorNode::_save_editor_layout() {
	if (waiting_for_first_scan) {
		return; // Scanning, do not touch docks.
	}
	Ref<ConfigFile> config;
	config.instantiate();
	// Load and amend existing config if it exists.
	config->load(EditorPaths::get_singleton()->get_project_settings_dir().path_join("editor_layout.cfg"));

	_save_docks_to_config(config, "docks");
	_save_open_scenes_to_config(config);
	_save_central_editor_layout_to_config(config);
	editor_data.get_plugin_window_layout(config);

	config->save(EditorPaths::get_singleton()->get_project_settings_dir().path_join("editor_layout.cfg"));
}

void EditorNode::_save_docks_to_config(Ref<ConfigFile> p_layout, const String &p_section) {
	for (int i = 0; i < DOCK_SLOT_MAX; i++) {
		String names;
		for (int j = 0; j < dock_slot[i]->get_tab_count(); j++) {
			String name = dock_slot[i]->get_tab_control(j)->get_name();
			if (!names.is_empty()) {
				names += ",";
			}
			names += name;
		}

		String config_key = "dock_" + itos(i + 1);

		if (p_layout->has_section_key(p_section, config_key)) {
			p_layout->erase_section_key(p_section, config_key);
		}

		if (!names.is_empty()) {
			p_layout->set_value(p_section, config_key, names);
		}

		int selected_tab_idx = dock_slot[i]->get_current_tab();
		if (selected_tab_idx >= 0) {
			p_layout->set_value(p_section, "dock_" + itos(i + 1) + "_selected_tab_idx", selected_tab_idx);
		}
	}

	Dictionary floating_docks_dump;

	for (WindowWrapper *wrapper : floating_docks) {
		Control *dock = wrapper->get_wrapped_control();

		Dictionary dock_dump;
		dock_dump["window_rect"] = wrapper->get_window_rect();

		int screen = wrapper->get_window_screen();
		dock_dump["window_screen"] = wrapper->get_window_screen();
		dock_dump["window_screen_rect"] = DisplayServer::get_singleton()->screen_get_usable_rect(screen);

		String name = dock->get_name();
		floating_docks_dump[name] = dock_dump;

		int dock_slot_id = wrapper->get_meta("dock_slot");
		String config_key = "dock_" + itos(dock_slot_id + 1);

		String names = p_layout->get_value(p_section, config_key, "");
		if (names.is_empty()) {
			names = name;
		} else {
			names += "," + name;
		}
		p_layout->set_value(p_section, config_key, names);
	}

	p_layout->set_value(p_section, "dock_floating", floating_docks_dump);

	for (int i = 0; i < vsplits.size(); i++) {
		if (vsplits[i]->is_visible_in_tree()) {
			p_layout->set_value(p_section, "dock_split_" + itos(i + 1), vsplits[i]->get_split_offset());
		}
	}

	for (int i = 0; i < hsplits.size(); i++) {
		p_layout->set_value(p_section, "dock_hsplit_" + itos(i + 1), hsplits[i]->get_split_offset());
	}

	// Save FileSystemDock state.

	p_layout->set_value(p_section, "dock_filesystem_split", FileSystemDock::get_singleton()->get_split_offset());
	p_layout->set_value(p_section, "dock_filesystem_display_mode", FileSystemDock::get_singleton()->get_display_mode());
	p_layout->set_value(p_section, "dock_filesystem_file_sort", FileSystemDock::get_singleton()->get_file_sort());
	p_layout->set_value(p_section, "dock_filesystem_file_list_display_mode", FileSystemDock::get_singleton()->get_file_list_display_mode());
	PackedStringArray selected_files = FileSystemDock::get_singleton()->get_selected_paths();
	p_layout->set_value(p_section, "dock_filesystem_selected_paths", selected_files);
	Vector<String> uncollapsed_paths = FileSystemDock::get_singleton()->get_uncollapsed_paths();
	p_layout->set_value(p_section, "dock_filesystem_uncollapsed_paths", uncollapsed_paths);
}

void EditorNode::_save_open_scenes_to_config(Ref<ConfigFile> p_layout) {
	PackedStringArray scenes;
	for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {
		String path = editor_data.get_scene_path(i);
		if (path.is_empty()) {
			continue;
		}
		scenes.push_back(path);
	}
	p_layout->set_value(EDITOR_NODE_CONFIG_SECTION, "open_scenes", scenes);

	String currently_edited_scene_path = editor_data.get_scene_path(editor_data.get_edited_scene());
	p_layout->set_value(EDITOR_NODE_CONFIG_SECTION, "current_scene", currently_edited_scene_path);
}

void EditorNode::save_editor_layout_delayed() {
	editor_layout_save_delay_timer->start();
}

void EditorNode::_dock_split_dragged(int ofs) {
	editor_layout_save_delay_timer->start();
}

void EditorNode::_load_editor_layout() {
	Ref<ConfigFile> config;
	config.instantiate();
	Error err = config->load(EditorPaths::get_singleton()->get_project_settings_dir().path_join("editor_layout.cfg"));
	if (err != OK) { // No config.
		// If config is not found, expand the res:// folder and favorites by default.
		TreeItem *root = FileSystemDock::get_singleton()->get_tree_control()->get_item_with_metadata("res://", 0);
		if (root) {
			root->set_collapsed(false);
		}

		TreeItem *favorites = FileSystemDock::get_singleton()->get_tree_control()->get_item_with_metadata("Favorites", 0);
		if (favorites) {
			favorites->set_collapsed(false);
		}

		if (overridden_default_layout >= 0) {
			_layout_menu_option(overridden_default_layout);
		}
		return;
	}

	_load_docks_from_config(config, "docks");
	_load_open_scenes_from_config(config);
	_load_central_editor_layout_from_config(config);

	editor_data.set_plugin_window_layout(config);
}

void EditorNode::_update_dock_slots_visibility(bool p_keep_selected_tabs) {
	if (!docks_visible) {
		for (int i = 0; i < DOCK_SLOT_MAX; i++) {
			dock_slot[i]->hide();
		}

		for (int i = 0; i < vsplits.size(); i++) {
			vsplits[i]->hide();
		}

		right_hsplit->hide();
	} else {
		for (int i = 0; i < DOCK_SLOT_MAX; i++) {
			int first_tab_visible = -1;
			for (int j = 0; j < dock_slot[i]->get_tab_count(); j++) {
				if (!dock_slot[i]->is_tab_hidden(j)) {
					first_tab_visible = j;
					break;
				}
			}
			if (first_tab_visible >= 0) {
				dock_slot[i]->show();
				if (p_keep_selected_tabs) {
					int current_tab = dock_slot[i]->get_current_tab();
					if (dock_slot[i]->is_tab_hidden(current_tab)) {
						dock_slot[i]->set_block_signals(true);
						dock_slot[i]->select_next_available();
						dock_slot[i]->set_block_signals(false);
					}
				} else {
					dock_slot[i]->set_block_signals(true);
					dock_slot[i]->set_current_tab(first_tab_visible);
					dock_slot[i]->set_block_signals(false);
				}
			} else {
				dock_slot[i]->hide();
			}
		}

		for (int i = 0; i < vsplits.size(); i++) {
			bool in_use = dock_slot[i * 2 + 0]->is_visible() || dock_slot[i * 2 + 1]->is_visible();
			vsplits[i]->set_visible(in_use);
		}

		right_hsplit->set_visible(right_l_vsplit->is_visible() || right_r_vsplit->is_visible());
	}
}

void EditorNode::_dock_tab_changed(int p_tab) {
	// Update visibility but don't set current tab.
	_update_dock_slots_visibility(true);
}

void EditorNode::_restore_floating_dock(const Dictionary &p_dock_dump, Control *p_dock, int p_slot_index) {
	WindowWrapper *wrapper = Object::cast_to<WindowWrapper>(p_dock);
	if (!wrapper) {
		_dock_make_float(p_dock, p_slot_index, false);
		wrapper = floating_docks[floating_docks.size() - 1];
	}

	wrapper->restore_window_from_saved_position(
			p_dock_dump.get("window_rect", Rect2i()),
			p_dock_dump.get("window_screen", -1),
			p_dock_dump.get("window_screen_rect", Rect2i()));
}

void EditorNode::_load_docks_from_config(Ref<ConfigFile> p_layout, const String &p_section) {
	Dictionary floating_docks_dump = p_layout->get_value(p_section, "dock_floating", Dictionary());

	bool restore_window_on_load = EDITOR_GET("interface/multi_window/restore_windows_on_load");

	for (int i = 0; i < DOCK_SLOT_MAX; i++) {
		if (!p_layout->has_section_key(p_section, "dock_" + itos(i + 1))) {
			continue;
		}

		Vector<String> names = String(p_layout->get_value(p_section, "dock_" + itos(i + 1))).split(",");

		for (int j = names.size() - 1; j >= 0; j--) {
			String name = names[j];

			// FIXME: Find it, in a horribly inefficient way.
			int atidx = -1;
			Control *node = nullptr;
			for (int k = 0; k < DOCK_SLOT_MAX; k++) {
				if (!dock_slot[k]->has_node(name)) {
					continue;
				}
				node = Object::cast_to<Control>(dock_slot[k]->get_node(name));
				if (!node) {
					continue;
				}
				atidx = k;
				break;
			}

			if (atidx == -1) {
				// Try floating docks.
				for (WindowWrapper *wrapper : floating_docks) {
					if (wrapper->get_meta("dock_name") == name) {
						if (restore_window_on_load && floating_docks_dump.has(name)) {
							_restore_floating_dock(floating_docks_dump[name], wrapper, i);
						} else {
							atidx = wrapper->get_meta("dock_slot");
							node = wrapper->get_wrapped_control();
							wrapper->set_window_enabled(false);
						}
						break;
					}
				}
			}
			if (!node) {
				// Well, it's not anywhere.
				continue;
			}

			if (atidx == i) {
				dock_slot[i]->move_child(node, 0);
			} else if (atidx != -1) {
				dock_slot[i]->move_tab_from_tab_container(dock_slot[atidx], dock_slot[atidx]->get_tab_idx_from_control(node), 0);
			}

			WindowWrapper *wrapper = Object::cast_to<WindowWrapper>(node);
			if (restore_window_on_load && floating_docks_dump.has(name)) {
				if (!dock_slot[i]->is_tab_hidden(dock_slot[i]->get_tab_idx_from_control(node))) {
					_restore_floating_dock(floating_docks_dump[name], node, i);
				}
			} else if (wrapper) {
				wrapper->set_window_enabled(false);
			}
		}

		if (!p_layout->has_section_key(p_section, "dock_" + itos(i + 1) + "_selected_tab_idx")) {
			continue;
		}

		int selected_tab_idx = p_layout->get_value(p_section, "dock_" + itos(i + 1) + "_selected_tab_idx");
		if (selected_tab_idx >= 0 && selected_tab_idx < dock_slot[i]->get_tab_count()) {
			dock_slot[i]->call_deferred("set_current_tab", selected_tab_idx);
		}
	}

	for (int i = 0; i < vsplits.size(); i++) {
		if (!p_layout->has_section_key(p_section, "dock_split_" + itos(i + 1))) {
			continue;
		}

		int ofs = p_layout->get_value(p_section, "dock_split_" + itos(i + 1));
		vsplits[i]->set_split_offset(ofs);
	}

	for (int i = 0; i < hsplits.size(); i++) {
		if (!p_layout->has_section_key(p_section, "dock_hsplit_" + itos(i + 1))) {
			continue;
		}
		int ofs = p_layout->get_value(p_section, "dock_hsplit_" + itos(i + 1));
		hsplits[i]->set_split_offset(ofs);
	}

	_update_dock_slots_visibility(false);

	// FileSystemDock.

	if (p_layout->has_section_key(p_section, "dock_filesystem_split")) {
		int fs_split_ofs = p_layout->get_value(p_section, "dock_filesystem_split");
		FileSystemDock::get_singleton()->set_split_offset(fs_split_ofs);
	}

	if (p_layout->has_section_key(p_section, "dock_filesystem_display_mode")) {
		FileSystemDock::DisplayMode dock_filesystem_display_mode = FileSystemDock::DisplayMode(int(p_layout->get_value(p_section, "dock_filesystem_display_mode")));
		FileSystemDock::get_singleton()->set_display_mode(dock_filesystem_display_mode);
	}

	if (p_layout->has_section_key(p_section, "dock_filesystem_file_sort")) {
		FileSystemDock::FileSortOption dock_filesystem_file_sort = FileSystemDock::FileSortOption(int(p_layout->get_value(p_section, "dock_filesystem_file_sort")));
		FileSystemDock::get_singleton()->set_file_sort(dock_filesystem_file_sort);
	}

	if (p_layout->has_section_key(p_section, "dock_filesystem_file_list_display_mode")) {
		FileSystemDock::FileListDisplayMode dock_filesystem_file_list_display_mode = FileSystemDock::FileListDisplayMode(int(p_layout->get_value(p_section, "dock_filesystem_file_list_display_mode")));
		FileSystemDock::get_singleton()->set_file_list_display_mode(dock_filesystem_file_list_display_mode);
	}

	if (p_layout->has_section_key(p_section, "dock_filesystem_selected_paths")) {
		PackedStringArray dock_filesystem_selected_paths = p_layout->get_value(p_section, "dock_filesystem_selected_paths");
		for (int i = 0; i < dock_filesystem_selected_paths.size(); i++) {
			FileSystemDock::get_singleton()->select_file(dock_filesystem_selected_paths[i]);
		}
	}

	// Restore collapsed state of FileSystemDock.
	PackedStringArray uncollapsed_tis;
	if (p_layout->has_section_key(p_section, "dock_filesystem_uncollapsed_paths")) {
		uncollapsed_tis = p_layout->get_value(p_section, "dock_filesystem_uncollapsed_paths");
	} else {
		uncollapsed_tis = { "res://" };
	}

	if (!uncollapsed_tis.is_empty()) {
		for (int i = 0; i < uncollapsed_tis.size(); i++) {
			TreeItem *uncollapsed_ti = FileSystemDock::get_singleton()->get_tree_control()->get_item_with_metadata(uncollapsed_tis[i], 0);
			if (uncollapsed_ti) {
				uncollapsed_ti->set_collapsed(false);
			}
		}
		FileSystemDock::get_singleton()->get_tree_control()->queue_redraw();
	}
}

void EditorNode::_save_central_editor_layout_to_config(Ref<ConfigFile> p_config_file) {
	// Bottom panel.

	int center_split_offset = center_split->get_split_offset();
	p_config_file->set_value(EDITOR_NODE_CONFIG_SECTION, "center_split_offset", center_split_offset);

	int selected_bottom_panel_item_idx = -1;
	for (int i = 0; i < bottom_panel_items.size(); i++) {
		if (bottom_panel_items[i].button->is_pressed()) {
			selected_bottom_panel_item_idx = i;
			break;
		}
	}
	if (selected_bottom_panel_item_idx != -1) {
		p_config_file->set_value(EDITOR_NODE_CONFIG_SECTION, "selected_bottom_panel_item", selected_bottom_panel_item_idx);
	} else {
		p_config_file->set_value(EDITOR_NODE_CONFIG_SECTION, "selected_bottom_panel_item", Variant());
	}

	// Debugger tab.

	int selected_default_debugger_tab_idx = EditorDebuggerNode::get_singleton()->get_default_debugger()->get_current_debugger_tab();
	p_config_file->set_value(EDITOR_NODE_CONFIG_SECTION, "selected_default_debugger_tab_idx", selected_default_debugger_tab_idx);

	// Main editor (plugin).

	int selected_main_editor_idx = -1;
	for (int i = 0; i < main_editor_buttons.size(); i++) {
		if (main_editor_buttons[i]->is_pressed()) {
			selected_main_editor_idx = i;
			break;
		}
	}
	if (selected_main_editor_idx != -1) {
		p_config_file->set_value(EDITOR_NODE_CONFIG_SECTION, "selected_main_editor_idx", selected_main_editor_idx);
	} else {
		p_config_file->set_value(EDITOR_NODE_CONFIG_SECTION, "selected_main_editor_idx", Variant());
	}
}

void EditorNode::_load_central_editor_layout_from_config(Ref<ConfigFile> p_config_file) {
	// Bottom panel.

	bool has_active_tab = false;
	if (p_config_file->has_section_key(EDITOR_NODE_CONFIG_SECTION, "selected_bottom_panel_item")) {
		int selected_bottom_panel_item_idx = p_config_file->get_value(EDITOR_NODE_CONFIG_SECTION, "selected_bottom_panel_item");
		if (selected_bottom_panel_item_idx >= 0 && selected_bottom_panel_item_idx < bottom_panel_items.size()) {
			// Make sure we don't try to open contextual editors which are not enabled in the current context.
			if (bottom_panel_items[selected_bottom_panel_item_idx].button->is_visible()) {
				_bottom_panel_switch(true, selected_bottom_panel_item_idx);
				has_active_tab = true;
			}
		}
	}

	if (p_config_file->has_section_key(EDITOR_NODE_CONFIG_SECTION, "center_split_offset")) {
		int center_split_offset = p_config_file->get_value(EDITOR_NODE_CONFIG_SECTION, "center_split_offset");
		center_split->set_split_offset(center_split_offset);

		// If there is no active tab we need to collapse the panel.
		if (!has_active_tab) {
			bottom_panel_items[0].control->show(); // _bottom_panel_switch() can collapse only visible tabs.
			_bottom_panel_switch(false, 0);
		}
	}

	// Debugger tab.

	if (p_config_file->has_section_key(EDITOR_NODE_CONFIG_SECTION, "selected_default_debugger_tab_idx")) {
		int selected_default_debugger_tab_idx = p_config_file->get_value(EDITOR_NODE_CONFIG_SECTION, "selected_default_debugger_tab_idx");
		EditorDebuggerNode::get_singleton()->get_default_debugger()->switch_to_debugger(selected_default_debugger_tab_idx);
	}

	// Main editor (plugin).

	if (p_config_file->has_section_key(EDITOR_NODE_CONFIG_SECTION, "selected_main_editor_idx")) {
		int selected_main_editor_idx = p_config_file->get_value(EDITOR_NODE_CONFIG_SECTION, "selected_main_editor_idx");
		if (selected_main_editor_idx >= 0 && selected_main_editor_idx < main_editor_buttons.size()) {
			callable_mp(this, &EditorNode::editor_select).call_deferred(selected_main_editor_idx);
		}
	}
}

void EditorNode::_load_open_scenes_from_config(Ref<ConfigFile> p_layout) {
	if (!bool(EDITOR_GET("interface/scene_tabs/restore_scenes_on_load"))) {
		return;
	}

	if (!p_layout->has_section(EDITOR_NODE_CONFIG_SECTION) ||
			!p_layout->has_section_key(EDITOR_NODE_CONFIG_SECTION, "open_scenes")) {
		return;
	}

	restoring_scenes = true;

	PackedStringArray scenes = p_layout->get_value(EDITOR_NODE_CONFIG_SECTION, "open_scenes");
	for (int i = 0; i < scenes.size(); i++) {
		load_scene(scenes[i]);
	}

	if (p_layout->has_section_key(EDITOR_NODE_CONFIG_SECTION, "current_scene")) {
		String current_scene = p_layout->get_value(EDITOR_NODE_CONFIG_SECTION, "current_scene");
		int current_scene_idx = scenes.find(current_scene);
		if (current_scene_idx >= 0) {
			_set_current_scene(current_scene_idx);
		}
	}

	save_editor_layout_delayed();

	restoring_scenes = false;
}

bool EditorNode::has_scenes_in_session() {
	if (!bool(EDITOR_GET("interface/scene_tabs/restore_scenes_on_load"))) {
		return false;
	}
	Ref<ConfigFile> config;
	config.instantiate();
	Error err = config->load(EditorPaths::get_singleton()->get_project_settings_dir().path_join("editor_layout.cfg"));
	if (err != OK) {
		return false;
	}
	if (!config->has_section(EDITOR_NODE_CONFIG_SECTION) || !config->has_section_key(EDITOR_NODE_CONFIG_SECTION, "open_scenes")) {
		return false;
	}
	Array scenes = config->get_value(EDITOR_NODE_CONFIG_SECTION, "open_scenes");
	return !scenes.is_empty();
}

bool EditorNode::ensure_main_scene(bool p_from_native) {
	pick_main_scene->set_meta("from_native", p_from_native); // Whether from play button or native run.
	String main_scene = GLOBAL_GET("application/run/main_scene");

	if (main_scene.is_empty()) {
		current_menu_option = -1;
		pick_main_scene->set_text(TTR("No main scene has ever been defined, select one?\nYou can change it later in \"Project Settings\" under the 'application' category."));
		pick_main_scene->popup_centered();

		if (editor_data.get_edited_scene_root()) {
			select_current_scene_button->set_disabled(false);
			select_current_scene_button->grab_focus();
		} else {
			select_current_scene_button->set_disabled(true);
		}

		return false;
	}

	if (!FileAccess::exists(main_scene)) {
		current_menu_option = -1;
		pick_main_scene->set_text(vformat(TTR("Selected scene '%s' does not exist, select a valid one?\nYou can change it later in \"Project Settings\" under the 'application' category."), main_scene));
		pick_main_scene->popup_centered();
		return false;
	}

	if (ResourceLoader::get_resource_type(main_scene) != "PackedScene") {
		current_menu_option = -1;
		pick_main_scene->set_text(vformat(TTR("Selected scene '%s' is not a scene file, select a valid one?\nYou can change it later in \"Project Settings\" under the 'application' category."), main_scene));
		pick_main_scene->popup_centered();
		return false;
	}

	return true;
}

void EditorNode::_immediate_dialog_confirmed() {
	immediate_dialog_confirmed = true;
}
bool EditorNode::immediate_confirmation_dialog(const String &p_text, const String &p_ok_text, const String &p_cancel_text, uint32_t p_wrap_width) {
	ConfirmationDialog *cd = memnew(ConfirmationDialog);
	cd->set_text(p_text);
	cd->set_ok_button_text(p_ok_text);
	cd->set_cancel_button_text(p_cancel_text);
	if (p_wrap_width > 0) {
		cd->set_autowrap(true);
		cd->get_label()->set_custom_minimum_size(Size2(p_wrap_width, 0) * EDSCALE);
	}

	cd->connect("confirmed", callable_mp(singleton, &EditorNode::_immediate_dialog_confirmed));
	singleton->gui_base->add_child(cd);

	cd->popup_centered();

	while (true) {
		OS::get_singleton()->delay_usec(1);
		DisplayServer::get_singleton()->process_events();
		Main::iteration();
		if (singleton->immediate_dialog_confirmed || !cd->is_visible()) {
			break;
		}
	}

	memdelete(cd);
	return singleton->immediate_dialog_confirmed;
}

void EditorNode::cleanup() {
	_init_callbacks.clear();
}

void EditorNode::_update_layouts_menu() {
	editor_layouts->clear();
	overridden_default_layout = -1;

	editor_layouts->reset_size();
	editor_layouts->add_shortcut(ED_SHORTCUT("layout/save", TTR("Save Layout")), SETTINGS_LAYOUT_SAVE);
	editor_layouts->add_shortcut(ED_SHORTCUT("layout/delete", TTR("Delete Layout")), SETTINGS_LAYOUT_DELETE);
	editor_layouts->add_separator();
	editor_layouts->add_shortcut(ED_SHORTCUT("layout/default", TTR("Default")), SETTINGS_LAYOUT_DEFAULT);

	Ref<ConfigFile> config;
	config.instantiate();
	Error err = config->load(EditorSettings::get_singleton()->get_editor_layouts_config());
	if (err != OK) {
		return; // No config.
	}

	List<String> layouts;
	config.ptr()->get_sections(&layouts);

	for (const String &layout : layouts) {
		if (layout == TTR("Default")) {
			editor_layouts->remove_item(editor_layouts->get_item_index(SETTINGS_LAYOUT_DEFAULT));
			overridden_default_layout = editor_layouts->get_item_count();
		}

		editor_layouts->add_item(layout);
	}
}

void EditorNode::_layout_menu_option(int p_id) {
	switch (p_id) {
		case SETTINGS_LAYOUT_SAVE: {
			current_menu_option = p_id;
			layout_dialog->set_title(TTR("Save Layout"));
			layout_dialog->set_ok_button_text(TTR("Save"));
			layout_dialog->set_name_line_enabled(true);
			layout_dialog->popup_centered();
		} break;
		case SETTINGS_LAYOUT_DELETE: {
			current_menu_option = p_id;
			layout_dialog->set_title(TTR("Delete Layout"));
			layout_dialog->set_ok_button_text(TTR("Delete"));
			layout_dialog->set_name_line_enabled(false);
			layout_dialog->popup_centered();
		} break;
		case SETTINGS_LAYOUT_DEFAULT: {
			_load_docks_from_config(default_layout, "docks");
			_save_editor_layout();
		} break;
		default: {
			Ref<ConfigFile> config;
			config.instantiate();
			Error err = config->load(EditorSettings::get_singleton()->get_editor_layouts_config());
			if (err != OK) {
				return; // No config.
			}

			_load_docks_from_config(config, editor_layouts->get_item_text(p_id));
			_save_editor_layout();
		}
	}
}

void EditorNode::_proceed_closing_scene_tabs() {
	List<String>::Element *E = tabs_to_close.front();
	if (!E) {
		if (_is_closing_editor()) {
			current_menu_option = tab_closing_menu_option;
			_menu_option_confirm(tab_closing_menu_option, true);
		} else {
			current_menu_option = -1;
			save_confirmation->hide();
		}
		return;
	}
	String scene_to_close = E->get();
	tabs_to_close.pop_front();

	int tab_idx = -1;
	for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {
		if (editor_data.get_scene_path(i) == scene_to_close) {
			tab_idx = i;
			break;
		}
	}
	ERR_FAIL_COND(tab_idx < 0);

	_scene_tab_closed(tab_idx);
}

bool EditorNode::_is_closing_editor() const {
	return tab_closing_menu_option == FILE_QUIT || tab_closing_menu_option == RUN_PROJECT_MANAGER || tab_closing_menu_option == RELOAD_CURRENT_PROJECT;
}

void EditorNode::_scene_tab_closed(int p_tab) {
	current_menu_option = SCENE_TAB_CLOSE;
	tab_closing_idx = p_tab;
	Node *scene = editor_data.get_edited_scene_root(p_tab);
	if (!scene) {
		_discard_changes();
		return;
	}

	String scene_filename = scene->get_scene_file_path();
	String unsaved_message;

	if (EditorUndoRedoManager::get_singleton()->is_history_unsaved(editor_data.get_scene_history_id(p_tab))) {
		if (scene_filename.is_empty()) {
			unsaved_message = TTR("This scene was never saved.");
		} else {
			// Consider editor startup to be a point of saving, so that when you
			// close and reopen the editor, you don't get an excessively long
			// "modified X hours ago".
			const uint64_t last_modified_seconds = Time::get_singleton()->get_unix_time_from_system() - MAX(started_timestamp, FileAccess::get_modified_time(scene->get_scene_file_path()));
			String last_modified_string;
			if (last_modified_seconds < 120) {
				last_modified_string = vformat(TTRN("%d second ago", "%d seconds ago", last_modified_seconds), last_modified_seconds);
			} else if (last_modified_seconds < 7200) {
				last_modified_string = vformat(TTRN("%d minute ago", "%d minutes ago", last_modified_seconds / 60), last_modified_seconds / 60);
			} else {
				last_modified_string = vformat(TTRN("%d hour ago", "%d hours ago", last_modified_seconds / 3600), last_modified_seconds / 3600);
			}
			unsaved_message = vformat(TTR("Scene \"%s\" has unsaved changes.\nLast saved: %s."), scene_filename, last_modified_string);
		}
	} else {
		// Check if any plugin has unsaved changes in that scene.
		for (int i = 0; i < editor_data.get_editor_plugin_count(); i++) {
			unsaved_message = editor_data.get_editor_plugin(i)->get_unsaved_status(scene_filename);
			if (!unsaved_message.is_empty()) {
				break;
			}
		}
	}

	if (!unsaved_message.is_empty()) {
		if (scene_tabs->get_current_tab() != p_tab) {
			_set_current_scene(p_tab);
		}

		save_confirmation->set_ok_button_text(TTR("Save & Close"));
		save_confirmation->set_text(unsaved_message + "\n\n" + TTR("Save before closing?"));
		save_confirmation->reset_size();
		save_confirmation->popup_centered();
	} else {
		_discard_changes();
	}

	save_editor_layout_delayed();
	scene_tabs->update_scene_tabs();
}

Button *EditorNode::add_bottom_panel_item(String p_text, Control *p_item) {
	Button *tb = memnew(Button);
	tb->set_flat(true);
	tb->connect("toggled", callable_mp(this, &EditorNode::_bottom_panel_switch).bind(bottom_panel_items.size()));
	tb->set_text(p_text);
	tb->set_toggle_mode(true);
	tb->set_focus_mode(Control::FOCUS_NONE);
	bottom_panel_vb->add_child(p_item);
	bottom_panel_hb->move_to_front();
	bottom_panel_hb_editors->add_child(tb);
	p_item->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	p_item->hide();
	BottomPanelItem bpi;
	bpi.button = tb;
	bpi.control = p_item;
	bpi.name = p_text;
	bottom_panel_items.push_back(bpi);

	return tb;
}

void EditorNode::hide_bottom_panel() {
	for (int i = 0; i < bottom_panel_items.size(); i++) {
		if (bottom_panel_items[i].control->is_visible()) {
			_bottom_panel_switch(false, i);
			break;
		}
	}
}

void EditorNode::make_bottom_panel_item_visible(Control *p_item) {
	for (int i = 0; i < bottom_panel_items.size(); i++) {
		if (bottom_panel_items[i].control == p_item) {
			_bottom_panel_switch(true, i);
			break;
		}
	}
}

void EditorNode::raise_bottom_panel_item(Control *p_item) {
	for (int i = 0; i < bottom_panel_items.size(); i++) {
		if (bottom_panel_items[i].control == p_item) {
			bottom_panel_items[i].button->move_to_front();
			SWAP(bottom_panel_items.write[i], bottom_panel_items.write[bottom_panel_items.size() - 1]);
			break;
		}
	}

	for (int i = 0; i < bottom_panel_items.size(); i++) {
		bottom_panel_items[i].button->disconnect("toggled", callable_mp(this, &EditorNode::_bottom_panel_switch));
		bottom_panel_items[i].button->connect("toggled", callable_mp(this, &EditorNode::_bottom_panel_switch).bind(i));
	}
}

void EditorNode::remove_bottom_panel_item(Control *p_item) {
	for (int i = 0; i < bottom_panel_items.size(); i++) {
		if (bottom_panel_items[i].control == p_item) {
			if (p_item->is_visible_in_tree()) {
				_bottom_panel_switch(false, i);
			}
			bottom_panel_vb->remove_child(bottom_panel_items[i].control);
			bottom_panel_hb_editors->remove_child(bottom_panel_items[i].button);
			memdelete(bottom_panel_items[i].button);
			bottom_panel_items.remove_at(i);
			break;
		}
	}

	for (int i = 0; i < bottom_panel_items.size(); i++) {
		bottom_panel_items[i].button->disconnect("toggled", callable_mp(this, &EditorNode::_bottom_panel_switch));
		bottom_panel_items[i].button->connect("toggled", callable_mp(this, &EditorNode::_bottom_panel_switch).bind(i));
	}
}

void EditorNode::_bottom_panel_switch(bool p_enable, int p_idx) {
	if (bottom_panel_updating) {
		return;
	}
	ERR_FAIL_INDEX(p_idx, bottom_panel_items.size());

	if (bottom_panel_items[p_idx].control->is_visible() == p_enable) {
		return;
	}

	if (p_enable) {
		bottom_panel_updating = true;

		for (int i = 0; i < bottom_panel_items.size(); i++) {
			bottom_panel_items[i].button->set_pressed(i == p_idx);
			bottom_panel_items[i].control->set_visible(i == p_idx);
		}
		if (EditorDebuggerNode::get_singleton() == bottom_panel_items[p_idx].control) {
			// This is the debug panel which uses tabs, so the top section should be smaller.
			bottom_panel->add_theme_style_override("panel", theme->get_stylebox(SNAME("BottomPanelDebuggerOverride"), EditorStringName(EditorStyles)));
		} else {
			bottom_panel->add_theme_style_override("panel", theme->get_stylebox(SNAME("BottomPanel"), EditorStringName(EditorStyles)));
		}
		center_split->set_dragger_visibility(SplitContainer::DRAGGER_VISIBLE);
		center_split->set_collapsed(false);
		if (bottom_panel_raise->is_pressed()) {
			top_split->hide();
		}
		bottom_panel_raise->show();
	} else {
		bottom_panel->add_theme_style_override("panel", theme->get_stylebox(SNAME("BottomPanel"), EditorStringName(EditorStyles)));
		bottom_panel_items[p_idx].button->set_pressed(false);
		bottom_panel_items[p_idx].control->set_visible(false);
		center_split->set_dragger_visibility(SplitContainer::DRAGGER_HIDDEN);
		center_split->set_collapsed(true);
		bottom_panel_raise->hide();
		if (bottom_panel_raise->is_pressed()) {
			top_split->show();
		}
	}
}

void EditorNode::set_docks_visible(bool p_show) {
	docks_visible = p_show;
	_update_dock_slots_visibility(true);
}

bool EditorNode::get_docks_visible() const {
	return docks_visible;
}

void EditorNode::_toggle_distraction_free_mode() {
	if (EDITOR_GET("interface/editor/separate_distraction_mode")) {
		int screen = -1;
		for (int i = 0; i < editor_table.size(); i++) {
			if (editor_plugin_screen == editor_table[i]) {
				screen = i;
				break;
			}
		}

		if (screen == EDITOR_SCRIPT) {
			script_distraction_free = !script_distraction_free;
			set_distraction_free_mode(script_distraction_free);
		} else {
			scene_distraction_free = !scene_distraction_free;
			set_distraction_free_mode(scene_distraction_free);
		}
	} else {
		set_distraction_free_mode(distraction_free->is_pressed());
	}
}

void EditorNode::set_distraction_free_mode(bool p_enter) {
	distraction_free->set_pressed(p_enter);

	if (p_enter) {
		if (docks_visible) {
			set_docks_visible(false);
		}
	} else {
		set_docks_visible(true);
	}
}

bool EditorNode::is_distraction_free_mode_enabled() const {
	return distraction_free->is_pressed();
}

void EditorNode::add_control_to_dock(DockSlot p_slot, Control *p_control) {
	ERR_FAIL_INDEX(p_slot, DOCK_SLOT_MAX);
	dock_slot[p_slot]->add_child(p_control);
	_update_dock_slots_visibility();
}

void EditorNode::remove_control_from_dock(Control *p_control) {
	// If the dock is floating, close it first.
	for (WindowWrapper *wrapper : floating_docks) {
		if (p_control == wrapper->get_wrapped_control()) {
			wrapper->set_window_enabled(false);
			break;
		}
	}

	Control *dock = nullptr;
	for (int i = 0; i < DOCK_SLOT_MAX; i++) {
		if (p_control->get_parent() == dock_slot[i]) {
			dock = dock_slot[i];
			break;
		}
	}

	ERR_FAIL_NULL_MSG(dock, "Control was not in dock.");

	dock->remove_child(p_control);
	_update_dock_slots_visibility();
}

Variant EditorNode::drag_resource(const Ref<Resource> &p_res, Control *p_from) {
	Control *drag_control = memnew(Control);
	TextureRect *drag_preview = memnew(TextureRect);
	Label *label = memnew(Label);

	Ref<Texture2D> preview;

	{
		// TODO: make proper previews
		Ref<ImageTexture> texture = theme->get_icon(SNAME("FileBigThumb"), EditorStringName(EditorIcons));
		Ref<Image> img = texture->get_image();
		img = img->duplicate();
		img->resize(48, 48); // meh
		preview = ImageTexture::create_from_image(img);
	}

	drag_preview->set_texture(preview);
	drag_control->add_child(drag_preview);
	if (p_res->get_path().is_resource_file()) {
		label->set_text(p_res->get_path().get_file());
	} else if (!p_res->get_name().is_empty()) {
		label->set_text(p_res->get_name());
	} else {
		label->set_text(p_res->get_class());
	}

	drag_control->add_child(label);

	p_from->set_drag_preview(drag_control); // Wait until it enters scene.

	label->set_position(Point2((preview->get_width() - label->get_minimum_size().width) / 2, preview->get_height()));

	Dictionary drag_data;
	drag_data["type"] = "resource";
	drag_data["resource"] = p_res;
	drag_data["from"] = p_from;

	return drag_data;
}

Variant EditorNode::drag_files_and_dirs(const Vector<String> &p_paths, Control *p_from) {
	bool has_folder = false;
	bool has_file = false;
	for (int i = 0; i < p_paths.size(); i++) {
		bool is_folder = p_paths[i].ends_with("/");
		has_folder |= is_folder;
		has_file |= !is_folder;
	}

	int max_rows = 6;
	int num_rows = p_paths.size() > max_rows ? max_rows - 1 : p_paths.size(); // Don't waste a row to say "1 more file" - list it instead.
	VBoxContainer *vbox = memnew(VBoxContainer);
	for (int i = 0; i < num_rows; i++) {
		HBoxContainer *hbox = memnew(HBoxContainer);
		TextureRect *icon = memnew(TextureRect);
		Label *label = memnew(Label);

		if (p_paths[i].ends_with("/")) {
			label->set_text(p_paths[i].substr(0, p_paths[i].length() - 1).get_file());
			icon->set_texture(theme->get_icon(SNAME("Folder"), EditorStringName(EditorIcons)));
		} else {
			label->set_text(p_paths[i].get_file());
			icon->set_texture(theme->get_icon(SNAME("File"), EditorStringName(EditorIcons)));
		}
		icon->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
		icon->set_size(Size2(16, 16));
		hbox->add_child(icon);
		hbox->add_child(label);
		vbox->add_child(hbox);
	}

	if (p_paths.size() > num_rows) {
		Label *label = memnew(Label);
		if (has_file && has_folder) {
			label->set_text(vformat(TTR("%d more files or folders"), p_paths.size() - num_rows));
		} else if (has_folder) {
			label->set_text(vformat(TTR("%d more folders"), p_paths.size() - num_rows));
		} else {
			label->set_text(vformat(TTR("%d more files"), p_paths.size() - num_rows));
		}
		vbox->add_child(label);
	}
	p_from->set_drag_preview(vbox); // Wait until it enters scene.

	Dictionary drag_data;
	drag_data["type"] = has_folder ? "files_and_dirs" : "files";
	drag_data["files"] = p_paths;
	drag_data["from"] = p_from;
	return drag_data;
}

void EditorNode::add_tool_menu_item(const String &p_name, const Callable &p_callback) {
	int idx = tool_menu->get_item_count();
	tool_menu->add_item(p_name, TOOLS_CUSTOM);
	tool_menu->set_item_metadata(idx, p_callback);
}

void EditorNode::add_tool_submenu_item(const String &p_name, PopupMenu *p_submenu) {
	ERR_FAIL_NULL(p_submenu);
	ERR_FAIL_COND(p_submenu->get_parent() != nullptr);

	tool_menu->add_child(p_submenu);
	tool_menu->add_submenu_item(p_name, p_submenu->get_name(), TOOLS_CUSTOM);
}

void EditorNode::remove_tool_menu_item(const String &p_name) {
	for (int i = 0; i < tool_menu->get_item_count(); i++) {
		if (tool_menu->get_item_id(i) != TOOLS_CUSTOM) {
			continue;
		}

		if (tool_menu->get_item_text(i) == p_name) {
			if (tool_menu->get_item_submenu(i) != "") {
				Node *n = tool_menu->get_node(tool_menu->get_item_submenu(i));
				tool_menu->remove_child(n);
				memdelete(n);
			}
			tool_menu->remove_item(i);
			tool_menu->reset_size();
			return;
		}
	}
}

PopupMenu *EditorNode::get_export_as_menu() {
	return export_as_menu;
}

void EditorNode::_dropped_files(const Vector<String> &p_files) {
	String to_path = ProjectSettings::get_singleton()->globalize_path(FileSystemDock::get_singleton()->get_current_directory());

	_add_dropped_files_recursive(p_files, to_path);

	EditorFileSystem::get_singleton()->scan_changes();
}

void EditorNode::_add_dropped_files_recursive(const Vector<String> &p_files, String to_path) {
	Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	ERR_FAIL_COND(dir.is_null());

	for (int i = 0; i < p_files.size(); i++) {
		String from = p_files[i];
		String to = to_path.path_join(from.get_file());

		if (dir->dir_exists(from)) {
			Vector<String> sub_files;

			Ref<DirAccess> sub_dir = DirAccess::open(from);
			ERR_FAIL_COND(sub_dir.is_null());

			sub_dir->list_dir_begin();

			String next_file = sub_dir->get_next();
			while (!next_file.is_empty()) {
				if (next_file == "." || next_file == "..") {
					next_file = sub_dir->get_next();
					continue;
				}

				sub_files.push_back(from.path_join(next_file));
				next_file = sub_dir->get_next();
			}

			if (!sub_files.is_empty()) {
				dir->make_dir(to);
				_add_dropped_files_recursive(sub_files, to);
			}

			continue;
		}

		dir->copy(from, to);
	}
}

void EditorNode::_file_access_close_error_notify(const String &p_str) {
	callable_mp_static(&EditorNode::_file_access_close_error_notify_impl).bind(p_str).call_deferred();
}

void EditorNode::_file_access_close_error_notify_impl(const String &p_str) {
	add_io_error(vformat(TTR("Unable to write to file '%s', file in use, locked or lacking permissions."), p_str));
}

// Since we felt that a bespoke NOTIFICATION might not be desirable, this function
// provides the hardcoded callbacks to address known bugs which occur on certain
// nodes during reimport.
// Ideally, we should probably agree on a standardized method name which could be
// called from here instead.
void EditorNode::_notify_scene_updated(Node *p_node) {
	Skeleton3D *skel_3d = Object::cast_to<Skeleton3D>(p_node);
	if (skel_3d) {
		skel_3d->reset_bone_poses();
	} else {
		BoneAttachment3D *attachment = Object::cast_to<BoneAttachment3D>(p_node);
		if (attachment) {
			attachment->notify_rebind_required();
		}
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_notify_scene_updated(p_node->get_child(i));
	}
}

void EditorNode::reload_scene(const String &p_path) {
	int scene_idx = -1;
	for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {
		if (editor_data.get_scene_path(i) == p_path) {
			scene_idx = i;
			break;
		}
	}

	int current_tab = editor_data.get_edited_scene();

	if (scene_idx == -1) {
		if (get_edited_scene()) {
			// Scene is not open, so at it might be instantiated. We'll refresh the whole scene later.
			EditorUndoRedoManager::get_singleton()->clear_history(false, editor_data.get_current_edited_scene_history_id());
		}
		return;
	}

	if (current_tab == scene_idx) {
		editor_data.apply_changes_in_editors();
		_save_editor_states(p_path);
	}

	// Reload scene.
	_remove_scene(scene_idx, false);
	load_scene(p_path, true, false, true, true);

	// Adjust index so tab is back a the previous position.
	editor_data.move_edited_scene_to_index(scene_idx);
	EditorUndoRedoManager::get_singleton()->clear_history(false, editor_data.get_scene_history_id(scene_idx));

	// Recover the tab.
	scene_tabs->set_current_tab(current_tab);
}

void EditorNode::find_all_instances_inheriting_path_in_node(Node *p_root, Node *p_node, const String &p_instance_path, List<Node *> &p_instance_list) {
	String scene_file_path = p_node->get_scene_file_path();

	// This is going to get messy...
	if (p_node->get_scene_file_path() == p_instance_path) {
		p_instance_list.push_back(p_node);
	} else {
		Node *current_node = p_node;

		Ref<SceneState> inherited_state = current_node->get_scene_inherited_state();
		while (inherited_state.is_valid()) {
			String inherited_path = inherited_state->get_path();
			if (inherited_path == p_instance_path) {
				p_instance_list.push_back(p_node);
				break;
			}

			inherited_state = inherited_state->get_base_scene_state();
		}
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *child = p_node->get_child(i);
		find_all_instances_inheriting_path_in_node(p_root, child, p_instance_path, p_instance_list);
	}
}

void EditorNode::reload_instances_with_path_in_edited_scenes(const String &p_instance_path) {
	int original_edited_scene_idx = editor_data.get_edited_scene();
	HashMap<int, List<Node *>> edited_scene_map;

	// Walk through each opened scene to get a global list of all instances which match
	// the current reimported scenes.
	for (int i = 0; i < editor_data.get_edited_scene_count(); i++) {
		if (editor_data.get_scene_path(i) != p_instance_path) {
			Node *edited_scene_root = editor_data.get_edited_scene_root(i);

			if (edited_scene_root) {
				List<Node *> valid_nodes;
				find_all_instances_inheriting_path_in_node(edited_scene_root, edited_scene_root, p_instance_path, valid_nodes);
				if (valid_nodes.size() > 0) {
					edited_scene_map[i] = valid_nodes;
				}
			}
		}
	}

	if (edited_scene_map.size() > 0) {
		// Reload the new instance.
		Error err;
		Ref<PackedScene> instance_scene_packed_scene = ResourceLoader::load(p_instance_path, "", ResourceFormatLoader::CACHE_MODE_IGNORE, &err);
		instance_scene_packed_scene->set_path(p_instance_path, true);

		ERR_FAIL_COND(err != OK);
		ERR_FAIL_COND(instance_scene_packed_scene.is_null());

		HashMap<String, Ref<PackedScene>> local_scene_cache;
		local_scene_cache[p_instance_path] = instance_scene_packed_scene;

		for (const KeyValue<int, List<Node *>> &edited_scene_map_elem : edited_scene_map) {
			// Set the current scene.
			int current_scene_idx = edited_scene_map_elem.key;
			editor_data.set_edited_scene(current_scene_idx);
			Node *current_edited_scene = editor_data.get_edited_scene_root(current_scene_idx);

			// Clear the history for this tab (should we allow history to be retained?).
			EditorUndoRedoManager::get_singleton()->clear_history();

			// Update the version
			editor_data.is_scene_changed(current_scene_idx);

			for (Node *original_node : edited_scene_map_elem.value) {
				// Walk the tree for the current node and extract relevant diff data, storing it in the modification table.
				// For additional nodes which are part of the current scene, they get added to the addition table.
				HashMap<NodePath, ModificationNodeEntry> modification_table;
				List<AdditiveNodeEntry> addition_list;
				update_diff_data_for_node(current_edited_scene, original_node, original_node, modification_table, addition_list);

				// Disconnect all relevant connections, all connections from and persistent connections to.
				for (const KeyValue<NodePath, ModificationNodeEntry> &modification_table_entry : modification_table) {
					for (Connection conn : modification_table_entry.value.connections_from) {
						conn.signal.get_object()->disconnect(conn.signal.get_name(), conn.callable);
					}
					for (ConnectionWithNodePath cwnp : modification_table_entry.value.connections_to) {
						Connection conn = cwnp.connection;
						if (conn.flags & CONNECT_PERSIST) {
							conn.signal.get_object()->disconnect(conn.signal.get_name(), conn.callable);
						}
					}
				}

				// Store all the paths for any selected nodes which are ancestors of the node we're replacing.
				List<NodePath> selected_node_paths;
				for (Node *selected_node : editor_selection->get_selected_node_list()) {
					if (selected_node == original_node || original_node->is_ancestor_of(selected_node)) {
						selected_node_paths.push_back(original_node->get_path_to(selected_node));
						editor_selection->remove_node(selected_node);
					}
				}

				// Remove all nodes which were added as additional elements (they will be restored later).
				for (AdditiveNodeEntry additive_node_entry : addition_list) {
					Node *addition_node = additive_node_entry.node;
					addition_node->get_parent()->remove_child(addition_node);
				}

				// Clear ownership of the nodes (kind of hack to workaround an issue with
				// replace_by when called on nodes in other tabs).
				List<Node *> nodes_owned_by_original_node;
				original_node->get_owned_by(original_node, &nodes_owned_by_original_node);
				for (Node *owned_node : nodes_owned_by_original_node) {
					owned_node->set_owner(nullptr);
				}

				// Delete all the remaining node children.
				while (original_node->get_child_count()) {
					Node *child = original_node->get_child(0);

					original_node->remove_child(child);
					child->queue_free();
				}

				// Reset the editable instance state.
				bool is_editable = true;
				Node *owner = original_node->get_owner();
				if (owner) {
					is_editable = owner->is_editable_instance(original_node);
				}

				// Load a replacement scene for the node.
				Ref<PackedScene> current_packed_scene;
				if (original_node->get_scene_file_path() == p_instance_path) {
					// If the node file name directly matches the scene we're replacing,
					// just load it since we already cached it.
					current_packed_scene = instance_scene_packed_scene;
				} else {
					// Otherwise, check the inheritance chain, reloading and caching any scenes
					// we require along the way.
					List<String> required_load_paths;
					String scene_path = original_node->get_scene_file_path();
					// Do we need to check if the paths are empty?
					if (!scene_path.is_empty()) {
						required_load_paths.push_front(scene_path);
					}
					Ref<SceneState> inherited_state = original_node->get_scene_inherited_state();
					while (inherited_state.is_valid()) {
						String inherited_path = inherited_state->get_path();
						// Do we need to check if the paths are empty?
						if (!inherited_path.is_empty()) {
							required_load_paths.push_front(inherited_path);
						}
						inherited_state = inherited_state->get_base_scene_state();
					}

					// Ensure the inheritance chain is loaded in the correct order so that cache can
					// be properly updated.
					for (String path : required_load_paths) {
						if (!local_scene_cache.find(path)) {
							current_packed_scene = ResourceLoader::load(path, "", ResourceFormatLoader::CACHE_MODE_IGNORE, &err);
							current_packed_scene->set_path(path, true);
							local_scene_cache[path] = current_packed_scene;
						} else {
							current_packed_scene = local_scene_cache[path];
						}
					}
				}

				ERR_FAIL_COND(current_packed_scene.is_null());

				// Instantiate the node.
				Node *instantiated_node = nullptr;
				if (current_packed_scene.is_valid()) {
					instantiated_node = current_packed_scene->instantiate(PackedScene::GEN_EDIT_STATE_INSTANCE);
				}

				ERR_FAIL_NULL(instantiated_node);

				bool original_node_is_displayed_folded = original_node->is_displayed_folded();
				bool original_node_scene_instance_load_placeholder = original_node->get_scene_instance_load_placeholder();

				// Update the name to match
				instantiated_node->set_name(original_node->get_name());

				// Is this replacing the edited root node?
				String original_node_file_path = original_node->get_scene_file_path();

				if (current_edited_scene == original_node) {
					instantiated_node->set_scene_instance_state(original_node->get_scene_instance_state());
					// Fix unsaved inherited scene
					if (original_node_file_path.is_empty()) {
						Ref<SceneState> state = current_packed_scene->get_state();
						state->set_path(current_packed_scene->get_path());
						instantiated_node->set_scene_inherited_state(state);
						instantiated_node->set_scene_file_path(String());
					}
					editor_data.set_edited_scene_root(instantiated_node);
					current_edited_scene = instantiated_node;

					if (original_node->is_inside_tree()) {
						SceneTreeDock::get_singleton()->set_edited_scene(current_edited_scene);
						original_node->get_tree()->set_edited_scene_root(instantiated_node);
					}
				}

				// Replace the original node with the instantiated version.
				original_node->replace_by(instantiated_node, false);

				// Mark the old node for deletion.
				original_node->queue_free();

				// Restore the folded and placeholder state from the original node.
				instantiated_node->set_display_folded(original_node_is_displayed_folded);
				instantiated_node->set_scene_instance_load_placeholder(original_node_scene_instance_load_placeholder);

				if (owner) {
					Ref<SceneState> ss_inst = owner->get_scene_instance_state();
					if (ss_inst.is_valid()) {
						ss_inst->update_instance_resource(p_instance_path, current_packed_scene);
					}

					owner->set_editable_instance(instantiated_node, is_editable);
				}

				// Attempt to re-add all the additional nodes.
				for (AdditiveNodeEntry additive_node_entry : addition_list) {
					Node *parent_node = instantiated_node->get_node_or_null(additive_node_entry.parent);

					if (!parent_node) {
						parent_node = current_edited_scene;
					}

					parent_node->add_child(additive_node_entry.node);
					parent_node->move_child(additive_node_entry.node, additive_node_entry.index);
					// If the additive node's owner was the node which got replaced, update it.
					if (additive_node_entry.owner == original_node) {
						additive_node_entry.owner = instantiated_node;
					}

					additive_node_entry.node->set_owner(additive_node_entry.owner);

					// If the parent node was lost, attempt to restore the original global transform.
					{
						Node2D *node_2d = Object::cast_to<Node2D>(additive_node_entry.node);
						if (node_2d) {
							node_2d->set_transform(additive_node_entry.transform_2d);
						}

						Node3D *node_3d = Object::cast_to<Node3D>(additive_node_entry.node);
						if (node_3d) {
							node_3d->set_transform(additive_node_entry.transform_3d);
						}
					}

					// Restore the ownership of its ancestors
					for (KeyValue<Node *, Node *> &E : additive_node_entry.ownership_table) {
						Node *current_ancestor = E.key;
						Node *ancestor_owner = E.value;

						if (ancestor_owner == original_node) {
							ancestor_owner = instantiated_node;
						}

						current_ancestor->set_owner(ancestor_owner);
					}
				}

				// Restore the selection.
				if (selected_node_paths.size()) {
					for (NodePath selected_node_path : selected_node_paths) {
						Node *selected_node = instantiated_node->get_node_or_null(selected_node_path);
						if (selected_node) {
							editor_selection->add_node(selected_node);
						}
					}
					editor_selection->update();
				}

				// Attempt to restore the modified properties and signals for the instantitated node and all its owned children.
				for (KeyValue<NodePath, ModificationNodeEntry> &E : modification_table) {
					NodePath new_current_path = E.key;
					Node *modifiable_node = instantiated_node->get_node_or_null(new_current_path);

					if (modifiable_node) {
						// Get properties for this node.
						List<PropertyInfo> pinfo;
						modifiable_node->get_property_list(&pinfo);

						// Get names of all valid property names (TODO: make this more efficient).
						List<String> property_names;
						for (const PropertyInfo &E2 : pinfo) {
							if (E2.usage & PROPERTY_USAGE_STORAGE) {
								property_names.push_back(E2.name);
							}
						}

						// Restore the modified properties for this node.
						for (const KeyValue<StringName, Variant> &E2 : E.value.property_table) {
							if (property_names.find(E2.key)) {
								modifiable_node->set(E2.key, E2.value);
							}
						}
						// Restore the connections to other nodes.
						for (const ConnectionWithNodePath &E2 : E.value.connections_to) {
							Connection conn = E2.connection;

							// Get the node the callable is targeting.
							Node *target_node = cast_to<Node>(conn.callable.get_object());

							// If the callable object no longer exists or is marked for deletion,
							// attempt to reaccquire the closest match by using the node path
							// we saved earlier.
							if (!target_node || !target_node->is_queued_for_deletion()) {
								target_node = modifiable_node->get_node_or_null(E2.node_path);
							}

							if (target_node) {
								// Reconstruct the callable.
								Callable new_callable = Callable(target_node, conn.callable.get_method());

								if (!modifiable_node->is_connected(conn.signal.get_name(), new_callable)) {
									ERR_FAIL_COND(modifiable_node->connect(conn.signal.get_name(), new_callable, conn.flags) != OK);
								}
							}
						}

						// Restore the connections from other nodes.
						for (const Connection &E2 : E.value.connections_from) {
							Connection conn = E2;

							bool valid = modifiable_node->has_method(conn.callable.get_method()) || Ref<Script>(modifiable_node->get_script()).is_null() || Ref<Script>(modifiable_node->get_script())->has_method(conn.callable.get_method());
							ERR_CONTINUE_MSG(!valid, vformat("Attempt to connect signal '%s.%s' to nonexistent method '%s.%s'.", conn.signal.get_object()->get_class(), conn.signal.get_name(), conn.callable.get_object()->get_class(), conn.callable.get_method()));

							// Get the object which the signal is connected from.
							Object *source_object = conn.signal.get_object();

							if (source_object) {
								ERR_FAIL_COND(source_object->connect(conn.signal.get_name(), Callable(modifiable_node, conn.callable.get_method()), conn.flags) != OK);
							}
						}

						// Re-add the groups.
						for (const Node::GroupInfo &E2 : E.value.groups) {
							modifiable_node->add_to_group(E2.name, E2.persistent);
						}
					}
				}
			}

			// Cleanup the history of the changes.
			editor_history.cleanup_history();

			_notify_scene_updated(current_edited_scene);
		}
		edited_scene_map.clear();
	}
	editor_data.set_edited_scene(original_edited_scene_idx);

	_edit_current();
}

int EditorNode::plugin_init_callback_count = 0;

void EditorNode::add_plugin_init_callback(EditorPluginInitializeCallback p_callback) {
	ERR_FAIL_COND(plugin_init_callback_count == MAX_INIT_CALLBACKS);

	plugin_init_callbacks[plugin_init_callback_count++] = p_callback;
}

EditorPluginInitializeCallback EditorNode::plugin_init_callbacks[EditorNode::MAX_INIT_CALLBACKS];

int EditorNode::build_callback_count = 0;

void EditorNode::add_build_callback(EditorBuildCallback p_callback) {
	ERR_FAIL_COND(build_callback_count == MAX_INIT_CALLBACKS);

	build_callbacks[build_callback_count++] = p_callback;
}

EditorBuildCallback EditorNode::build_callbacks[EditorNode::MAX_BUILD_CALLBACKS];

bool EditorNode::call_build() {
	bool builds_successful = true;

	for (int i = 0; i < build_callback_count && builds_successful; i++) {
		if (!build_callbacks[i]()) {
			ERR_PRINT("A Godot Engine build callback failed.");
			builds_successful = false;
		}
	}

	if (builds_successful && !editor_data.call_build()) {
		ERR_PRINT("An EditorPlugin build callback failed.");
		builds_successful = false;
	}

	return builds_successful;
}

void EditorNode::_inherit_imported(const String &p_action) {
	open_imported->hide();
	load_scene(open_import_request, true, true);
}

void EditorNode::_open_imported() {
	load_scene(open_import_request, true, false, true, true);
}

void EditorNode::dim_editor(bool p_dimming) {
	dimmed = p_dimming;
	gui_base->set_modulate(p_dimming ? Color(0.5, 0.5, 0.5) : Color(1, 1, 1));
}

bool EditorNode::is_editor_dimmed() const {
	return dimmed;
}

void EditorNode::open_export_template_manager() {
	export_template_manager->popup_manager();
}

void EditorNode::add_resource_conversion_plugin(const Ref<EditorResourceConversionPlugin> &p_plugin) {
	resource_conversion_plugins.push_back(p_plugin);
}

void EditorNode::remove_resource_conversion_plugin(const Ref<EditorResourceConversionPlugin> &p_plugin) {
	resource_conversion_plugins.erase(p_plugin);
}

Vector<Ref<EditorResourceConversionPlugin>> EditorNode::find_resource_conversion_plugin(const Ref<Resource> &p_for_resource) {
	Vector<Ref<EditorResourceConversionPlugin>> ret;

	for (int i = 0; i < resource_conversion_plugins.size(); i++) {
		if (resource_conversion_plugins[i].is_valid() && resource_conversion_plugins[i]->handles(p_for_resource)) {
			ret.push_back(resource_conversion_plugins[i]);
		}
	}

	return ret;
}

void EditorNode::_bottom_panel_raise_toggled(bool p_pressed) {
	top_split->set_visible(!p_pressed);
}

void EditorNode::_update_renderer_color() {
	String rendering_method = renderer->get_selected_metadata();

	if (rendering_method == "forward_plus") {
		renderer->add_theme_color_override("font_color", Color::hex(0x5d8c3fff));
	}
	if (rendering_method == "mobile") {
		renderer->add_theme_color_override("font_color", Color::hex(0xa5557dff));
	}
	if (rendering_method == "gl_compatibility") {
		renderer->add_theme_color_override("font_color", Color::hex(0x5586a4ff));
	}
}

void EditorNode::_renderer_selected(int p_which) {
	String rendering_method = renderer->get_item_metadata(p_which);

	String current_renderer = GLOBAL_GET("rendering/renderer/rendering_method");

	if (rendering_method == current_renderer) {
		return;
	}

	renderer_request = rendering_method;
	video_restart_dialog->set_text(
			vformat(TTR("Changing the renderer requires restarting the editor.\n\nChoosing Save & Restart will change the rendering method to:\n- Desktop platforms: %s\n- Mobile platforms: %s\n- Web platform: gl_compatibility"),
					renderer_request, renderer_request.replace("forward_plus", "mobile")));
	video_restart_dialog->popup_centered();
	renderer->select(renderer_current);
	_update_renderer_color();
}

void EditorNode::_add_renderer_entry(const String &p_renderer_name, bool p_mark_overridden) {
	String item_text;
	if (p_renderer_name == "forward_plus") {
		item_text = TTR("Forward+");
	}
	if (p_renderer_name == "mobile") {
		item_text = TTR("Mobile");
	}
	if (p_renderer_name == "gl_compatibility") {
		item_text = TTR("Compatibility");
	}
	if (p_mark_overridden) {
		item_text += " " + TTR("(Overridden)");
	}
	renderer->add_item(item_text);
}

void EditorNode::_resource_saved(Ref<Resource> p_resource, const String &p_path) {
	if (singleton->saving_resources_in_path.has(p_resource)) {
		// This is going to be handled by save_resource_in_path when the time is right.
		return;
	}

	if (EditorFileSystem::get_singleton()) {
		EditorFileSystem::get_singleton()->update_file(p_path);
	}

	singleton->editor_folding.save_resource_folding(p_resource, p_path);
}

void EditorNode::_resource_loaded(Ref<Resource> p_resource, const String &p_path) {
	singleton->editor_folding.load_resource_folding(p_resource, p_path);
}

void EditorNode::_feature_profile_changed() {
	Ref<EditorFeatureProfile> profile = feature_profile_manager->get_current_profile();
	// FIXME: Close all floating docks to avoid crash.
	for (WindowWrapper *wrapper : floating_docks) {
		wrapper->set_window_enabled(false);
	}
	TabContainer *import_tabs = cast_to<TabContainer>(ImportDock::get_singleton()->get_parent());
	TabContainer *node_tabs = cast_to<TabContainer>(NodeDock::get_singleton()->get_parent());
	TabContainer *fs_tabs = cast_to<TabContainer>(FileSystemDock::get_singleton()->get_parent());
	TabContainer *history_tabs = cast_to<TabContainer>(history_dock->get_parent());
	if (profile.is_valid()) {
		node_tabs->set_tab_hidden(node_tabs->get_tab_idx_from_control(NodeDock::get_singleton()), profile->is_feature_disabled(EditorFeatureProfile::FEATURE_NODE_DOCK));
		// The Import dock is useless without the FileSystem dock. Ensure the configuration is valid.
		bool fs_dock_disabled = profile->is_feature_disabled(EditorFeatureProfile::FEATURE_FILESYSTEM_DOCK);
		fs_tabs->set_tab_hidden(fs_tabs->get_tab_idx_from_control(FileSystemDock::get_singleton()), fs_dock_disabled);
		import_tabs->set_tab_hidden(import_tabs->get_tab_idx_from_control(ImportDock::get_singleton()), fs_dock_disabled || profile->is_feature_disabled(EditorFeatureProfile::FEATURE_IMPORT_DOCK));
		history_tabs->set_tab_hidden(history_tabs->get_tab_idx_from_control(history_dock), profile->is_feature_disabled(EditorFeatureProfile::FEATURE_HISTORY_DOCK));

		main_editor_buttons[EDITOR_3D]->set_visible(!profile->is_feature_disabled(EditorFeatureProfile::FEATURE_3D));
		main_editor_buttons[EDITOR_SCRIPT]->set_visible(!profile->is_feature_disabled(EditorFeatureProfile::FEATURE_SCRIPT));
		if (AssetLibraryEditorPlugin::is_available()) {
			main_editor_buttons[EDITOR_ASSETLIB]->set_visible(!profile->is_feature_disabled(EditorFeatureProfile::FEATURE_ASSET_LIB));
		}
		if ((profile->is_feature_disabled(EditorFeatureProfile::FEATURE_3D) && singleton->main_editor_buttons[EDITOR_3D]->is_pressed()) ||
				(profile->is_feature_disabled(EditorFeatureProfile::FEATURE_SCRIPT) && singleton->main_editor_buttons[EDITOR_SCRIPT]->is_pressed()) ||
				(AssetLibraryEditorPlugin::is_available() && profile->is_feature_disabled(EditorFeatureProfile::FEATURE_ASSET_LIB) && singleton->main_editor_buttons[EDITOR_ASSETLIB]->is_pressed())) {
			editor_select(EDITOR_2D);
		}
	} else {
		import_tabs->set_tab_hidden(import_tabs->get_tab_idx_from_control(ImportDock::get_singleton()), false);
		node_tabs->set_tab_hidden(node_tabs->get_tab_idx_from_control(NodeDock::get_singleton()), false);
		fs_tabs->set_tab_hidden(fs_tabs->get_tab_idx_from_control(FileSystemDock::get_singleton()), false);
		history_tabs->set_tab_hidden(history_tabs->get_tab_idx_from_control(history_dock), false);
		main_editor_buttons[EDITOR_3D]->set_visible(true);
		main_editor_buttons[EDITOR_SCRIPT]->set_visible(true);
		if (AssetLibraryEditorPlugin::is_available()) {
			main_editor_buttons[EDITOR_ASSETLIB]->set_visible(true);
		}
	}

	_update_dock_slots_visibility();
}

void EditorNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("push_item", "object", "property", "inspector_only"), &EditorNode::push_item, DEFVAL(""), DEFVAL(false));

	ClassDB::bind_method("set_edited_scene", &EditorNode::set_edited_scene);
	ClassDB::bind_method("open_request", &EditorNode::open_request);
	ClassDB::bind_method("edit_foreign_resource", &EditorNode::edit_foreign_resource);
	ClassDB::bind_method("is_resource_read_only", &EditorNode::is_resource_read_only);

	ClassDB::bind_method("stop_child_process", &EditorNode::stop_child_process);

	ClassDB::bind_method("_set_main_scene_state", &EditorNode::_set_main_scene_state);
	ClassDB::bind_method("_update_recent_scenes", &EditorNode::_update_recent_scenes);

	ADD_SIGNAL(MethodInfo("request_help_search"));
	ADD_SIGNAL(MethodInfo("script_add_function_request", PropertyInfo(Variant::OBJECT, "obj"), PropertyInfo(Variant::STRING, "function"), PropertyInfo(Variant::PACKED_STRING_ARRAY, "args")));
	ADD_SIGNAL(MethodInfo("resource_saved", PropertyInfo(Variant::OBJECT, "obj")));
	ADD_SIGNAL(MethodInfo("scene_saved", PropertyInfo(Variant::STRING, "path")));
	ADD_SIGNAL(MethodInfo("scene_changed"));
	ADD_SIGNAL(MethodInfo("scene_closed", PropertyInfo(Variant::STRING, "path")));
}

static Node *_resource_get_edited_scene() {
	return EditorNode::get_singleton()->get_edited_scene();
}

void EditorNode::_print_handler(void *p_this, const String &p_string, bool p_error, bool p_rich) {
	callable_mp_static(&EditorNode::_print_handler_impl).bind(p_string, p_error, p_rich).call_deferred();
}

void EditorNode::_print_handler_impl(const String &p_string, bool p_error, bool p_rich) {
	if (!singleton) {
		return;
	}
	if (p_error) {
		singleton->log->add_message(p_string, EditorLog::MSG_TYPE_ERROR);
	} else if (p_rich) {
		singleton->log->add_message(p_string, EditorLog::MSG_TYPE_STD_RICH);
	} else {
		singleton->log->add_message(p_string, EditorLog::MSG_TYPE_STD);
	}
}

static void _execute_thread(void *p_ud) {
	EditorNode::ExecuteThreadArgs *eta = (EditorNode::ExecuteThreadArgs *)p_ud;
	Error err = OS::get_singleton()->execute(eta->path, eta->args, &eta->output, &eta->exitcode, true, &eta->execute_output_mutex);
	print_verbose("Thread exit status: " + itos(eta->exitcode));
	if (err != OK) {
		eta->exitcode = err;
	}

	eta->done.set();
}

int EditorNode::execute_and_show_output(const String &p_title, const String &p_path, const List<String> &p_arguments, bool p_close_on_ok, bool p_close_on_errors, String *r_output) {
	if (execute_output_dialog) {
		execute_output_dialog->set_title(p_title);
		execute_output_dialog->get_ok_button()->set_disabled(true);
		execute_outputs->clear();
		execute_outputs->set_scroll_follow(true);
		EditorInterface::get_singleton()->popup_dialog_centered_ratio(execute_output_dialog);
	}

	ExecuteThreadArgs eta;
	eta.path = p_path;
	eta.args = p_arguments;
	eta.exitcode = 255;

	int prev_len = 0;

	eta.execute_output_thread.start(_execute_thread, &eta);

	while (!eta.done.is_set()) {
		{
			MutexLock lock(eta.execute_output_mutex);
			if (prev_len != eta.output.length()) {
				String to_add = eta.output.substr(prev_len, eta.output.length());
				prev_len = eta.output.length();
				execute_outputs->add_text(to_add);
				DisplayServer::get_singleton()->process_events(); // Get rid of pending events.
				Main::iteration();
			}
		}
		OS::get_singleton()->delay_usec(1000);
	}

	eta.execute_output_thread.wait_to_finish();
	execute_outputs->add_text("\nExit Code: " + itos(eta.exitcode));

	if (execute_output_dialog) {
		if (p_close_on_errors && eta.exitcode != 0) {
			execute_output_dialog->hide();
		}
		if (p_close_on_ok && eta.exitcode == 0) {
			execute_output_dialog->hide();
		}

		execute_output_dialog->get_ok_button()->set_disabled(false);
	}

	if (r_output) {
		*r_output = eta.output;
	}
	return eta.exitcode;
}

EditorNode::EditorNode() {
	DEV_ASSERT(!singleton);
	singleton = this;

	Resource::_get_local_scene_func = _resource_get_edited_scene;

	{
		PortableCompressedTexture2D::set_keep_all_compressed_buffers(true);
		RenderingServer::get_singleton()->set_debug_generate_wireframes(true);

		AudioServer::get_singleton()->set_enable_tagging_used_audio_streams(true);

		// No navigation by default if in editor.
		if (NavigationServer3D::get_singleton()->get_debug_enabled()) {
			NavigationServer3D::get_singleton()->set_active(true);
		} else {
			NavigationServer3D::get_singleton()->set_active(false);
		}

		// No physics by default if in editor.
		PhysicsServer3D::get_singleton()->set_active(false);
		PhysicsServer2D::get_singleton()->set_active(false);

		// No scripting by default if in editor (except for tool).
		ScriptServer::set_scripting_enabled(false);

		Input::get_singleton()->set_use_accumulated_input(true);
		if (!DisplayServer::get_singleton()->is_touchscreen_available()) {
			// Only if no touchscreen ui hint, disable emulation just in case.
			Input::get_singleton()->set_emulate_touch_from_mouse(false);
		}
		DisplayServer::get_singleton()->cursor_set_custom_image(Ref<Resource>());
	}

	SceneState::set_disable_placeholders(true);
	ResourceLoader::clear_translation_remaps(); // Using no remaps if in editor.
	ResourceLoader::clear_path_remaps();
	ResourceLoader::set_create_missing_resources_if_class_unavailable(true);

	EditorPropertyNameProcessor *epnp = memnew(EditorPropertyNameProcessor);
	add_child(epnp);

	EditorUndoRedoManager::get_singleton()->connect("version_changed", callable_mp(this, &EditorNode::_update_undo_redo_allowed));
	EditorUndoRedoManager::get_singleton()->connect("history_changed", callable_mp(this, &EditorNode::_update_undo_redo_allowed));
	ProjectSettings::get_singleton()->connect("settings_changed", callable_mp(this, &EditorNode::_update_from_settings));
	GDExtensionManager::get_singleton()->connect("extensions_reloaded", callable_mp(this, &EditorNode::_gdextensions_reloaded));

	TranslationServer::get_singleton()->set_enabled(false);
	// Load settings.
	if (!EditorSettings::get_singleton()) {
		EditorSettings::create();
	}

	FileAccess::set_backup_save(EDITOR_GET("filesystem/on_save/safe_save_on_backup_then_rename"));

	// Warm up the surface upgrade tool as early as possible.
	surface_upgrade_tool = memnew(SurfaceUpgradeTool);
	run_surface_upgrade_tool = EditorSettings::get_singleton()->get_project_metadata("surface_upgrade_tool", "run_on_restart", false);
	if (run_surface_upgrade_tool) {
		SurfaceUpgradeTool::get_singleton()->begin_upgrade();
	}

	{
		int display_scale = EDITOR_GET("interface/editor/display_scale");

		switch (display_scale) {
			case 0:
				// Try applying a suitable display scale automatically.
				EditorScale::set_scale(EditorSettings::get_singleton()->get_auto_display_scale());
				break;
			case 1:
				EditorScale::set_scale(0.75);
				break;
			case 2:
				EditorScale::set_scale(1.0);
				break;
			case 3:
				EditorScale::set_scale(1.25);
				break;
			case 4:
				EditorScale::set_scale(1.5);
				break;
			case 5:
				EditorScale::set_scale(1.75);
				break;
			case 6:
				EditorScale::set_scale(2.0);
				break;
			default:
				EditorScale::set_scale(EDITOR_GET("interface/editor/custom_display_scale"));
				break;
		}
	}

	// Define a minimum window size to prevent UI elements from overlapping or being cut off.
	Window *w = Object::cast_to<Window>(SceneTree::get_singleton()->get_root());
	if (w) {
		const Size2 minimum_size = Size2(1024, 600) * EDSCALE;
		w->set_min_size(minimum_size); // Calling it this early doesn't sync the property with DS.
		DisplayServer::get_singleton()->window_set_min_size(minimum_size);
	}

	EditorFileDialog::set_default_show_hidden_files(EDITOR_GET("filesystem/file_dialog/show_hidden_files"));
	EditorFileDialog::set_default_display_mode((EditorFileDialog::DisplayMode)EDITOR_GET("filesystem/file_dialog/display_mode").operator int());

	int swap_cancel_ok = EDITOR_GET("interface/editor/accept_dialog_cancel_ok_buttons");
	if (swap_cancel_ok != 0) { // 0 is auto, set in register_scene based on DisplayServer.
		// Swap on means OK first.
		AcceptDialog::set_swap_cancel_ok(swap_cancel_ok == 2);
	}

	ResourceLoader::set_abort_on_missing_resources(false);
	ResourceLoader::set_error_notify_func(&EditorNode::add_io_error);
	ResourceLoader::set_dependency_error_notify_func(&EditorNode::_dependency_error_report);

	SceneState::set_instantiation_warning_notify_func([](const String &p_warning) {
		add_io_warning(p_warning);
		callable_mp(EditorInterface::get_singleton(), &EditorInterface::mark_scene_as_unsaved).call_deferred();
	});

	{
		// Register importers at the beginning, so dialogs are created with the right extensions.
		Ref<ResourceImporterTexture> import_texture = memnew(ResourceImporterTexture(true));
		ResourceFormatImporter::get_singleton()->add_importer(import_texture);

		Ref<ResourceImporterLayeredTexture> import_cubemap;
		import_cubemap.instantiate();
		import_cubemap->set_mode(ResourceImporterLayeredTexture::MODE_CUBEMAP);
		ResourceFormatImporter::get_singleton()->add_importer(import_cubemap);

		Ref<ResourceImporterLayeredTexture> import_array;
		import_array.instantiate();
		import_array->set_mode(ResourceImporterLayeredTexture::MODE_2D_ARRAY);
		ResourceFormatImporter::get_singleton()->add_importer(import_array);

		Ref<ResourceImporterLayeredTexture> import_cubemap_array;
		import_cubemap_array.instantiate();
		import_cubemap_array->set_mode(ResourceImporterLayeredTexture::MODE_CUBEMAP_ARRAY);
		ResourceFormatImporter::get_singleton()->add_importer(import_cubemap_array);

		Ref<ResourceImporterLayeredTexture> import_3d = memnew(ResourceImporterLayeredTexture(true));
		import_3d->set_mode(ResourceImporterLayeredTexture::MODE_3D);
		ResourceFormatImporter::get_singleton()->add_importer(import_3d);

		Ref<ResourceImporterImage> import_image;
		import_image.instantiate();
		ResourceFormatImporter::get_singleton()->add_importer(import_image);

		Ref<ResourceImporterTextureAtlas> import_texture_atlas;
		import_texture_atlas.instantiate();
		ResourceFormatImporter::get_singleton()->add_importer(import_texture_atlas);

		Ref<ResourceImporterDynamicFont> import_font_data_dynamic;
		import_font_data_dynamic.instantiate();
		ResourceFormatImporter::get_singleton()->add_importer(import_font_data_dynamic);

		Ref<ResourceImporterBMFont> import_font_data_bmfont;
		import_font_data_bmfont.instantiate();
		ResourceFormatImporter::get_singleton()->add_importer(import_font_data_bmfont);

		Ref<ResourceImporterImageFont> import_font_data_image;
		import_font_data_image.instantiate();
		ResourceFormatImporter::get_singleton()->add_importer(import_font_data_image);

		Ref<ResourceImporterCSVTranslation> import_csv_translation;
		import_csv_translation.instantiate();
		ResourceFormatImporter::get_singleton()->add_importer(import_csv_translation);

		Ref<ResourceImporterWAV> import_wav;
		import_wav.instantiate();
		ResourceFormatImporter::get_singleton()->add_importer(import_wav);

		Ref<ResourceImporterOBJ> import_obj;
		import_obj.instantiate();
		ResourceFormatImporter::get_singleton()->add_importer(import_obj);

		Ref<ResourceImporterShaderFile> import_shader_file;
		import_shader_file.instantiate();
		ResourceFormatImporter::get_singleton()->add_importer(import_shader_file);

		Ref<ResourceImporterScene> import_scene = memnew(ResourceImporterScene(false, true));
		ResourceFormatImporter::get_singleton()->add_importer(import_scene);

		Ref<ResourceImporterScene> import_animation = memnew(ResourceImporterScene(true, true));
		ResourceFormatImporter::get_singleton()->add_importer(import_animation);

		{
			Ref<EditorSceneFormatImporterCollada> import_collada;
			import_collada.instantiate();
			ResourceImporterScene::add_importer(import_collada);

			Ref<EditorOBJImporter> import_obj2;
			import_obj2.instantiate();
			ResourceImporterScene::add_importer(import_obj2);

			Ref<EditorSceneFormatImporterESCN> import_escn;
			import_escn.instantiate();
			ResourceImporterScene::add_importer(import_escn);
		}

		Ref<ResourceImporterBitMap> import_bitmap;
		import_bitmap.instantiate();
		ResourceFormatImporter::get_singleton()->add_importer(import_bitmap);
	}

	{
		Ref<EditorInspectorDefaultPlugin> eidp;
		eidp.instantiate();
		EditorInspector::add_inspector_plugin(eidp);

		Ref<EditorInspectorRootMotionPlugin> rmp;
		rmp.instantiate();
		EditorInspector::add_inspector_plugin(rmp);

		Ref<EditorInspectorVisualShaderModePlugin> smp;
		smp.instantiate();
		EditorInspector::add_inspector_plugin(smp);
	}

	editor_selection = memnew(EditorSelection);

	EditorFileSystem *efs = memnew(EditorFileSystem);
	add_child(efs);

	// Used for previews.
	FileDialog::get_icon_func = _file_dialog_get_icon;
	FileDialog::register_func = _file_dialog_register;
	FileDialog::unregister_func = _file_dialog_unregister;

	EditorFileDialog::get_icon_func = _file_dialog_get_icon;
	EditorFileDialog::register_func = _editor_file_dialog_register;
	EditorFileDialog::unregister_func = _editor_file_dialog_unregister;

	editor_export = memnew(EditorExport);
	add_child(editor_export);

	// Exporters might need the theme.
	EditorColorMap::create();
	EditorTheme::initialize();
	theme = create_custom_theme();
	DisplayServer::set_early_window_clear_color_override(true, theme->get_color(SNAME("background"), EditorStringName(Editor)));

	register_exporters();

	EDITOR_DEF("interface/editor/save_on_focus_loss", false);
	EDITOR_DEF("interface/editor/show_update_spinner", false);
	EDITOR_DEF("interface/editor/update_continuously", false);
	EDITOR_DEF("interface/editor/localize_settings", true);
	EDITOR_DEF_RST("interface/scene_tabs/restore_scenes_on_load", true);
	EDITOR_DEF_RST("interface/inspector/default_property_name_style", EditorPropertyNameProcessor::STYLE_CAPITALIZED);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::INT, "interface/inspector/default_property_name_style", PROPERTY_HINT_ENUM, "Raw,Capitalized,Localized"));
	EDITOR_DEF_RST("interface/inspector/default_float_step", 0.001);
	// The lowest value is equal to the minimum float step for 32-bit floats.
	// The step must be set manually, as changing this setting should not change the step here.
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::FLOAT, "interface/inspector/default_float_step", PROPERTY_HINT_RANGE, "0.0000001,1,0.0000001"));
	EDITOR_DEF_RST("interface/inspector/disable_folding", false);
	EDITOR_DEF_RST("interface/inspector/auto_unfold_foreign_scenes", true);
	EDITOR_DEF("interface/inspector/horizontal_vector2_editing", false);
	EDITOR_DEF("interface/inspector/horizontal_vector_types_editing", true);
	EDITOR_DEF("interface/inspector/open_resources_in_current_inspector", true);

	PackedStringArray open_in_new_inspector_defaults;
	// Required for the script editor to work.
	open_in_new_inspector_defaults.push_back("Script");
	// Required for the GridMap editor to work.
	open_in_new_inspector_defaults.push_back("MeshLibrary");
	EDITOR_DEF("interface/inspector/resources_to_open_in_new_inspector", open_in_new_inspector_defaults);

	EDITOR_DEF("interface/inspector/default_color_picker_mode", 0);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::INT, "interface/inspector/default_color_picker_mode", PROPERTY_HINT_ENUM, "RGB,HSV,RAW,OKHSL", PROPERTY_USAGE_DEFAULT));
	EDITOR_DEF("interface/inspector/default_color_picker_shape", (int32_t)ColorPicker::SHAPE_OKHSL_CIRCLE);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::INT, "interface/inspector/default_color_picker_shape", PROPERTY_HINT_ENUM, "HSV Rectangle,HSV Rectangle Wheel,VHS Circle,OKHSL Circle", PROPERTY_USAGE_DEFAULT));

	ED_SHORTCUT("canvas_item_editor/pan_view", TTR("Pan View"), Key::SPACE);

	const Vector<String> textfile_ext = ((String)(EDITOR_GET("docks/filesystem/textfile_extensions"))).split(",", false);
	for (const String &E : textfile_ext) {
		textfile_extensions.insert(E);
	}

	resource_preview = memnew(EditorResourcePreview);
	add_child(resource_preview);
	progress_dialog = memnew(ProgressDialog);
	progress_dialog->set_unparent_when_invisible(true);

	gui_base = memnew(Panel);
	add_child(gui_base);

	// Take up all screen.
	gui_base->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	gui_base->set_anchor(SIDE_RIGHT, Control::ANCHOR_END);
	gui_base->set_anchor(SIDE_BOTTOM, Control::ANCHOR_END);
	gui_base->set_end(Point2(0, 0));

	main_vbox = memnew(VBoxContainer);
	gui_base->add_child(main_vbox);

	title_bar = memnew(EditorTitleBar);
	main_vbox->add_child(title_bar);

	left_l_hsplit = memnew(HSplitContainer);
	main_vbox->add_child(left_l_hsplit);

	left_l_hsplit->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	left_l_vsplit = memnew(VSplitContainer);
	left_l_hsplit->add_child(left_l_vsplit);
	dock_slot[DOCK_SLOT_LEFT_UL] = memnew(TabContainer);
	left_l_vsplit->add_child(dock_slot[DOCK_SLOT_LEFT_UL]);
	dock_slot[DOCK_SLOT_LEFT_BL] = memnew(TabContainer);
	left_l_vsplit->add_child(dock_slot[DOCK_SLOT_LEFT_BL]);

	left_r_hsplit = memnew(HSplitContainer);
	left_l_hsplit->add_child(left_r_hsplit);
	left_r_vsplit = memnew(VSplitContainer);
	left_r_hsplit->add_child(left_r_vsplit);
	dock_slot[DOCK_SLOT_LEFT_UR] = memnew(TabContainer);
	left_r_vsplit->add_child(dock_slot[DOCK_SLOT_LEFT_UR]);
	dock_slot[DOCK_SLOT_LEFT_BR] = memnew(TabContainer);
	left_r_vsplit->add_child(dock_slot[DOCK_SLOT_LEFT_BR]);

	main_hsplit = memnew(HSplitContainer);
	left_r_hsplit->add_child(main_hsplit);
	VBoxContainer *center_vb = memnew(VBoxContainer);
	main_hsplit->add_child(center_vb);
	center_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	center_split = memnew(VSplitContainer);
	center_split->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	center_split->set_collapsed(false);
	center_vb->add_child(center_split);

	right_hsplit = memnew(HSplitContainer);
	main_hsplit->add_child(right_hsplit);

	right_l_vsplit = memnew(VSplitContainer);
	right_hsplit->add_child(right_l_vsplit);
	dock_slot[DOCK_SLOT_RIGHT_UL] = memnew(TabContainer);
	right_l_vsplit->add_child(dock_slot[DOCK_SLOT_RIGHT_UL]);
	dock_slot[DOCK_SLOT_RIGHT_BL] = memnew(TabContainer);
	right_l_vsplit->add_child(dock_slot[DOCK_SLOT_RIGHT_BL]);

	right_r_vsplit = memnew(VSplitContainer);
	right_hsplit->add_child(right_r_vsplit);
	dock_slot[DOCK_SLOT_RIGHT_UR] = memnew(TabContainer);
	right_r_vsplit->add_child(dock_slot[DOCK_SLOT_RIGHT_UR]);
	dock_slot[DOCK_SLOT_RIGHT_BR] = memnew(TabContainer);
	right_r_vsplit->add_child(dock_slot[DOCK_SLOT_RIGHT_BR]);

	// Store them for easier access.
	vsplits.push_back(left_l_vsplit);
	vsplits.push_back(left_r_vsplit);
	vsplits.push_back(right_l_vsplit);
	vsplits.push_back(right_r_vsplit);

	hsplits.push_back(left_l_hsplit);
	hsplits.push_back(left_r_hsplit);
	hsplits.push_back(main_hsplit);
	hsplits.push_back(right_hsplit);

	for (int i = 0; i < vsplits.size(); i++) {
		vsplits[i]->connect("dragged", callable_mp(this, &EditorNode::_dock_split_dragged));
		hsplits[i]->connect("dragged", callable_mp(this, &EditorNode::_dock_split_dragged));
	}

	dock_select_popup = memnew(PopupPanel);
	gui_base->add_child(dock_select_popup);
	VBoxContainer *dock_vb = memnew(VBoxContainer);
	dock_select_popup->add_child(dock_vb);

	HBoxContainer *dock_hb = memnew(HBoxContainer);
	dock_tab_move_left = memnew(Button);
	dock_tab_move_left->set_flat(true);
	dock_tab_move_left->set_focus_mode(Control::FOCUS_NONE);
	dock_tab_move_left->connect("pressed", callable_mp(this, &EditorNode::_dock_move_left));
	dock_hb->add_child(dock_tab_move_left);

	Label *dock_label = memnew(Label);
	dock_label->set_text(TTR("Dock Position"));
	dock_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	dock_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	dock_hb->add_child(dock_label);

	dock_tab_move_right = memnew(Button);
	dock_tab_move_right->set_flat(true);
	dock_tab_move_right->set_focus_mode(Control::FOCUS_NONE);
	dock_tab_move_right->connect("pressed", callable_mp(this, &EditorNode::_dock_move_right));

	dock_hb->add_child(dock_tab_move_right);
	dock_vb->add_child(dock_hb);

	dock_select = memnew(Control);
	dock_select->set_custom_minimum_size(Size2(128, 64) * EDSCALE);
	dock_select->connect("gui_input", callable_mp(this, &EditorNode::_dock_select_input));
	dock_select->connect("draw", callable_mp(this, &EditorNode::_dock_select_draw));
	dock_select->connect("mouse_exited", callable_mp(this, &EditorNode::_dock_popup_exit));
	dock_select->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	dock_vb->add_child(dock_select);

	if (!SceneTree::get_singleton()->get_root()->is_embedding_subwindows() && !EDITOR_GET("interface/editor/single_window_mode") && EDITOR_GET("interface/multi_window/enable")) {
		dock_float = memnew(Button);
		dock_float->set_icon(theme->get_icon("MakeFloating", EditorStringName(EditorIcons)));
		dock_float->set_text(TTR("Make Floating"));
		dock_float->set_focus_mode(Control::FOCUS_NONE);
		dock_float->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
		dock_float->connect("pressed", callable_mp(this, &EditorNode::_dock_make_selected_float));

		dock_vb->add_child(dock_float);
	}

	dock_select_popup->reset_size();

	for (int i = 0; i < DOCK_SLOT_MAX; i++) {
		dock_slot[i]->set_custom_minimum_size(Size2(170, 0) * EDSCALE);
		dock_slot[i]->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		dock_slot[i]->set_popup(dock_select_popup);
		dock_slot[i]->connect("pre_popup_pressed", callable_mp(this, &EditorNode::_dock_pre_popup).bind(i));
		dock_slot[i]->set_drag_to_rearrange_enabled(true);
		dock_slot[i]->set_tabs_rearrange_group(1);
		dock_slot[i]->connect("tab_changed", callable_mp(this, &EditorNode::_dock_tab_changed));
		dock_slot[i]->set_use_hidden_tabs_for_min_size(true);
	}

	editor_layout_save_delay_timer = memnew(Timer);
	add_child(editor_layout_save_delay_timer);
	editor_layout_save_delay_timer->set_wait_time(0.5);
	editor_layout_save_delay_timer->set_one_shot(true);
	editor_layout_save_delay_timer->connect("timeout", callable_mp(this, &EditorNode::_save_editor_layout));

	top_split = memnew(VSplitContainer);
	center_split->add_child(top_split);
	top_split->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	top_split->set_collapsed(true);

	VBoxContainer *srt = memnew(VBoxContainer);
	srt->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	srt->add_theme_constant_override("separation", 0);
	top_split->add_child(srt);

	scene_tabs = memnew(EditorSceneTabs);
	srt->add_child(scene_tabs);
	scene_tabs->connect("tab_changed", callable_mp(this, &EditorNode::_set_current_scene));
	scene_tabs->connect("tab_closed", callable_mp(this, &EditorNode::_scene_tab_closed));

	distraction_free = memnew(Button);
	distraction_free->set_flat(true);
	ED_SHORTCUT_AND_COMMAND("editor/distraction_free_mode", TTR("Distraction Free Mode"), KeyModifierMask::CTRL | KeyModifierMask::SHIFT | Key::F11);
	ED_SHORTCUT_OVERRIDE("editor/distraction_free_mode", "macos", KeyModifierMask::META | KeyModifierMask::CTRL | Key::D);
	distraction_free->set_shortcut(ED_GET_SHORTCUT("editor/distraction_free_mode"));
	distraction_free->set_tooltip_text(TTR("Toggle distraction-free mode."));
	distraction_free->set_toggle_mode(true);
	scene_tabs->add_extra_button(distraction_free);
	distraction_free->connect("pressed", callable_mp(this, &EditorNode::_toggle_distraction_free_mode));

	scene_root_parent = memnew(PanelContainer);
	scene_root_parent->set_custom_minimum_size(Size2(0, 80) * EDSCALE);
	scene_root_parent->add_theme_style_override("panel", theme->get_stylebox(SNAME("Content"), EditorStringName(EditorStyles)));
	scene_root_parent->set_draw_behind_parent(true);
	srt->add_child(scene_root_parent);
	scene_root_parent->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	scene_root = memnew(SubViewport);
	scene_root->set_embedding_subwindows(true);
	scene_root->set_disable_3d(true);
	scene_root->set_disable_input(true);
	scene_root->set_as_audio_listener_2d(true);

	main_screen_vbox = memnew(VBoxContainer);
	main_screen_vbox->set_name("MainScreen");
	main_screen_vbox->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	main_screen_vbox->add_theme_constant_override("separation", 0);
	scene_root_parent->add_child(main_screen_vbox);

	bool global_menu = !bool(EDITOR_GET("interface/editor/use_embedded_menu")) && DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_GLOBAL_MENU);
	bool can_expand = bool(EDITOR_GET("interface/editor/expand_to_title")) && DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_EXTEND_TO_TITLE);

	if (can_expand) {
		// Add spacer to avoid other controls under window minimize/maximize/close buttons (left side).
		left_menu_spacer = memnew(Control);
		left_menu_spacer->set_mouse_filter(Control::MOUSE_FILTER_PASS);
		title_bar->add_child(left_menu_spacer);
	}

	main_menu = memnew(MenuBar);
	title_bar->add_child(main_menu);

	main_menu->add_theme_style_override("hover", theme->get_stylebox(SNAME("MenuHover"), EditorStringName(EditorStyles)));
	main_menu->set_flat(true);
	main_menu->set_start_index(0); // Main menu, add to the start of global menu.
	main_menu->set_prefer_global_menu(global_menu);
	main_menu->set_switch_on_hover(true);

	file_menu = memnew(PopupMenu);
	file_menu->set_name(TTR("Scene"));
	main_menu->add_child(file_menu);
	main_menu->set_menu_tooltip(0, TTR("Operations with scene files."));

	accept = memnew(AcceptDialog);
	accept->set_unparent_when_invisible(true);

	save_accept = memnew(AcceptDialog);
	save_accept->set_unparent_when_invisible(true);
	save_accept->connect("confirmed", callable_mp(this, &EditorNode::_menu_option).bind((int)MenuOptions::FILE_SAVE_AS_SCENE));

	project_export = memnew(ProjectExportDialog);
	gui_base->add_child(project_export);

	dependency_error = memnew(DependencyErrorDialog);
	gui_base->add_child(dependency_error);

	dependency_fixer = memnew(DependencyEditor);
	gui_base->add_child(dependency_fixer);

	editor_settings_dialog = memnew(EditorSettingsDialog);
	gui_base->add_child(editor_settings_dialog);

	project_settings_editor = memnew(ProjectSettingsEditor(&editor_data));
	gui_base->add_child(project_settings_editor);

	scene_import_settings = memnew(SceneImportSettings);
	gui_base->add_child(scene_import_settings);

	audio_stream_import_settings = memnew(AudioStreamImportSettings);
	gui_base->add_child(audio_stream_import_settings);

	fontdata_import_settings = memnew(DynamicFontImportSettings);
	gui_base->add_child(fontdata_import_settings);

	export_template_manager = memnew(ExportTemplateManager);
	gui_base->add_child(export_template_manager);

	feature_profile_manager = memnew(EditorFeatureProfileManager);
	gui_base->add_child(feature_profile_manager);

	build_profile_manager = memnew(EditorBuildProfileManager);
	gui_base->add_child(build_profile_manager);

	about = memnew(EditorAbout);
	gui_base->add_child(about);
	feature_profile_manager->connect("current_feature_profile_changed", callable_mp(this, &EditorNode::_feature_profile_changed));

#if !defined(ANDROID_ENABLED) && !defined(WEB_ENABLED)
	fbx_importer_manager = memnew(FBXImporterManager);
	gui_base->add_child(fbx_importer_manager);
#endif

	warning = memnew(AcceptDialog);
	warning->set_unparent_when_invisible(true);
	warning->add_button(TTR("Copy Text"), true, "copy");
	warning->connect("custom_action", callable_mp(this, &EditorNode::_copy_warning));

	ED_SHORTCUT("editor/next_tab", TTR("Next Scene Tab"), KeyModifierMask::CMD_OR_CTRL + Key::TAB);
	ED_SHORTCUT("editor/prev_tab", TTR("Previous Scene Tab"), KeyModifierMask::CMD_OR_CTRL + KeyModifierMask::SHIFT + Key::TAB);
	ED_SHORTCUT("editor/filter_files", TTR("Focus FileSystem Filter"), KeyModifierMask::CMD_OR_CTRL + KeyModifierMask::ALT + Key::P);

	command_palette = EditorCommandPalette::get_singleton();
	command_palette->set_title(TTR("Command Palette"));
	gui_base->add_child(command_palette);

	file_menu->add_shortcut(ED_SHORTCUT_AND_COMMAND("editor/new_scene", TTR("New Scene"), KeyModifierMask::CMD_OR_CTRL + Key::N), FILE_NEW_SCENE);
	file_menu->add_shortcut(ED_SHORTCUT_AND_COMMAND("editor/new_inherited_scene", TTR("New Inherited Scene..."), KeyModifierMask::CMD_OR_CTRL + KeyModifierMask::SHIFT + Key::N), FILE_NEW_INHERITED_SCENE);
	file_menu->add_shortcut(ED_SHORTCUT_AND_COMMAND("editor/open_scene", TTR("Open Scene..."), KeyModifierMask::CMD_OR_CTRL + Key::O), FILE_OPEN_SCENE);
	file_menu->add_shortcut(ED_SHORTCUT_AND_COMMAND("editor/reopen_closed_scene", TTR("Reopen Closed Scene"), KeyModifierMask::CMD_OR_CTRL + KeyModifierMask::SHIFT + Key::T), FILE_OPEN_PREV);
	file_menu->add_submenu_item(TTR("Open Recent"), "RecentScenes", FILE_OPEN_RECENT);

	file_menu->add_separator();
	file_menu->add_shortcut(ED_SHORTCUT_AND_COMMAND("editor/save_scene", TTR("Save Scene"), KeyModifierMask::CMD_OR_CTRL + Key::S), FILE_SAVE_SCENE);
	file_menu->add_shortcut(ED_SHORTCUT_AND_COMMAND("editor/save_scene_as", TTR("Save Scene As..."), KeyModifierMask::CMD_OR_CTRL + KeyModifierMask::SHIFT + Key::S), FILE_SAVE_AS_SCENE);
	file_menu->add_shortcut(ED_SHORTCUT_AND_COMMAND("editor/save_all_scenes", TTR("Save All Scenes"), KeyModifierMask::CMD_OR_CTRL + KeyModifierMask::SHIFT + KeyModifierMask::ALT + Key::S), FILE_SAVE_ALL_SCENES);

	file_menu->add_separator();

	file_menu->add_shortcut(ED_SHORTCUT_ARRAY_AND_COMMAND("editor/quick_open", TTR("Quick Open..."), { int32_t(KeyModifierMask::SHIFT + KeyModifierMask::ALT + Key::O), int32_t(KeyModifierMask::CMD_OR_CTRL + Key::P) }), FILE_QUICK_OPEN);
	ED_SHORTCUT_OVERRIDE_ARRAY("editor/quick_open", "macos", { int32_t(KeyModifierMask::META + KeyModifierMask::CTRL + Key::O), int32_t(KeyModifierMask::CMD_OR_CTRL + Key::P) });
	file_menu->add_shortcut(ED_SHORTCUT_AND_COMMAND("editor/quick_open_scene", TTR("Quick Open Scene..."), KeyModifierMask::CMD_OR_CTRL + KeyModifierMask::SHIFT + Key::O), FILE_QUICK_OPEN_SCENE);
	file_menu->add_shortcut(ED_SHORTCUT_AND_COMMAND("editor/quick_open_script", TTR("Quick Open Script..."), KeyModifierMask::CMD_OR_CTRL + KeyModifierMask::ALT + Key::O), FILE_QUICK_OPEN_SCRIPT);

	file_menu->add_separator();
	export_as_menu = memnew(PopupMenu);
	export_as_menu->set_name("Export");
	file_menu->add_child(export_as_menu);
	file_menu->add_submenu_item(TTR("Export As..."), "Export");
	export_as_menu->add_shortcut(ED_SHORTCUT("editor/export_as_mesh_library", TTR("MeshLibrary...")), FILE_EXPORT_MESH_LIBRARY);
	export_as_menu->connect("index_pressed", callable_mp(this, &EditorNode::_export_as_menu_option));

	file_menu->add_separator();
	file_menu->add_shortcut(ED_GET_SHORTCUT("ui_undo"), EDIT_UNDO, true, true);
	file_menu->add_shortcut(ED_GET_SHORTCUT("ui_redo"), EDIT_REDO, true, true);

	file_menu->add_separator();
	file_menu->add_shortcut(ED_SHORTCUT_AND_COMMAND("editor/reload_saved_scene", TTR("Reload Saved Scene")), EDIT_RELOAD_SAVED_SCENE);
	file_menu->add_shortcut(ED_SHORTCUT_AND_COMMAND("editor/close_scene", TTR("Close Scene"), KeyModifierMask::CMD_OR_CTRL + KeyModifierMask::SHIFT + Key::W), FILE_CLOSE);

	recent_scenes = memnew(PopupMenu);
	recent_scenes->set_name("RecentScenes");
	file_menu->add_child(recent_scenes);
	recent_scenes->connect("id_pressed", callable_mp(this, &EditorNode::_open_recent_scene));

	if (!global_menu || !OS::get_singleton()->has_feature("macos")) {
		// On macOS  "Quit" and "About" options are in the "app" menu.
		file_menu->add_separator();
		file_menu->add_shortcut(ED_SHORTCUT_AND_COMMAND("editor/file_quit", TTR("Quit"), KeyModifierMask::CMD_OR_CTRL + Key::Q), FILE_QUIT, true);
	}

	project_menu = memnew(PopupMenu);
	project_menu->set_name(TTR("Project"));
	main_menu->add_child(project_menu);

	project_menu->add_shortcut(ED_SHORTCUT_AND_COMMAND("editor/project_settings", TTR("Project Settings..."), Key::NONE, TTR("Project Settings")), RUN_SETTINGS);
	project_menu->connect("id_pressed", callable_mp(this, &EditorNode::_menu_option));

	project_menu->add_separator();
	project_menu->add_item(TTR("Version Control"), VCS_MENU);

	project_menu->add_separator();
	project_menu->add_shortcut(ED_SHORTCUT_AND_COMMAND("editor/export", TTR("Export..."), Key::NONE, TTR("Export")), FILE_EXPORT_PROJECT);
#ifndef ANDROID_ENABLED
	project_menu->add_item(TTR("Install Android Build Template..."), FILE_INSTALL_ANDROID_SOURCE);
	project_menu->add_item(TTR("Open User Data Folder"), RUN_USER_DATA_FOLDER);
#endif

	project_menu->add_separator();
	project_menu->add_item(TTR("Customize Engine Build Configuration..."), TOOLS_BUILD_PROFILE_MANAGER);
	project_menu->add_separator();

	tool_menu = memnew(PopupMenu);
	tool_menu->set_name("Tools");
	tool_menu->connect("index_pressed", callable_mp(this, &EditorNode::_tool_menu_option));
	project_menu->add_child(tool_menu);
	project_menu->add_submenu_item(TTR("Tools"), "Tools");
	tool_menu->add_item(TTR("Orphan Resource Explorer..."), TOOLS_ORPHAN_RESOURCES);
	tool_menu->add_item(TTR("Upgrade Mesh Surfaces..."), TOOLS_SURFACE_UPGRADE);

	project_menu->add_separator();
	project_menu->add_shortcut(ED_SHORTCUT("editor/reload_current_project", TTR("Reload Current Project")), RELOAD_CURRENT_PROJECT);
	ED_SHORTCUT_AND_COMMAND("editor/quit_to_project_list", TTR("Quit to Project List"), KeyModifierMask::CTRL + KeyModifierMask::SHIFT + Key::Q);
	ED_SHORTCUT_OVERRIDE("editor/quit_to_project_list", "macos", KeyModifierMask::META + KeyModifierMask::CTRL + KeyModifierMask::ALT + Key::Q);
	project_menu->add_shortcut(ED_GET_SHORTCUT("editor/quit_to_project_list"), RUN_PROJECT_MANAGER, true);

	// Spacer to center 2D / 3D / Script buttons.
	HBoxContainer *left_spacer = memnew(HBoxContainer);
	left_spacer->set_mouse_filter(Control::MOUSE_FILTER_PASS);
	left_spacer->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	title_bar->add_child(left_spacer);

	if (can_expand && global_menu) {
		project_title = memnew(Label);
		project_title->add_theme_font_override("font", theme->get_font(SNAME("bold"), EditorStringName(EditorFonts)));
		project_title->add_theme_font_size_override("font_size", theme->get_font_size(SNAME("bold_size"), EditorStringName(EditorFonts)));
		project_title->set_focus_mode(Control::FOCUS_NONE);
		project_title->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
		project_title->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
		project_title->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		project_title->set_mouse_filter(Control::MOUSE_FILTER_PASS);
		left_spacer->add_child(project_title);
	}

	main_editor_button_hb = memnew(HBoxContainer);
	title_bar->add_child(main_editor_button_hb);

	// Options are added and handled by DebuggerEditorPlugin.
	debug_menu = memnew(PopupMenu);
	debug_menu->set_name(TTR("Debug"));
	main_menu->add_child(debug_menu);

	settings_menu = memnew(PopupMenu);
	settings_menu->set_name(TTR("Editor"));
	main_menu->add_child(settings_menu);

	ED_SHORTCUT_AND_COMMAND("editor/editor_settings", TTR("Editor Settings..."));
	ED_SHORTCUT_OVERRIDE("editor/editor_settings", "macos", KeyModifierMask::META + Key::COMMA);
	settings_menu->add_shortcut(ED_GET_SHORTCUT("editor/editor_settings"), SETTINGS_PREFERENCES);
	settings_menu->add_shortcut(ED_SHORTCUT("editor/command_palette", TTR("Command Palette..."), KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT | Key::P), HELP_COMMAND_PALETTE);
	settings_menu->add_separator();

	editor_layouts = memnew(PopupMenu);
	editor_layouts->set_name("Layouts");
	editor_layouts->set_auto_translate(false);
	settings_menu->add_child(editor_layouts);
	editor_layouts->connect("id_pressed", callable_mp(this, &EditorNode::_layout_menu_option));
	settings_menu->add_submenu_item(TTR("Editor Layout"), "Layouts");
	settings_menu->add_separator();

	ED_SHORTCUT_AND_COMMAND("editor/take_screenshot", TTR("Take Screenshot"), KeyModifierMask::CTRL | Key::F12);
	ED_SHORTCUT_OVERRIDE("editor/take_screenshot", "macos", KeyModifierMask::META | Key::F12);
	settings_menu->add_shortcut(ED_GET_SHORTCUT("editor/take_screenshot"), EDITOR_SCREENSHOT);

	settings_menu->set_item_tooltip(-1, TTR("Screenshots are stored in the Editor Data/Settings Folder."));

#ifndef ANDROID_ENABLED
	ED_SHORTCUT_AND_COMMAND("editor/fullscreen_mode", TTR("Toggle Fullscreen"), KeyModifierMask::SHIFT | Key::F11);
	ED_SHORTCUT_OVERRIDE("editor/fullscreen_mode", "macos", KeyModifierMask::META | KeyModifierMask::CTRL | Key::F);
	settings_menu->add_shortcut(ED_GET_SHORTCUT("editor/fullscreen_mode"), SETTINGS_TOGGLE_FULLSCREEN);
#endif
	settings_menu->add_separator();

#ifndef ANDROID_ENABLED
	if (OS::get_singleton()->get_data_path() == OS::get_singleton()->get_config_path()) {
		// Configuration and data folders are located in the same place (Windows/macOS).
		settings_menu->add_item(TTR("Open Editor Data/Settings Folder"), SETTINGS_EDITOR_DATA_FOLDER);
	} else {
		// Separate configuration and data folders (Linux).
		settings_menu->add_item(TTR("Open Editor Data Folder"), SETTINGS_EDITOR_DATA_FOLDER);
		settings_menu->add_item(TTR("Open Editor Settings Folder"), SETTINGS_EDITOR_CONFIG_FOLDER);
	}
	settings_menu->add_separator();
#endif

	settings_menu->add_item(TTR("Manage Editor Features..."), SETTINGS_MANAGE_FEATURE_PROFILES);
#ifndef ANDROID_ENABLED
	settings_menu->add_item(TTR("Manage Export Templates..."), SETTINGS_MANAGE_EXPORT_TEMPLATES);
#endif
#if !defined(ANDROID_ENABLED) && !defined(WEB_ENABLED)
	settings_menu->add_item(TTR("Configure FBX Importer..."), SETTINGS_MANAGE_FBX_IMPORTER);
#endif

	help_menu = memnew(PopupMenu);
	help_menu->set_name(TTR("Help"));
	main_menu->add_child(help_menu);

	help_menu->connect("id_pressed", callable_mp(this, &EditorNode::_menu_option));

	ED_SHORTCUT_AND_COMMAND("editor/editor_help", TTR("Search Help"), Key::F1);
	ED_SHORTCUT_OVERRIDE("editor/editor_help", "macos", KeyModifierMask::ALT | Key::SPACE);
	help_menu->add_icon_shortcut(theme->get_icon(SNAME("HelpSearch"), EditorStringName(EditorIcons)), ED_GET_SHORTCUT("editor/editor_help"), HELP_SEARCH);
	help_menu->add_separator();
	help_menu->add_icon_shortcut(theme->get_icon(SNAME("ExternalLink"), EditorStringName(EditorIcons)), ED_SHORTCUT_AND_COMMAND("editor/online_docs", TTR("Online Documentation")), HELP_DOCS);
	help_menu->add_icon_shortcut(theme->get_icon(SNAME("ExternalLink"), EditorStringName(EditorIcons)), ED_SHORTCUT_AND_COMMAND("editor/q&a", TTR("Questions & Answers")), HELP_QA);
	help_menu->add_icon_shortcut(theme->get_icon(SNAME("ExternalLink"), EditorStringName(EditorIcons)), ED_SHORTCUT_AND_COMMAND("editor/community", TTR("Community")), HELP_COMMUNITY);
	help_menu->add_separator();
	help_menu->add_icon_shortcut(theme->get_icon(SNAME("ActionCopy"), EditorStringName(EditorIcons)), ED_SHORTCUT_AND_COMMAND("editor/copy_system_info", TTR("Copy System Info")), HELP_COPY_SYSTEM_INFO);
	help_menu->set_item_tooltip(-1, TTR("Copies the system info as a single-line text into the clipboard."));
	help_menu->add_icon_shortcut(theme->get_icon(SNAME("ExternalLink"), EditorStringName(EditorIcons)), ED_SHORTCUT_AND_COMMAND("editor/report_a_bug", TTR("Report a Bug")), HELP_REPORT_A_BUG);
	help_menu->add_icon_shortcut(theme->get_icon(SNAME("ExternalLink"), EditorStringName(EditorIcons)), ED_SHORTCUT_AND_COMMAND("editor/suggest_a_feature", TTR("Suggest a Feature")), HELP_SUGGEST_A_FEATURE);
	help_menu->add_icon_shortcut(theme->get_icon(SNAME("ExternalLink"), EditorStringName(EditorIcons)), ED_SHORTCUT_AND_COMMAND("editor/send_docs_feedback", TTR("Send Docs Feedback")), HELP_SEND_DOCS_FEEDBACK);
	help_menu->add_separator();
	if (!global_menu || !OS::get_singleton()->has_feature("macos")) {
		// On macOS  "Quit" and "About" options are in the "app" menu.
		help_menu->add_icon_shortcut(theme->get_icon(SNAME("Godot"), EditorStringName(EditorIcons)), ED_SHORTCUT_AND_COMMAND("editor/about", TTR("About Godot")), HELP_ABOUT);
	}
	help_menu->add_icon_shortcut(theme->get_icon(SNAME("Heart"), EditorStringName(EditorIcons)), ED_SHORTCUT_AND_COMMAND("editor/support_development", TTR("Support Godot Development")), HELP_SUPPORT_GODOT_DEVELOPMENT);

	// Spacer to center 2D / 3D / Script buttons.
	Control *right_spacer = memnew(Control);
	right_spacer->set_mouse_filter(Control::MOUSE_FILTER_PASS);
	right_spacer->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	title_bar->add_child(right_spacer);

	project_run_bar = memnew(EditorRunBar);
	title_bar->add_child(project_run_bar);
	project_run_bar->connect("play_pressed", callable_mp(this, &EditorNode::_project_run_started));
	project_run_bar->connect("stop_pressed", callable_mp(this, &EditorNode::_project_run_stopped));

	HBoxContainer *right_menu_hb = memnew(HBoxContainer);
	title_bar->add_child(right_menu_hb);

	renderer = memnew(OptionButton);
	renderer->set_visible(true);
	renderer->set_flat(true);
	renderer->set_fit_to_longest_item(false);
	renderer->set_focus_mode(Control::FOCUS_NONE);
	renderer->add_theme_font_override("font", theme->get_font(SNAME("bold"), EditorStringName(EditorFonts)));
	renderer->add_theme_font_size_override("font_size", theme->get_font_size(SNAME("bold_size"), EditorStringName(EditorFonts)));
	renderer->set_tooltip_text(TTR("Choose a rendering method.\n\nNotes:\n- On mobile platforms, the Mobile rendering method is used if Forward+ is selected here.\n- On the web platform, the Compatibility rendering method is always used."));

	right_menu_hb->add_child(renderer);

	if (can_expand) {
		// Add spacer to avoid other controls under the window minimize/maximize/close buttons (right side).
		right_menu_spacer = memnew(Control);
		right_menu_spacer->set_mouse_filter(Control::MOUSE_FILTER_PASS);
		title_bar->add_child(right_menu_spacer);
	}

	String current_renderer_ps = GLOBAL_GET("rendering/renderer/rendering_method");
	current_renderer_ps = current_renderer_ps.to_lower();
	String current_renderer_os = OS::get_singleton()->get_current_rendering_method().to_lower();

	// Add the renderers name to the UI.
	if (current_renderer_ps == current_renderer_os) {
		renderer->connect("item_selected", callable_mp(this, &EditorNode::_renderer_selected));
		// As we are doing string comparisons, keep in standard case to prevent problems with capitals
		// "vulkan" in particular uses lowercase "v" in the code, and uppercase in the UI.
		PackedStringArray renderers = ProjectSettings::get_singleton()->get_custom_property_info().get(StringName("rendering/renderer/rendering_method")).hint_string.split(",", false);
		for (int i = 0; i < renderers.size(); i++) {
			String rendering_method = renderers[i];
			_add_renderer_entry(rendering_method, false);
			renderer->set_item_metadata(i, rendering_method);
			// Lowercase for standard comparison.
			rendering_method = rendering_method.to_lower();
			if (current_renderer_ps == rendering_method) {
				renderer->select(i);
				renderer_current = i;
			}
		}
	} else {
		// It's an CLI-overridden rendering method.
		_add_renderer_entry(current_renderer_os, true);
		renderer->set_item_metadata(0, current_renderer_os);
		renderer->select(0);
		renderer_current = 0;
	}
	_update_renderer_color();

	video_restart_dialog = memnew(ConfirmationDialog);
	video_restart_dialog->set_ok_button_text(TTR("Save & Restart"));
	video_restart_dialog->connect("confirmed", callable_mp(this, &EditorNode::_menu_option).bind(SET_RENDERER_NAME_SAVE_AND_RESTART));
	gui_base->add_child(video_restart_dialog);

	progress_hb = memnew(BackgroundProgress);

	layout_dialog = memnew(EditorLayoutsDialog);
	gui_base->add_child(layout_dialog);
	layout_dialog->set_hide_on_ok(false);
	layout_dialog->set_size(Size2(225, 270) * EDSCALE);
	layout_dialog->connect("name_confirmed", callable_mp(this, &EditorNode::_dialog_action));

	update_spinner = memnew(MenuButton);
	right_menu_hb->add_child(update_spinner);
	update_spinner->set_icon(theme->get_icon(SNAME("Progress1"), EditorStringName(EditorIcons)));
	update_spinner->get_popup()->connect("id_pressed", callable_mp(this, &EditorNode::_menu_option));
	PopupMenu *p = update_spinner->get_popup();
	p->add_radio_check_item(TTR("Update Continuously"), SETTINGS_UPDATE_CONTINUOUSLY);
	p->add_radio_check_item(TTR("Update When Changed"), SETTINGS_UPDATE_WHEN_CHANGED);
	p->add_separator();
	p->add_item(TTR("Hide Update Spinner"), SETTINGS_UPDATE_SPINNER_HIDE);
	_update_update_spinner();

	// Instantiate and place editor docks.

	memnew(SceneTreeDock(scene_root, editor_selection, editor_data));
	memnew(InspectorDock(editor_data));
	memnew(ImportDock);
	memnew(NodeDock);

	FileSystemDock *filesystem_dock = memnew(FileSystemDock);
	filesystem_dock->connect("inherit", callable_mp(this, &EditorNode::_inherit_request));
	filesystem_dock->connect("instantiate", callable_mp(this, &EditorNode::_instantiate_request));
	filesystem_dock->connect("display_mode_changed", callable_mp(this, &EditorNode::_save_editor_layout));
	get_project_settings()->connect_filesystem_dock_signals(filesystem_dock);

	history_dock = memnew(HistoryDock);

	// Scene: Top left.
	dock_slot[DOCK_SLOT_LEFT_UR]->add_child(SceneTreeDock::get_singleton());
	dock_slot[DOCK_SLOT_LEFT_UR]->set_tab_title(dock_slot[DOCK_SLOT_LEFT_UR]->get_tab_idx_from_control(SceneTreeDock::get_singleton()), TTR("Scene"));

	// Import: Top left, behind Scene.
	dock_slot[DOCK_SLOT_LEFT_UR]->add_child(ImportDock::get_singleton());
	dock_slot[DOCK_SLOT_LEFT_UR]->set_tab_title(dock_slot[DOCK_SLOT_LEFT_UR]->get_tab_idx_from_control(ImportDock::get_singleton()), TTR("Import"));

	// FileSystem: Bottom left.
	dock_slot[DOCK_SLOT_LEFT_BR]->add_child(FileSystemDock::get_singleton());
	dock_slot[DOCK_SLOT_LEFT_BR]->set_tab_title(dock_slot[DOCK_SLOT_LEFT_BR]->get_tab_idx_from_control(FileSystemDock::get_singleton()), TTR("FileSystem"));

	// Inspector: Full height right.
	dock_slot[DOCK_SLOT_RIGHT_UL]->add_child(InspectorDock::get_singleton());
	dock_slot[DOCK_SLOT_RIGHT_UL]->set_tab_title(dock_slot[DOCK_SLOT_RIGHT_UL]->get_tab_idx_from_control(InspectorDock::get_singleton()), TTR("Inspector"));

	// Node: Full height right, behind Inspector.
	dock_slot[DOCK_SLOT_RIGHT_UL]->add_child(NodeDock::get_singleton());
	dock_slot[DOCK_SLOT_RIGHT_UL]->set_tab_title(dock_slot[DOCK_SLOT_RIGHT_UL]->get_tab_idx_from_control(NodeDock::get_singleton()), TTR("Node"));

	// History: Full height right, behind Node.
	dock_slot[DOCK_SLOT_RIGHT_UL]->add_child(history_dock);
	dock_slot[DOCK_SLOT_RIGHT_UL]->set_tab_title(dock_slot[DOCK_SLOT_RIGHT_UL]->get_tab_idx_from_control(history_dock), TTR("History"));

	// Hide unused dock slots and vsplits.
	dock_slot[DOCK_SLOT_LEFT_UL]->hide();
	dock_slot[DOCK_SLOT_LEFT_BL]->hide();
	dock_slot[DOCK_SLOT_RIGHT_BL]->hide();
	dock_slot[DOCK_SLOT_RIGHT_UR]->hide();
	dock_slot[DOCK_SLOT_RIGHT_BR]->hide();
	left_l_vsplit->hide();
	right_r_vsplit->hide();

	// Add some offsets to left_r and main hsplits to make LEFT_R and RIGHT_L docks wider than minsize.
	left_r_hsplit->set_split_offset(270 * EDSCALE);
	main_hsplit->set_split_offset(-270 * EDSCALE);

	// Define corresponding default layout.

	const String docks_section = "docks";
	default_layout.instantiate();
	// Dock numbers are based on DockSlot enum value + 1.
	default_layout->set_value(docks_section, "dock_3", "Scene,Import");
	default_layout->set_value(docks_section, "dock_4", "FileSystem");
	default_layout->set_value(docks_section, "dock_5", "Inspector,Node,History");

	for (int i = 0; i < vsplits.size(); i++) {
		default_layout->set_value(docks_section, "dock_split_" + itos(i + 1), 0);
	}
	default_layout->set_value(docks_section, "dock_hsplit_1", 0);
	default_layout->set_value(docks_section, "dock_hsplit_2", 270 * EDSCALE);
	default_layout->set_value(docks_section, "dock_hsplit_3", -270 * EDSCALE);
	default_layout->set_value(docks_section, "dock_hsplit_4", 0);

	_update_layouts_menu();

	// Bottom panels.

	bottom_panel = memnew(PanelContainer);
	bottom_panel->add_theme_style_override("panel", theme->get_stylebox(SNAME("BottomPanel"), EditorStringName(EditorStyles)));
	center_split->add_child(bottom_panel);
	center_split->set_dragger_visibility(SplitContainer::DRAGGER_HIDDEN);

	bottom_panel_vb = memnew(VBoxContainer);
	bottom_panel->add_child(bottom_panel_vb);

	bottom_panel_hb = memnew(HBoxContainer);
	bottom_panel_hb->set_custom_minimum_size(Size2(0, 24 * EDSCALE)); // Adjust for the height of the "Expand Bottom Dock" icon.
	bottom_panel_vb->add_child(bottom_panel_hb);

	bottom_panel_hb_editors = memnew(HBoxContainer);
	bottom_panel_hb_editors->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	bottom_panel_hb->add_child(bottom_panel_hb_editors);

	editor_toaster = memnew(EditorToaster);
	bottom_panel_hb->add_child(editor_toaster);

	VBoxContainer *version_info_vbc = memnew(VBoxContainer);
	bottom_panel_hb->add_child(version_info_vbc);

	// Add a dummy control node for vertical spacing.
	Control *v_spacer = memnew(Control);
	version_info_vbc->add_child(v_spacer);

	version_btn = memnew(LinkButton);
	version_btn->set_text(VERSION_FULL_CONFIG);
	String hash = String(VERSION_HASH);
	if (hash.length() != 0) {
		hash = " " + vformat("[%s]", hash.left(9));
	}
	// Set the text to copy in metadata as it slightly differs from the button's text.
	version_btn->set_meta(META_TEXT_TO_COPY, "v" VERSION_FULL_BUILD + hash);
	// Fade out the version label to be less prominent, but still readable.
	version_btn->set_self_modulate(Color(1, 1, 1, 0.65));
	version_btn->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
	version_btn->set_tooltip_text(TTR("Click to copy."));
	version_btn->connect("pressed", callable_mp(this, &EditorNode::_version_button_pressed));
	version_info_vbc->add_child(version_btn);

	// Add a dummy control node for horizontal spacing.
	Control *h_spacer = memnew(Control);
	bottom_panel_hb->add_child(h_spacer);

	bottom_panel_raise = memnew(Button);
	bottom_panel_hb->add_child(bottom_panel_raise);
	bottom_panel_raise->hide();
	bottom_panel_raise->set_flat(true);
	bottom_panel_raise->set_toggle_mode(true);
	bottom_panel_raise->set_shortcut(ED_SHORTCUT_AND_COMMAND("editor/bottom_panel_expand", TTR("Expand Bottom Panel"), KeyModifierMask::SHIFT | Key::F12));
	bottom_panel_raise->connect("toggled", callable_mp(this, &EditorNode::_bottom_panel_raise_toggled));

	log = memnew(EditorLog);
	Button *output_button = add_bottom_panel_item(TTR("Output"), log);
	log->set_tool_button(output_button);

	center_split->connect("resized", callable_mp(this, &EditorNode::_vp_resized));

	native_shader_source_visualizer = memnew(EditorNativeShaderSourceVisualizer);
	gui_base->add_child(native_shader_source_visualizer);

	orphan_resources = memnew(OrphanResourcesDialog);
	gui_base->add_child(orphan_resources);

	surface_upgrade_dialog = memnew(SurfaceUpgradeDialog);
	gui_base->add_child(surface_upgrade_dialog);

	confirmation = memnew(ConfirmationDialog);
	gui_base->add_child(confirmation);
	confirmation->connect("confirmed", callable_mp(this, &EditorNode::_menu_confirm_current));

	save_confirmation = memnew(ConfirmationDialog);
	save_confirmation->add_button(TTR("Don't Save"), DisplayServer::get_singleton()->get_swap_cancel_ok(), "discard");
	gui_base->add_child(save_confirmation);
	save_confirmation->set_min_size(Vector2(450.0 * EDSCALE, 0));
	save_confirmation->connect("confirmed", callable_mp(this, &EditorNode::_menu_confirm_current));
	save_confirmation->connect("custom_action", callable_mp(this, &EditorNode::_discard_changes));

	gradle_build_manage_templates = memnew(ConfirmationDialog);
	gradle_build_manage_templates->set_text(TTR("Android build template is missing, please install relevant templates."));
	gradle_build_manage_templates->set_ok_button_text(TTR("Manage Templates"));
	gradle_build_manage_templates->add_button(TTR("Install from file"))->connect("pressed", callable_mp(this, &EditorNode::_menu_option).bind(SETTINGS_INSTALL_ANDROID_BUILD_TEMPLATE));
	gradle_build_manage_templates->connect("confirmed", callable_mp(this, &EditorNode::_menu_option).bind(SETTINGS_MANAGE_EXPORT_TEMPLATES));
	gui_base->add_child(gradle_build_manage_templates);

	file_android_build_source = memnew(EditorFileDialog);
	file_android_build_source->set_title(TTR("Select Android sources file"));
	file_android_build_source->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	file_android_build_source->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	file_android_build_source->add_filter("*.zip");
	file_android_build_source->connect("file_selected", callable_mp(this, &EditorNode::_android_build_source_selected));
	gui_base->add_child(file_android_build_source);

	install_android_build_template = memnew(ConfirmationDialog);
	install_android_build_template->set_text(TTR("This will set up your project for gradle Android builds by installing the source template to \"res://android/build\".\nYou can then apply modifications and build your own custom APK on export (adding modules, changing the AndroidManifest.xml, etc.).\nNote that in order to make gradle builds instead of using pre-built APKs, the \"Use Gradle Build\" option should be enabled in the Android export preset."));
	install_android_build_template->set_ok_button_text(TTR("Install"));
	install_android_build_template->connect("confirmed", callable_mp(this, &EditorNode::_menu_confirm_current));
	gui_base->add_child(install_android_build_template);

	remove_android_build_template = memnew(ConfirmationDialog);
	remove_android_build_template->set_text(TTR("The Android build template is already installed in this project and it won't be overwritten.\nRemove the \"res://android/build\" directory manually before attempting this operation again."));
	remove_android_build_template->set_ok_button_text(TTR("Show in File Manager"));
	remove_android_build_template->connect("confirmed", callable_mp(this, &EditorNode::_menu_option).bind(FILE_EXPLORE_ANDROID_BUILD_TEMPLATES));
	gui_base->add_child(remove_android_build_template);

	file_templates = memnew(EditorFileDialog);
	file_templates->set_title(TTR("Import Templates From ZIP File"));

	gui_base->add_child(file_templates);
	file_templates->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	file_templates->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	file_templates->clear_filters();
	file_templates->add_filter("*.tpz", TTR("Template Package"));

	file = memnew(EditorFileDialog);
	gui_base->add_child(file);
	file->set_current_dir("res://");

	file_export_lib = memnew(EditorFileDialog);
	file_export_lib->set_title(TTR("Export Library"));
	file_export_lib->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	file_export_lib->connect("file_selected", callable_mp(this, &EditorNode::_dialog_action));
	file_export_lib_merge = memnew(CheckBox);
	file_export_lib_merge->set_text(TTR("Merge With Existing"));
	file_export_lib_merge->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	file_export_lib_merge->set_pressed(true);
	file_export_lib->get_vbox()->add_child(file_export_lib_merge);
	file_export_lib_apply_xforms = memnew(CheckBox);
	file_export_lib_apply_xforms->set_text(TTR("Apply MeshInstance Transforms"));
	file_export_lib_apply_xforms->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	file_export_lib_apply_xforms->set_pressed(false);
	file_export_lib->get_vbox()->add_child(file_export_lib_apply_xforms);
	gui_base->add_child(file_export_lib);

	file_script = memnew(EditorFileDialog);
	file_script->set_title(TTR("Open & Run a Script"));
	file_script->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	file_script->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	List<String> sexts;
	ResourceLoader::get_recognized_extensions_for_type("Script", &sexts);
	for (const String &E : sexts) {
		file_script->add_filter("*." + E);
	}
	gui_base->add_child(file_script);
	file_script->connect("file_selected", callable_mp(this, &EditorNode::_dialog_action));

	file_menu->connect("id_pressed", callable_mp(this, &EditorNode::_menu_option));
	file_menu->connect("about_to_popup", callable_mp(this, &EditorNode::_update_file_menu_opened));
	file_menu->connect("popup_hide", callable_mp(this, &EditorNode::_update_file_menu_closed));

	settings_menu->connect("id_pressed", callable_mp(this, &EditorNode::_menu_option));

	file->connect("file_selected", callable_mp(this, &EditorNode::_dialog_action));
	file_templates->connect("file_selected", callable_mp(this, &EditorNode::_dialog_action));

	audio_preview_gen = memnew(AudioStreamPreviewGenerator);
	add_child(audio_preview_gen);

	add_editor_plugin(memnew(DebuggerEditorPlugin(debug_menu)));

	disk_changed = memnew(ConfirmationDialog);
	{
		VBoxContainer *vbc = memnew(VBoxContainer);
		disk_changed->add_child(vbc);

		Label *dl = memnew(Label);
		dl->set_text(TTR("The following files are newer on disk.\nWhat action should be taken?"));
		vbc->add_child(dl);

		disk_changed_list = memnew(Tree);
		vbc->add_child(disk_changed_list);
		disk_changed_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);

		disk_changed->connect("confirmed", callable_mp(this, &EditorNode::_reload_modified_scenes));
		disk_changed->connect("confirmed", callable_mp(this, &EditorNode::_reload_project_settings));
		disk_changed->set_ok_button_text(TTR("Reload"));

		disk_changed->add_button(TTR("Resave"), !DisplayServer::get_singleton()->get_swap_cancel_ok(), "resave");
		disk_changed->connect("custom_action", callable_mp(this, &EditorNode::_resave_scenes));
	}

	gui_base->add_child(disk_changed);

	add_editor_plugin(memnew(AnimationPlayerEditorPlugin));
	add_editor_plugin(memnew(AnimationTrackKeyEditEditorPlugin));
	add_editor_plugin(memnew(CanvasItemEditorPlugin));
	add_editor_plugin(memnew(Node3DEditorPlugin));
	add_editor_plugin(memnew(ScriptEditorPlugin));

	EditorAudioBuses *audio_bus_editor = EditorAudioBuses::register_editor();

	ScriptTextEditor::register_editor(); // Register one for text scripts.
	TextEditor::register_editor();

	if (AssetLibraryEditorPlugin::is_available()) {
		add_editor_plugin(memnew(AssetLibraryEditorPlugin));
	} else {
		print_verbose("Asset Library not available (due to using Web editor, or SSL support disabled).");
	}

	// More visually meaningful to have this later.
	raise_bottom_panel_item(AnimationPlayerEditor::get_singleton());

	add_editor_plugin(VersionControlEditorPlugin::get_singleton());

	vcs_actions_menu = VersionControlEditorPlugin::get_singleton()->get_version_control_actions_panel();
	vcs_actions_menu->set_name("Version Control");
	vcs_actions_menu->connect("index_pressed", callable_mp(this, &EditorNode::_version_control_menu_option));
	vcs_actions_menu->add_item(TTR("Create Version Control Metadata..."), RUN_VCS_METADATA);
	vcs_actions_menu->add_item(TTR("Version Control Settings..."), RUN_VCS_SETTINGS);
	project_menu->add_child(vcs_actions_menu);
	project_menu->set_item_submenu(project_menu->get_item_index(VCS_MENU), "Version Control");

	add_editor_plugin(memnew(AudioBusesEditorPlugin(audio_bus_editor)));

	for (int i = 0; i < EditorPlugins::get_plugin_count(); i++) {
		add_editor_plugin(EditorPlugins::create(i));
	}

	for (const StringName &extension_class_name : GDExtensionEditorPlugins::get_extension_classes()) {
		add_extension_editor_plugin(extension_class_name);
	}
	GDExtensionEditorPlugins::editor_node_add_plugin = &EditorNode::add_extension_editor_plugin;
	GDExtensionEditorPlugins::editor_node_remove_plugin = &EditorNode::remove_extension_editor_plugin;

	for (int i = 0; i < plugin_init_callback_count; i++) {
		plugin_init_callbacks[i]();
	}

	resource_preview->add_preview_generator(Ref<EditorTexturePreviewPlugin>(memnew(EditorTexturePreviewPlugin)));
	resource_preview->add_preview_generator(Ref<EditorImagePreviewPlugin>(memnew(EditorImagePreviewPlugin)));
	resource_preview->add_preview_generator(Ref<EditorPackedScenePreviewPlugin>(memnew(EditorPackedScenePreviewPlugin)));
	resource_preview->add_preview_generator(Ref<EditorMaterialPreviewPlugin>(memnew(EditorMaterialPreviewPlugin)));
	resource_preview->add_preview_generator(Ref<EditorScriptPreviewPlugin>(memnew(EditorScriptPreviewPlugin)));
	resource_preview->add_preview_generator(Ref<EditorAudioStreamPreviewPlugin>(memnew(EditorAudioStreamPreviewPlugin)));
	resource_preview->add_preview_generator(Ref<EditorMeshPreviewPlugin>(memnew(EditorMeshPreviewPlugin)));
	resource_preview->add_preview_generator(Ref<EditorBitmapPreviewPlugin>(memnew(EditorBitmapPreviewPlugin)));
	resource_preview->add_preview_generator(Ref<EditorFontPreviewPlugin>(memnew(EditorFontPreviewPlugin)));
	resource_preview->add_preview_generator(Ref<EditorGradientPreviewPlugin>(memnew(EditorGradientPreviewPlugin)));

	{
		Ref<StandardMaterial3DConversionPlugin> spatial_mat_convert;
		spatial_mat_convert.instantiate();
		resource_conversion_plugins.push_back(spatial_mat_convert);

		Ref<ORMMaterial3DConversionPlugin> orm_mat_convert;
		orm_mat_convert.instantiate();
		resource_conversion_plugins.push_back(orm_mat_convert);

		Ref<CanvasItemMaterialConversionPlugin> canvas_item_mat_convert;
		canvas_item_mat_convert.instantiate();
		resource_conversion_plugins.push_back(canvas_item_mat_convert);

		Ref<ParticleProcessMaterialConversionPlugin> particles_mat_convert;
		particles_mat_convert.instantiate();
		resource_conversion_plugins.push_back(particles_mat_convert);

		Ref<ProceduralSkyMaterialConversionPlugin> procedural_sky_mat_convert;
		procedural_sky_mat_convert.instantiate();
		resource_conversion_plugins.push_back(procedural_sky_mat_convert);

		Ref<PanoramaSkyMaterialConversionPlugin> panorama_sky_mat_convert;
		panorama_sky_mat_convert.instantiate();
		resource_conversion_plugins.push_back(panorama_sky_mat_convert);

		Ref<PhysicalSkyMaterialConversionPlugin> physical_sky_mat_convert;
		physical_sky_mat_convert.instantiate();
		resource_conversion_plugins.push_back(physical_sky_mat_convert);

		Ref<FogMaterialConversionPlugin> fog_mat_convert;
		fog_mat_convert.instantiate();
		resource_conversion_plugins.push_back(fog_mat_convert);

		Ref<VisualShaderConversionPlugin> vshader_convert;
		vshader_convert.instantiate();
		resource_conversion_plugins.push_back(vshader_convert);
	}

	update_spinner_step_msec = OS::get_singleton()->get_ticks_msec();
	update_spinner_step_frame = Engine::get_singleton()->get_frames_drawn();

	editor_plugin_screen = nullptr;
	editor_plugins_over = memnew(EditorPluginList);
	editor_plugins_force_over = memnew(EditorPluginList);
	editor_plugins_force_input_forwarding = memnew(EditorPluginList);

	Ref<GDExtensionExportPlugin> gdextension_export_plugin;
	gdextension_export_plugin.instantiate();

	EditorExport::get_singleton()->add_export_plugin(gdextension_export_plugin);

	Ref<DedicatedServerExportPlugin> dedicated_server_export_plugin;
	dedicated_server_export_plugin.instantiate();

	EditorExport::get_singleton()->add_export_plugin(dedicated_server_export_plugin);

	Ref<PackedSceneEditorTranslationParserPlugin> packed_scene_translation_parser_plugin;
	packed_scene_translation_parser_plugin.instantiate();
	EditorTranslationParser::get_singleton()->add_parser(packed_scene_translation_parser_plugin, EditorTranslationParser::STANDARD);

	_edit_current();
	current = nullptr;
	saving_resource = Ref<Resource>();

	set_process(true);

	open_imported = memnew(ConfirmationDialog);
	open_imported->set_ok_button_text(TTR("Open Anyway"));
	new_inherited_button = open_imported->add_button(TTR("New Inherited"), !DisplayServer::get_singleton()->get_swap_cancel_ok(), "inherit");
	open_imported->connect("confirmed", callable_mp(this, &EditorNode::_open_imported));
	open_imported->connect("custom_action", callable_mp(this, &EditorNode::_inherit_imported));
	gui_base->add_child(open_imported);

	quick_open = memnew(EditorQuickOpen);
	gui_base->add_child(quick_open);
	quick_open->connect("quick_open", callable_mp(this, &EditorNode::_quick_opened));

	_update_recent_scenes();

	set_process_shortcut_input(true);

	load_errors = memnew(RichTextLabel);
	load_error_dialog = memnew(AcceptDialog);
	load_error_dialog->set_unparent_when_invisible(true);
	load_error_dialog->add_child(load_errors);
	load_error_dialog->set_title(TTR("Load Errors"));

	execute_outputs = memnew(RichTextLabel);
	execute_outputs->set_selection_enabled(true);
	execute_outputs->set_context_menu_enabled(true);
	execute_output_dialog = memnew(AcceptDialog);
	execute_output_dialog->set_unparent_when_invisible(true);
	execute_output_dialog->add_child(execute_outputs);
	execute_output_dialog->set_title("");

	EditorFileSystem::get_singleton()->connect("sources_changed", callable_mp(this, &EditorNode::_sources_changed));
	EditorFileSystem::get_singleton()->connect("filesystem_changed", callable_mp(this, &EditorNode::_fs_changed));
	EditorFileSystem::get_singleton()->connect("resources_reimported", callable_mp(this, &EditorNode::_resources_reimported));
	EditorFileSystem::get_singleton()->connect("resources_reload", callable_mp(this, &EditorNode::_resources_changed));

	_build_icon_type_cache();

	pick_main_scene = memnew(ConfirmationDialog);
	gui_base->add_child(pick_main_scene);
	pick_main_scene->set_ok_button_text(TTR("Select"));
	pick_main_scene->connect("confirmed", callable_mp(this, &EditorNode::_menu_option).bind(SETTINGS_PICK_MAIN_SCENE));
	select_current_scene_button = pick_main_scene->add_button(TTR("Select Current"), true, "select_current");
	pick_main_scene->connect("custom_action", callable_mp(this, &EditorNode::_pick_main_scene_custom_action));

	for (int i = 0; i < _init_callbacks.size(); i++) {
		_init_callbacks[i]();
	}

	editor_data.add_edited_scene(-1);
	editor_data.set_edited_scene(0);
	scene_tabs->update_scene_tabs();

	ImportDock::get_singleton()->initialize_import_options();

	FileAccess::set_file_close_fail_notify_callback(_file_access_close_error_notify);

	print_handler.printfunc = _print_handler;
	print_handler.userdata = this;
	add_print_handler(&print_handler);

	ResourceSaver::set_save_callback(_resource_saved);
	ResourceLoader::set_load_callback(_resource_loaded);

	// Use the Ctrl modifier so F2 can be used to rename nodes in the scene tree dock.
	ED_SHORTCUT_AND_COMMAND("editor/editor_2d", TTR("Open 2D Editor"), KeyModifierMask::CTRL | Key::F1);
	ED_SHORTCUT_AND_COMMAND("editor/editor_3d", TTR("Open 3D Editor"), KeyModifierMask::CTRL | Key::F2);
	ED_SHORTCUT_AND_COMMAND("editor/editor_script", TTR("Open Script Editor"), KeyModifierMask::CTRL | Key::F3);
	ED_SHORTCUT_AND_COMMAND("editor/editor_assetlib", TTR("Open Asset Library"), KeyModifierMask::CTRL | Key::F4);

	ED_SHORTCUT_OVERRIDE("editor/editor_2d", "macos", KeyModifierMask::META | KeyModifierMask::CTRL | Key::KEY_1);
	ED_SHORTCUT_OVERRIDE("editor/editor_3d", "macos", KeyModifierMask::META | KeyModifierMask::CTRL | Key::KEY_2);
	ED_SHORTCUT_OVERRIDE("editor/editor_script", "macos", KeyModifierMask::META | KeyModifierMask::CTRL | Key::KEY_3);
	ED_SHORTCUT_OVERRIDE("editor/editor_assetlib", "macos", KeyModifierMask::META | KeyModifierMask::CTRL | Key::KEY_4);

	ED_SHORTCUT_AND_COMMAND("editor/editor_next", TTR("Open the next Editor"));
	ED_SHORTCUT_AND_COMMAND("editor/editor_prev", TTR("Open the previous Editor"));

	screenshot_timer = memnew(Timer);
	screenshot_timer->set_one_shot(true);
	screenshot_timer->set_wait_time(settings_menu->get_submenu_popup_delay() + 0.1f);
	screenshot_timer->connect("timeout", callable_mp(this, &EditorNode::_request_screenshot));
	add_child(screenshot_timer);
	screenshot_timer->set_owner(get_owner());

	// Adjust spacers to center 2D / 3D / Script buttons.
	int max_w = MAX(project_run_bar->get_minimum_size().x + right_menu_hb->get_minimum_size().x, main_menu->get_minimum_size().x);
	left_spacer->set_custom_minimum_size(Size2(MAX(0, max_w - main_menu->get_minimum_size().x), 0));
	right_spacer->set_custom_minimum_size(Size2(MAX(0, max_w - project_run_bar->get_minimum_size().x - right_menu_hb->get_minimum_size().x), 0));

	// Extend menu bar to window title.
	if (can_expand) {
		DisplayServer::get_singleton()->process_events();
		DisplayServer::get_singleton()->window_set_flag(DisplayServer::WINDOW_FLAG_EXTEND_TO_TITLE, true, DisplayServer::MAIN_WINDOW_ID);
		title_bar->set_can_move_window(true);
	}

	String exec = OS::get_singleton()->get_executable_path();
	// Save editor executable path for third-party tools.
	EditorSettings::get_singleton()->set_project_metadata("editor_metadata", "executable_path", exec);
}

EditorNode::~EditorNode() {
	EditorInspector::cleanup_plugins();
	EditorTranslationParser::get_singleton()->clean_parsers();
	ResourceImporterScene::clean_up_importer_plugins();

	remove_print_handler(&print_handler);
	EditorHelp::cleanup_doc();
	memdelete(editor_selection);
	memdelete(editor_plugins_over);
	memdelete(editor_plugins_force_over);
	memdelete(editor_plugins_force_input_forwarding);
	memdelete(progress_hb);
	memdelete(surface_upgrade_tool);

	EditorSettings::destroy();
	EditorColorMap::finish();
	EditorTheme::finalize();

	GDExtensionEditorPlugins::editor_node_add_plugin = nullptr;
	GDExtensionEditorPlugins::editor_node_remove_plugin = nullptr;

	FileDialog::get_icon_func = nullptr;
	FileDialog::register_func = nullptr;
	FileDialog::unregister_func = nullptr;

	EditorFileDialog::get_icon_func = nullptr;
	EditorFileDialog::register_func = nullptr;
	EditorFileDialog::unregister_func = nullptr;

	file_dialogs.clear();
	editor_file_dialogs.clear();

	singleton = nullptr;
}

/*
 * EDITOR PLUGIN LIST
 */

void EditorPluginList::make_visible(bool p_visible) {
	for (int i = 0; i < plugins_list.size(); i++) {
		plugins_list[i]->make_visible(p_visible);
	}
}

void EditorPluginList::edit(Object *p_object) {
	for (int i = 0; i < plugins_list.size(); i++) {
		plugins_list[i]->edit(p_object);
	}
}

bool EditorPluginList::forward_gui_input(const Ref<InputEvent> &p_event) {
	bool discard = false;

	for (int i = 0; i < plugins_list.size(); i++) {
		if (plugins_list[i]->forward_canvas_gui_input(p_event)) {
			discard = true;
		}
	}

	return discard;
}

EditorPlugin::AfterGUIInput EditorPluginList::forward_3d_gui_input(Camera3D *p_camera, const Ref<InputEvent> &p_event, bool serve_when_force_input_enabled) {
	EditorPlugin::AfterGUIInput after = EditorPlugin::AFTER_GUI_INPUT_PASS;

	for (int i = 0; i < plugins_list.size(); i++) {
		if ((!serve_when_force_input_enabled) && plugins_list[i]->is_input_event_forwarding_always_enabled()) {
			continue;
		}

		EditorPlugin::AfterGUIInput current_after = plugins_list[i]->forward_3d_gui_input(p_camera, p_event);
		if (current_after == EditorPlugin::AFTER_GUI_INPUT_STOP) {
			after = EditorPlugin::AFTER_GUI_INPUT_STOP;
		}
		if (after != EditorPlugin::AFTER_GUI_INPUT_STOP && current_after == EditorPlugin::AFTER_GUI_INPUT_CUSTOM) {
			after = EditorPlugin::AFTER_GUI_INPUT_CUSTOM;
		}
	}

	return after;
}

void EditorPluginList::forward_canvas_draw_over_viewport(Control *p_overlay) {
	for (int i = 0; i < plugins_list.size(); i++) {
		plugins_list[i]->forward_canvas_draw_over_viewport(p_overlay);
	}
}

void EditorPluginList::forward_canvas_force_draw_over_viewport(Control *p_overlay) {
	for (int i = 0; i < plugins_list.size(); i++) {
		plugins_list[i]->forward_canvas_force_draw_over_viewport(p_overlay);
	}
}

void EditorPluginList::forward_3d_draw_over_viewport(Control *p_overlay) {
	for (int i = 0; i < plugins_list.size(); i++) {
		plugins_list[i]->forward_3d_draw_over_viewport(p_overlay);
	}
}

void EditorPluginList::forward_3d_force_draw_over_viewport(Control *p_overlay) {
	for (int i = 0; i < plugins_list.size(); i++) {
		plugins_list[i]->forward_3d_force_draw_over_viewport(p_overlay);
	}
}

void EditorPluginList::add_plugin(EditorPlugin *p_plugin) {
	ERR_FAIL_COND(plugins_list.has(p_plugin));
	plugins_list.push_back(p_plugin);
}

void EditorPluginList::remove_plugin(EditorPlugin *p_plugin) {
	plugins_list.erase(p_plugin);
}

bool EditorPluginList::is_empty() {
	return plugins_list.is_empty();
}

void EditorPluginList::clear() {
	plugins_list.clear();
}

EditorPluginList::EditorPluginList() {
}

EditorPluginList::~EditorPluginList() {
}
