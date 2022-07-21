/*************************************************************************/
/*  editor_scene_importer_blend.cpp                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "editor_scene_importer_blend.h"

#ifdef TOOLS_ENABLED

#include "../gltf_document.h"
#include "../gltf_state.h"

#include "core/config/project_settings.h"
#include "editor/editor_file_dialog.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "main/main.h"
#include "scene/main/node.h"
#include "scene/resources/animation.h"

#ifdef WINDOWS_ENABLED
// Code by Pedro Estebanez (https://github.com/godotengine/godot/pull/59766)
#include <shlwapi.h>
#endif

uint32_t EditorSceneFormatImporterBlend::get_import_flags() const {
	return ImportFlags::IMPORT_SCENE | ImportFlags::IMPORT_ANIMATION;
}

void EditorSceneFormatImporterBlend::get_extensions(List<String> *r_extensions) const {
	r_extensions->push_back("blend");
}

Node *EditorSceneFormatImporterBlend::import_scene(const String &p_path, uint32_t p_flags,
		const HashMap<StringName, Variant> &p_options, int p_bake_fps,
		List<String> *r_missing_deps, Error *r_err) {
	// Get global paths for source and sink.

	// Escape paths to be valid Python strings to embed in the script.
	const String source_global = ProjectSettings::get_singleton()->globalize_path(p_path).c_escape();
	const String sink = ProjectSettings::get_singleton()->get_imported_files_path().plus_file(
			vformat("%s-%s.gltf", p_path.get_file().get_basename(), p_path.md5_text()));
	const String sink_global = ProjectSettings::get_singleton()->globalize_path(sink).c_escape();

	// Handle configuration options.

	String parameters_arg;

	if (p_options.has(SNAME("blender/nodes/custom_properties")) && p_options[SNAME("blender/nodes/custom_properties")]) {
		parameters_arg += "export_extras=True,";
	} else {
		parameters_arg += "export_extras=False,";
	}
	if (p_options.has(SNAME("blender/meshes/skins")) && p_options[SNAME("blender/meshes/skins")]) {
		int32_t skins = p_options["blender/meshes/skins"];
		if (skins == BLEND_BONE_INFLUENCES_NONE) {
			parameters_arg += "export_all_influences=False,";
		} else if (skins == BLEND_BONE_INFLUENCES_COMPATIBLE) {
			parameters_arg += "export_all_influences=False,";
		} else if (skins == BLEND_BONE_INFLUENCES_ALL) {
			parameters_arg += "export_all_influences=True,";
		}
		parameters_arg += "export_skins=True,";
	} else {
		parameters_arg += "export_skins=False,";
	}
	if (p_options.has(SNAME("blender/materials/export_materials")) && p_options[SNAME("blender/materials/export_materials")]) {
		int32_t exports = p_options["blender/materials/export_materials"];
		if (exports == BLEND_MATERIAL_EXPORT_PLACEHOLDER) {
			parameters_arg += "export_materials='PLACEHOLDER',";
		} else if (exports == BLEND_MATERIAL_EXPORT_EXPORT) {
			parameters_arg += "export_materials='EXPORT',";
		}
	} else {
		parameters_arg += "export_materials='PLACEHOLDER',";
	}
	if (p_options.has(SNAME("blender/nodes/cameras")) && p_options[SNAME("blender/nodes/cameras")]) {
		parameters_arg += "export_cameras=True,";
	} else {
		parameters_arg += "export_cameras=False,";
	}
	if (p_options.has(SNAME("blender/nodes/punctual_lights")) && p_options[SNAME("blender/nodes/punctual_lights")]) {
		parameters_arg += "export_lights=True,";
	} else {
		parameters_arg += "export_lights=False,";
	}
	if (p_options.has(SNAME("blender/meshes/colors")) && p_options[SNAME("blender/meshes/colors")]) {
		parameters_arg += "export_colors=True,";
	} else {
		parameters_arg += "export_colors=False,";
	}
	if (p_options.has(SNAME("blender/nodes/visible")) && p_options[SNAME("blender/nodes/visible")]) {
		int32_t visible = p_options["blender/nodes/visible"];
		if (visible == BLEND_VISIBLE_VISIBLE_ONLY) {
			parameters_arg += "use_visible=True,";
		} else if (visible == BLEND_VISIBLE_RENDERABLE) {
			parameters_arg += "use_renderable=True,";
		} else if (visible == BLEND_VISIBLE_ALL) {
			parameters_arg += "use_visible=False,use_renderable=False,";
		}
	} else {
		parameters_arg += "use_visible=False,use_renderable=False,";
	}

	if (p_options.has(SNAME("blender/meshes/uvs")) && p_options[SNAME("blender/meshes/uvs")]) {
		parameters_arg += "export_texcoords=True,";
	} else {
		parameters_arg += "export_texcoords=False,";
	}
	if (p_options.has(SNAME("blender/meshes/normals")) && p_options[SNAME("blender/meshes/normals")]) {
		parameters_arg += "export_normals=True,";
	} else {
		parameters_arg += "export_normals=False,";
	}
	if (p_options.has(SNAME("blender/meshes/tangents")) && p_options[SNAME("blender/meshes/tangents")]) {
		parameters_arg += "export_tangents=True,";
	} else {
		parameters_arg += "export_tangents=False,";
	}
	if (p_options.has(SNAME("blender/animation/group_tracks")) && p_options[SNAME("blender/animation/group_tracks")]) {
		parameters_arg += "export_nla_strips=True,";
	} else {
		parameters_arg += "export_nla_strips=False,";
	}
	if (p_options.has(SNAME("blender/animation/limit_playback")) && p_options[SNAME("blender/animation/limit_playback")]) {
		parameters_arg += "export_frame_range=True,";
	} else {
		parameters_arg += "export_frame_range=False,";
	}
	if (p_options.has(SNAME("blender/animation/always_sample")) && p_options[SNAME("blender/animation/always_sample")]) {
		parameters_arg += "export_force_sampling=True,";
	} else {
		parameters_arg += "export_force_sampling=False,";
	}
	if (p_options.has(SNAME("blender/meshes/export_bones_deforming_mesh_only")) && p_options[SNAME("blender/meshes/export_bones_deforming_mesh_only")]) {
		parameters_arg += "export_def_bones=True,";
	} else {
		parameters_arg += "export_def_bones=False,";
	}
	if (p_options.has(SNAME("blender/nodes/modifiers")) && p_options[SNAME("blender/nodes/modifiers")]) {
		parameters_arg += "export_apply=True";
	} else {
		parameters_arg += "export_apply=False";
	}

	String unpack_all;
	if (p_options.has(SNAME("blender/materials/unpack_enabled")) && p_options[SNAME("blender/materials/unpack_enabled")]) {
		unpack_all = "bpy.ops.file.unpack_all(method='USE_LOCAL');";
	}

	// Prepare Blender export script.

	String common_args = vformat("filepath='%s',", sink_global) +
			"export_format='GLTF_SEPARATE',"
			"export_yup=True," +
			parameters_arg;
	String script =
			String("import bpy, sys;") +
			"print('Blender 3.0 or higher is required.', file=sys.stderr) if bpy.app.version < (3, 0, 0) else None;" +
			vformat("bpy.ops.wm.open_mainfile(filepath='%s');", source_global) +
			unpack_all +
			vformat("bpy.ops.export_scene.gltf(export_keep_originals=True,%s);", common_args);
	print_verbose(script);

	// Run script with configured Blender binary.

	String blender_path = EDITOR_GET("filesystem/import/blender/blender3_path");

#ifdef WINDOWS_ENABLED
	blender_path = blender_path.plus_file("blender.exe");
#else
	blender_path = blender_path.plus_file("blender");
#endif

	List<String> args;
	args.push_back("--background");
	args.push_back("--python-expr");
	args.push_back(script);

	String standard_out;
	int ret;
	OS::get_singleton()->execute(blender_path, args, &standard_out, &ret, true);
	print_verbose(blender_path);
	print_verbose(standard_out);

	if (ret != 0) {
		if (r_err) {
			*r_err = ERR_SCRIPT_FAILED;
		}
		ERR_PRINT(vformat("Blend export to glTF failed with error: %d.", ret));
		return nullptr;
	}

	// Import the generated glTF.

	// Use GLTFDocument instead of glTF importer to keep image references.
	Ref<GLTFDocument> gltf;
	gltf.instantiate();
	Ref<GLTFState> state;
	state.instantiate();
	String base_dir;
	if (p_options.has(SNAME("blender/materials/unpack_enabled")) && p_options[SNAME("blender/materials/unpack_enabled")]) {
		base_dir = sink.get_base_dir();
	}
	Error err = gltf->append_from_file(sink.get_basename() + ".gltf", state, p_flags, p_bake_fps, base_dir);
	if (err != OK) {
		if (r_err) {
			*r_err = FAILED;
		}
		return nullptr;
	}
	return gltf->generate_scene(state, p_bake_fps);
}

Variant EditorSceneFormatImporterBlend::get_option_visibility(const String &p_path, bool p_for_animation, const String &p_option,
		const HashMap<StringName, Variant> &p_options) {
	if (p_path.get_extension().to_lower() != "blend") {
		return true;
	}

	if (p_option.begins_with("animation/")) {
		if (p_option != "animation/import" && !bool(p_options["animation/import"])) {
			return false;
		}
	}
	return true;
}

void EditorSceneFormatImporterBlend::get_import_options(const String &p_path, List<ResourceImporter::ImportOption> *r_options) {
	if (p_path.get_extension().to_lower() != "blend") {
		return;
	}
#define ADD_OPTION_BOOL(PATH, VALUE) \
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, SNAME(PATH)), VALUE));
#define ADD_OPTION_ENUM(PATH, ENUM_HINT, VALUE) \
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::INT, SNAME(PATH), PROPERTY_HINT_ENUM, ENUM_HINT), VALUE));

	ADD_OPTION_ENUM("blender/nodes/visible", "Visible Only,Renderable,All", BLEND_VISIBLE_ALL);
	ADD_OPTION_BOOL("blender/nodes/punctual_lights", true);
	ADD_OPTION_BOOL("blender/nodes/cameras", true);
	ADD_OPTION_BOOL("blender/nodes/custom_properties", true);
	ADD_OPTION_ENUM("blender/nodes/modifiers", "No Modifiers,All Modifiers", BLEND_MODIFIERS_ALL);
	ADD_OPTION_BOOL("blender/meshes/colors", false);
	ADD_OPTION_BOOL("blender/meshes/uvs", true);
	ADD_OPTION_BOOL("blender/meshes/normals", true);
	ADD_OPTION_BOOL("blender/meshes/tangents", true);
	ADD_OPTION_ENUM("blender/meshes/skins", "None,4 Influences (Compatible),All Influences", BLEND_BONE_INFLUENCES_ALL);
	ADD_OPTION_BOOL("blender/meshes/export_bones_deforming_mesh_only", false);
	ADD_OPTION_BOOL("blender/materials/unpack_enabled", true);
	ADD_OPTION_ENUM("blender/materials/export_materials", "Placeholder,Export", BLEND_MATERIAL_EXPORT_EXPORT);
	ADD_OPTION_BOOL("blender/animation/limit_playback", true);
	ADD_OPTION_BOOL("blender/animation/always_sample", true);
	ADD_OPTION_BOOL("blender/animation/group_tracks", true);

#undef ADD_OPTION_BOOL
#undef ADD_OPTION_ENUM
}

///////////////////////////

static bool _test_blender_path(const String &p_path, String *r_err = nullptr) {
	String path = p_path;
#ifdef WINDOWS_ENABLED
	path = path.plus_file("blender.exe");
#else
	path = path.plus_file("blender");
#endif

#if defined(MACOS_ENABLED)
	if (!FileAccess::exists(path)) {
		path = path.plus_file("Blender");
	}
#endif

	if (!FileAccess::exists(path)) {
		if (r_err) {
			*r_err = TTR("Path does not contain a Blender installation.");
		}
		return false;
	}
	List<String> args;
	args.push_back("--version");
	String pipe;
	Error err = OS::get_singleton()->execute(path, args, &pipe);
	if (err != OK) {
		if (r_err) {
			*r_err = TTR("Can't execute Blender binary.");
		}
		return false;
	}

	if (pipe.find("Blender ") != 0) {
		if (r_err) {
			*r_err = vformat(TTR("Unexpected --version output from Blender binary at: %s"), path);
		}
		return false;
	}
	pipe = pipe.replace_first("Blender ", "");
	int pp = pipe.find(".");
	if (pp == -1) {
		if (r_err) {
			*r_err = TTR("Path supplied lacks a Blender binary.");
		}
		return false;
	}
	String v = pipe.substr(0, pp);
	int version = v.to_int();
	if (version < 3) {
		if (r_err) {
			*r_err = TTR("This Blender installation is too old for this importer (not 3.0+).");
		}
		return false;
	}
	if (version > 3) {
		if (r_err) {
			*r_err = TTR("This Blender installation is too new for this importer (not 3.x).");
		}
		return false;
	}

	return true;
}

bool EditorFileSystemImportFormatSupportQueryBlend::is_active() const {
	bool blend_enabled = GLOBAL_GET("filesystem/import/blender/enabled");

	String blender_path = EDITOR_GET("filesystem/import/blender/blender3_path");

	if (blend_enabled && !_test_blender_path(blender_path)) {
		// Intending to import Blender, but blend not configured.
		return true;
	}

	return false;
}
Vector<String> EditorFileSystemImportFormatSupportQueryBlend::get_file_extensions() const {
	Vector<String> ret;
	ret.push_back("blend");
	return ret;
}

void EditorFileSystemImportFormatSupportQueryBlend::_validate_path(String p_path) {
	String error;
	bool success = false;
	if (p_path == "") {
		error = TTR("Path is empty.");
	} else {
		if (_test_blender_path(p_path, &error)) {
			success = true;
			if (auto_detected_path == p_path) {
				error = TTR("Path to Blender installation is valid (Autodetected).");
			} else {
				error = TTR("Path to Blender installation is valid.");
			}
		}
	}

	path_status->set_text(error);

	if (success) {
		path_status->add_theme_color_override("font_color", path_status->get_theme_color(SNAME("success_color"), SNAME("Editor")));
		configure_blender_dialog->get_ok_button()->set_disabled(false);
	} else {
		path_status->add_theme_color_override("font_color", path_status->get_theme_color(SNAME("error_color"), SNAME("Editor")));
		configure_blender_dialog->get_ok_button()->set_disabled(true);
	}
}

bool EditorFileSystemImportFormatSupportQueryBlend::_autodetect_path(String p_path) {
	if (_test_blender_path(p_path)) {
		auto_detected_path = p_path;
		return true;
	}
	return false;
}

void EditorFileSystemImportFormatSupportQueryBlend::_path_confirmed() {
	confirmed = true;
}

void EditorFileSystemImportFormatSupportQueryBlend::_select_install(String p_path) {
	blender_path->set_text(p_path);
	_validate_path(p_path);
}
void EditorFileSystemImportFormatSupportQueryBlend::_browse_install() {
	if (blender_path->get_text() != String()) {
		browse_dialog->set_current_dir(blender_path->get_text());
	}

	browse_dialog->popup_centered_ratio();
}

bool EditorFileSystemImportFormatSupportQueryBlend::query() {
	if (!configure_blender_dialog) {
		configure_blender_dialog = memnew(ConfirmationDialog);
		configure_blender_dialog->set_title(TTR("Configure Blender Importer"));
		configure_blender_dialog->set_flag(Window::FLAG_BORDERLESS, true); // Avoid closing accidentally .
		configure_blender_dialog->set_close_on_escape(false);

		VBoxContainer *vb = memnew(VBoxContainer);
		vb->add_child(memnew(Label(TTR("Blender 3.0+ is required to import '.blend' files.\nPlease provide a valid path to a Blender installation:"))));

		HBoxContainer *hb = memnew(HBoxContainer);

		blender_path = memnew(LineEdit);
		blender_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		hb->add_child(blender_path);
		blender_path_browse = memnew(Button);
		hb->add_child(blender_path_browse);
		blender_path_browse->set_text(TTR("Browse"));
		blender_path_browse->connect("pressed", callable_mp(this, &EditorFileSystemImportFormatSupportQueryBlend::_browse_install));
		hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		hb->set_custom_minimum_size(Size2(400 * EDSCALE, 0));

		vb->add_child(hb);

		path_status = memnew(Label);
		vb->add_child(path_status);

		configure_blender_dialog->add_child(vb);

		blender_path->connect("text_changed", callable_mp(this, &EditorFileSystemImportFormatSupportQueryBlend::_validate_path));

		EditorNode::get_singleton()->get_gui_base()->add_child(configure_blender_dialog);

		configure_blender_dialog->set_ok_button_text(TTR("Confirm Path"));
		configure_blender_dialog->set_cancel_button_text(TTR("Disable '.blend' Import"));
		configure_blender_dialog->get_cancel_button()->set_tooltip(TTR("Disables Blender '.blend' files import for this project. Can be re-enabled in Project Settings."));
		configure_blender_dialog->connect("confirmed", callable_mp(this, &EditorFileSystemImportFormatSupportQueryBlend::_path_confirmed));

		browse_dialog = memnew(EditorFileDialog);
		browse_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
		browse_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_DIR);
		browse_dialog->connect("dir_selected", callable_mp(this, &EditorFileSystemImportFormatSupportQueryBlend::_select_install));

		EditorNode::get_singleton()->get_gui_base()->add_child(browse_dialog);
	}

	String path = EDITOR_GET("filesystem/import/blender/blender3_path");

	if (path == "") {
		// Autodetect
		auto_detected_path = "";

#if defined(MACOS_ENABLED)

		{
			Vector<String> mdfind_paths;
			{
				List<String> mdfind_args;
				mdfind_args.push_back("kMDItemCFBundleIdentifier=org.blenderfoundation.blender");

				String output;
				Error err = OS::get_singleton()->execute("mdfind", mdfind_args, &output);
				if (err == OK) {
					mdfind_paths = output.split("\n");
				}
			}

			bool found = false;
			for (const String &path : mdfind_paths) {
				found = _autodetect_path(path.plus_file("Contents/MacOS"));
				if (found) {
					break;
				}
			}
			if (!found) {
				found = _autodetect_path("/opt/homebrew/bin");
			}
			if (!found) {
				found = _autodetect_path("/opt/local/bin");
			}
			if (!found) {
				found = _autodetect_path("/usr/local/bin");
			}
			if (!found) {
				found = _autodetect_path("/usr/local/opt");
			}
			if (!found) {
				found = _autodetect_path("/Applications/Blender.app/Contents/MacOS");
			}
		}
#elif defined(WINDOWS_ENABLED)
		{
			char blender_opener_path[MAX_PATH];
			DWORD path_len = MAX_PATH;
			HRESULT res = AssocQueryString(0, ASSOCSTR_EXECUTABLE, ".blend", "open", blender_opener_path, &path_len);
			if (res == S_OK && _autodetect_path(String(blender_opener_path).get_base_dir())) {
				// Good.
			} else if (_autodetect_path("C:\\Program Files\\Blender Foundation")) {
				// Good.
			} else {
				_autodetect_path("C:\\Program Files (x86)\\Blender Foundation");
			}
		}

#elif defined(UNIX_ENABLED)
		if (_autodetect_path("/usr/bin")) {
			// Good.
		} else if (_autodetect_path("/usr/local/bin")) {
			// Good
		} else {
			_autodetect_path("/opt/blender/bin");
		}
#endif
		if (auto_detected_path != "") {
			path = auto_detected_path;
		}
	}

	blender_path->set_text(path);

	_validate_path(path);

	configure_blender_dialog->popup_centered();
	confirmed = false;

	while (true) {
		OS::get_singleton()->delay_usec(1);
		DisplayServer::get_singleton()->process_events();
		Main::iteration();
		if (!configure_blender_dialog->is_visible() || confirmed) {
			break;
		}
	}

	if (confirmed) {
		// Can only confirm a valid path.
		EditorSettings::get_singleton()->set("filesystem/import/blender/blender3_path", blender_path->get_text());
		EditorSettings::get_singleton()->save();
	} else {
		// Disable Blender import
		ProjectSettings::get_singleton()->set("filesystem/import/blender/enabled", false);
		ProjectSettings::get_singleton()->save();

		if (EditorNode::immediate_confirmation_dialog(TTR("Disabling '.blend' file import requires restarting the editor."), TTR("Save & Restart"), TTR("Restart"))) {
			EditorNode::get_singleton()->save_all_scenes();
		}
		EditorNode::get_singleton()->restart_editor();
		return true;
	}

	return false;
}

EditorFileSystemImportFormatSupportQueryBlend::EditorFileSystemImportFormatSupportQueryBlend() {
}

#endif // TOOLS_ENABLED
