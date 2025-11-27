/**************************************************************************/
/*  svg_import_conversion_tool.cpp                                        */
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

#include "svg_import_conversion_tool.h"

#include "core/io/config_file.h"
#include "core/io/dir_access.h"
#include "editor/editor_node.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/dialogs.h"

void SvgImportConversionTool::_find_svg_files(EditorFileSystemDirectory *p_dir, Vector<String> &r_svg_paths) {
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	for (int i = 0; i < p_dir->get_file_count(); i++) {
		const String path = p_dir->get_file_path(i);
		const String ext = path.get_extension();
		if (ext == "svg") {
			if (da->file_exists(path + ".import")) {
				if (_is_texture2d_import(path + ".import")) {
					r_svg_paths.append(path);
				}
			}
		}
	}

	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		_find_svg_files(p_dir->get_subdir(i), r_svg_paths);
	}
}

void SvgImportConversionTool::_bind_methods() {
	ADD_SIGNAL(MethodInfo("conversion_finished"));
	ClassDB::bind_method(D_METHOD("_on_dialog_confirmed"), &SvgImportConversionTool::_on_dialog_confirmed);
}

void SvgImportConversionTool::popup_dialog() {
	if (!convert_dialog) {
		convert_dialog = memnew(ConfirmationDialog);
		convert_dialog->set_autowrap(true);
		convert_dialog->set_text(TTRC("This tool will convert all SVG files that are currently imported as Texture2D to use the SVGTexture importer instead.\n\nThe following will be preserved:\n- SVG scale (renamed to base_scale)\n\nDo you want to continue?"));
		convert_dialog->get_ok_button()->set_text(TTRC("Convert"));
		convert_dialog->get_label()->set_custom_minimum_size(Size2(750 * EDSCALE, 0));
		EditorNode::get_singleton()->get_gui_base()->add_child(convert_dialog);
		convert_dialog->connect(SceneStringName(confirmed), callable_mp(this, &SvgImportConversionTool::convert_svgs));
	}
	convert_dialog->popup_centered();
}

void SvgImportConversionTool::prepare_conversion() {
	EditorSettings::get_singleton()->set_project_metadata(META_SVG_IMPORT_CONVERSION_TOOL, META_RUN_ON_RESTART, true);

	Vector<String> reimport_svg_paths;
	_find_svg_files(EditorFileSystem::get_singleton()->get_filesystem(), reimport_svg_paths);

	EditorSettings::get_singleton()->set_project_metadata(META_SVG_IMPORT_CONVERSION_TOOL, META_REIMPORT_PATHS, reimport_svg_paths);

	// Delay to avoid deadlocks, since this dialog can be triggered by loading a scene.
	callable_mp(EditorNode::get_singleton(), &EditorNode::restart_editor).call_deferred(false);
}

void SvgImportConversionTool::begin_conversion() {
	EditorSettings::get_singleton()->set_project_metadata(META_SVG_IMPORT_CONVERSION_TOOL, META_RUN_ON_RESTART, false);
}

bool SvgImportConversionTool::_is_texture2d_import(const String &p_import_path) {
	Ref<ConfigFile> cf;
	cf.instantiate();
	Error err = cf->load(p_import_path);
	if (err != OK) {
		return false;
	}

	String importer = cf->get_value("remap", "importer", "");
	return importer == "texture";
}

void SvgImportConversionTool::_convert_svg_file(const String &p_path) {
	String import_path = p_path + ".import";
	float svg_scale = 1.0;
	ResourceUID::ID uid = ResourceUID::INVALID_ID;

	// Read old import settings
	Ref<ConfigFile> cf;
	cf.instantiate();
	if (cf->load(import_path) == OK) {
		if (cf->has_section_key("remap", "uid")) {
			String uid_text = cf->get_value("remap", "uid", "");
			uid = ResourceUID::get_singleton()->text_to_id(uid_text);
		}
		if (cf->has_section_key("params", "svg/scale")) {
			svg_scale = cf->get_value("params", "svg/scale", 1.0);
		}
	}

	String base_path = p_path.get_basename();
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	if (da.is_valid()) {
		da->remove(base_path + ".ctex");

		const char *variants[] = { "s3tc", "etc2", "bptc", "astc", nullptr };
		int i = 0;
		while (variants[i]) {
			da->remove(base_path + "." + variants[i] + ".ctex");
			i++;
		}

		da->remove(base_path + ".editor.ctex");
		da->remove(base_path + ".editor.meta");
	}

	cf->clear();

	cf->set_value("remap", "importer", "svg");
	cf->set_value("remap", "type", "SVGTexture");
	if (uid != ResourceUID::INVALID_ID) {
		cf->set_value("remap", "uid", ResourceUID::get_singleton()->id_to_text(uid));
	}

	// The path will be set by the importer during reimport
	cf->set_value("remap", "path", "");

	cf->set_value("params", "base_scale", svg_scale);

	cf->save(import_path);
}

void SvgImportConversionTool::_on_dialog_confirmed() {
	convert_svgs();
}

void SvgImportConversionTool::convert_svgs() {
	if (!EditorFileSystem::get_singleton() || !EditorFileSystem::get_singleton()->get_filesystem()) {
		if (EditorNode::get_singleton()) {
			EditorNode::get_singleton()->show_warning("Editor file system is not ready. Please try again.");
		}
		return;
	}

	Vector<String> svg_paths;
	_find_svg_files(EditorFileSystem::get_singleton()->get_filesystem(), svg_paths);

	if (svg_paths.is_empty()) {
		if (EditorNode::get_singleton()) {
			EditorNode::get_singleton()->show_warning("No SVG files found that are imported as Texture2D.");
		}
		return;
	}

	print_line(vformat("Found %d SVG files to convert.", svg_paths.size()));

	for (int i = 0; i < svg_paths.size(); i++) {
		print_line(vformat("Converting: %s", svg_paths[i]));
		_convert_svg_file(svg_paths[i]);
	}

	print_line(vformat("Converted %d SVG files to use SVGTexture importer.", svg_paths.size()));
	print_line(vformat("Recommend reloading the project to prevent (non-harmful) cache errors."));

	if (!svg_paths.is_empty()) {
		EditorFileSystem::get_singleton()->reimport_files(svg_paths);
	}
}

void SvgImportConversionTool::finish_conversion() {
	emit_signal(CONVERSION_FINISHED);
}
