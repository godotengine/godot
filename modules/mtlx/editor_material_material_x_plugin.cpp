/**************************************************************************/
/*  editor_material_material_x_plugin.cpp                                 */
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

#ifdef TOOLS_ENABLED

#include "editor_material_material_x_plugin.h"

#include "material_x_3d.h"

#include "core/config/project_settings.h"
#include "core/error/error_list.h"
#include "core/object/object.h"
#include "core/templates/vector.h"
#include "editor/editor_file_system.h"
#include "editor/editor_node.h"
#include "editor/gui/editor_file_dialog.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/gui/check_box.h"
#include "scene/main/node.h"

String MaterialXPlugin::get_name() const {
	return "MaterialXImporter";
}

bool MaterialXPlugin::has_main_screen() const {
	return false;
}

MaterialXPlugin::MaterialXPlugin() {
	file_export_lib = memnew(EditorFileDialog);
	EditorNode::get_singleton()->get_gui_base()->add_child(file_export_lib);
	file_export_lib->connect("file_selected", callable_mp(this, &MaterialXPlugin::save_materialx_as_resource));
	file_export_lib->set_title(TTR("Import MaterialX Material"));
	file_export_lib->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	file_export_lib->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	file_export_lib->clear_filters();
	file_export_lib->add_filter("*.mtlx");
	file_export_lib->set_title(TTR("Import MaterialX to Material resource"));

	add_tool_menu_item(TTR("Import MaterialX Material ..."), callable_mp(this, &MaterialXPlugin::_material_x_dialog_action));
}

void MaterialXPlugin::save_materialx_as_resource(String p_file) {
	Ref<MTLXLoader> loader;
	loader.instantiate();
	String dir = p_file.get_base_dir();
	String filename = p_file.get_file().get_basename();
	String resource_path = dir.path_join(filename + ".res");
	Ref<Material> resource = loader->_load(resource_path, p_file, true, 0);
	if (resource.is_null()) {
		ERR_PRINT("Material save error");
		return;
	}
	ResourceSaver::save(resource, resource_path);
	EditorFileSystem::get_singleton()->scan_changes();
}

void MaterialXPlugin::_material_x_dialog_action() {
	file_export_lib->popup_centered_ratio();
}

#endif // TOOLS_ENABLED
