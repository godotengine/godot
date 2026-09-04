/**************************************************************************/
/*  lightmap_gi_editor_plugin.cpp                                         */
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

#include "lightmap_gi_editor_plugin.h"

#include "core/io/resource_loader.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_file_dialog.h"
#include "scene/3d/lightmap_gi.h"
#include "scene/main/scene_tree.h"
#include "servers/display/display_server.h"
#include "servers/rendering/rendering_server.h"

#include "modules/modules_enabled.gen.h" // For lightmapper_rd.

void LightmapGIEditorPlugin::_bake() {
	EditorNode::get_singleton()->lightmap_gi_bake_select_file("", lightmap);
}

void LightmapGIEditorPlugin::edit(Object *p_object) {
	LightmapGI *s = Object::cast_to<LightmapGI>(p_object);
	if (!s) {
		return;
	}

	lightmap = s;
}

bool LightmapGIEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("LightmapGI");
}

void LightmapGIEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		bake->show();
	} else {
		bake->hide();
	}
}

void LightmapGIEditorPlugin::_bind_methods() {
	ClassDB::bind_method("_bake", &LightmapGIEditorPlugin::_bake);
}

LightmapGIEditorPlugin::LightmapGIEditorPlugin() {
	bake = memnew(Button);
	bake->set_theme_type_variation(SceneStringName(FlatButton));
	// TODO: Rework this as a dedicated toolbar control so we can hook into theme changes and update it
	// when the editor theme updates.
	bake->set_button_icon(EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Bake"), EditorStringName(EditorIcons)));
	bake->set_text(TTR("Bake Lightmaps"));

#ifdef MODULE_LIGHTMAPPER_RD_ENABLED
	// Disable lightmap baking if not supported on the current GPU.
	if (!DisplayServer::get_singleton()->can_create_rendering_device()) {
		bake->set_disabled(true);
		bake->set_tooltip_text(vformat(TTR("Lightmap baking is not supported on this GPU (%s)."), RenderingServer::get_singleton()->get_video_adapter_name()));
	}
#else
	// Disable lightmap baking if the module is disabled at compile-time.
	bake->set_disabled(true);
#if defined(ANDROID_ENABLED) || defined(APPLE_EMBEDDED_ENABLED)
	bake->set_tooltip_text(vformat(TTR("Lightmaps cannot be baked on %s."), OS::get_singleton()->get_name()));
#else
	bake->set_tooltip_text(TTR("Lightmaps cannot be baked, as the `lightmapper_rd` module was disabled at compile-time."));
#endif
#endif // MODULE_LIGHTMAPPER_RD_ENABLED

	bake->hide();
	bake->connect(SceneStringName(pressed), Callable(this, "_bake"));
	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, bake);
	lightmap = nullptr;
}
