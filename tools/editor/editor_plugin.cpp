/*************************************************************************/
/*  editor_plugin.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "editor_plugin.h"
#include "plugins/canvas_item_editor_plugin.h"
#include "plugins/spatial_editor_plugin.h"

void EditorPlugin::add_custom_type(const String& p_type, const String& p_base,const Ref<Script>& p_script, const Ref<Texture>& p_icon) {

	EditorNode::get_editor_data().add_custom_type(p_type,p_base,p_script,p_icon);
}

void EditorPlugin::remove_custom_type(const String& p_type){

	EditorNode::get_editor_data().remove_custom_type(p_type);
}



void EditorPlugin::add_custom_control(CustomControlContainer p_location,Control *p_control) {

	switch(p_location) {

		case CONTAINER_TOOLBAR: {

			EditorNode::get_menu_hb()->add_child(p_control);
		} break;
		case CONTAINER_SPATIAL_EDITOR_MENU: {

			SpatialEditor::get_singleton()->add_control_to_menu_panel(p_control);

		} break;
		case CONTAINER_SPATIAL_EDITOR_SIDE: {

			SpatialEditor::get_singleton()->get_palette_split()->add_child(p_control);
			SpatialEditor::get_singleton()->get_palette_split()->move_child(p_control,0);

		} break;
		case CONTAINER_SPATIAL_EDITOR_BOTTOM: {

			SpatialEditor::get_singleton()->get_shader_split()->add_child(p_control);

		} break;
		case CONTAINER_CANVAS_EDITOR_MENU: {

			CanvasItemEditor::get_singleton()->add_control_to_menu_panel(p_control);

		} break;
		case CONTAINER_CANVAS_EDITOR_SIDE: {

			CanvasItemEditor::get_singleton()->get_palette_split()->add_child(p_control);

		} break;
		case CONTAINER_CANVAS_EDITOR_BOTTOM: {

			CanvasItemEditor::get_singleton()->get_bottom_split()->add_child(p_control);

		} break;

	}
}

bool EditorPlugin::create_spatial_gizmo(Spatial* p_spatial) {
	//??
	return false;
}

bool EditorPlugin::forward_input_event(const InputEvent& p_event) {

	if (get_script_instance() && get_script_instance()->has_method("forward_input_event")) {
		return get_script_instance()->call("forward_input_event",p_event);
	}
	return false;
}
bool EditorPlugin::forward_spatial_input_event(Camera* p_camera,const InputEvent& p_event) {

	if (get_script_instance() && get_script_instance()->has_method("forward_spatial_input_event")) {
		return get_script_instance()->call("forward_spatial_input_event",p_camera,p_event);
	}

	return false;
}
String EditorPlugin::get_name() const {

	if (get_script_instance() && get_script_instance()->has_method("get_name")) {
		return get_script_instance()->call("get_name");
	}

	return String();

}
bool EditorPlugin::has_main_screen() const {

	if (get_script_instance() && get_script_instance()->has_method("has_main_screen")) {
		return get_script_instance()->call("has_main_screen");
	}

	return false;

}
void EditorPlugin::make_visible(bool p_visible) {

	if (get_script_instance() && get_script_instance()->has_method("make_visible")) {
		get_script_instance()->call("make_visible",p_visible);
	}
}


void EditorPlugin::edit(Object *p_object) {

	if (get_script_instance() && get_script_instance()->has_method("edit")) {
		get_script_instance()->call("edit",p_object);
	}

}

bool EditorPlugin::handles(Object *p_object) const {

	if (get_script_instance() && get_script_instance()->has_method("handles")) {
		return get_script_instance()->call("handles",p_object);
	}

	return false;
}
Dictionary EditorPlugin::get_state() const {

	if (get_script_instance() && get_script_instance()->has_method("get_state")) {
		return get_script_instance()->call("get_state");
	}

	return Dictionary();
}

void EditorPlugin::set_state(const Dictionary& p_state) {

	if (get_script_instance() && get_script_instance()->has_method("set_state")) {
		get_script_instance()->call("set_state",p_state);
	}
}

void EditorPlugin::clear() {

	if (get_script_instance() && get_script_instance()->has_method("clear")) {
		get_script_instance()->call("clear");
	}

}

void EditorPlugin::save_external_data() {} // if editor references external resources/scenes, save them
void EditorPlugin::apply_changes() {

	if (get_script_instance() && get_script_instance()->has_method("apply_changes")) {
		get_script_instance()->call("apply_changes");
	}


} // if changes are pending in editor, apply them
void EditorPlugin::get_breakpoints(List<String> *p_breakpoints) {

	if (get_script_instance() && get_script_instance()->has_method("get_breakpoints")) {
		StringArray arr = get_script_instance()->call("get_breakpoints");
		for(int i=0;i<arr.size();i++)
			p_breakpoints->push_back(arr[i]);
	}

}
bool EditorPlugin::get_remove_list(List<Node*> *p_list) {

	return false;
}

void EditorPlugin::restore_global_state() {}
void EditorPlugin::save_global_state() {}


void EditorPlugin::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("get_undo_redo"),&EditorPlugin::_get_undo_redo);
	ObjectTypeDB::bind_method(_MD("add_custom_control","container","control"),&EditorPlugin::add_custom_control);
	ObjectTypeDB::bind_method(_MD("add_custom_type","type","base","script:Script","icon:Texture"),&EditorPlugin::add_custom_type);
	ObjectTypeDB::bind_method(_MD("remove_custom_type","type"),&EditorPlugin::remove_custom_type);

	ObjectTypeDB::add_virtual_method(get_type_static(),MethodInfo(Variant::BOOL,"forward_input_event",PropertyInfo(Variant::INPUT_EVENT,"event")));
	ObjectTypeDB::add_virtual_method(get_type_static(),MethodInfo(Variant::BOOL,"forward_spatial_input_event",PropertyInfo(Variant::OBJECT,"camera",PROPERTY_HINT_RESOURCE_TYPE,"Camera"),PropertyInfo(Variant::INPUT_EVENT,"event")));
	ObjectTypeDB::add_virtual_method(get_type_static(),MethodInfo(Variant::STRING,"get_name"));
	ObjectTypeDB::add_virtual_method(get_type_static(),MethodInfo(Variant::BOOL,"has_main_screen"));
	ObjectTypeDB::add_virtual_method(get_type_static(),MethodInfo("make_visible",PropertyInfo(Variant::BOOL,"visible")));
	ObjectTypeDB::add_virtual_method(get_type_static(),MethodInfo("edit",PropertyInfo(Variant::OBJECT,"object")));
	ObjectTypeDB::add_virtual_method(get_type_static(),MethodInfo(Variant::BOOL,"handles",PropertyInfo(Variant::OBJECT,"object")));
	ObjectTypeDB::add_virtual_method(get_type_static(),MethodInfo(Variant::DICTIONARY,"get_state"));
	ObjectTypeDB::add_virtual_method(get_type_static(),MethodInfo("set_state",PropertyInfo(Variant::DICTIONARY,"state")));
	ObjectTypeDB::add_virtual_method(get_type_static(),MethodInfo("clear"));
	ObjectTypeDB::add_virtual_method(get_type_static(),MethodInfo("apply_changes"));
	ObjectTypeDB::add_virtual_method(get_type_static(),MethodInfo(Variant::STRING_ARRAY,"get_breakpoints"));

	BIND_CONSTANT( CONTAINER_TOOLBAR );
	BIND_CONSTANT( CONTAINER_SPATIAL_EDITOR_MENU );
	BIND_CONSTANT( CONTAINER_SPATIAL_EDITOR_SIDE );
	BIND_CONSTANT( CONTAINER_SPATIAL_EDITOR_BOTTOM );
	BIND_CONSTANT( CONTAINER_CANVAS_EDITOR_MENU );
	BIND_CONSTANT( CONTAINER_CANVAS_EDITOR_SIDE );

}

EditorPlugin::EditorPlugin()
{
	undo_redo=NULL;
}


EditorPlugin::~EditorPlugin()
{
}



EditorPluginCreateFunc EditorPlugins::creation_funcs[MAX_CREATE_FUNCS];

int EditorPlugins::creation_func_count=0;
