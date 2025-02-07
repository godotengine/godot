#include "resource_inspector_dock.h"
#include "editor/editor_inspector.h"
#include "editor/editor_settings.h"
#include "editor/editor_node.h"

ResourceInspectorDock *ResourceInspectorDock::singleton = nullptr;

ResourceInspectorDock::ResourceInspectorDock() {
	singleton = this;

	set_name("Resource Inspector");

	set_v_size_flags(SIZE_EXPAND_FILL);

	tab_container = memnew(TabContainer);
	tab_container->set_v_size_flags(SIZE_EXPAND_FILL);
	tab_container->set_drag_to_rearrange_enabled(true);
	tab_container->get_tab_bar()->connect("gui_input", callable_mp(this, &ResourceInspectorDock::_tab_gui_input));
	add_child(tab_container);
}

Error ResourceInspectorDock::inspect_resource(const String &p_resource) {
	Ref<Resource> res = ResourceLoader::load(p_resource);
	ERR_FAIL_COND_V(res.is_null(), ERR_CANT_OPEN);

	edit_resource(res);
	
	return OK;
}

Error ResourceInspectorDock::edit_resource(const Ref<Resource> &p_resource) {

	String path = p_resource->get_path();
	 // Check if it's a script or scene file
    if (!path.ends_with(".tres")) {
        return ERR_INVALID_PARAMETER;
    }


	// Check if already open
	for (int i = 0; i < tab_container->get_tab_count(); i++) {
		EditorInspector *inspector = Object::cast_to<EditorInspector>(tab_container->get_tab_control(i)->get_child(0));
		if (inspector && inspector->get_edited_object() == p_resource.ptr()) {
			tab_container->set_current_tab(i);
			return OK;
		}

	}

	// Create new tab
	VBoxContainer *tab_vbox = memnew(VBoxContainer);
	EditorInspector *inspector = memnew(EditorInspector);
	inspector->set_v_size_flags(SIZE_EXPAND_FILL);
	inspector->set_use_doc_hints(true);
	inspector->set_hide_script(true);
	inspector->set_hide_metadata(true);
	inspector->set_use_folding(!bool(EDITOR_GET("interface/inspector/disable_folding")));
	inspector->set_property_name_style(EditorPropertyNameProcessor::get_default_inspector_style());
	inspector->set_use_filter(true);
	inspector->set_show_categories(false, true);
	inspector->set_autoclear(true);
	tab_vbox->add_child(inspector);

	String tab_title = path.get_file();
	tab_container->add_child(tab_vbox);
	tab_container->set_current_tab(tab_container->get_tab_count() - 1);
	tab_container->set_tab_title(tab_container->get_tab_count() - 1, tab_title);


	inspector->edit(p_resource.ptr());

	return OK;
}


void ResourceInspectorDock::_tab_gui_input(const Ref<InputEvent> &p_event) {
	const Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->get_button_index() == MouseButton::RIGHT && mb->is_pressed()) {
		int tab_clicked = tab_container->get_current_tab();
		if (tab_clicked >= 0) {
			Control *tab = Object::cast_to<Control>(tab_container->get_tab_control(tab_clicked));
			if (tab) {
				tab_container->remove_child(tab);
			}
		}
	}
}

void ResourceInspectorDock::_cleanup_tabs() {
	for (int i = 0; i < tab_container->get_tab_count(); i++) {
		Control *tab = Object::cast_to<Control>(tab_container->get_tab_control(i));
		if (tab) {
			tab_container->remove_child(tab);
		}
	}
}


void ResourceInspectorDock::_bind_methods() {
}

ResourceInspectorDock::~ResourceInspectorDock() {
	singleton = nullptr;
}