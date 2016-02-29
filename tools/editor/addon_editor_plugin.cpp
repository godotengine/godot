#include "addon_editor_plugin.h"
#include "editor_node.h"
EditorAddonLibrary::EditorAddonLibrary() {

	tabs = memnew( TabContainer );
	tabs->set_v_size_flags(SIZE_EXPAND_FILL);
	add_child(tabs);

	installed = memnew( EditorPluginSettings );
	installed->set_name("Installed");
	tabs->add_child(installed);

	library = memnew( VBoxContainer );
	library->set_name("Online");
	tabs->add_child(library);

	HBoxContainer *search_hb = memnew( HBoxContainer );

	library->add_child(search_hb);

	search_hb->add_child( memnew( Label("Search: ")));
	filter =memnew( LineEdit );
	search_hb->add_child(filter);
	filter->set_h_size_flags(SIZE_EXPAND_FILL);

	categories = memnew( OptionButton );
	categories->add_item("All Categories");
	search_hb->add_child(categories);

	search = memnew( Button("Search"));
	search_hb->add_child(search);




}


///////


void AddonEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {

		addon_library->show();
	} else {

		addon_library->hide();
	}

}

AddonEditorPlugin::AddonEditorPlugin(EditorNode *p_node) {

	editor=p_node;
	addon_library = memnew( EditorAddonLibrary );
	addon_library->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	editor->get_viewport()->add_child(addon_library);
	addon_library->set_area_as_parent_rect();
	addon_library->hide();

}

AddonEditorPlugin::~AddonEditorPlugin() {

}
