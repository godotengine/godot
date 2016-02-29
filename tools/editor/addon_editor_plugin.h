#ifndef ADDON_EDITOR_PLUGIN_H
#define ADDON_EDITOR_PLUGIN_H


#include "editor_plugin.h"
#include "scene/gui/box_container.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/option_button.h"
#include "scene/gui/tab_container.h"
#include "editor_plugin_settings.h"

class EditorAddonLibrary : public VBoxContainer {
	OBJ_TYPE(EditorAddonLibrary,VBoxContainer);

	TabContainer *tabs;
	EditorPluginSettings *installed;
	VBoxContainer *library;
	LineEdit *filter;
	OptionButton *categories;
	Button *search;


public:
	EditorAddonLibrary();
};

class AddonEditorPlugin : public EditorPlugin {

	OBJ_TYPE( AddonEditorPlugin, EditorPlugin );

	EditorAddonLibrary *addon_library;
	EditorNode *editor;

public:

	virtual String get_name() const { return "Addons"; }
	bool has_main_screen() const { return true; }
	virtual void edit(Object *p_object) {}
	virtual bool handles(Object *p_object) const { return false; }
	virtual void make_visible(bool p_visible);
	//virtual bool get_remove_list(List<Node*> *p_list) { return canvas_item_editor->get_remove_list(p_list); }
	//virtual Dictionary get_state() const;
	//virtual void set_state(const Dictionary& p_state);

	AddonEditorPlugin(EditorNode *p_node);
	~AddonEditorPlugin();

};

#endif // EDITORASSETLIBRARY_H
