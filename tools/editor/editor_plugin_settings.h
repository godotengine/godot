#ifndef EDITORPLUGINSETTINGS_H
#define EDITORPLUGINSETTINGS_H


#include "scene/gui/dialogs.h"
#include "property_editor.h"
#include "optimized_save_dialog.h"
#include "undo_redo.h"
#include "editor_data.h"

class EditorPluginSettings : public VBoxContainer {

	OBJ_TYPE(EditorPluginSettings,VBoxContainer);

	Button* update_list;
	Tree *plugin_list;
	bool updating;


	void _plugin_activity_changed();
protected:

	void _notification(int p_what);

	static void _bind_methods();


public:

	void update_plugins();

	EditorPluginSettings();
};

#endif // EDITORPLUGINSETTINGS_H
