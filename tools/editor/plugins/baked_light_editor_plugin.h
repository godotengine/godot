#ifndef BAKED_LIGHT_EDITOR_PLUGIN_H
#define BAKED_LIGHT_EDITOR_PLUGIN_H

#include "tools/editor/editor_plugin.h"
#include "tools/editor/editor_node.h"
#include "scene/3d/baked_light.h"
#include "scene/gui/spin_box.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/


class BakedLightBaker;
class MeshInstance;

class BakedLightEditor : public Control {

	OBJ_TYPE(BakedLightEditor, Control );


	MeshInstance *preview;
	BakedLightBaker *baker;
	AcceptDialog *err_dialog;

	MenuButton * options;
	BakedLight *node;

	enum Menu {

		MENU_OPTION_BAKE,
		MENU_OPTION_CLEAR
	};

	void _menu_option(int);

friend class BakedLightEditorPlugin;
protected:
	void _node_removed(Node *p_node);
	static void _bind_methods();
public:

	void edit(BakedLight *p_baked_light);
	BakedLightEditor();
	~BakedLightEditor();
};

class BakedLightEditorPlugin : public EditorPlugin {

	OBJ_TYPE( BakedLightEditorPlugin, EditorPlugin );

	BakedLightEditor *baked_light_editor;
	EditorNode *editor;

public:

	virtual String get_name() const { return "BakedLight"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	BakedLightEditorPlugin(EditorNode *p_node);
	~BakedLightEditorPlugin();

};

#endif // MULTIMESH_EDITOR_PLUGIN_H


