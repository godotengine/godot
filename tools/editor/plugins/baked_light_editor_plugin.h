#ifndef BAKED_LIGHT_EDITOR_PLUGIN_H
#define BAKED_LIGHT_EDITOR_PLUGIN_H

#include "tools/editor/editor_plugin.h"
#include "tools/editor/editor_node.h"
#include "tools/editor/plugins/baked_light_baker.h"
#include "scene/gui/spin_box.h"



/**
	@author Juan Linietsky <reduzio@gmail.com>
*/



class MeshInstance;

class BakedLightEditor : public Control {

	OBJ_TYPE(BakedLightEditor, Control );


	float update_timeout;
	DVector<uint8_t> octree_texture;
	DVector<uint8_t> light_texture;
	DVector<int> octree_sampler;

	BakedLightBaker *baker;
	AcceptDialog *err_dialog;

	HBoxContainer *bake_hbox;
	Button *button_bake;
	Button *button_reset;
	Button *button_make_lightmaps;
	Label *bake_info;

	uint64_t last_rays_time;



	BakedLightInstance *node;

	enum Menu {

		MENU_OPTION_BAKE,
		MENU_OPTION_CLEAR
	};

	void _bake_lightmaps();

	void _bake_pressed();
	void _clear_pressed();

	void _end_baking();
	void _menu_option(int);

friend class BakedLightEditorPlugin;
protected:
	void _node_removed(Node *p_node);
	static void _bind_methods();
	void _notification(int p_what);
public:

	void edit(BakedLightInstance *p_baked_light);
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


