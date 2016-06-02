#ifndef MESH_EDITOR_PLUGIN_H
#define MESH_EDITOR_PLUGIN_H

#include "tools/editor/editor_plugin.h"
#include "tools/editor/editor_node.h"
#include "scene/resources/material.h"
#include "scene/3d/light.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/camera.h"

class MeshEditor : public Control {

	OBJ_TYPE(MeshEditor, Control);



	float rot_x;
	float rot_y;

	Viewport *viewport;
	MeshInstance *mesh_instance;
	DirectionalLight *light1;
	DirectionalLight *light2;
	Camera *camera;

	Ref<Mesh> mesh;


	TextureButton *light_1_switch;
	TextureButton *light_2_switch;

	void _button_pressed(Node* p_button);
	bool first_enter;

	void _update_rotation();
protected:
	void _notification(int p_what);
	void _input_event(InputEvent p_event);
	static void _bind_methods();
public:

	void edit(Ref<Mesh> p_mesh);
	MeshEditor();
};


class MeshEditorPlugin : public EditorPlugin {

	OBJ_TYPE( MeshEditorPlugin, EditorPlugin );

	MeshEditor *mesh_editor;
	EditorNode *editor;

public:

	virtual String get_name() const { return "Mesh"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	MeshEditorPlugin(EditorNode *p_node);
	~MeshEditorPlugin();

};

#endif // MESH_EDITOR_PLUGIN_H
