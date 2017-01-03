#ifndef GIPROBEEDITORPLUGIN_H
#define GIPROBEEDITORPLUGIN_H

#include "tools/editor/editor_plugin.h"
#include "tools/editor/editor_node.h"
#include "scene/resources/material.h"
#include "scene/3d/gi_probe.h"



class GIProbeEditorPlugin : public EditorPlugin {

	GDCLASS( GIProbeEditorPlugin, EditorPlugin );

	GIProbe *gi_probe;

	Button *bake;
	EditorNode *editor;

	void _bake();
protected:

	static void _bind_methods();
public:

	virtual String get_name() const { return "GIProbe"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	GIProbeEditorPlugin(EditorNode *p_node);
	~GIProbeEditorPlugin();

};

#endif // GIPROBEEDITORPLUGIN_H
