#ifndef BAKED_LIGHTMAP_EDITOR_PLUGIN_H
#define BAKED_LIGHTMAP_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/3d/baked_lightmap.h"
#include "scene/resources/material.h"

class BakedLightmapEditorPlugin : public EditorPlugin {

	GDCLASS(BakedLightmapEditorPlugin, EditorPlugin);

	BakedLightmap *lightmap;

	Button *bake;
	EditorNode *editor;

	static EditorProgress *tmp_progress;
	static void bake_func_begin(int p_steps);
	static bool bake_func_step(int p_step, const String &p_description);
	static void bake_func_end();

	void _bake();

protected:
	static void _bind_methods();

public:
	virtual String get_name() const { return "BakedLightmap"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	BakedLightmapEditorPlugin(EditorNode *p_node);
	~BakedLightmapEditorPlugin();
};

#endif // BAKED_LIGHTMAP_EDITOR_PLUGIN_H
