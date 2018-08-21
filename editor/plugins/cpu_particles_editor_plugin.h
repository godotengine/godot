#ifndef CPU_PARTICLES_EDITOR_PLUGIN_H
#define CPU_PARTICLES_EDITOR_PLUGIN_H

#include "editor/plugins/particles_editor_plugin.h"
#include "scene/3d/cpu_particles.h"

class CPUParticlesEditor : public ParticlesEditorBase {

	GDCLASS(CPUParticlesEditor, ParticlesEditorBase);

	enum Menu {

		MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_NODE,
		MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_MESH,
		MENU_OPTION_CLEAR_EMISSION_VOLUME,

	};

	CPUParticles *node;

	void _menu_option(int);

	friend class CPUParticlesEditorPlugin;

	virtual void _generate_emission_points();

protected:
	void _notification(int p_notification);
	void _node_removed(Node *p_node);
	static void _bind_methods();

public:
	void edit(CPUParticles *p_particles);
	CPUParticlesEditor();
};

class CPUParticlesEditorPlugin : public EditorPlugin {

	GDCLASS(CPUParticlesEditorPlugin, EditorPlugin);

	CPUParticlesEditor *particles_editor;
	EditorNode *editor;

public:
	virtual String get_name() const { return "CPUParticles"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	CPUParticlesEditorPlugin(EditorNode *p_node);
	~CPUParticlesEditorPlugin();
};

#endif // CPU_PARTICLES_EDITOR_PLUGIN_H
