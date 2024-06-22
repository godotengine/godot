#ifndef SPICY_PARTICLE_SYSTEM_PLUGIN_H
#define SPICY_PARTICLE_SYSTEM_PLUGIN_H

#include "scene/3d/spicyparticlesystem/SpicyParticleSystemNode.h"

#include "editor/plugins/editor_plugin.h"
#include "scene/2d/sprite_2d.h"
#include "scene/gui/spin_box.h"
#include "editor/editor_data.h"
#include "editor/editor_properties.h"
#include "editor/property_selector.h"
#include "scene/3d/node_3d.h"
#include "scene/gui/control.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/scroll_bar.h"
#include "scene/gui/tree.h"
#include "scene/resources/animation.h"


class SpicyParticleSystemModuleInspectorPlugin : public EditorInspectorPlugin
{
	GDCLASS(SpicyParticleSystemModuleInspectorPlugin, EditorInspectorPlugin)
private:
	const Control* base_ref = nullptr;
protected:
	void _control_ref_added(Control* ref);
	static void _bind_methods() {}
public:
	SpicyParticleSystemModuleInspectorPlugin();
	~SpicyParticleSystemModuleInspectorPlugin();

	virtual bool _can_handle(Object* p_object) const;
	virtual void _parse_end(Object* object);

	inline void set_base_ref(const Control* control) { base_ref = control; }
};

class SpicyParticleSystemInspectorPlugin : public EditorInspectorPlugin
{
	GDCLASS(SpicyParticleSystemInspectorPlugin, EditorInspectorPlugin)
private:
	const Control* base_ref = nullptr;
protected:
	static void _bind_methods() {}
public:
	SpicyParticleSystemInspectorPlugin();
	~SpicyParticleSystemInspectorPlugin();

	virtual bool _can_handle(Object* p_object) const;
	virtual bool _parse_property(Object* object, Variant::Type type, const String& name, PropertyHint hint_type, const String& hint_string, BitField<PropertyUsageFlags> usage_flags, bool wide);
	//virtual void _parse_group(Object* object, const String& group);

	inline void set_base_ref(const Control* control) { base_ref = control; }
};

class EditorPropertyRandomInteger : public EditorProperty 
{
	GDCLASS(EditorPropertyRandomInteger, EditorProperty);
private:
	EditorSpinSlider* spin = nullptr;
	Button* button = nullptr;
	bool setting = false;
	const Control* base_ref = nullptr;


protected:
	static void _bind_methods();

public:
	EditorPropertyRandomInteger();

	void _randomize_integer();
	void _value_changed(int64_t p_val);
	virtual void _set_read_only(bool p_read_only) override;
	virtual void update_property() override;
	void setup(int64_t p_min, int64_t p_max, int64_t p_step, bool p_hide_slider, bool p_allow_greater, bool p_allow_lesser, const String& p_suffix = String());

	inline void set_base_ref(const Control* control) { base_ref = control; }
};


#pragma region EditorPlugin

class SpicyParticleSystemPlugin : public EditorPlugin
{
	GDCLASS(SpicyParticleSystemPlugin, EditorPlugin)
private:
	SpicyParticleSystemNode* system_node = nullptr;
	Ref<SpicyParticleSystemModuleInspectorPlugin> module_inspector_plugin;
	Ref<SpicyParticleSystemInspectorPlugin> inspector_plugin;
	PanelContainer* on_screen_editor = nullptr;

	Button* play_pause_button = nullptr;
	EditorSpinSlider* spin_box_time = nullptr;
	Label* label_count = nullptr;
	Button* button_move = nullptr;

	Ref<Texture2D> play_icon;
	Ref<Texture2D> pause_icon;
	Ref<Texture2D> restart_icon;
	Ref<Texture2D> stop_icon;

	bool only_selected = false;
	TypedArray<Node> selection;

	bool move_button_down = false;
	Vector2 mouse_offset = Vector2(0, 0);

	const Control* view_control = nullptr;
private:
	void set_pause_icon();
	void set_play_icon();
	void _play_pause();
	void _restart();
	void _stop();
	void _set_simulation_time(double time);
	void _move_button_down();
	void _move_button_up();
	void _set_only_selected(bool show);
	void _update_selection();
protected:
	static void _bind_methods() {}
public:
	SpicyParticleSystemPlugin();
	~SpicyParticleSystemPlugin();

	String get_name() const override { return "SpicyParticleSystemPlugin"; }
	void _enter_tree() override;
	void _exit_tree() override;
	void _process(double delta);

	void _create_on_screen_editor();
	void _update_on_screen_editor();
	virtual void _edit(Object* object);
	virtual void _make_visible(bool visible);
	virtual bool _handles(Object* object) const;
	virtual void _forward_3d_draw_over_viewport(Control* viewport_control);
	virtual int32_t _forward_3d_gui_input(Camera3D* viewport_camera, const Ref<InputEvent>& event);
};


#pragma endregion

#endif