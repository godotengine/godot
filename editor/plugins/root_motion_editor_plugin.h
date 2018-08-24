#ifndef ROOT_MOTION_EDITOR_PLUGIN_H
#define ROOT_MOTION_EDITOR_PLUGIN_H

#include "editor/editor_inspector.h"
#include "editor/editor_spin_slider.h"
#include "editor/property_selector.h"
#include "scene/animation/animation_tree.h"

class EditorPropertyRootMotion : public EditorProperty {
	GDCLASS(EditorPropertyRootMotion, EditorProperty)
	Button *assign;
	Button *clear;
	NodePath base_hint;

	ConfirmationDialog *filter_dialog;
	Tree *filters;

	void _confirmed();
	void _node_assign();
	void _node_clear();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	virtual void update_property();
	void setup(const NodePath &p_base_hint);
	EditorPropertyRootMotion();
};

class EditorInspectorRootMotionPlugin : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorRootMotionPlugin, EditorInspectorPlugin)

public:
	virtual bool can_handle(Object *p_object);
	virtual void parse_begin(Object *p_object);
	virtual bool parse_property(Object *p_object, Variant::Type p_type, const String &p_path, PropertyHint p_hint, const String &p_hint_text, int p_usage);
	virtual void parse_end();
};

#endif // ROOT_MOTION_EDITOR_PLUGIN_H
