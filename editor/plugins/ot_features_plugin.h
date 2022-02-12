/*************************************************************************/
/*  ot_features_plugin.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef OT_FEATURES_PLUGIN_H
#define OT_FEATURES_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "editor/editor_properties.h"

/*************************************************************************/

class OpenTypeFeaturesEditor : public EditorProperty {
	GDCLASS(OpenTypeFeaturesEditor, EditorProperty);
	EditorSpinSlider *spin;
	bool setting = true;
	void _value_changed(double p_val);
	Button *button = nullptr;

	void _remove_feature();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property() override;
	OpenTypeFeaturesEditor();
};

/*************************************************************************/

class OpenTypeFeaturesAdd : public EditorProperty {
	GDCLASS(OpenTypeFeaturesAdd, EditorProperty);

	Button *button = nullptr;
	PopupMenu *menu = nullptr;
	PopupMenu *menu_ss = nullptr;
	PopupMenu *menu_cv = nullptr;
	PopupMenu *menu_cu = nullptr;

	void _add_feature(int p_option);
	void _features_menu();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_property() override;

	OpenTypeFeaturesAdd();
};

/*************************************************************************/

class EditorInspectorPluginOpenTypeFeatures : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginOpenTypeFeatures, EditorInspectorPlugin);

public:
	virtual bool can_handle(Object *p_object) override;
	virtual bool parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const uint32_t p_usage, const bool p_wide = false) override;
};

/*************************************************************************/

class OpenTypeFeaturesEditorPlugin : public EditorPlugin {
	GDCLASS(OpenTypeFeaturesEditorPlugin, EditorPlugin);

public:
	OpenTypeFeaturesEditorPlugin(EditorNode *p_node);

	virtual String get_name() const override { return "OpenTypeFeatures"; }
};

#endif // OT_FEATURES_PLUGIN_H
