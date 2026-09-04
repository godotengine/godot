/**************************************************************************/
/*  particles_editor_plugin.h                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "editor/plugins/editor_plugin.h"

class CheckBox;
class ConfirmationDialog;
class HBoxContainer;
class MenuButton;
class OptionButton;
class ParticleProcessMaterial;
class SceneTreeDialog;
class SpinBox;
class Texture2D;

class ParticlesEditorPlugin : public EditorPlugin {
	GDCLASS(ParticlesEditorPlugin, EditorPlugin);

private:
	enum {
		MENU_OPTION_CONVERT,
		MENU_RESTART
	};

	HBoxContainer *toolbar = nullptr;
	MenuButton *menu = nullptr;

protected:
	String handled_type;
	String conversion_option_name;

	Node *edited_node = nullptr;

	void _notification(int p_what);

	Ref<ParticleProcessMaterial> get_material_for_object_created_in_editor() const;
	Ref<Texture2D> get_texture_for_object_created_in_editor() const;
	bool need_show_lifetime_dialog(SpinBox *p_seconds);
	virtual void _menu_callback(int p_idx);

	virtual void _add_menu_options(PopupMenu *p_menu) {}
	virtual Node *_convert_particles() = 0;

public:
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	ParticlesEditorPlugin();
};
