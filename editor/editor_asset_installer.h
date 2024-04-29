/**************************************************************************/
/*  editor_asset_installer.h                                              */
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

#ifndef EDITOR_ASSET_INSTALLER_H
#define EDITOR_ASSET_INSTALLER_H

#include "scene/gui/dialogs.h"
#include "scene/gui/tree.h"

class CheckBox;
class Label;

class EditorAssetInstaller : public ConfirmationDialog {
	GDCLASS(EditorAssetInstaller, ConfirmationDialog);

	Tree *tree = nullptr;
	Label *asset_title_label = nullptr;
	Label *asset_conflicts_label = nullptr;
	CheckBox *skip_toplevel_check = nullptr;

	String package_path;
	String asset_name;
	HashSet<String> asset_files;
	HashMap<String, TreeItem *> file_item_map;

	Ref<Texture2D> generic_extension_icon;
	HashMap<String, Ref<Texture2D>> extension_icon_map;

	bool updating = false;
	String toplevel_prefix;
	bool skip_toplevel = false;

	void _check_has_toplevel();
	void _set_skip_toplevel(bool p_checked);

	void _rebuild_tree();
	TreeItem *_create_dir_item(TreeItem *p_parent, const String &p_path, HashMap<String, TreeItem *> &p_item_map);
	TreeItem *_create_file_item(TreeItem *p_parent, const String &p_path, int *r_conflicts);

	void _item_checked();
	void _check_propagated_to_item(Object *p_obj, int column);

	void _install_asset();
	virtual void ok_pressed() override;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void open_asset(const String &p_path, bool p_autoskip_toplevel = false);

	void set_asset_name(const String &p_asset_name);
	String get_asset_name() const;

	EditorAssetInstaller();
};

#endif // EDITOR_ASSET_INSTALLER_H
