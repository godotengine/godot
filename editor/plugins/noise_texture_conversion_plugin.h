/**************************************************************************/
/*  noise_texture_conversion_plugin.h                                     */
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

#include "editor/editor_node.h"
#include "editor/plugins/editor_resource_conversion_plugin.h"
#include "scene/gui/check_box.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/line_edit.h"

class ConvertTextureDialog;
class EditorValidationPanel;
class LineEdit;

class NoiseTextureConversionPlugin : public EditorResourceConversionPlugin {
	GDCLASS(NoiseTextureConversionPlugin, EditorResourceConversionPlugin);

	struct PendingResourceUpdate {
		Callable cb;
		String fpath;
	};

public:
	virtual String converts_to() const override;
	virtual bool handles(const Ref<Resource> &p_resource) const override;
	Ref<Resource> convert(const Ref<Resource> &p_resource) const override;
	bool convert_async(const Ref<Resource> &p_resource, const Callable &p_on_complete) override;
	ConvertTextureDialog *create_confirmation_dialog();

private:
	void _confirm_conversion();
	void _on_filesystem_updated();
	ConvertTextureDialog *dialog;
	Callable callback;
	Ref<Resource> resource;
	Vector<PendingResourceUpdate> pending_updates;
};

class ConvertTextureDialog : public ConfirmationDialog {
	GDCLASS(ConvertTextureDialog, ConfirmationDialog);

	enum {
		MSG_ID_PATH,
		MSG_ID_EMPTY,
		MSG_ID_INFO_0,
		MSG_ID_INFO_1
	};

	Ref<Resource> resource = nullptr;
	EditorValidationPanel *validation_panel = nullptr;
	EditorFileDialog *file_browse = nullptr;
	LineEdit *file_path = nullptr;
	Button *path_button = nullptr;
	CheckBox *checkbox_overwrite = nullptr;

public:
	ConvertTextureDialog();
	String get_file_path() const { return file_path->get_text(); }

	void config(const Ref<Resource> &p_resource);

protected:
	void _notification(int p_what);

private:
	void _browse_path();
	void _browse_path_selected(String selected_path);
	void _overwrite_button_pressed();
	void _check_path_and_content();
};
