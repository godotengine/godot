/**************************************************************************/
/*  noise_texture_conversion_plugin.cpp                                   */
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

#include "noise_texture_conversion_plugin.h"

#include "core/config/project_settings.h"
#include "core/io/resource_importer.h"
#include "editor/editor_file_system.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/gui/editor_validation_panel.h"
#include "editor/themes/editor_scale.h"
#include "modules/noise/noise_texture_2d.h"
#include "scene/gui/check_box.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/line_edit.h"
#include "scene/resources/image_texture.h"

String NoiseTextureConversionPlugin::converts_to() const {
	return "CompressedTexture2D";
}

bool NoiseTextureConversionPlugin::handles(const Ref<Resource> &p_resource) const {
	Ref<NoiseTexture2D> mat = p_resource;
	return mat.is_valid();
}

void ConvertTextureDialog::_check_path_and_content() {
	String file_path_text = file_path->get_text().strip_edges();

	String file_type = file_path->get_text().get_extension() == "png" ? "PNG" : "WebP";
	validation_panel->set_message(MSG_ID_INFO_0, vformat(TTR("The image data from the resource will be saved as a %s file to the path specified."), file_type), EditorValidationPanel::MSG_INFO);
	validation_panel->set_message(MSG_ID_INFO_1, TTR("The new image will be imported as a CompressedTexture2D which will replace the original resource."), EditorValidationPanel::MSG_INFO);

	if (file_path_text.is_empty()) {
		validation_panel->set_message(MSG_ID_PATH, TTR("Image path is empty."), EditorValidationPanel::MSG_ERROR);
		return;
	} else if (file_path_text.get_file().get_basename().is_empty()) {
		validation_panel->set_message(MSG_ID_PATH, TTR("Image file name is empty."), EditorValidationPanel::MSG_ERROR);
		return;
	} else if (!file_path_text.get_file().get_basename().is_valid_filename()) {
		validation_panel->set_message(MSG_ID_PATH, TTR("Image file name is invalid."), EditorValidationPanel::MSG_ERROR);
		return;
	}
	file_path_text = ProjectSettings::get_singleton()->localize_path(file_path_text);
	if (!file_path_text.begins_with("res://")) {
		validation_panel->set_message(MSG_ID_PATH, TTR("Path is not local."), EditorValidationPanel::MSG_ERROR);
	} else {
		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		if (da->dir_exists(file_path_text)) {
			validation_panel->set_message(MSG_ID_PATH, TTR("A directory with the same name exists."), EditorValidationPanel::MSG_ERROR);
		}
		if (da->file_exists(file_path_text) && !checkbox_overwrite->is_pressed()) {
			validation_panel->set_message(MSG_ID_PATH, TTR("A file with the same name exists."), EditorValidationPanel::MSG_ERROR);
		}
		if (!da->dir_exists(file_path_text.get_base_dir())) {
			validation_panel->set_message(MSG_ID_PATH, TTR("Parent directory does not exist."), EditorValidationPanel::MSG_ERROR);
		}
	}

	Ref<NoiseTexture2D> noise_tex = resource;
	auto base_image = noise_tex->get_image();
	if (base_image.is_null()) {
		validation_panel->set_message(MSG_ID_EMPTY, TTR("Empty image data! The texture's noise property cannot be empty."), EditorValidationPanel::MSG_ERROR);
	}
}

void ConvertTextureDialog::_browse_path_selected(String selected_path) {
	file_path->set_text(selected_path);
	validation_panel->update();
}

void ConvertTextureDialog::_browse_path() {
	file_browse->set_current_path(file_path->get_text());
	file_browse->popup_file_dialog();
}

void ConvertTextureDialog::_overwrite_button_pressed() {
	validation_panel->update();
}

ConvertTextureDialog::ConvertTextureDialog() {
	GridContainer *gc = memnew(GridContainer);
	gc->set_columns(2);

	validation_panel = memnew(EditorValidationPanel);
	validation_panel->add_line(MSG_ID_PATH, TTR("Image path/name is valid."));
	validation_panel->add_line(MSG_ID_EMPTY);
	validation_panel->add_line(MSG_ID_INFO_0);
	validation_panel->add_line(MSG_ID_INFO_1);
	validation_panel->set_update_callback(callable_mp(this, &ConvertTextureDialog::_check_path_and_content));
	validation_panel->set_accept_button(get_ok_button());
	validation_panel->update();

	VBoxContainer *vb = memnew(VBoxContainer);
	vb->add_child(gc);
	vb->add_child(validation_panel);
	add_child(vb);

	HBoxContainer *hb = memnew(HBoxContainer);
	file_path = memnew(LineEdit);
	file_path->connect(SceneStringName(text_changed), callable_mp(validation_panel, &EditorValidationPanel::update).unbind(1));
	file_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hb->add_child(file_path);
	hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	register_text_enter(file_path);
	path_button = memnew(Button);

	path_button->connect(SceneStringName(pressed), callable_mp(this, &ConvertTextureDialog::_browse_path));
	hb->add_child(path_button);
	Label *label = memnew(Label(TTR("New Image Path:")));
	gc->add_child(label);
	gc->add_child(hb);

	Label *label_overwrite = memnew(Label(TTR("Overwrite Existing Files:")));
	checkbox_overwrite = memnew(CheckBox);
	checkbox_overwrite->connect(SNAME("pressed"), callable_mp(this, &ConvertTextureDialog::_overwrite_button_pressed));
	gc->add_child(label_overwrite);
	gc->add_child(checkbox_overwrite);

	gc->set_custom_minimum_size(Size2(650, 0) * EDSCALE);
	set_title(TTR("Convert to CompressedTexture2D"));

	file_browse = memnew(EditorFileDialog);
	file_browse->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	// We may not want to allow overwriting, but that gets handled in the main dialog.
	file_browse->set_disable_overwrite_warning(true);
	file_browse->set_filters({ "*.webp", "*.png" });
	file_browse->connect(SNAME("file_selected"), callable_mp(this, &ConvertTextureDialog::_browse_path_selected));
	add_child(file_browse);
}

ConvertTextureDialog *NoiseTextureConversionPlugin::create_confirmation_dialog() {
	dialog = memnew(ConvertTextureDialog);
	dialog->connect(SceneStringName(confirmed), callable_mp(this, &NoiseTextureConversionPlugin::_confirm_conversion));
	return dialog;
}

void ConvertTextureDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			path_button->set_button_icon(get_editor_theme_icon(SNAME("Folder")));
		} break;
	}
}

void ConvertTextureDialog::config(const Ref<Resource> &p_resource) {
	resource = p_resource;
	file_path->set_text(resource->get_path().get_basename() + ".webp");
}

void NoiseTextureConversionPlugin::_confirm_conversion() {
	Ref<NoiseTexture2D> noise_tex = resource;
	if (noise_tex.is_null()) {
		callback.call(Ref<Resource>({}));
		return;
	}

	auto base_image = noise_tex->get_image();
	auto file_path = dialog->get_file_path();
	if (file_path.get_extension() == "png") {
		base_image->save_png(file_path);
	} else {
		base_image->save_webp(file_path);
	}
	base_image->notify_property_list_changed();
	pending_updates.append({ callback, dialog->get_file_path() });
	EditorFileSystem::get_singleton()->scan_changes();

	return;
}

void NoiseTextureConversionPlugin::_on_filesystem_updated() {
	// Keep track of resource updates that haven't completed yet.
	Vector<PendingResourceUpdate> remaining_updates;

	for (PendingResourceUpdate &update : pending_updates) {
		Ref<Resource> new_img = ResourceLoader::load(dialog->get_file_path(), "image");
		if (new_img.is_null()) {
			remaining_updates.push_back(update);
		} else {
			update.cb.call(new_img);
		}
	}
	pending_updates = remaining_updates;
	if (pending_updates.size() == 0) {
		EditorFileSystem::get_singleton()->disconnect(SNAME("filesystem_changed"),
				callable_mp(this, &NoiseTextureConversionPlugin::_on_filesystem_updated));
	}
}

bool NoiseTextureConversionPlugin::convert_async(const Ref<Resource> &p_resource, const Callable &p_on_complete) {
	resource = p_resource;
	callback = p_on_complete;
	EditorNode::get_singleton()->add_child(create_confirmation_dialog());
	dialog->popup_centered();
	dialog->config(p_resource);

	// The callback may already be set by an in-flight conversion.
	Callable cb = callable_mp(this, &NoiseTextureConversionPlugin::_on_filesystem_updated);
	if (!EditorFileSystem::get_singleton()->is_connected(SNAME("filesystem_changed"), cb)) {
		EditorFileSystem::get_singleton()->connect(SNAME("filesystem_changed"), cb);
	}

	return true;
}

Ref<Resource> NoiseTextureConversionPlugin::convert(const Ref<Resource> &p_resource) const {
	// Synchronous conversions not supported.
	return nullptr;
}
