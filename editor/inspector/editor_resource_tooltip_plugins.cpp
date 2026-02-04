/**************************************************************************/
/*  editor_resource_tooltip_plugins.cpp                                   */
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

#include "editor_resource_tooltip_plugins.h"

#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/inspector/editor_resource_preview.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/gui/texture_rect.h"

void EditorResourceTooltipPlugin::_thumbnail_ready(const String &p_path, const Ref<Texture2D> &p_preview, const Ref<Texture2D> &p_small_preview, ObjectID p_trect_id) {
	TextureRect *tr = ObjectDB::get_instance<TextureRect>(p_trect_id);
	if (tr) {
		tr->set_texture(p_preview);
	}
}

void EditorResourceTooltipPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("request_thumbnail", "path", "control"), &EditorResourceTooltipPlugin::request_thumbnail);

	GDVIRTUAL_BIND(_handles, "type");
	GDVIRTUAL_BIND(_make_tooltip_for_path, "path", "metadata", "base");
}

VBoxContainer *EditorResourceTooltipPlugin::make_default_tooltip(const String &p_resource_path) {
	VBoxContainer *vb = memnew(VBoxContainer);
	vb->add_theme_constant_override("separation", -4 * EDSCALE);
	{
		Label *label = memnew(Label(p_resource_path.get_file()));
		vb->add_child(label);
	}

	ResourceUID::ID id = EditorFileSystem::get_singleton()->get_file_uid(p_resource_path);
	if (id != ResourceUID::INVALID_ID) {
		Label *label = memnew(Label(ResourceUID::get_singleton()->id_to_text(id)));
		vb->add_child(label);
	}

	{
		Ref<FileAccess> f = FileAccess::open(p_resource_path, FileAccess::READ);
		if (f.is_valid()) {
			Label *label = memnew(Label(vformat(TTR("Size: %s"), String::humanize_size(f->get_length()))));
			vb->add_child(label);
		} else {
			Label *label = memnew(Label(TTR("Invalid file or broken link.")));
			label->add_theme_color_override(SceneStringName(font_color), EditorNode::get_singleton()->get_gui_base()->get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
			vb->add_child(label);
			return vb;
		}
	}

	if (ResourceLoader::exists(p_resource_path)) {
		String type = ResourceLoader::get_resource_type(p_resource_path);
		Label *label = memnew(Label(vformat(TTR("Type: %s"), type)));
		vb->add_child(label);
	}

	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	if (da->is_link(p_resource_path)) {
		Label *link = memnew(Label(vformat(TTR("Link to: %s"), da->read_link(p_resource_path))));
		vb->add_child(link);
	}
	return vb;
}

void EditorResourceTooltipPlugin::append_editor_description_tooltip(const String &p_resource_path, VBoxContainer *p_default_tooltip) {
	if (!ResourceLoader::exists(p_resource_path)) {
		return;
	}

	Ref<Resource> cached_resource = ResourceCache::get_ref(p_resource_path);
	String editor_description;
	if (cached_resource.is_valid()) {
		// Use the editor description in the currently-loaded instance. (Might be unsaved)
		editor_description = cached_resource->get_editor_description();
	} else {
		// Not loaded yet.
		editor_description = ResourceLoader::get_resource_editor_description(p_resource_path);
	}
	if (!editor_description.is_empty()) {
		String tooltip;
		const PackedInt32Array boundaries = TS->string_get_word_breaks(editor_description, "", 80);
		for (int i = 0; i < boundaries.size(); i += 2) {
			const int start = boundaries[i];
			const int end = boundaries[i + 1];
			tooltip += "\n" + editor_description.substr(start, end - start + 1).rstrip("\n");
		}
		Label *label = memnew(Label(tooltip));
		label->set_auto_translate_mode(Node::AUTO_TRANSLATE_MODE_DISABLED);
		p_default_tooltip->add_child(label);
	}
}

void EditorResourceTooltipPlugin::request_thumbnail(const String &p_path, TextureRect *p_for_control) const {
	ERR_FAIL_NULL(p_for_control);
	EditorResourcePreview::get_singleton()->queue_resource_preview(p_path, callable_mp(const_cast<EditorResourceTooltipPlugin *>(this), &EditorResourceTooltipPlugin::_thumbnail_ready).bind(p_for_control->get_instance_id()));
}

bool EditorResourceTooltipPlugin::handles(const String &p_resource_type) const {
	bool ret = false;
	GDVIRTUAL_CALL(_handles, p_resource_type, ret);
	return ret;
}

Control *EditorResourceTooltipPlugin::make_tooltip_for_path(const String &p_resource_path, const Dictionary &p_metadata, Control *p_base) const {
	Control *ret = nullptr;
	GDVIRTUAL_CALL(_make_tooltip_for_path, p_resource_path, p_metadata, p_base, ret);
	return ret;
}

// EditorTextureTooltipPlugin

bool EditorTextureTooltipPlugin::handles(const String &p_resource_type) const {
	return ClassDB::is_parent_class(p_resource_type, "Texture2D") || ClassDB::is_parent_class(p_resource_type, "Image");
}

Control *EditorTextureTooltipPlugin::make_tooltip_for_path(const String &p_resource_path, const Dictionary &p_metadata, Control *p_base) const {
	HBoxContainer *hb = memnew(HBoxContainer);
	VBoxContainer *vb = Object::cast_to<VBoxContainer>(p_base);
	DEV_ASSERT(vb);
	vb->set_alignment(BoxContainer::ALIGNMENT_CENTER);

	Vector2 dimensions = p_metadata.get("dimensions", Vector2());
	Label *label = memnew(Label(vformat(TTR(U"Dimensions: %d × %d"), dimensions.x, dimensions.y)));
	vb->add_child(label);

	TextureRect *tr = memnew(TextureRect);
	tr->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	hb->add_child(tr);
	request_thumbnail(p_resource_path, tr);

	hb->add_child(vb);
	return hb;
}

// EditorAudioStreamTooltipPlugin

bool EditorAudioStreamTooltipPlugin::handles(const String &p_resource_type) const {
	return ClassDB::is_parent_class(p_resource_type, "AudioStream");
}

Control *EditorAudioStreamTooltipPlugin::make_tooltip_for_path(const String &p_resource_path, const Dictionary &p_metadata, Control *p_base) const {
	VBoxContainer *vb = Object::cast_to<VBoxContainer>(p_base);
	DEV_ASSERT(vb);

	double length = p_metadata.get("length", 0.0);
	if (length >= 60.0) {
		vb->add_child(memnew(Label(vformat(TTR("Length: %0dm %0ds"), int(length / 60.0), int(std::fmod(length, 60))))));
	} else if (length >= 1.0) {
		vb->add_child(memnew(Label(vformat(TTR("Length: %0.1fs"), length))));
	} else {
		vb->add_child(memnew(Label(vformat(TTR("Length: %0.3fs"), length))));
	}

	TextureRect *tr = memnew(TextureRect);
	vb->add_child(tr);
	request_thumbnail(p_resource_path, tr);

	return vb;
}
