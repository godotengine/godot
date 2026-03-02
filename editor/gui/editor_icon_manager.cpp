/**************************************************************************/
/*  editor_icon_manager.cpp                                               */
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

#include "editor_icon_manager.h"

#include "editor/editor_node.h"
#include "scene/resources/atlas_texture.h"

void EditorIconManager::_notification(int p_what) {
	if (p_what == NOTIFICATION_THEME_CHANGED) {
		for (KeyValue<StringName, Ref<AtlasTexture>> &icon : icons) {
			icon.value->set_atlas(get_editor_theme_icon(icon.key));
		}
	}
}

EditorIconManager::EditorIconManager() {
	singleton = this;
}

Ref<Texture2D> EditorIconManager::get_icon(const StringName &p_icon_name) {
	Ref<AtlasTexture> *existing = singleton->icons.getptr(p_icon_name);
	if (existing) {
		return *existing;
	}
	Ref<AtlasTexture> icon;
	icon.instantiate();
	icon->set_atlas(singleton->get_editor_theme_icon(p_icon_name));
	singleton->icons[p_icon_name] = icon;
	return icon;
}
