/**************************************************************************/
/*  audio_stream_effect_editor_plugin.cpp                                 */
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

#include "audio_stream_effect_editor_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "scene/resources/audio_stream_effect.h"

void AudioStreamEffectEditorPlugin::edit(Object *p_object) {
}

bool AudioStreamEffectEditorPlugin::handles(Object *p_object) const {
	return false;
}

void AudioStreamEffectEditorPlugin::make_visible(bool p_visible) {
}

void AudioStreamEffectEditorPlugin::_move_effect_array_element(Object *p_undo_redo, Object *p_edited, const String &p_array_prefix, int p_from_index, int p_to_pos) {
	EditorUndoRedoManager *undo_redo_man = Object::cast_to<EditorUndoRedoManager>(p_undo_redo);
	ERR_FAIL_NULL(undo_redo_man);

	AudioStreamEffect *stream = Object::cast_to<AudioStreamEffect>(p_edited);
	if (!stream) {
		return;
	}

	// Compute the array indices to save.
	int begin = 0;
	int end;
	if (p_array_prefix == "effect_") {
		end = stream->get_effect_count();
	} else {
		ERR_FAIL_MSG("Invalid array prefix for AudioStreamEffect.");
	}
	if (p_from_index < 0) {
		// Adding new.
		if (p_to_pos >= 0) {
			begin = p_to_pos;
		} else {
			end = 0; // Nothing to save when adding at the end.
		}
	} else if (p_to_pos < 0) {
		// Removing.
		begin = p_from_index;
	} else {
		// Moving.
		begin = MIN(p_from_index, p_to_pos);
		end = MIN(MAX(p_from_index, p_to_pos) + 1, end);
	}

#define ADD_UNDO(obj, property) undo_redo_man->add_undo_property(obj, property, obj->get(property));
	// Save layers' properties.
	if (p_from_index < 0) {
		undo_redo_man->add_undo_method(stream, "remove_effect", p_to_pos < 0 ? stream->get_effect_count() : p_to_pos);
	} else if (p_to_pos < 0) {
		undo_redo_man->add_undo_method(stream, "add_effect", p_from_index, Ref<AudioEffect>());
	}

	List<PropertyInfo> properties;
	stream->get_property_list(&properties);
	for (PropertyInfo pi : properties) {
		if (pi.name.begins_with(p_array_prefix)) {
			String str = pi.name.trim_prefix(p_array_prefix);
			int to_char_index = 0;
			while (to_char_index < str.length()) {
				if (str[to_char_index] < '0' || str[to_char_index] > '9') {
					break;
				}
				to_char_index++;
			}
			if (to_char_index > 0) {
				int array_index = str.left(to_char_index).to_int();
				if (array_index >= begin && array_index < end) {
					ADD_UNDO(stream, pi.name);
				}
			}
		}
	}
#undef ADD_UNDO

	if (p_from_index < 0) {
		undo_redo_man->add_do_method(stream, "add_effect", p_to_pos, Ref<AudioEffect>(), false);
	} else if (p_to_pos < 0) {
		undo_redo_man->add_do_method(stream, "remove_effect", p_from_index);
	} else {
		undo_redo_man->add_do_method(stream, "move_effect", p_from_index, p_to_pos);
	}
}

AudioStreamEffectEditorPlugin::AudioStreamEffectEditorPlugin() {
	EditorNode::get_editor_data().add_move_array_element_function(SNAME("AudioStreamEffect"), callable_mp(this, &AudioStreamEffectEditorPlugin::_move_effect_array_element));
}
