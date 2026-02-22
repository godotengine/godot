/**************************************************************************/
/*  audio_play_button_editor_plugin.cpp                                   */
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

#include "audio_play_button_editor_plugin.h"

#include "core/object/callable_mp.h"
#include "scene/2d/audio_stream_player_2d.h"
#include "scene/3d/audio_stream_player_3d.h"
#include "scene/audio/audio_stream_player.h"

bool EditorInspectorAudioButtonPlugin::can_handle(Object *p_object) {
	return Object::cast_to<AudioStreamPlayer>(p_object) != nullptr ||
			Object::cast_to<AudioStreamPlayer2D>(p_object) != nullptr ||
			Object::cast_to<AudioStreamPlayer3D>(p_object) != nullptr;
}

void EditorInspectorAudioButtonPlugin::parse_category(Object *p_object, const String &p_category) {
	Callable callable;
	bool stream_is_empty;

	if (Object::cast_to<AudioStreamPlayer>(p_object) != nullptr && p_category == "AudioStreamPlayer") {
		AudioStreamPlayer *node = Object::cast_to<AudioStreamPlayer>(p_object);
		stream_is_empty = node->get_stream().is_null();
		callable = callable_mp(node, &AudioStreamPlayer::play).bind(0);
	} else if (Object::cast_to<AudioStreamPlayer2D>(p_object) != nullptr && p_category == "AudioStreamPlayer2D") {
		AudioStreamPlayer2D *node = Object::cast_to<AudioStreamPlayer2D>(p_object);
		stream_is_empty = node->get_stream().is_null();
		callable = callable_mp(node, &AudioStreamPlayer2D::play).bind(0);
	} else if (Object::cast_to<AudioStreamPlayer3D>(p_object) != nullptr && p_category == "AudioStreamPlayer3D") {
		AudioStreamPlayer3D *node = Object::cast_to<AudioStreamPlayer3D>(p_object);
		stream_is_empty = node->get_stream().is_null();
		callable = callable_mp(node, &AudioStreamPlayer3D::play).bind(0);
	} else {
		return;
	}

	EditorInspectorActionButton *action_button = memnew(EditorInspectorActionButton(TTRC("Play Audio"), "AudioStreamPlayer"));
	action_button->connect(SceneStringName(pressed), callable);
	action_button->set_disabled(stream_is_empty);
	action_button->set_tooltip_text(TTRC("Plays the audio stream in the editor. This is equivalent to calling \"play()\" from script."));

	add_custom_control(action_button);
}

AudioPlayButtonEditorPlugin::AudioPlayButtonEditorPlugin() {
	Ref<EditorInspectorAudioButtonPlugin> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}
