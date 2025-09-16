/**************************************************************************/
/*  spx_audio.h                                                       */
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

#ifndef SPX_AUDIO_H
#define SPX_AUDIO_H

#include "core/templates/list.h"
#include "core/templates/rb_map.h"
#include "core/string/string_name.h"
#include "gdextension_spx_ext.h"

class AudioStreamPlayer2D;
class Node;
class SpxAudio {
private:
	RBMap<GdInt, AudioStreamPlayer2D *> aid_audios;
	List<AudioStreamPlayer2D *> audios;
	List<AudioStreamPlayer2D *> loop_audios;
	Node *root;
	GdObj id;

	int bus_id;
	StringName bus_name;

	GdFloat cur_pitch = 1.0;

private:
	void on_bus_dirty();
	AudioStreamPlayer2D *_get_aid_audio(GdInt aid);
public:
	void on_create(GdInt id, Node *root);
	void on_destroy();
	void on_update(float delta);

public:
	void stop_all();
	void set_pitch(GdFloat pitch);
	GdFloat get_pitch();
	void set_pan(GdFloat pan);
	GdFloat get_pan();
	void set_volume(GdFloat volume);
	GdFloat get_volume();

	void play(GdInt aid, GdString path, Node* owner = nullptr, GdFloat attenuation = 1.0f, GdFloat max_distance = 2000.0f);
	void pause(GdInt aid);
	void resume(GdInt aid);
	void stop(GdInt aid);
    void set_loop(GdInt aid, GdBool loop);
	GdBool get_loop(GdInt aid);

	GdFloat get_timer(GdInt aid);
	void set_timer(GdInt aid, GdFloat time);
	GdBool is_playing(GdInt aid);
};

#endif // SPX_AUDIO_H
