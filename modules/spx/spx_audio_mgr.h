/**************************************************************************/
/*  spx_audio_mgr.h                                                       */
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

#ifndef SPX_AUDIO_MGR_H
#define SPX_AUDIO_MGR_H

#include "gdextension_spx_ext.h"
#include "spx_base_mgr.h"
#include "scene/main/node.h"
#include "scene/2d/node_2d.h"
#include "core/templates/rb_map.h"
#include "core/templates/list.h"
#include "core/os/mutex.h"

// Forward declarations
class SpxAudio;
class AudioStreamPlayer2D;

class SpxAudioMgr : SpxBaseMgr {
	SPXCLASS(SpxAudioMgr, SpxBaseMgr)

private:
	RBMap<GdObj, SpxAudio *> id_audios;
	RBMap<GdInt, SpxAudio *> aid_audios;
	Node *root = nullptr;
	GdObj g_audio_id;

	static Mutex lock;
	SpxAudio *_get_audio(GdObj obj);
	SpxAudio *_get_aid_audio(GdInt aid);
	
public:
	virtual ~SpxAudioMgr() = default; // Added virtual destructor to fix -Werror=non-virtual-dtor

public:
	void on_awake() override;
	void on_destroy() override;
	void on_update(float delta) override;

public:
	void stop_all();
	GdObj create_audio();
	void destroy_audio(GdObj obj);

	void set_pitch(GdObj obj, GdFloat pitch);
	GdFloat get_pitch(GdObj obj);
	void set_pan(GdObj obj, GdFloat pan);
	GdFloat get_pan(GdObj obj);
	void set_volume(GdObj obj, GdFloat volume);
	GdFloat get_volume(GdObj obj);

	// play audio and return the audioid
	GdInt play_with_attenuation(GdObj obj, GdString path,GdObj owner_id, GdFloat attenuation ,GdFloat max_distance );
	GdInt play(GdObj obj, GdString path);
	void pause(GdInt aid);
	void resume(GdInt aid);
	void stop(GdInt aid);
	void set_loop(GdInt aid, GdBool loop);
	GdBool get_loop(GdInt aid);
	
	GdFloat get_timer(GdInt aid);
	void set_timer(GdInt aid, GdFloat time);
	GdBool is_playing(GdInt aid);
	
};

#endif // SPX_AUDIO_MGR_H
