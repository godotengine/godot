/**************************************************************************/
/*  tts_driver_onecore.h                                                  */
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

#include "tts_driver.h"
#include "winrt_utils.h"

struct TTSUtterance;

class GodotMediaEndedEventHandler;
class GodotMediaFailedEventHandler;
class GodotMediaMarkerReachedEventHandler;
class GodotMediaCueEventHandler;

class TTSDriverOneCore : public TTSDriver {
	friend class GodotMediaEndedEventHandler;
	friend class GodotMediaFailedEventHandler;
	friend class GodotMediaMarkerReachedEventHandler;
	friend class GodotMediaCueEventHandler;

	List<TTSUtterance> queue;

	bool playing = false;
	bool paused = false;
	bool update_requested = false;

	ComPtr<ROMediaPlayer> media;
	struct TrackData {
		ROEventToken token_c;
		ComPtr<ROTypedEventHandler_TimedMetadataTrack_MediaCueEventArgs> handler_c;
		ComPtr<ROTimedMetadataTrack> track;
	};
	Vector<TrackData> tracks;
	ComPtr<ROTypedEventHandler_MediaPlayer_PlaybackMediaMarkerReachedEventArgs> handler_s;
	ComPtr<ROTypedEventHandler_MediaPlayer_MediaPlayerFailedEventArgs> handler_f;
	ComPtr<ROTypedEventHandler_MediaPlayer_IInspectable> handler_e;
	ROEventToken token_s;
	ROEventToken token_f;
	ROEventToken token_e;
	int64_t offset = 0;
	int64_t id = -1;
	Char16String string;

	void _dispose_current(bool p_silent = false, bool p_canceled = false);
	void _speech_cancel(int p_msg_id);
	void _speech_end(int p_msg_id);
	void _speech_index_mark(int p_msg_id, int p_index_mark);

	static TTSDriverOneCore *singleton;

public:
	virtual bool is_speaking() const override;
	virtual bool is_paused() const override;
	virtual Array get_voices() const override;

	virtual void speak(const String &p_text, const String &p_voice, int p_volume = 50, float p_pitch = 1.f, float p_rate = 1.f, int64_t p_utterance_id = 0, bool p_interrupt = false) override;
	virtual void pause() override;
	virtual void resume() override;
	virtual void stop() override;

	virtual void process_events() override;

	virtual bool init() override;

	TTSDriverOneCore();
	~TTSDriverOneCore();
};
