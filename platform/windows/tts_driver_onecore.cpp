/**************************************************************************/
/*  tts_driver_onecore.cpp                                                */
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

#include "tts_driver_onecore.h"

#include "core/object/callable_mp.h"
#include "servers/display/display_server.h"

static const WCHAR *ROSpeechSynthesizerName = L"Windows.Media.SpeechSynthesis.SpeechSynthesizer";
static const WCHAR *ROMediaPlayerName = L"Windows.Media.Playback.MediaPlayer";
static const WCHAR *ROMediaSourceName = L"Windows.Media.Core.MediaSource";
static const WCHAR *ROMediaPlaybackItemName = L"Windows.Media.Playback.MediaPlaybackItem";
static const WCHAR *ROSpeechSynthesisStreamName = L"Windows.Media.SpeechSynthesis.SpeechSynthesisStream";

/**************************************************************************/
/* MediaPlayer events                                                     */
/**************************************************************************/

GODOT_GCC_WARNING_PUSH
GODOT_GCC_WARNING_IGNORE("-Wnon-virtual-dtor")
GODOT_GCC_WARNING_IGNORE("-Wctor-dtor-privacy")
GODOT_GCC_WARNING_IGNORE("-Wshadow")
GODOT_GCC_WARNING_IGNORE("-Wstrict-aliasing")
GODOT_CLANG_WARNING_PUSH
GODOT_CLANG_WARNING_IGNORE("-Wnon-virtual-dtor")

class GodotMediaEndedEventHandler : public ROTypedEventHandler_MediaPlayer_IInspectable {
	TYPED_EVENT_HANDLER_CLASS(GodotMediaEndedEventHandler, ROTypedEventHandler_MediaPlayer_IInspectable)

	HRESULT STDMETHODCALLTYPE Invoke(ROMediaPlayer *p_sender, IInspectable *p_args) {
		ERR_FAIL_NULL_V(TTSDriverOneCore::singleton, E_ABORT);
		TTSDriverOneCore::singleton->_dispose_current(false, false);
		return S_OK;
	}
};

class GodotMediaFailedEventHandler : public ROTypedEventHandler_MediaPlayer_MediaPlayerFailedEventArgs {
	TYPED_EVENT_HANDLER_CLASS(GodotMediaFailedEventHandler, ROTypedEventHandler_MediaPlayer_MediaPlayerFailedEventArgs)

	HRESULT STDMETHODCALLTYPE Invoke(ROMediaPlayer *p_sender, ROMediaPlayerFailedEventArgs *p_args) {
		ERR_FAIL_NULL_V(TTSDriverOneCore::singleton, E_ABORT);
		TTSDriverOneCore::singleton->_dispose_current(false, true);
		return S_OK;
	}
};

class GodotMediaMarkerReachedEventHandler : public ROTypedEventHandler_MediaPlayer_PlaybackMediaMarkerReachedEventArgs {
	TYPED_EVENT_HANDLER_CLASS(GodotMediaMarkerReachedEventHandler, ROTypedEventHandler_MediaPlayer_PlaybackMediaMarkerReachedEventArgs)

	HRESULT STDMETHODCALLTYPE Invoke(ROMediaPlayer *p_sender, ROPlaybackMediaMarkerReachedEventArgs *p_args) {
		ERR_FAIL_NULL_V(TTSDriverOneCore::singleton, E_ABORT);

		ComPtr<ROPlaybackMediaMarkerReachedEventArgs> args_marker = p_args;
		ComPtr<ROPlaybackMediaMarker> marker;
		ERR_FAIL_COND_V(FAILED(args_marker->get_PlaybackMediaMarker((void **)marker.GetAddressOf())), E_ABORT);

		Ref<HStringWrapper> htext;
		htext.instantiate();
		ERR_FAIL_COND_V(FAILED(marker->get_Text((void **)htext->get_ptrw())), E_ABORT);

		String text = htext->get_string();
		TTSDriverOneCore::singleton->offset += text.length() + 1;
		int pos = 0;
		for (int j = 0; j < MIN(TTSDriverOneCore::singleton->offset, TTSDriverOneCore::singleton->string.length()); j++) {
			char16_t c = TTSDriverOneCore::singleton->string[j];
			if ((c & 0xfffffc00) == 0xd800) {
				j++;
			}
			pos++;
		}
		callable_mp(TTSDriverOneCore::singleton, &TTSDriverOneCore::_speech_index_mark).call_deferred(TTSDriverOneCore::singleton->id, pos);
		return S_OK;
	}
};

/**************************************************************************/
/* TimedMetadataTrack events                                              */
/**************************************************************************/

class GodotMediaCueEventHandler : public ROTypedEventHandler_TimedMetadataTrack_MediaCueEventArgs {
	TYPED_EVENT_HANDLER_CLASS(GodotMediaCueEventHandler, ROTypedEventHandler_TimedMetadataTrack_MediaCueEventArgs)

	HRESULT STDMETHODCALLTYPE Invoke(ROTimedMetadataTrack *p_sender, ROMediaCueEventArgs *p_args) {
		ERR_FAIL_NULL_V(TTSDriverOneCore::singleton, E_ABORT);

		ComPtr<ROMediaCueEventArgs> args_cue = p_args;
		ComPtr<ROMediaCue> media_cue;
		ERR_FAIL_COND_V(FAILED(args_cue->get_Cue((void **)media_cue.GetAddressOf())), E_ABORT);
		ComPtr<ROSpeechCue> sp_cue;
		ERR_FAIL_COND_V(FAILED(media_cue.As(&sp_cue)), E_ABORT);

		ComPtr<IInspectable> start_ref_ii;
		ERR_FAIL_COND_V(FAILED(sp_cue->get_StartPositionInInput((void **)start_ref_ii.GetAddressOf())), E_ABORT);
		ComPtr<ROReference_1_Int32> start_ref;
		ERR_FAIL_COND_V(FAILED(start_ref_ii.As(&start_ref)), E_ABORT);

		int32_t pos16 = 0;
		ERR_FAIL_COND_V(FAILED(start_ref->get_Value(&pos16)), E_ABORT);

		int pos = 0;
		for (int j = 0; j < MIN(pos16, TTSDriverOneCore::singleton->string.length()); j++) {
			char16_t c = TTSDriverOneCore::singleton->string[j];
			if ((c & 0xfffffc00) == 0xd800) {
				j++;
			}
			pos++;
		}
		callable_mp(TTSDriverOneCore::singleton, &TTSDriverOneCore::_speech_index_mark).call_deferred(TTSDriverOneCore::singleton->id, pos);
		return S_OK;
	}
};

GODOT_GCC_WARNING_POP
GODOT_CLANG_WARNING_POP

/**************************************************************************/
/* TTSDriverOneCore                                                       */
/**************************************************************************/

TTSDriverOneCore *TTSDriverOneCore::singleton = nullptr;

void TTSDriverOneCore::_speech_index_mark(int p_msg_id, int p_index_mark) {
	DisplayServer::get_singleton()->tts_post_utterance_event(DisplayServerEnums::TTS_UTTERANCE_BOUNDARY, p_msg_id, p_index_mark);
}

void TTSDriverOneCore::_speech_cancel(int p_msg_id) {
	DisplayServer::get_singleton()->tts_post_utterance_event(DisplayServerEnums::TTS_UTTERANCE_CANCELED, p_msg_id);
}

void TTSDriverOneCore::_speech_end(int p_msg_id) {
	DisplayServer::get_singleton()->tts_post_utterance_event(DisplayServerEnums::TTS_UTTERANCE_ENDED, p_msg_id);
}

void TTSDriverOneCore::_dispose_current(bool p_silent, bool p_canceled) {
	if (media) {
		for (const TrackData &T : tracks) {
			T.track->remove_CueEntered(T.token_c);
		}
		tracks.clear();
		media->remove_MediaFailed(token_f);
		media->remove_MediaEnded(token_e);
		if (token_s) {
			media->remove_PlaybackMediaMarkerReached(token_s);
		}
		media.Reset();
		token_e.reset();
		token_f.reset();
		token_s.reset();

		if (!p_silent) {
			if (p_canceled) {
				callable_mp(this, &TTSDriverOneCore::_speech_cancel).call_deferred(id);
			} else {
				callable_mp(this, &TTSDriverOneCore::_speech_end).call_deferred(id);
			}
		}
		id = -1;
		string = Char16String();
		playing = false;
		paused = false;
		offset = 0;
	}
}

void TTSDriverOneCore::process_events() {
	if (update_requested && !paused && queue.size() > 0 && !is_speaking()) {
		TTSUtterance &message = queue.front()->get();
		_dispose_current(true);
		playing = true;

		ComPtr<IActivationFactory> synth_fact;
		ERR_FAIL_COND(FAILED(WinRTUtils::activation_factory(ROSpeechSynthesizerName, IID_PPV_ARGS(&synth_fact))));

		ComPtr<ROSpeechSynthesizer> synth;
		ERR_FAIL_COND(FAILED(synth_fact->ActivateInstance((IInspectable **)synth.GetAddressOf())));

		if (WinRTUtils::is_api_contract_present(L"Windows.Foundation.UniversalApiContract", 4)) {
			ComPtr<ROSpeechSynthesizer2> synth2;
			HRESULT res = synth.As(&synth2);
			if (SUCCEEDED(res)) {
				ComPtr<ROSpeechSynthesizerOptions> opts;
				res = synth2->get_Options((void **)opts.GetAddressOf());
				if (SUCCEEDED(res)) {
					opts->put_IncludeWordBoundaryMetadata(true);
					if (WinRTUtils::is_api_contract_present(L"Windows.Foundation.UniversalApiContract", 5)) {
						ComPtr<ROSpeechSynthesizerOptions2> opts2;
						res = opts.As(&opts2);
						if (SUCCEEDED(res)) {
							opts2->put_AudioVolume(CLAMP((double)message.volume / 100.0, 0.0, 1.0));
							opts2->put_SpeakingRate(CLAMP(message.rate, 0.5, 6.0));
							opts2->put_AudioPitch(CLAMP(message.pitch, 0.0, 2.0));
						}
					}
				}
			}
		}

		ComPtr<ROInstalledVoicesStatic> synth_static;
		ERR_FAIL_COND(FAILED(WinRTUtils::activation_factory(ROSpeechSynthesizerName, IID_PPV_ARGS(&synth_static))));

		ComPtr<ROVectorView_IInspectable> voices;
		ERR_FAIL_COND(FAILED(synth_static->get_AllVoices((void **)voices.GetAddressOf())));

		uint32_t size = 0;
		ERR_FAIL_COND(FAILED(voices->get_Size(&size)));

		for (uint32_t i = 0; i < size; i++) {
			ComPtr<ROVoiceInformation> voice;
			ERR_CONTINUE(FAILED(voices->GetAt(i, (IInspectable **)voice.GetAddressOf())));

			Ref<HStringWrapper> vid;
			vid.instantiate();
			ERR_CONTINUE(FAILED(voice->get_Id((void **)vid->get_ptrw())));

			if (vid->get_string() == message.voice) {
				synth->put_Voice((void **)voice.Get());
			}
		}

		Ref<HStringWrapper> text;
		text.instantiate();
		text->set_string(message.text);
		string = message.text.utf16();

		ComPtr<ROAsyncOperation> stream_op;
		ERR_FAIL_COND(FAILED(synth->SynthesizeTextToStreamAsync(text->get_ptr(), (void **)stream_op.GetAddressOf())));

		ComPtr<ROAsyncInfo> stream_op_info;
		ERR_FAIL_COND(FAILED(stream_op.As(&stream_op_info)));

		ROAsyncStatus stream_op_status = ROAsyncStatus::Started;
		while (stream_op_status == ROAsyncStatus::Started) {
			Sleep(1);
			stream_op_info->get_Status((int32_t *)&stream_op_status);
		}
		ERR_FAIL_COND_MSG(stream_op_status != ROAsyncStatus::Completed, "AsyncOperation<SpeechSynthesisStream> failed.");

		ComPtr<IInspectable> stream_ii;
		ERR_FAIL_COND(FAILED(stream_op->GetResults((void **)stream_ii.GetAddressOf())));
		ComPtr<ROSpeechSynthesisStream> stream;
		ERR_FAIL_COND(FAILED(stream_ii.As(&stream)));
		ComPtr<ROContentTypeProvider> stream_ct;
		ERR_FAIL_COND(FAILED(stream.As(&stream_ct)));

		ComPtr<IActivationFactory> media_fact;
		ERR_FAIL_COND(FAILED(WinRTUtils::activation_factory(ROMediaPlayerName, IID_PPV_ARGS(&media_fact))));

		ComPtr<IInspectable> media_ii;
		ERR_FAIL_COND(FAILED(media_fact->ActivateInstance((IInspectable **)media_ii.GetAddressOf())));
		ERR_FAIL_COND(FAILED(media_ii.As(&media)));

		GodotMediaEndedEventHandler *handler_end = new GodotMediaEndedEventHandler();
		ERR_FAIL_COND(FAILED(handler_end->QueryInterface(IID_PPV_ARGS(&handler_e))));
		ERR_FAIL_COND(FAILED(media->add_MediaEnded(handler_e.Get(), &token_e)));
		GodotMediaFailedEventHandler *handler_fail = new GodotMediaFailedEventHandler();
		ERR_FAIL_COND(FAILED(handler_fail->QueryInterface(IID_PPV_ARGS(&handler_f))));
		ERR_FAIL_COND(FAILED(media->add_MediaFailed(handler_f.Get(), &token_f)));

		ComPtr<ROMediaSourceStatics> source_static;
		ERR_FAIL_COND(FAILED(WinRTUtils::activation_factory(ROMediaSourceName, IID_PPV_ARGS(&source_static))));

		Ref<HStringWrapper> ctype;
		ctype.instantiate();
		ERR_FAIL_COND(FAILED(stream_ct->get_ContentType((void **)ctype->get_ptrw())));

		ComPtr<RORandomAccessStream> stream_ra;
		ERR_FAIL_COND(FAILED(stream.As(&stream_ra)));

		ComPtr<ROMediaSource> source;
		ERR_FAIL_COND(FAILED(source_static->CreateFromStream(stream_ra.Get(), ctype->get_ptr(), (void **)source.GetAddressOf())));

		ComPtr<ROMediaPlayerSource2> player_source;
		ERR_FAIL_COND(FAILED(media.As(&player_source)));

		if (WinRTUtils::is_api_contract_present(L"Windows.Foundation.UniversalApiContract", 4) && WinRTUtils::is_type_present(ROMediaPlaybackItemName)) {
			ComPtr<ROMediaPlaybackItemFactory> mitem_fact;
			ERR_FAIL_COND(FAILED(WinRTUtils::activation_factory(ROMediaPlaybackItemName, IID_PPV_ARGS(&mitem_fact))));

			ComPtr<ROMediaPlaybackItem> mitem;
			ERR_FAIL_COND(FAILED(mitem_fact->Create(source.Get(), (void **)mitem.GetAddressOf())));
			ERR_FAIL_COND(FAILED(player_source->put_Source(mitem.Get())));

			ComPtr<IInspectable> list_ii;
			ERR_FAIL_COND(FAILED(mitem->get_TimedMetadataTracks((void **)list_ii.GetAddressOf())));
			ComPtr<ROMediaPlaybackTimedMetadataTrackList> list;
			ComPtr<ROVectorView_TimedMetadataTrack> list_items;
			ERR_FAIL_COND(FAILED(list_ii.As(&list)));
			ERR_FAIL_COND(FAILED(list_ii.As(&list_items)));

			uint32_t list_size = 0;
			ERR_FAIL_COND(FAILED(list_items->get_Size(&list_size)));
			for (uint32_t i = 0; i < list_size; i++) {
				ComPtr<ROTimedMetadataTrack> track;
				ERR_CONTINUE(FAILED(list_items->GetAt(i, track.GetAddressOf())));

				int32_t track_kind = 0;
				ERR_CONTINUE(FAILED(track->get_TimedMetadataKind(&track_kind)));
				if ((ROTimedMetadataKind)track_kind == ROTimedMetadataKind::Speech) {
					ROEventToken token_c;
					ComPtr<ROTypedEventHandler_TimedMetadataTrack_MediaCueEventArgs> handler_c;
					GodotMediaCueEventHandler *handler_cue = new GodotMediaCueEventHandler();
					ERR_CONTINUE(FAILED(handler_cue->QueryInterface(IID_PPV_ARGS(&handler_c))));
					ERR_CONTINUE(FAILED(track->add_CueEntered(handler_c.Get(), &token_c)));
					ERR_CONTINUE(FAILED(list->SetPresentationMode(i, (int32_t)RoTimedMetadataTrackPresentationMode::ApplicationPresented)));

					tracks.push_back({ token_c, handler_c, track });
				}
			}
		} else {
			ERR_FAIL_COND(FAILED(player_source->put_Source(source.Get())));

			GodotMediaMarkerReachedEventHandler *handler_marker = new GodotMediaMarkerReachedEventHandler();
			ERR_FAIL_COND(FAILED(handler_marker->QueryInterface(IID_PPV_ARGS(&handler_s))));
			ERR_FAIL_COND(FAILED(media->add_PlaybackMediaMarkerReached(handler_s.Get(), &token_s)));
		}

		ERR_FAIL_COND(FAILED(media->put_AutoPlay(true)));

		id = message.id;
		update_requested = false;
		paused = false;

		ERR_FAIL_COND(FAILED(media->Play()));

		DisplayServer::get_singleton()->tts_post_utterance_event(DisplayServerEnums::TTS_UTTERANCE_STARTED, message.id);
		queue.pop_front();
	}
}

bool TTSDriverOneCore::is_speaking() const {
	return playing;
}

bool TTSDriverOneCore::is_paused() const {
	return paused;
}

Array TTSDriverOneCore::get_voices() const {
	Array list;

	ComPtr<ROInstalledVoicesStatic> synth_static;
	ERR_FAIL_COND_V(FAILED(WinRTUtils::activation_factory(ROSpeechSynthesizerName, IID_PPV_ARGS(&synth_static))), list);

	ComPtr<ROVectorView_IInspectable> voices;
	ERR_FAIL_COND_V(FAILED(synth_static->get_AllVoices((void **)voices.GetAddressOf())), list);

	uint32_t size = 0;
	ERR_FAIL_COND_V(FAILED(voices->get_Size(&size)), list);

	for (uint32_t i = 0; i < size; i++) {
		ComPtr<ROVoiceInformation> voice;
		ERR_CONTINUE(FAILED(voices->GetAt(i, (IInspectable **)voice.GetAddressOf())));

		Ref<HStringWrapper> vname;
		vname.instantiate();
		ERR_CONTINUE(FAILED(voice->get_DisplayName((void **)vname->get_ptrw())));

		Ref<HStringWrapper> vid;
		vid.instantiate();
		ERR_CONTINUE(FAILED(voice->get_Id((void **)vid->get_ptrw())));

		Ref<HStringWrapper> vlang;
		vlang.instantiate();
		ERR_CONTINUE(FAILED(voice->get_Language((void **)vlang->get_ptrw())));

		Dictionary voice_d;
		voice_d["id"] = vid->get_string();
		voice_d["name"] = vname->get_string();
		voice_d["language"] = vlang->get_string().replace_char('-', '_');
		list.push_back(voice_d);
	}
	return list;
}

void TTSDriverOneCore::speak(const String &p_text, const String &p_voice, int p_volume, float p_pitch, float p_rate, int64_t p_utterance_id, bool p_interrupt) {
	if (p_interrupt) {
		stop();
	}

	if (p_text.is_empty()) {
		DisplayServer::get_singleton()->tts_post_utterance_event(DisplayServerEnums::TTS_UTTERANCE_CANCELED, p_utterance_id);
		return;
	}

	TTSUtterance message;
	message.text = p_text;
	message.voice = p_voice;
	message.volume = CLAMP(p_volume, 0, 100);
	message.pitch = CLAMP(p_pitch, 0.f, 2.f);
	message.rate = CLAMP(p_rate, 0.1f, 10.f);
	message.id = p_utterance_id;
	queue.push_back(message);

	if (is_paused()) {
		resume();
	} else {
		update_requested = true;
	}
}

void TTSDriverOneCore::pause() {
	if (!paused && playing) {
		media->Pause();
		paused = true;
	}
}

void TTSDriverOneCore::resume() {
	if (paused && playing) {
		media->Play();
		paused = false;
	}
}

void TTSDriverOneCore::stop() {
	for (TTSUtterance &message : queue) {
		DisplayServer::get_singleton()->tts_post_utterance_event(DisplayServerEnums::TTS_UTTERANCE_CANCELED, message.id);
	}
	queue.clear();
	_dispose_current(false, true);
}

bool TTSDriverOneCore::init() {
	if (!WinRTUtils::is_initialized()) {
		print_verbose("Text-to-Speech: Cannot initialize OneCore driver, WinRT API not supported!");
		return false;
	}
	if (!WinRTUtils::is_api_contract_present(L"Windows.Foundation.UniversalApiContract", 1) || !WinRTUtils::is_type_present(ROSpeechSynthesizerName) || !WinRTUtils::is_type_present(ROSpeechSynthesisStreamName) || !WinRTUtils::is_type_present(ROMediaPlayerName)) {
		print_verbose("Text-to-Speech: Cannot initialize OneCore driver, API contract or type not present!");
	}
	ComPtr<ROInstalledVoicesStatic> synth_static;
	if (FAILED(WinRTUtils::activation_factory(ROSpeechSynthesizerName, IID_PPV_ARGS(&synth_static)))) {
		print_verbose("Text-to-Speech: Cannot initialize OneCore driver, interface not supported!");
		return false;
	}

	ComPtr<ROVectorView_IInspectable> voices;
	ERR_FAIL_COND_V(FAILED(synth_static->get_AllVoices((void **)voices.GetAddressOf())), false);

	uint32_t size = 0;
	ERR_FAIL_COND_V(FAILED(voices->get_Size(&size)), false);
	if (size == 0) {
		print_verbose("Text-to-Speech: Cannot initialize OneCore driver, no voices found!");
		return false;
	}

	print_verbose("Text-to-Speech: OneCore initialized.");
	return true;
}

TTSDriverOneCore::TTSDriverOneCore() {
	singleton = this;
}

TTSDriverOneCore::~TTSDriverOneCore() {
	_dispose_current(false, true);
	singleton = nullptr;
}
