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
	if (media.get() != nullptr) {
		for (const TrackData &T : tracks) {
			T.track.CueEntered(T.token);
		}
		tracks.clear();
		media->MediaFailed(singleton->token_f);
		media->MediaEnded(singleton->token_e);
		if (!ApiInformation::IsApiContractPresent(L"Windows.Foundation.UniversalApiContract", 4)) {
			media->PlaybackMediaMarkerReached(singleton->token_s);
		}
		media->Close();
		media.reset();

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

		SpeechSynthesizer synth = SpeechSynthesizer();

		if (ApiInformation::IsApiContractPresent(L"Windows.Foundation.UniversalApiContract", 4)) {
			synth.Options().IncludeWordBoundaryMetadata(true);
		}
		if (ApiInformation::IsApiContractPresent(L"Windows.Foundation.UniversalApiContract", 5)) {
			synth.Options().SpeakingRate(CLAMP(message.rate, 0.5, 6.0));
			synth.Options().AudioPitch(CLAMP(message.pitch, 0.0, 2.0));
			synth.Options().AudioVolume(CLAMP((double)message.volume / 100.0, 0.0, 1.0));
		}

		winrt::hstring name = winrt::hstring((const wchar_t *)message.voice.utf16().get_data());
		IVectorView<VoiceInformation> voices = SpeechSynthesizer::AllVoices();
		for (uint32_t i = 0; i < voices.Size(); i++) {
			VoiceInformation voice = voices.GetAt(i);
			if (voice.Id() == name) {
				synth.Voice(voice);
				break;
			}
		}

		string = message.text.utf16();
		winrt::hstring text = winrt::hstring((const wchar_t *)string.get_data());

		SpeechSynthesisStream stream = synth.SynthesizeTextToStreamAsync(text).get();

		media = std::make_shared<MediaPlayer>();
		token_f = media->MediaFailed([=, this](const MediaPlayer &p_sender, const MediaPlayerFailedEventArgs &p_args) {
			_dispose_current(false, true);
		});
		token_e = media->MediaEnded([=, this](const MediaPlayer &p_sender, const IInspectable &p_args) {
			_dispose_current(false, false);
		});
		if (ApiInformation::IsApiContractPresent(L"Windows.Foundation.UniversalApiContract", 4)) {
			MediaPlaybackItem mitem = MediaPlaybackItem(MediaSource::CreateFromStream(stream, stream.ContentType()));
			media->Source(mitem);
			MediaPlaybackTimedMetadataTrackList list = mitem.TimedMetadataTracks();

			for (uint32_t i = 0; i < list.Size(); i++) {
				TimedMetadataTrack track = list.GetAt(i);
				if (track.TimedMetadataKind() == TimedMetadataKind::Speech) {
					winrt::event_token token = track.CueEntered([=, this](const TimedMetadataTrack &p_sender, const MediaCueEventArgs &p_args) {
						SpeechCue sq;
						p_args.Cue().as(sq);
						int32_t pos16 = sq.StartPositionInInput().Value();
						int pos = 0;
						for (int j = 0; j < MIN(pos16, string.length()); j++) {
							char16_t c = string[j];
							if ((c & 0xfffffc00) == 0xd800) {
								j++;
							}
							pos++;
						}
						callable_mp(singleton, &TTSDriverOneCore::_speech_index_mark).call_deferred(id, pos);
					});
					tracks.push_back({ track, token });
					list.SetPresentationMode(i, TimedMetadataTrackPresentationMode::ApplicationPresented);
				}
			}
		} else {
			media->Source(MediaSource::CreateFromStream(stream, stream.ContentType()));
			token_s = media->PlaybackMediaMarkerReached([=, this](const MediaPlayer &p_sender, const PlaybackMediaMarkerReachedEventArgs &p_args) {
				offset += p_args.PlaybackMediaMarker().Text().size() + 1;
				int pos = 0;
				for (int j = 0; j < MIN(offset, string.length()); j++) {
					char16_t c = string[j];
					if ((c & 0xfffffc00) == 0xd800) {
						j++;
					}
					pos++;
				}
				callable_mp(singleton, &TTSDriverOneCore::_speech_index_mark).call_deferred(id, pos);
			});
		}
		media->AutoPlay(true);

		id = message.id;
		update_requested = false;
		paused = false;

		media->Play();

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

	IVectorView<VoiceInformation> voices = SpeechSynthesizer::AllVoices();
	for (uint32_t i = 0; i < voices.Size(); i++) {
		VoiceInformation voice = voices.GetAt(i);
		winrt::hstring vname = voice.DisplayName();
		winrt::hstring vid = voice.Id();
		winrt::hstring vlang = voice.Language();

		Dictionary voice_d;
		voice_d["id"] = String::utf16((const char16_t *)vid.c_str(), vid.size());
		voice_d["name"] = String::utf16((const char16_t *)vname.c_str(), vname.size());
		voice_d["language"] = String::utf16((const char16_t *)vlang.c_str(), vlang.size());
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
	if (!ApiInformation::IsApiContractPresent(L"Windows.Foundation.UniversalApiContract", 1)) {
		print_verbose("Text-to-Speech: Cannot initialize OneCore driver, API contract not present!");
		return false;
	}
	if (SpeechSynthesizer::AllVoices().Size() == 0) {
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
