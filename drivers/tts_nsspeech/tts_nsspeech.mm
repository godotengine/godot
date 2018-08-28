/*************************************************************************/
/*  tts_nsspeech.mm                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "tts_nsspeech.h"

/*
API Documentation:
https://developer.apple.com/documentation/appkit/nsspeechsynthesizer?language=objc
*/

#ifdef NSTTS_ENABLED

@implementation GodotSpeechSynthesizerDelegate
- (void)discardMessages {

	messages.clear();
};

- (void)appendMessage:(const String &)message {

	messages.push_back(message);
};

- (void)speechSynthesizer:(NSSpeechSynthesizer *)sender didFinishSpeaking:(BOOL)finishedSpeaking {

	if (messages.size() > 0) {
		messages.pop_front();
		if (messages.size() > 0) {
			[sender startSpeakingString:[NSString stringWithUTF8String:messages[0].utf8().get_data()]];
		}
	}
};

- (bool)isSpeaking {

	return !messages.empty();
};
@end

void TTSDriverNSSpeech::speak(const String &p_text, bool p_interrupt) {

	if (synth && delegate) {
		if (p_interrupt) {
			[delegate discardMessages];
			[synth stopSpeaking];

			[delegate appendMessage:p_text];
			[synth startSpeakingString:[NSString stringWithUTF8String:p_text.utf8().get_data()]];
		} else {
			[delegate appendMessage:p_text];
			if (![synth isSpeaking]) {
				[synth startSpeakingString:[NSString stringWithUTF8String:p_text.utf8().get_data()]];
			}
		}
	}
};

void TTSDriverNSSpeech::stop() {

	if (synth && delegate) {
		[delegate discardMessages];
		[synth stopSpeaking];
	}
};

Array TTSDriverNSSpeech::get_voices() {

	Array list;
	NSArray *voices = [NSSpeechSynthesizer availableVoices];

	for (NSString *voiceIdentifierString in [NSSpeechSynthesizer availableVoices]) {
		NSString *voiceLocaleIdentifier = [[NSSpeechSynthesizer attributesForVoice:voiceIdentifierString] objectForKey:NSVoiceLocaleIdentifier];

		Dictionary voice_d;
		voice_d["name"] = String::utf8([voiceIdentifierString UTF8String]);
		voice_d["language"] = String::utf8([voiceLocaleIdentifier UTF8String]);
		list.push_back(voice_d);
	}

	return list;
};

void TTSDriverNSSpeech::set_voice(const String &p_voice) {

	if (synth) {
		[synth setVoice:[NSString stringWithUTF8String:p_voice.utf8().get_data()]];
	}
};

bool TTSDriverNSSpeech::is_speaking() {

	if (synth && delegate) {
		return [synth isSpeaking] || [delegate isSpeaking];
	} else {
		return false;
	}
};

void TTSDriverNSSpeech::set_volume(int p_volume) {

	if (synth) {
		[synth setVolume:p_volume / 100.0];
	}
};

int TTSDriverNSSpeech::get_volume() {

	if (synth) {
		return [synth volume] * 100.0;
	} else {
		return 0;
	}
};

void TTSDriverNSSpeech::set_rate(int p_rate) {

	if (synth) {
		[synth setRate:200 + (p_rate / 2)];
	}
};

int TTSDriverNSSpeech::get_rate() {

	if (synth) {
		return ([synth rate] - 200) * 2;
	} else {
		return 0;
	}
};

TTSDriverNSSpeech::TTSDriverNSSpeech() {

	synth = [[NSSpeechSynthesizer alloc] init];
	delegate = NULL;
	ERR_FAIL_COND(!synth);
	if (synth) {
		delegate = [[GodotSpeechSynthesizerDelegate alloc] init];
		ERR_FAIL_COND(!delegate);
		[synth setDelegate:delegate];
	}
};

TTSDriverNSSpeech::~TTSDriverNSSpeech() {

	if (synth) {
		[synth release];
		synth = NULL;
	}
	if (delegate) {
		[delegate release];
		delegate = NULL;
	}
};

#endif
