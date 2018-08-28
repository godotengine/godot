/*************************************************************************/
/*  tts_sapi.cpp                                                         */
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

#include "tts_sapi.h"

/*
API documentation:
https://docs.microsoft.com/en-us/previous-versions/windows/desktop/ee413476(v%3dvs.85)
*/

#ifdef SAPITTS_ENABLED

#include <sphelper.h>
#include <winnls.h>

void TTSDriverSAPI::speak(const String &p_text, bool p_interrupt) {

	if (synth) {
		if (p_interrupt) {
			synth->Speak(p_text.c_str(), SPF_IS_NOT_XML | SPF_ASYNC | SPF_PURGEBEFORESPEAK, NULL);
		} else {
			synth->Speak(p_text.c_str(), SPF_IS_NOT_XML | SPF_ASYNC, NULL);
		}
	}
};

void TTSDriverSAPI::stop() {

	if (synth) {
		synth->Speak(NULL, SPF_PURGEBEFORESPEAK, NULL);
	}
};

Array TTSDriverSAPI::get_voices() {

	Array list;

	IEnumSpObjectTokens *cpEnum;
	ISpObjectToken *cpVoiceToken;
	ISpDataKey *cpDataKeyAttribs;
	ULONG ulCount = 0;
	HRESULT hr = SpEnumTokens(SPCAT_VOICES, NULL, NULL, &cpEnum);
	if (SUCCEEDED(hr)) {
		hr = cpEnum->GetCount(&ulCount);

		while (SUCCEEDED(hr) && ulCount--) {
			hr = cpEnum->Next(1, &cpVoiceToken, NULL);

			HRESULT hr_attr = cpVoiceToken->OpenKey(SPTOKENKEY_ATTRIBUTES, &cpDataKeyAttribs);
			if (SUCCEEDED(hr_attr)) {
				wchar_t *w_id = 0L;
				wchar_t *w_lang = 0L;

				cpVoiceToken->GetId(&w_id);
				cpDataKeyAttribs->GetStringValue(L"Language", &w_lang);

				LCID locale = String(w_lang).split(";")[0].hex_to_int64(false);

				int locale_chars = GetLocaleInfoW(locale, LOCALE_SISO639LANGNAME, NULL, 0);
				int region_chars = GetLocaleInfoW(locale, LOCALE_SISO3166CTRYNAME, NULL, 0);
				wchar_t *w_lang_code = new wchar_t[locale_chars];
				wchar_t *w_reg_code = new wchar_t[region_chars];
				GetLocaleInfoW(locale, LOCALE_SISO639LANGNAME, w_lang_code, locale_chars);
				GetLocaleInfoW(locale, LOCALE_SISO3166CTRYNAME, w_reg_code, region_chars);

				Dictionary voice_d;
				voice_d["name"] = String(w_id);
				voice_d["language"] = String(w_lang_code) + "_" + String(w_reg_code);
				list.push_back(voice_d);

				delete[] w_lang_code;
				delete[] w_reg_code;

				cpDataKeyAttribs->Release();
			}
			cpVoiceToken->Release();
		}
		cpEnum->Release();
	}

	return list;
};

void TTSDriverSAPI::set_voice(const String &p_voice) {

	if (synth) {
		IEnumSpObjectTokens *cpEnum;
		ISpObjectToken *cpVoiceToken;
		ULONG ulCount = 0;
		HRESULT hr = SpEnumTokens(SPCAT_VOICES, NULL, NULL, &cpEnum);
		if (SUCCEEDED(hr)) {
			hr = cpEnum->GetCount(&ulCount);

			while (SUCCEEDED(hr) && ulCount--) {
				wchar_t *w_id = 0L;
				hr = cpEnum->Next(1, &cpVoiceToken, NULL);
				cpVoiceToken->GetId(&w_id);
				if (String(w_id) == p_voice) {
					synth->SetVoice(cpVoiceToken);

					cpVoiceToken->Release();
					cpEnum->Release();

					return;
				}
				cpVoiceToken->Release();
			}
			cpEnum->Release();
		}
	}
};

bool TTSDriverSAPI::is_speaking() {

	if (synth) {
		SPVOICESTATUS pStatus;
		synth->GetStatus(&pStatus, NULL);
		return (pStatus.dwRunningState == SPRS_IS_SPEAKING);
	} else {
		return false;
	}
};

void TTSDriverSAPI::set_volume(int p_volume) {

	if (synth) {
		synth->SetVolume(p_volume);
	}
};

int TTSDriverSAPI::get_volume() {

	USHORT volume = 0;
	if (synth) {
		synth->GetVolume(&volume);
	}
	return volume;
};

void TTSDriverSAPI::set_rate(int p_rate) {

	if (synth) {
		synth->SetRate(p_rate / 10);
	}
};

int TTSDriverSAPI::get_rate() {

	long rate = 0;
	if (synth) {
		synth->GetRate(&rate);
	}
	return rate * 10;
};

TTSDriverSAPI::TTSDriverSAPI() {

	HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
	ERR_FAIL_COND(hr != S_OK);

	hr = CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_ALL, IID_ISpVoice, (void **)&synth);
	ERR_FAIL_COND(hr != S_OK);
};

TTSDriverSAPI::~TTSDriverSAPI() {

	if (synth) {
		synth->Release();
		synth = NULL;
	}
	CoUninitialize();
};

#endif
