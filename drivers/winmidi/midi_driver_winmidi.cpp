/*************************************************************************/
/*  midi_driver_winmidi.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifdef WINMIDI_ENABLED

#include "midi_driver_winmidi.h"

#include "core/string/print_string.h"

void MIDIDriverWinMidi::read(HMIDIIN hMidiIn, UINT wMsg, DWORD_PTR dwInstance, DWORD_PTR dwParam1, DWORD_PTR dwParam2) {
	if (wMsg == MIM_DATA) {
		receive_input_packet((uint64_t)dwParam2, (uint8_t *)&dwParam1, 3);
	}
}

Error MIDIDriverWinMidi::open() {
	for (UINT i = 0; i < midiInGetNumDevs(); i++) {
		HMIDIIN midi_in;

		MMRESULT res = midiInOpen(&midi_in, i, (DWORD_PTR)read, (DWORD_PTR)this, CALLBACK_FUNCTION);
		if (res == MMSYSERR_NOERROR) {
			midiInStart(midi_in);
			connected_sources.insert(i, midi_in);
		} else {
			char err[256];
			midiInGetErrorText(res, err, 256);
			ERR_PRINT("midiInOpen error: " + String(err));

			MIDIINCAPS caps;
			res = midiInGetDevCaps(i, &caps, sizeof(MIDIINCAPS));
			if (res == MMSYSERR_NOERROR) {
				ERR_PRINT("Can't open MIDI device \"" + String(caps.szPname) + "\", is it being used by another application?");
			}
		}
	}

	return OK;
}

PackedStringArray MIDIDriverWinMidi::get_connected_inputs() {
	PackedStringArray list;

	for (int i = 0; i < connected_sources.size(); i++) {
		HMIDIIN midi_in = connected_sources[i];
		UINT id = 0;
		MMRESULT res = midiInGetID(midi_in, &id);
		if (res == MMSYSERR_NOERROR) {
			MIDIINCAPS caps;
			res = midiInGetDevCaps(i, &caps, sizeof(MIDIINCAPS));
			if (res == MMSYSERR_NOERROR) {
				list.push_back(caps.szPname);
			}
		}
	}

	return list;
}

void MIDIDriverWinMidi::close() {
	for (int i = 0; i < connected_sources.size(); i++) {
		HMIDIIN midi_in = connected_sources[i];
		midiInStop(midi_in);
		midiInClose(midi_in);
	}
	connected_sources.clear();
}

MIDIDriverWinMidi::MIDIDriverWinMidi() {
}

MIDIDriverWinMidi::~MIDIDriverWinMidi() {
	close();
}

#endif // WINMIDI_ENABLED
