/*************************************************************************/
/*  tts_ios.h                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TTS_IOS_H
#define TTS_IOS_H

#if __has_include(<AVFAudio/AVSpeechSynthesis.h>)
#import <AVFAudio/AVSpeechSynthesis.h>
#else
#import <AVFoundation/AVFoundation.h>
#endif

#include "core/array.h"
#include "core/list.h"
#include "core/map.h"
#include "core/os/os.h"
#include "core/ustring.h"

@interface TTS_IOS : NSObject <AVSpeechSynthesizerDelegate> {
	bool speaking;
	Map<id, int> ids;

	AVSpeechSynthesizer *av_synth;
	List<OS::TTSUtterance> queue;
}

- (void)pauseSpeaking;
- (void)resumeSpeaking;
- (void)stopSpeaking;
- (bool)isSpeaking;
- (bool)isPaused;
- (void)speak:(const String &)text voice:(const String &)voice volume:(int)volume pitch:(float)pitch rate:(float)rate utterance_id:(int)utterance_id interrupt:(bool)interrupt;
- (Array)getVoices;
@end

#endif // TTS_IOS_H
