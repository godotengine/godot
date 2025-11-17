/**************************************************************************/
/*  GodotTTS.java                                                         */
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

package org.godotengine.godot.tts;

import org.godotengine.godot.GodotLib;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.speech.tts.Voice;

import androidx.annotation.Keep;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.Set;

/**
 * Wrapper for Android Text to Speech API and custom utterance query implementation.
 * <p>
 * A [GodotTTS] provides the following features:
 * <p>
 * <ul>
 * <li>Access to the Android Text to Speech API.
 * <li>Utterance pause / resume functions, unsupported by Android TTS API.
 * </ul>
 */
@Keep
public class GodotTTS extends UtteranceProgressListener implements TextToSpeech.OnInitListener {
	// Note: These constants must be in sync with DisplayServer::TTSUtteranceEvent enum from "servers/display/display_server.h".
	final private static int EVENT_START = 0;
	final private static int EVENT_END = 1;
	final private static int EVENT_CANCEL = 2;
	final private static int EVENT_BOUNDARY = 3;

	// Note: These constants must be in sync with TTS_Android constants from "platform/android/tts_android.h".
	final private static int INIT_STATE_UNKNOWN = 0;
	final private static int INIT_STATE_SUCCESS = 1;
	final private static int INIT_STATE_FAIL = -1;

	private final Context context;
	private TextToSpeech synth;
	private int state;
	private LinkedList<GodotUtterance> queue;
	final private Object lock = new Object();
	private GodotUtterance lastUtterance;

	private boolean speaking;
	private boolean paused;

	public GodotTTS(Context context) {
		this.context = context;
	}

	private void updateTTS() {
		if (!speaking && queue.size() > 0) {
			int mode = TextToSpeech.QUEUE_FLUSH;
			GodotUtterance message = queue.pollFirst();

			Set<Voice> voices = synth.getVoices();
			if (voices == null) {
				return;
			}
			for (Voice v : voices) {
				if (v.getName().equals(message.voice)) {
					synth.setVoice(v);
					break;
				}
			}
			synth.setPitch(message.pitch);
			synth.setSpeechRate(message.rate);

			Bundle params = new Bundle();
			params.putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, message.volume / 100.f);

			lastUtterance = message;
			lastUtterance.start = 0;
			lastUtterance.offset = 0;
			paused = false;

			synth.speak(message.text, mode, params, String.valueOf(message.id));
			speaking = true;
		}
	}

	/**
	 * Called by TTS engine when the TTS service is about to speak the specified range.
	 */
	@Override
	public void onRangeStart(String utteranceId, int start, int end, int frame) {
		synchronized (lock) {
			if (lastUtterance != null && Long.parseLong(utteranceId) == lastUtterance.id) {
				lastUtterance.offset = start;
				GodotLib.ttsCallback(EVENT_BOUNDARY, lastUtterance.id, start + lastUtterance.start);
			}
		}
	}

	/**
	 * Called by TTS engine when an utterance was canceled in progress.
	 */
	@Override
	public void onStop(String utteranceId, boolean interrupted) {
		synchronized (lock) {
			if (lastUtterance != null && !paused && Long.parseLong(utteranceId) == lastUtterance.id) {
				GodotLib.ttsCallback(EVENT_CANCEL, lastUtterance.id, 0);
				speaking = false;
				updateTTS();
			}
		}
	}

	/**
	 * Called by TTS engine when an utterance has begun to be spoken..
	 */
	@Override
	public void onStart(String utteranceId) {
		synchronized (lock) {
			if (lastUtterance != null && lastUtterance.start == 0 && Long.parseLong(utteranceId) == lastUtterance.id) {
				GodotLib.ttsCallback(EVENT_START, lastUtterance.id, 0);
			}
		}
	}

	/**
	 * Called by TTS engine when an utterance was successfully finished.
	 */
	@Override
	public void onDone(String utteranceId) {
		synchronized (lock) {
			if (lastUtterance != null && !paused && Long.parseLong(utteranceId) == lastUtterance.id) {
				GodotLib.ttsCallback(EVENT_END, lastUtterance.id, 0);
				speaking = false;
				updateTTS();
			}
		}
	}

	/**
	 * Called by TTS engine when an error has occurred during processing.
	 */
	@Override
	public void onError(String utteranceId, int errorCode) {
		synchronized (lock) {
			if (lastUtterance != null && !paused && Long.parseLong(utteranceId) == lastUtterance.id) {
				GodotLib.ttsCallback(EVENT_CANCEL, lastUtterance.id, 0);
				speaking = false;
				updateTTS();
			}
		}
	}

	/**
	 * Called by TTS engine when an error has occurred during processing (pre API level 21 version).
	 */
	@Override
	public void onError(String utteranceId) {
		synchronized (lock) {
			if (lastUtterance != null && !paused && Long.parseLong(utteranceId) == lastUtterance.id) {
				GodotLib.ttsCallback(EVENT_CANCEL, lastUtterance.id, 0);
				speaking = false;
				updateTTS();
			}
		}
	}

	/**
	 * Initialize synth and query.
	 */
	public void init() {
		state = INIT_STATE_UNKNOWN;
		synth = new TextToSpeech(context, this);
		queue = new LinkedList<GodotUtterance>();

		synth.setOnUtteranceProgressListener(this);
	}

	/**
	 * Called by TTS engine when initialization is finished.
	 */
	@Override
	public void onInit(int status) {
		synchronized (lock) {
			if (status == TextToSpeech.SUCCESS) {
				state = INIT_STATE_SUCCESS;
			} else {
				state = INIT_STATE_FAIL;
			}
		}
	}

	/**
	 * Adds an utterance to the queue.
	 */
	public void speak(String text, String voice, int volume, float pitch, float rate, long utterance_id, boolean interrupt) {
		synchronized (lock) {
			if (state != INIT_STATE_SUCCESS) {
				return;
			}
			GodotUtterance message = new GodotUtterance(text, voice, volume, pitch, rate, utterance_id);
			queue.addLast(message);

			if (isPaused()) {
				resumeSpeaking();
			} else {
				updateTTS();
			}
		}
	}

	/**
	 * Puts the synthesizer into a paused state.
	 */
	public void pauseSpeaking() {
		synchronized (lock) {
			if (state != INIT_STATE_SUCCESS) {
				return;
			}
			if (!paused) {
				paused = true;
				synth.stop();
			}
		}
	}

	/**
	 * Resumes the synthesizer if it was paused.
	 */
	public void resumeSpeaking() {
		synchronized (lock) {
			if (state != INIT_STATE_SUCCESS) {
				return;
			}
			if (lastUtterance != null && paused) {
				int mode = TextToSpeech.QUEUE_FLUSH;

				Set<Voice> voices = synth.getVoices();
				if (voices == null) {
					return;
				}
				for (Voice v : voices) {
					if (v.getName().equals(lastUtterance.voice)) {
						synth.setVoice(v);
						break;
					}
				}
				synth.setPitch(lastUtterance.pitch);
				synth.setSpeechRate(lastUtterance.rate);

				Bundle params = new Bundle();
				params.putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, lastUtterance.volume / 100.f);

				lastUtterance.start = lastUtterance.offset;
				lastUtterance.offset = 0;
				paused = false;

				synth.speak(lastUtterance.text.substring(lastUtterance.start), mode, params, String.valueOf(lastUtterance.id));
				speaking = true;
			} else {
				paused = false;
			}
		}
	}

	/**
	 * Stops synthesis in progress and removes all utterances from the queue.
	 */
	public void stopSpeaking() {
		synchronized (lock) {
			if (state != INIT_STATE_SUCCESS) {
				return;
			}
			for (GodotUtterance u : queue) {
				GodotLib.ttsCallback(EVENT_CANCEL, u.id, 0);
			}
			queue.clear();

			if (lastUtterance != null) {
				GodotLib.ttsCallback(EVENT_CANCEL, lastUtterance.id, 0);
			}
			lastUtterance = null;

			paused = false;
			speaking = false;

			synth.stop();
		}
	}

	/**
	 * Returns voice information.
	 */
	public String[] getVoices() {
		synchronized (lock) {
			if (state != INIT_STATE_SUCCESS) {
				return new String[0];
			}
			Set<Voice> voices = synth.getVoices();
			if (voices == null) {
				return new String[0];
			}
			String[] list = new String[voices.size()];
			int i = 0;
			for (Voice v : voices) {
				list[i++] = v.getLocale().toString() + ";" + v.getName();
			}
			return list;
		}
	}

	/**
	 * Returns true if the synthesizer is generating speech, or have utterance waiting in the queue.
	 */
	public boolean isSpeaking() {
		return speaking;
	}

	/**
	 * Returns true if the synthesizer is in a paused state.
	 */
	public boolean isPaused() {
		return paused;
	}

	/**
	 * Returns INIT_STATE_SUCCESS if the synthesizer initialization finished successfully, INIT_STATE_FAIL if initialization failed, and INIT_STATE_UNKNOWN otherwise.
	 */
	public int getState() {
		synchronized (lock) {
			return state;
		}
	}
}
