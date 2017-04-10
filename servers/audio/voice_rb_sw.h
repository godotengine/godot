/*************************************************************************/
/*  voice_rb_sw.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef VOICE_RB_SW_H
#define VOICE_RB_SW_H

#include "os/os.h"
#include "servers/audio_server.h"
class VoiceRBSW {
public:
	enum {
		VOICE_RB_SIZE = 1024
	};

	struct Command {

		enum Type {
			CMD_NONE,
			CMD_PLAY,
			CMD_STOP,
			CMD_SET_VOLUME,
			CMD_SET_PAN,
			CMD_SET_FILTER,
			CMD_SET_CHORUS,
			CMD_SET_REVERB,
			CMD_SET_MIX_RATE,
			CMD_SET_POSITIONAL,
			CMD_CHANGE_ALL_FX_VOLUMES
		};

		Type type;
		RID voice;

		struct {

			RID sample;

		} play;

		union {

			struct {

				float volume;
			} volume;

			struct {

				float pan, depth, height;
			} pan;

			struct {

				AS::FilterType type;
				float cutoff;
				float resonance;
				float gain;
			} filter;

			struct {
				float send;
			} chorus;
			struct {
				float send;
				AS::ReverbRoomType room;
			} reverb;

			struct {

				int mix_rate;
			} mix_rate;

			struct {

				bool positional;
			} positional;
		};

		Command() { type = CMD_NONE; }
	};

private:
	Command voice_cmd_rb[VOICE_RB_SIZE];
	volatile int read_pos;
	volatile int write_pos;

public:
	_FORCE_INLINE_ bool commands_left() const { return read_pos != write_pos; }
	_FORCE_INLINE_ Command pop_command() {
		ERR_FAIL_COND_V(read_pos == write_pos, Command());
		Command cmd = voice_cmd_rb[read_pos];
		read_pos = (read_pos + 1) % VOICE_RB_SIZE;
		return cmd;
	}
	_FORCE_INLINE_ void push_command(const Command &p_command) {

		bool full = ((write_pos + 1) % VOICE_RB_SIZE) == read_pos;
		if (full) {
#ifdef DEBUG_ENABLED
			if (OS::get_singleton()->is_stdout_verbose()) {
				ERR_EXPLAIN("Audio Ring Buffer Full (too many commands");
				ERR_FAIL_COND(((write_pos + 1) % VOICE_RB_SIZE) == read_pos);
			}
#endif
			return;
		}

		voice_cmd_rb[write_pos] = p_command;
		write_pos = (write_pos + 1) % VOICE_RB_SIZE;
	}

	VoiceRBSW() { read_pos = write_pos = 0; }
};

#endif // VOICE_RB_SW_H
