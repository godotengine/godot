/*************************************************************************/
/*  cp_song.h                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef CPSONG_H
#define CPSONG_H

/**CPSong Class
  *@author Juan Linietsky
  */

#include "cp_instrument.h"
#include "cp_order.h"
#include "cp_pattern.h"
#include "cp_sample.h"

class CPSong {
public:
	enum {
		MAX_SONG_NAME = 26,
		MAX_ORDERS = 200,
		MAX_PATTERNS = 200,
		MAX_SAMPLES = 99,
		MAX_INSTRUMENTS = 99,

		CHANNEL_MAX_PAN = 64,
		CHANNEL_MAX_VOLUME = 64,
		CHANNEL_MAX_CHORUS = 64,
		CHANNEL_MAX_REVERB = 64,

		MIN_TEMPO = 31,
		MAX_TEMPO = 255,
		MIN_SPEED = 1,
		MAX_SPEED = 255,
		MAX_MESSAGE_LEN = 8000,

	};

	enum ReverbMode {
		REVERB_MODE_ROOM,
		REVERB_MODE_STUDIO_SMALL,
		REVERB_MODE_STUDIO_MEDIUM,
		REVERB_MODE_STUDIO_LARGE,
		REVERB_MODE_HALL,
		REVERB_MODE_SPACE_ECHO,
		REVERB_MODE_ECHO,
		REVERB_MODE_DELAY,
		REVERB_MODE_HALF_ECHO
	};

private:
	CPOrder order[MAX_ORDERS];
	CPPattern pattern[MAX_PATTERNS];
	CPSample sample[MAX_SAMPLES];
	CPInstrument instrument[MAX_INSTRUMENTS];

	struct Song_Variables { // variables that wont change in playback

		char name[MAX_SONG_NAME];
		char message[MAX_MESSAGE_LEN];
		/* string message; */

		int row_highlight_minor;
		int row_highlight_major;

		int mixing_volume;
		int stereo_separation;

		bool use_stereo;
		bool use_instruments;
		bool use_linear_slides;

		bool old_effects;
		bool compatible_gxx;

	} variables;

	struct Initial_Variables { // Initial values used for playback

		struct Channel_State {

			int pan, volume; // 0-- CHANNEL_MAX_PAN, CHANNEL_MAX_VOLUME
			bool surround;
			bool mute;
			int chorus; //0 - 64
			int reverb; //0 - 64
		};

		int global_volume;
		int speed;
		int tempo;

		Channel_State channel[CPPattern::WIDTH];
	} initial_variables;

	struct Effects {

		ReverbMode reverb_mode;

		struct Chorus {

			int delay_ms;
			int separation_ms;
			int depth_ms10;
			int speed_hz10;
		} chorus;

	} effects;

public:
	/* Properties */

	const char *get_name();
	void set_name(const char *p_name);

	const char *get_message();
	void set_message(const char *p_message);

	void set_row_highlight_minor(int p_hl_minor); /* 0 .. 256 */
	int get_row_highlight_minor(); /* 0 .. 256 */

	void set_row_highlight_major(int p_hl_major); /* 0 .. 256 */
	int get_row_highlight_major(); /* 0 .. 256 */

	void set_mixing_volume(int p_mix_volume); /* 0 .. 128 */
	int get_mixing_volume(); /* 0 .. 128 */

	void set_global_volume(int p_global_volume); /* 0 .. 128 */
	int get_global_volume(); /* 0 .. 128 */

	void set_stereo_separation(int p_separation); /* 0 .. 128 */
	int get_stereo_separation(); /* 0 .. 128 */

	void set_stereo(bool p_stereo);
	bool is_stereo();

	void set_instruments(bool p_instruments);
	bool has_instruments();

	void set_linear_slides(bool p_linear_slides);
	bool has_linear_slides();

	void set_old_effects(bool p_old_effects);
	bool has_old_effects();

	void set_compatible_gxx(bool p_compatible_gxx);
	bool has_compatible_gxx();

	void set_speed(int p_speed); /* 1 .. 255 */
	int get_speed(); /* 1 .. 255 */

	void set_tempo(int p_tempo); /* 31 .. 255 */
	int get_tempo(); /* 31 .. 255 */

	void set_channel_pan(int p_channel, int p_pan); /* 0 .. 64 */
	int get_channel_pan(int p_channel);

	void set_channel_volume(int p_channel, int p_volume); /* 0 .. 64 */
	int get_channel_volume(int p_channel);

	void set_channel_surround(int p_channel, bool p_surround);
	bool is_channel_surround(int p_channel);

	void set_channel_mute(int p_channel, bool p_mute);
	bool is_channel_mute(int p_channel);

	void set_channel_chorus(int p_channel, int p_chorus); /* 0 .. 64 */
	int get_channel_chorus(int p_channel);

	void set_channel_reverb(int p_channel, int p_reverb); /* 0 .. 64 */
	int get_channel_reverb(int p_channel);

	/* arrays of stuff */

	CPPattern *get_pattern(int p_pattern);
	CPSample *get_sample(int p_sample);
	CPInstrument *get_instrument(int p_instrument);

	int get_order(int p_position);
	void set_order(int p_position, int p_order);

	/* Effects */

	ReverbMode get_reverb_mode();
	void set_reverb_mode(ReverbMode p_mode);

	void set_chorus_delay_ms(int p_amount);
	void set_chorus_separation_ms(int p_amount);
	void set_chorus_depth_ms10(int p_amount);
	void set_chorus_speed_hz10(int p_amount);

	int get_chorus_delay_ms();
	int get_chorus_separation_ms();
	int get_chorus_depth_ms10();
	int get_chorus_speed_hz10();

	/* utils */

	void reset(bool p_clear_patterns = true, bool p_clear_samples = true, bool p_clear_instruments = true, bool p_clear_variables = true);

	void cleanup_unused_patterns();
	void cleanup_unused_instruments();
	void cleanup_unused_samples();
	void cleanup_unused_orders();
	void clear_all_default_pan();
	void clear_all_default_vol();

	void clear_instrument_with_samples(int p_instrument);

	void make_instruments_from_samples();
	void make_instrument_from_sample(int p_sample);

	void separate_in_one_sample_instruments(int p_instrument);

	int get_order_in_use_count();
	int get_pattern_in_use_count();
	int get_instrument_in_use_count();
	int get_channels_in_use();

	CPSong();
	~CPSong();
};

/* Some helper for something used a lot */

int get_song_next_order_idx(CPSong *p_song, int p_order_idx);

#endif
