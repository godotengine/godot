/*************************************************************************/
/*  cp_player_data.h                                                     */
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

#ifndef CP_PLAYER_DATA_H
#define CP_PLAYER_DATA_H

#include "cp_config.h"
#include "cp_mixer.h"
#include "cp_song.h"
#include "cp_tables.h"

/**CPPlayer Data
  *@author Juan Linietsky
  */

/******************************
 player_data.h
 ------------------------

The player and its data.
I hope you dont get sick reading this
********************************/

//Default pan values

class CPPlayer {

	enum {
		PAN_SURROUND = 512,
		PAN_RIGHT = 255,
		PAN_LEFT = 0,
		PAN_CENTER = 128
	};

	CPSong *song;

	CPMixer *mixer;

	struct Filter_Control {

		int32_t it_reso;
		int32_t it_cutoff;
		int32_t envelope_cutoff;
		int32_t final_cutoff;

		void process();
		void set_filter_parameters(int *p_cutoff, uint8_t *p_reso);
	};

	//tells you if a channel is doing
	//noteoff/notekill/notefade/etc
	enum {

		END_NOTE_NOTHING = 0,
		END_NOTE_OFF = 1,
		END_NOTE_FADE = 2,
		END_NOTE_KILL = 4
	};

	//Tells you what should a channel restart
	enum {

		KICK_NOTHING,
		KICK_NOTE,
		KICK_NOTEOFF,
		KICK_ENVELOPE
	};

	enum {

		MAX_VOICES = 256
	};

	struct Channel_Control;

	struct Voice_Control {

		struct Envelope_Control {

			int pos_index;
			int status;
			int value;
			bool sustain_looping;
			bool looping;
			bool terminated;
			bool active;
			bool kill;
		};

		Filter_Control filter;
		uint16_t reverb_send;
		uint16_t chorus_send;

		CPInstrument *instrument_ptr;
		CPSample *sample_ptr;

		//		Sample_Data *sample_data;

		int32_t period;

		int32_t sample_start_index; /* The starting byte index in the sample */

		bool has_master_channel;
		int master_channel_index;
		int instruement_index;

		int instrument_index;
		int sample_index;
		int8_t NNA_type;

		int note_end_flags;

		uint8_t sample; /* which instrument number */

		int16_t output_volume; /* output volume (vol + sampcol + instvol) */
		int8_t channel_volume; /* channel's "global" volume */
		uint16_t fadeout_volume; /* fading volume rate */
		int32_t total_volume; /* total volume of channel (before global mixings) */
		uint8_t kick; /* if true = sample has to be restarted */

		uint8_t note; /* the audible note (as heard, direct rep of period) */

		int16_t panning; /* panning position */

		uint8_t nna; /* New note action type + master/slave flags */
		uint8_t volflg; /* volume envelope settings */
		uint8_t panflg; /* panning envelope settings */
		uint8_t pitflg; /* pitch envelope settings */
		uint8_t keyoff; /* if true = fade out and stuff */
		int16_t handle; /* which sample-handle */
		int32_t start; /* The start byte index in the sample */

		/* Below here is info NOT in MP_CONTROL!! */
		//ENVPR       venv;
		//ENVPR       penv;
		//ENVPR       cenv;

		Envelope_Control volume_envelope_ctrl;
		Envelope_Control panning_envelope_ctrl;
		Envelope_Control pitch_envelope_ctrl;

		uint16_t auto_vibrato_pos; /* autovibrato pos */
		uint16_t auto_vibrato_sweep_pos; /* autovibrato sweep pos */

		int16_t masterchn;
		uint16_t masterperiod;

		Channel_Control *master_channel; /* index of "master" effects channel */

		void start_envelope(CPEnvelope *p_envelope, Envelope_Control *p_envelope_ctrl, Envelope_Control *p_from_env);
		bool process_envelope(CPEnvelope *p_envelope, Envelope_Control *p_envelope_ctrl);

		uint16_t display_volume;

		Voice_Control() {

			reset();
		}

		void reset();
		void update_info_from_master_channel();
	};

	struct Channel_Control {

		/* NOTE info */
		uint8_t note; /* the audible note as heard, direct rep of period */
		uint8_t real_note; /* the note that indexes the audible */
		int32_t sample_start_index; /* The starting byte index in the sample */
		uint8_t old_note;

		uint8_t kick;

		Filter_Control filter;
		uint16_t reverb_send;
		uint16_t chorus_send;

		int note_end_flags;

		/* INSTRUMENT INFO */

		CPInstrument *instrument_ptr;
		CPSample *sample_ptr;

		uint8_t instrument_index;
		uint8_t sample_index;
		bool new_instrument;

		/* SAMPLE SPECIFIC INFO */
		int32_t base_speed; /* what finetune to use */

		/* INSTRUMENT SPECIFIC INFO */

		int8_t NNA_type;
		int8_t duplicate_check_type;
		int8_t duplicate_check_action;

		bool volume_envelope_on;
		bool panning_envelope_on;
		bool pitch_envelope_on;

		bool has_own_period;

		bool row_has_note;

		/* VOLUME COLUMN */

		int16_t volume; /* amiga volume (0 t/m 64) to play the sample at */
		int16_t aux_volume;
		bool has_own_volume;
		bool mute;
		int16_t random_volume_variation; /* 0-100 - 100 has no effect */

		/* VOLUME/PAN/PITCH MODIFIERS */

		int8_t default_volume; // CHANNEL default volume (0-64)
		int16_t channel_volume; // CHANNEL current volume //chanvol - current!
		int16_t output_volume; /* output volume (vol + sampcol + instvol) //volume */
		int16_t channel_global_volume;

		uint16_t fadeout_volume; /* fading volume rate */

		int32_t period; /* period to play the sample at */

		/* PAN */

		int16_t panning; /* panning position */
		int16_t channel_panning;
		int8_t sliding;

		uint16_t aux_period; /* temporary period */

		/* TIMING */
		uint8_t note_delay; /* (used for note delay) */

		/* Slave Voice Control */

		Voice_Control *slave_voice; /* Audio Slave of current effects control channel */

		struct Carry {

			Voice_Control::Envelope_Control vol;
			Voice_Control::Envelope_Control pan;
			Voice_Control::Envelope_Control pitch;
			bool maybe;

		} carry;

		uint8_t slave_voice_index; /* Audio Slave of current effects control channel */

		uint8_t *row; /* row currently playing on this channel */

		/* effect memory variables */

		uint8_t current_command;
		uint8_t current_parameter;
		uint8_t current_volume_command;
		uint8_t current_volume_parameter;
		uint8_t volcol_volume_slide;

		/* CPSample Offset */

		int32_t lo_offset;
		int32_t hi_offset;

		/* Panbrello waveform */
		uint8_t panbrello_type; /* current panbrello waveform */
		uint8_t panbrello_position; /* current panbrello position */
		int8_t panbrello_speed; /* "" speed */
		uint8_t panbrello_depth; /* "" depth */
		uint8_t panbrello_info;
		/* Arpegio */

		uint8_t arpegio_info;
		/* CPPattern Loop */

		int pattern_loop_position;
		int8_t pattern_loop_count;

		/* Vibrato */
		bool doing_vibrato;
		int8_t vibrato_position; /* current vibrato position */
		uint8_t vibrato_speed; /* "" speed */
		uint8_t vibrato_depth; /* "" depth */
		uint8_t vibrato_type;
		/* Tremor */
		int8_t tremor_position;
		uint8_t tremor_speed; /* s3m tremor ontime/offtime */
		uint8_t tremor_depth;
		uint8_t tremor_info;

		/* Tremolo */
		int8_t tremolo_position;
		uint8_t tremolo_speed; /* s3m tremor ontime/offtime */
		uint8_t tremolo_depth;
		uint8_t tremolo_info;
		uint8_t tremolo_type;

		/* Retrig */
		int8_t retrig_counter; /* retrig value (0 means don't retrig) */
		uint8_t retrig_speed; /* last used retrig speed */
		uint8_t retrig_volslide; /* last used retrig slide */

		/* CPSample Offset */
		int32_t sample_offset_hi; /* last used high order of sample offset */
		uint16_t sample_offset; /* last used low order of sample-offset (effect 9) */
		uint16_t sample_offset_fine; /* fine sample offset memory */

		/* Portamento */
		uint16_t slide_to_period; /* period to slide to (with effect 3 or 5) */
		uint8_t portamento_speed;

		/* Volume Slide */

		uint8_t volume_slide_info;

		/* Channel Volume Slide */

		uint8_t channel_volume_slide_info;

		/* Global Volume Slide */

		uint8_t global_volume_slide_info;

		/* Channel Pan Slide */

		uint8_t channel_pan_slide_info;

		/* Pitch Slide */

		uint8_t pitch_slide_info;
		/* Tempo Slide */

		uint8_t tempo_slide_info;

		/* S effects memory */

		uint8_t current_S_effect;
		uint8_t current_S_data;

		/* Volume column memory */

		uint8_t volume_column_effect_mem;
		uint8_t volume_column_data_mem;

		int64_t last_event_usecs;
		bool reserved;

		void reset();

		Channel_Control() {
			channel_global_volume = 255;
			last_event_usecs = -1;
		}
	};

	struct Control_Variables { // control variables (dynamic version) of initial variables

		bool reached_end;

		char play_mode;
		bool filters;
		int global_volume;
		int speed;
		int tempo;

		int ticks_counter;

		int pattern_delay_1;
		int pattern_delay_2;

		Channel_Control channel[CPPattern::WIDTH];

		int max_voices;

		int voices_used; /* reference value */

		bool force_no_nna;
		bool external_vibrato;

		struct Position {

			int current_order;
			int current_pattern;
			int current_row;
			int force_next_order;
			bool forbid_jump;
		};

		int32_t random_seed;

		Position position;
		Position previous_position;
	};

	Voice_Control voice[MAX_VOICES];

	Control_Variables control;

	/* VOICE SETUP */

	void setup_voices();

	/* MIXER SETUP */
	void handle_tick();
	void update_mixer();

	/* NOTE / INSTRUMENT PROCESSING */

	void process_new_note(int p_track, uint8_t p_note);
	bool process_new_instrument(int p_track, uint8_t p_instrument);
	bool process_note_and_instrument(int p_track, int p_note, int p_instrument);

	/* EFFECT PROCESSING */
	void do_effect_S(int p_track);
	void do_panbrello(int p_track);
	void do_global_volume_slide(int p_track);
	void do_tremolo(int p_track);
	void do_retrig(int p_track);
	void do_pan_slide(int p_track);
	void do_channel_volume_slide(int p_track);
	void do_volume_slide(int p_track, int inf);
	void do_pitch_slide_down(int p_track, uint8_t inf);
	void do_pitch_slide_up(int p_track, uint8_t inf);
	void do_tremor(int p_track);
	void do_vibrato(int p_track, bool fine);
	void do_pitch_slide_to_note(int p_track);
	void run_effects(int p_track);
	void run_volume_column_effects(int p_track);
	void pre_process_effects();
	void do_arpegio(int p_track);
	uint64_t song_usecs;
	/* NNA */

	void process_NNAs();

	/* MISC UTILS */

	int find_empty_voice();
	void process_volume_column(int p_track, uint8_t p_volume);
	void process_note(int p_track, CPNote p_note);

	/* CPTables */
	static uint8_t auto_vibrato_table[128];
	static uint8_t vibrato_table[32];
	static int8_t panbrello_table[256];

	static void callback_function(void *p_userdata);

public:
	//Play modes

	enum {

		PLAY_NOTHING = 0,
		PLAY_PATTERN = 1,
		PLAY_SONG = 2
	};

	int32_t get_frequency(int32_t period);
	int32_t get_period(uint16_t note, int32_t p_c5freq);

	int get_current_tempo() { return control.tempo; };
	int get_current_speed() { return control.speed; };

	int get_voices_used() { return control.voices_used; };
	int get_voice_envelope_pos(int p_voice, CPEnvelope *p_envelope);
	int get_voice_amount_limit() { return control.max_voices; };
	void set_voice_amount_limit(int p_limit);
	void set_reserved_voices(int p_amount);
	int get_reserved_voices_amount();

	bool is_voice_active(int p_voice);
	int get_channel_voice(int p_channel);
	const char *get_voice_sample_name(int p_voice);
	const char *get_voice_instrument_name(int p_voice);
	CPEnvelope *get_voice_envelope(int p_voice, CPInstrument::EnvelopeType p_env_type);
	int get_voice_envelope_pos(int p_voice, CPInstrument::EnvelopeType p_env_type);
	int get_voice_volume(int p_voice);

	int get_voice_sample_index(int p_voice);

	void set_virtual_channels(int p_amount);
	int get_virtual_channels() { return control.max_voices; };

	/* Play Info/Position */
	bool is_playing() { return (control.play_mode > 0); };
	int get_play_mode() { return (control.play_mode); };
	int get_current_order() { return control.position.current_order; };
	int get_current_row() { return control.position.current_row; };
	int get_current_pattern() { return control.position.current_pattern; };

	void goto_next_order();
	void goto_previous_order();

	void process_tick();

	CPMixer *get_mixer_ptr() {

		return mixer;
	}

	void reset();

	/* External player control - editor - */

	void play_start_pattern(int p_pattern);
	void play_start_song();
	void play_start_song_from_order(int p_order);
	void play_start_song_from_order_and_row(int p_order, int p_row);
	void play_start(int p_pattern, int p_order, int p_row, bool p_lock = true);

	void play_stop();
	void play_note(int p_channel, CPNote note, bool p_reserve = false);

	bool reached_end_of_song();

	void set_force_no_nna(bool p_force);
	void set_force_external_vibratos(bool p_force);

	void set_filters_enabled(bool p_enable);
	bool are_filters_enabled() { return control.filters; }

	void set_channel_global_volume(int p_channel, int p_volume); //0-255
	int get_channel_global_volume(int p_channel) const;

	int64_t get_channel_last_note_time_usec(int p_channel) const;

	CPSong *get_song() { return song; };

	CPPlayer(CPMixer *p_mixer, CPSong *p_song);
	~CPPlayer();
};

#endif
