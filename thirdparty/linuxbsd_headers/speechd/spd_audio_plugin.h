/*
 * spd_audio_plugin.h -- The SPD Audio Plugin Header
 *
 * Copyright (C) 2004 Brailcom, o.p.s.
 *
 * This is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation; either version 2.1, or (at your option) any later
 * version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this package; see the file COPYING.  If not, write to the Free
 * Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 *
 */

#ifndef __SPD_AUDIO_PLUGIN_H
#define __SPD_AUDIO_PLUGIN_H

#define SPD_AUDIO_PLUGIN_ENTRY_STR "spd_audio_plugin_get"

/* *INDENT-OFF* */
#ifdef __cplusplus
extern "C" {
#endif
/* *INDENT-ON* */

typedef enum { SPD_AUDIO_LE, SPD_AUDIO_BE } AudioFormat;

typedef struct {
	int bits;
	int num_channels;
	int sample_rate;

	int num_samples;
	signed short *samples;
} AudioTrack;

struct spd_audio_plugin;

typedef struct {

	int volume;
	AudioFormat format;

	struct spd_audio_plugin const *function;
	void *private_data;

	int working;
} AudioID;

typedef struct spd_audio_plugin {
	const char *name;
	AudioID *(*open) (void **pars);
	int (*play) (AudioID * id, AudioTrack track);
	int (*stop) (AudioID * id);
	int (*close) (AudioID * id);
	int (*set_volume) (AudioID * id, int);
	void (*set_loglevel) (int level);
	char const *(*get_playcmd) (void);
} spd_audio_plugin_t;

/* *INDENT-OFF* */
#ifdef __cplusplus
}
#endif /* __cplusplus */
/* *INDENT-ON* */

#endif /* ifndef #__SPD_AUDIO_PLUGIN_H */
