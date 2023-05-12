/**
 * \file include/seq_midi_event.h
 * \brief Application interface library for the ALSA driver
 * \author Jaroslav Kysela <perex@perex.cz>
 * \author Abramo Bagnara <abramo@alsa-project.org>
 * \author Takashi Iwai <tiwai@suse.de>
 * \date 1998-2001
 *
 * Application interface library for the ALSA driver
 */
/*
 *   This library is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as
 *   published by the Free Software Foundation; either version 2.1 of
 *   the License, or (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU Lesser General Public License for more details.
 *
 *   You should have received a copy of the GNU Lesser General Public
 *   License along with this library; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 *
 */

#ifndef __ALSA_SEQ_MIDI_EVENT_H
#define __ALSA_SEQ_MIDI_EVENT_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  \defgroup MIDI_Event Sequencer event <-> MIDI byte stream coder
 *  \ingroup Sequencer
 *  Sequencer event <-> MIDI byte stream coder
 *  \{
 */

/** container for sequencer midi event parsers */
typedef struct snd_midi_event snd_midi_event_t;

int snd_midi_event_new(size_t bufsize, snd_midi_event_t **rdev);
int snd_midi_event_resize_buffer(snd_midi_event_t *dev, size_t bufsize);
void snd_midi_event_free(snd_midi_event_t *dev);
void snd_midi_event_init(snd_midi_event_t *dev);
void snd_midi_event_reset_encode(snd_midi_event_t *dev);
void snd_midi_event_reset_decode(snd_midi_event_t *dev);
void snd_midi_event_no_status(snd_midi_event_t *dev, int on);
/* encode from byte stream - return number of written bytes if success */
long snd_midi_event_encode(snd_midi_event_t *dev, const unsigned char *buf, long count, snd_seq_event_t *ev);
int snd_midi_event_encode_byte(snd_midi_event_t *dev, int c, snd_seq_event_t *ev);
/* decode from event to bytes - return number of written bytes if success */
long snd_midi_event_decode(snd_midi_event_t *dev, unsigned char *buf, long count, const snd_seq_event_t *ev);

/** \} */

#ifdef __cplusplus
}
#endif

#endif /* __ALSA_SEQ_MIDI_EVENT_H */

