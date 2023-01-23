/**
 * \file include/rawmidi.h
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

#ifndef __ALSA_RAWMIDI_H
#define __ALSA_RAWMIDI_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  \defgroup RawMidi RawMidi Interface
 *  The RawMidi Interface. See \ref rawmidi page for more details.
 *  \{
 */

/** dlsym version for interface entry callback */
#define SND_RAWMIDI_DLSYM_VERSION	_dlsym_rawmidi_001

/** RawMidi information container */
typedef struct _snd_rawmidi_info snd_rawmidi_info_t;
/** RawMidi settings container */
typedef struct _snd_rawmidi_params snd_rawmidi_params_t;
/** RawMidi status container */
typedef struct _snd_rawmidi_status snd_rawmidi_status_t;

/** RawMidi stream (direction) */
typedef enum _snd_rawmidi_stream {
	/** Output stream */
	SND_RAWMIDI_STREAM_OUTPUT = 0,
	/** Input stream */
	SND_RAWMIDI_STREAM_INPUT,
	SND_RAWMIDI_STREAM_LAST = SND_RAWMIDI_STREAM_INPUT
} snd_rawmidi_stream_t;

/** Append (flag to open mode) \hideinitializer */
#define SND_RAWMIDI_APPEND	0x0001
/** Non blocking mode (flag to open mode) \hideinitializer */
#define SND_RAWMIDI_NONBLOCK	0x0002
/** Write sync mode (Flag to open mode) \hideinitializer */
#define SND_RAWMIDI_SYNC	0x0004

/** RawMidi handle */
typedef struct _snd_rawmidi snd_rawmidi_t;

/** RawMidi type */
typedef enum _snd_rawmidi_type {
	/** Kernel level RawMidi */
	SND_RAWMIDI_TYPE_HW,
	/** Shared memory client RawMidi (not yet implemented) */
	SND_RAWMIDI_TYPE_SHM,
	/** INET client RawMidi (not yet implemented) */
	SND_RAWMIDI_TYPE_INET,
	/** Virtual (sequencer) RawMidi */
	SND_RAWMIDI_TYPE_VIRTUAL
} snd_rawmidi_type_t;

int snd_rawmidi_open(snd_rawmidi_t **in_rmidi, snd_rawmidi_t **out_rmidi,
		     const char *name, int mode);
int snd_rawmidi_open_lconf(snd_rawmidi_t **in_rmidi, snd_rawmidi_t **out_rmidi,
			   const char *name, int mode, snd_config_t *lconf);
int snd_rawmidi_close(snd_rawmidi_t *rmidi);
int snd_rawmidi_poll_descriptors_count(snd_rawmidi_t *rmidi);
int snd_rawmidi_poll_descriptors(snd_rawmidi_t *rmidi, struct pollfd *pfds, unsigned int space);
int snd_rawmidi_poll_descriptors_revents(snd_rawmidi_t *rawmidi, struct pollfd *pfds, unsigned int nfds, unsigned short *revent);
int snd_rawmidi_nonblock(snd_rawmidi_t *rmidi, int nonblock);
size_t snd_rawmidi_info_sizeof(void);
/** \hideinitializer
 * \brief allocate an invalid #snd_rawmidi_info_t using standard alloca
 * \param ptr returned pointer
 */
#define snd_rawmidi_info_alloca(ptr) __snd_alloca(ptr, snd_rawmidi_info)
int snd_rawmidi_info_malloc(snd_rawmidi_info_t **ptr);
void snd_rawmidi_info_free(snd_rawmidi_info_t *obj);
void snd_rawmidi_info_copy(snd_rawmidi_info_t *dst, const snd_rawmidi_info_t *src);
unsigned int snd_rawmidi_info_get_device(const snd_rawmidi_info_t *obj);
unsigned int snd_rawmidi_info_get_subdevice(const snd_rawmidi_info_t *obj);
snd_rawmidi_stream_t snd_rawmidi_info_get_stream(const snd_rawmidi_info_t *obj);
int snd_rawmidi_info_get_card(const snd_rawmidi_info_t *obj);
unsigned int snd_rawmidi_info_get_flags(const snd_rawmidi_info_t *obj);
const char *snd_rawmidi_info_get_id(const snd_rawmidi_info_t *obj);
const char *snd_rawmidi_info_get_name(const snd_rawmidi_info_t *obj);
const char *snd_rawmidi_info_get_subdevice_name(const snd_rawmidi_info_t *obj);
unsigned int snd_rawmidi_info_get_subdevices_count(const snd_rawmidi_info_t *obj);
unsigned int snd_rawmidi_info_get_subdevices_avail(const snd_rawmidi_info_t *obj);
void snd_rawmidi_info_set_device(snd_rawmidi_info_t *obj, unsigned int val);
void snd_rawmidi_info_set_subdevice(snd_rawmidi_info_t *obj, unsigned int val);
void snd_rawmidi_info_set_stream(snd_rawmidi_info_t *obj, snd_rawmidi_stream_t val);
int snd_rawmidi_info(snd_rawmidi_t *rmidi, snd_rawmidi_info_t * info);
size_t snd_rawmidi_params_sizeof(void);
/** \hideinitializer
 * \brief allocate an invalid #snd_rawmidi_params_t using standard alloca
 * \param ptr returned pointer
 */
#define snd_rawmidi_params_alloca(ptr) __snd_alloca(ptr, snd_rawmidi_params)
int snd_rawmidi_params_malloc(snd_rawmidi_params_t **ptr);
void snd_rawmidi_params_free(snd_rawmidi_params_t *obj);
void snd_rawmidi_params_copy(snd_rawmidi_params_t *dst, const snd_rawmidi_params_t *src);
int snd_rawmidi_params_set_buffer_size(snd_rawmidi_t *rmidi, snd_rawmidi_params_t *params, size_t val);
size_t snd_rawmidi_params_get_buffer_size(const snd_rawmidi_params_t *params);
int snd_rawmidi_params_set_avail_min(snd_rawmidi_t *rmidi, snd_rawmidi_params_t *params, size_t val);
size_t snd_rawmidi_params_get_avail_min(const snd_rawmidi_params_t *params);
int snd_rawmidi_params_set_no_active_sensing(snd_rawmidi_t *rmidi, snd_rawmidi_params_t *params, int val);
int snd_rawmidi_params_get_no_active_sensing(const snd_rawmidi_params_t *params);
int snd_rawmidi_params(snd_rawmidi_t *rmidi, snd_rawmidi_params_t * params);
int snd_rawmidi_params_current(snd_rawmidi_t *rmidi, snd_rawmidi_params_t *params);
size_t snd_rawmidi_status_sizeof(void);
/** \hideinitializer
 * \brief allocate an invalid #snd_rawmidi_status_t using standard alloca
 * \param ptr returned pointer
 */
#define snd_rawmidi_status_alloca(ptr) __snd_alloca(ptr, snd_rawmidi_status)
int snd_rawmidi_status_malloc(snd_rawmidi_status_t **ptr);
void snd_rawmidi_status_free(snd_rawmidi_status_t *obj);
void snd_rawmidi_status_copy(snd_rawmidi_status_t *dst, const snd_rawmidi_status_t *src);
void snd_rawmidi_status_get_tstamp(const snd_rawmidi_status_t *obj, snd_htimestamp_t *ptr);
size_t snd_rawmidi_status_get_avail(const snd_rawmidi_status_t *obj);
size_t snd_rawmidi_status_get_xruns(const snd_rawmidi_status_t *obj);
int snd_rawmidi_status(snd_rawmidi_t *rmidi, snd_rawmidi_status_t * status);
int snd_rawmidi_drain(snd_rawmidi_t *rmidi);
int snd_rawmidi_drop(snd_rawmidi_t *rmidi);
ssize_t snd_rawmidi_write(snd_rawmidi_t *rmidi, const void *buffer, size_t size);
ssize_t snd_rawmidi_read(snd_rawmidi_t *rmidi, void *buffer, size_t size);
const char *snd_rawmidi_name(snd_rawmidi_t *rmidi);
snd_rawmidi_type_t snd_rawmidi_type(snd_rawmidi_t *rmidi);
snd_rawmidi_stream_t snd_rawmidi_stream(snd_rawmidi_t *rawmidi);

/** \} */

#ifdef __cplusplus
}
#endif

#endif /* __RAWMIDI_H */

