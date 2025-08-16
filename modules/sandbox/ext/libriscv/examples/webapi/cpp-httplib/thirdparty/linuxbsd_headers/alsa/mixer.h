/**
 * \file include/mixer.h
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

#ifndef __ALSA_MIXER_H
#define __ALSA_MIXER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  \defgroup Mixer Mixer Interface
 *  The mixer interface.
 *  \{
 */

/** Mixer handle */
typedef struct _snd_mixer snd_mixer_t;
/** Mixer elements class handle */
typedef struct _snd_mixer_class snd_mixer_class_t;
/** Mixer element handle */
typedef struct _snd_mixer_elem snd_mixer_elem_t;

/** 
 * \brief Mixer callback function
 * \param mixer Mixer handle
 * \param mask event mask
 * \param elem related mixer element (if any)
 * \return 0 on success otherwise a negative error code
 */
typedef int (*snd_mixer_callback_t)(snd_mixer_t *ctl,
				    unsigned int mask,
				    snd_mixer_elem_t *elem);

/** 
 * \brief Mixer element callback function
 * \param elem Mixer element
 * \param mask event mask
 * \return 0 on success otherwise a negative error code
 */
typedef int (*snd_mixer_elem_callback_t)(snd_mixer_elem_t *elem,
					 unsigned int mask);

/**
 * \brief Compare function for sorting mixer elements
 * \param e1 First element
 * \param e2 Second element
 * \return -1 if e1 < e2, 0 if e1 == e2, 1 if e1 > e2
 */
typedef int (*snd_mixer_compare_t)(const snd_mixer_elem_t *e1,
				   const snd_mixer_elem_t *e2);

/**
 * \brief Event callback for the mixer class
 * \param class_ Mixer class
 * \param mask Event mask (SND_CTL_EVENT_*)
 * \param helem HCTL element which invoked the event
 * \param melem Mixer element associated to HCTL element
 * \return zero if success, otherwise a negative error value
 */
typedef int (*snd_mixer_event_t)(snd_mixer_class_t *class_, unsigned int mask,
				 snd_hctl_elem_t *helem, snd_mixer_elem_t *melem);


/** Mixer element type */
typedef enum _snd_mixer_elem_type {
	/* Simple mixer elements */
	SND_MIXER_ELEM_SIMPLE,
	SND_MIXER_ELEM_LAST = SND_MIXER_ELEM_SIMPLE
} snd_mixer_elem_type_t;

int snd_mixer_open(snd_mixer_t **mixer, int mode);
int snd_mixer_close(snd_mixer_t *mixer);
snd_mixer_elem_t *snd_mixer_first_elem(snd_mixer_t *mixer);
snd_mixer_elem_t *snd_mixer_last_elem(snd_mixer_t *mixer);
int snd_mixer_handle_events(snd_mixer_t *mixer);
int snd_mixer_attach(snd_mixer_t *mixer, const char *name);
int snd_mixer_attach_hctl(snd_mixer_t *mixer, snd_hctl_t *hctl);
int snd_mixer_detach(snd_mixer_t *mixer, const char *name);
int snd_mixer_detach_hctl(snd_mixer_t *mixer, snd_hctl_t *hctl);
int snd_mixer_get_hctl(snd_mixer_t *mixer, const char *name, snd_hctl_t **hctl);
int snd_mixer_poll_descriptors_count(snd_mixer_t *mixer);
int snd_mixer_poll_descriptors(snd_mixer_t *mixer, struct pollfd *pfds, unsigned int space);
int snd_mixer_poll_descriptors_revents(snd_mixer_t *mixer, struct pollfd *pfds, unsigned int nfds, unsigned short *revents);
int snd_mixer_load(snd_mixer_t *mixer);
void snd_mixer_free(snd_mixer_t *mixer);
int snd_mixer_wait(snd_mixer_t *mixer, int timeout);
int snd_mixer_set_compare(snd_mixer_t *mixer, snd_mixer_compare_t msort);
void snd_mixer_set_callback(snd_mixer_t *obj, snd_mixer_callback_t val);
void * snd_mixer_get_callback_private(const snd_mixer_t *obj);
void snd_mixer_set_callback_private(snd_mixer_t *obj, void * val);
unsigned int snd_mixer_get_count(const snd_mixer_t *obj);
int snd_mixer_class_unregister(snd_mixer_class_t *clss);

snd_mixer_elem_t *snd_mixer_elem_next(snd_mixer_elem_t *elem);
snd_mixer_elem_t *snd_mixer_elem_prev(snd_mixer_elem_t *elem);
void snd_mixer_elem_set_callback(snd_mixer_elem_t *obj, snd_mixer_elem_callback_t val);
void * snd_mixer_elem_get_callback_private(const snd_mixer_elem_t *obj);
void snd_mixer_elem_set_callback_private(snd_mixer_elem_t *obj, void * val);
snd_mixer_elem_type_t snd_mixer_elem_get_type(const snd_mixer_elem_t *obj);

int snd_mixer_class_register(snd_mixer_class_t *class_, snd_mixer_t *mixer);
int snd_mixer_elem_new(snd_mixer_elem_t **elem,
		       snd_mixer_elem_type_t type,
		       int compare_weight,
		       void *private_data,
		       void (*private_free)(snd_mixer_elem_t *elem));
int snd_mixer_elem_add(snd_mixer_elem_t *elem, snd_mixer_class_t *class_);
int snd_mixer_elem_remove(snd_mixer_elem_t *elem);
void snd_mixer_elem_free(snd_mixer_elem_t *elem);
int snd_mixer_elem_info(snd_mixer_elem_t *elem);
int snd_mixer_elem_value(snd_mixer_elem_t *elem);
int snd_mixer_elem_attach(snd_mixer_elem_t *melem, snd_hctl_elem_t *helem);
int snd_mixer_elem_detach(snd_mixer_elem_t *melem, snd_hctl_elem_t *helem);
int snd_mixer_elem_empty(snd_mixer_elem_t *melem);
void *snd_mixer_elem_get_private(const snd_mixer_elem_t *melem);

size_t snd_mixer_class_sizeof(void);
/** \hideinitializer
 * \brief allocate an invalid #snd_mixer_class_t using standard alloca
 * \param ptr returned pointer
 */
#define snd_mixer_class_alloca(ptr) __snd_alloca(ptr, snd_mixer_class)
int snd_mixer_class_malloc(snd_mixer_class_t **ptr);
void snd_mixer_class_free(snd_mixer_class_t *obj);
void snd_mixer_class_copy(snd_mixer_class_t *dst, const snd_mixer_class_t *src);
snd_mixer_t *snd_mixer_class_get_mixer(const snd_mixer_class_t *class_);
snd_mixer_event_t snd_mixer_class_get_event(const snd_mixer_class_t *class_);
void *snd_mixer_class_get_private(const snd_mixer_class_t *class_);
snd_mixer_compare_t snd_mixer_class_get_compare(const snd_mixer_class_t *class_);
int snd_mixer_class_set_event(snd_mixer_class_t *class_, snd_mixer_event_t event);
int snd_mixer_class_set_private(snd_mixer_class_t *class_, void *private_data);
int snd_mixer_class_set_private_free(snd_mixer_class_t *class_, void (*private_free)(snd_mixer_class_t *));
int snd_mixer_class_set_compare(snd_mixer_class_t *class_, snd_mixer_compare_t compare);

/**
 *  \defgroup SimpleMixer Simple Mixer Interface
 *  \ingroup Mixer
 *  The simple mixer interface.
 *  \{
 */

/* Simple mixer elements API */

/** Mixer simple element channel identifier */
typedef enum _snd_mixer_selem_channel_id {
	/** Unknown */
	SND_MIXER_SCHN_UNKNOWN = -1,
	/** Front left */
	SND_MIXER_SCHN_FRONT_LEFT = 0,
	/** Front right */
	SND_MIXER_SCHN_FRONT_RIGHT,
	/** Rear left */
	SND_MIXER_SCHN_REAR_LEFT,
	/** Rear right */
	SND_MIXER_SCHN_REAR_RIGHT,
	/** Front center */
	SND_MIXER_SCHN_FRONT_CENTER,
	/** Woofer */
	SND_MIXER_SCHN_WOOFER,
	/** Side Left */
	SND_MIXER_SCHN_SIDE_LEFT,
	/** Side Right */
	SND_MIXER_SCHN_SIDE_RIGHT,
	/** Rear Center */
	SND_MIXER_SCHN_REAR_CENTER,
	SND_MIXER_SCHN_LAST = 31,
	/** Mono (Front left alias) */
	SND_MIXER_SCHN_MONO = SND_MIXER_SCHN_FRONT_LEFT
} snd_mixer_selem_channel_id_t;

/** Mixer simple element - register options - abstraction level */
enum snd_mixer_selem_regopt_abstract {
	/** no abstraction - try use all universal controls from driver */
	SND_MIXER_SABSTRACT_NONE = 0,
	/** basic abstraction - Master,PCM,CD,Aux,Record-Gain etc. */
	SND_MIXER_SABSTRACT_BASIC,
};

/** Mixer simple element - register options */
struct snd_mixer_selem_regopt {
	/** structure version */
	int ver;
	/** v1: abstract layer selection */
	enum snd_mixer_selem_regopt_abstract abstract;
	/** v1: device name (must be NULL when playback_pcm or capture_pcm != NULL) */
	const char *device;
	/** v1: playback PCM connected to mixer device (NULL == none) */
	snd_pcm_t *playback_pcm;
	/** v1: capture PCM connected to mixer device (NULL == none) */
	snd_pcm_t *capture_pcm;
};

/** Mixer simple element identifier */
typedef struct _snd_mixer_selem_id snd_mixer_selem_id_t;

const char *snd_mixer_selem_channel_name(snd_mixer_selem_channel_id_t channel);

int snd_mixer_selem_register(snd_mixer_t *mixer,
			     struct snd_mixer_selem_regopt *options,
			     snd_mixer_class_t **classp);
void snd_mixer_selem_get_id(snd_mixer_elem_t *element,
			    snd_mixer_selem_id_t *id);
const char *snd_mixer_selem_get_name(snd_mixer_elem_t *elem);
unsigned int snd_mixer_selem_get_index(snd_mixer_elem_t *elem);
snd_mixer_elem_t *snd_mixer_find_selem(snd_mixer_t *mixer,
				       const snd_mixer_selem_id_t *id);

int snd_mixer_selem_is_active(snd_mixer_elem_t *elem);
int snd_mixer_selem_is_playback_mono(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_playback_channel(snd_mixer_elem_t *obj, snd_mixer_selem_channel_id_t channel);
int snd_mixer_selem_is_capture_mono(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_capture_channel(snd_mixer_elem_t *obj, snd_mixer_selem_channel_id_t channel);
int snd_mixer_selem_get_capture_group(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_common_volume(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_playback_volume(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_playback_volume_joined(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_capture_volume(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_capture_volume_joined(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_common_switch(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_playback_switch(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_playback_switch_joined(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_capture_switch(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_capture_switch_joined(snd_mixer_elem_t *elem);
int snd_mixer_selem_has_capture_switch_exclusive(snd_mixer_elem_t *elem);

int snd_mixer_selem_ask_playback_vol_dB(snd_mixer_elem_t *elem, long value, long *dBvalue);
int snd_mixer_selem_ask_capture_vol_dB(snd_mixer_elem_t *elem, long value, long *dBvalue);
int snd_mixer_selem_ask_playback_dB_vol(snd_mixer_elem_t *elem, long dBvalue, int dir, long *value);
int snd_mixer_selem_ask_capture_dB_vol(snd_mixer_elem_t *elem, long dBvalue, int dir, long *value);
int snd_mixer_selem_get_playback_volume(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, long *value);
int snd_mixer_selem_get_capture_volume(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, long *value);
int snd_mixer_selem_get_playback_dB(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, long *value);
int snd_mixer_selem_get_capture_dB(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, long *value);
int snd_mixer_selem_get_playback_switch(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, int *value);
int snd_mixer_selem_get_capture_switch(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, int *value);
int snd_mixer_selem_set_playback_volume(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, long value);
int snd_mixer_selem_set_capture_volume(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, long value);
int snd_mixer_selem_set_playback_dB(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, long value, int dir);
int snd_mixer_selem_set_capture_dB(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, long value, int dir);
int snd_mixer_selem_set_playback_volume_all(snd_mixer_elem_t *elem, long value);
int snd_mixer_selem_set_capture_volume_all(snd_mixer_elem_t *elem, long value);
int snd_mixer_selem_set_playback_dB_all(snd_mixer_elem_t *elem, long value, int dir);
int snd_mixer_selem_set_capture_dB_all(snd_mixer_elem_t *elem, long value, int dir);
int snd_mixer_selem_set_playback_switch(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, int value);
int snd_mixer_selem_set_capture_switch(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, int value);
int snd_mixer_selem_set_playback_switch_all(snd_mixer_elem_t *elem, int value);
int snd_mixer_selem_set_capture_switch_all(snd_mixer_elem_t *elem, int value);
int snd_mixer_selem_get_playback_volume_range(snd_mixer_elem_t *elem, 
					      long *min, long *max);
int snd_mixer_selem_get_playback_dB_range(snd_mixer_elem_t *elem, 
					  long *min, long *max);
int snd_mixer_selem_set_playback_volume_range(snd_mixer_elem_t *elem, 
					      long min, long max);
int snd_mixer_selem_get_capture_volume_range(snd_mixer_elem_t *elem, 
					     long *min, long *max);
int snd_mixer_selem_get_capture_dB_range(snd_mixer_elem_t *elem, 
					 long *min, long *max);
int snd_mixer_selem_set_capture_volume_range(snd_mixer_elem_t *elem, 
					     long min, long max);

int snd_mixer_selem_is_enumerated(snd_mixer_elem_t *elem);
int snd_mixer_selem_is_enum_playback(snd_mixer_elem_t *elem);
int snd_mixer_selem_is_enum_capture(snd_mixer_elem_t *elem);
int snd_mixer_selem_get_enum_items(snd_mixer_elem_t *elem);
int snd_mixer_selem_get_enum_item_name(snd_mixer_elem_t *elem, unsigned int idx, size_t maxlen, char *str);
int snd_mixer_selem_get_enum_item(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, unsigned int *idxp);
int snd_mixer_selem_set_enum_item(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, unsigned int idx);

size_t snd_mixer_selem_id_sizeof(void);
/** \hideinitializer
 * \brief allocate an invalid #snd_mixer_selem_id_t using standard alloca
 * \param ptr returned pointer
 */
#define snd_mixer_selem_id_alloca(ptr) __snd_alloca(ptr, snd_mixer_selem_id)
int snd_mixer_selem_id_malloc(snd_mixer_selem_id_t **ptr);
void snd_mixer_selem_id_free(snd_mixer_selem_id_t *obj);
void snd_mixer_selem_id_copy(snd_mixer_selem_id_t *dst, const snd_mixer_selem_id_t *src);
const char *snd_mixer_selem_id_get_name(const snd_mixer_selem_id_t *obj);
unsigned int snd_mixer_selem_id_get_index(const snd_mixer_selem_id_t *obj);
void snd_mixer_selem_id_set_name(snd_mixer_selem_id_t *obj, const char *val);
void snd_mixer_selem_id_set_index(snd_mixer_selem_id_t *obj, unsigned int val);

/** \} */

/** \} */

#ifdef __cplusplus
}
#endif

#endif /* __ALSA_MIXER_H */

