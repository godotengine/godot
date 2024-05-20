/**
 * \file include/mixer_abst.h
 * \brief Mixer abstract implementation interface library for the ALSA library
 * \author Jaroslav Kysela <perex@perex.cz>
 * \date 2005
 *
 * Mixer abstact implementation interface library for the ALSA library
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

#ifndef __ALSA_MIXER_ABST_H
#define __ALSA_MIXER_ABST_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  \defgroup Mixer_Abstract Mixer Abstact Module Interface
 *  The mixer abstact module interface.
 *  \{
 */

#define	SM_PLAY			0
#define SM_CAPT			1

#define SM_CAP_GVOLUME		(1<<1)
#define SM_CAP_GSWITCH		(1<<2)
#define SM_CAP_PVOLUME		(1<<3)
#define SM_CAP_PVOLUME_JOIN	(1<<4)
#define SM_CAP_PSWITCH		(1<<5) 
#define SM_CAP_PSWITCH_JOIN	(1<<6) 
#define SM_CAP_CVOLUME		(1<<7) 
#define SM_CAP_CVOLUME_JOIN	(1<<8) 
#define SM_CAP_CSWITCH		(1<<9) 
#define SM_CAP_CSWITCH_JOIN	(1<<10)
#define SM_CAP_CSWITCH_EXCL	(1<<11)
#define SM_CAP_PENUM		(1<<12)
#define SM_CAP_CENUM		(1<<13)
/* SM_CAP_* 24-31 => private for module use */

#define SM_OPS_IS_ACTIVE	0
#define SM_OPS_IS_MONO		1
#define SM_OPS_IS_CHANNEL	2
#define SM_OPS_IS_ENUMERATED	3
#define SM_OPS_IS_ENUMCNT	4

#define sm_selem(x)		((sm_selem_t *)((x)->private_data))
#define sm_selem_ops(x)		((sm_selem_t *)((x)->private_data))->ops

typedef struct _sm_selem {
	snd_mixer_selem_id_t *id;
	struct sm_elem_ops *ops;
	unsigned int caps;
	unsigned int capture_group;
} sm_selem_t;

typedef struct _sm_class_basic {
	char *device;
	snd_ctl_t *ctl;
	snd_hctl_t *hctl;
	snd_ctl_card_info_t *info;
} sm_class_basic_t;

struct sm_elem_ops {	
	int (*is)(snd_mixer_elem_t *elem, int dir, int cmd, int val);
	int (*get_range)(snd_mixer_elem_t *elem, int dir, long *min, long *max);
	int (*set_range)(snd_mixer_elem_t *elem, int dir, long min, long max);
	int (*get_dB_range)(snd_mixer_elem_t *elem, int dir, long *min, long *max);
	int (*ask_vol_dB)(snd_mixer_elem_t *elem, int dir, long value, long *dbValue);
	int (*ask_dB_vol)(snd_mixer_elem_t *elem, int dir, long dbValue, long *value, int xdir);
	int (*get_volume)(snd_mixer_elem_t *elem, int dir, snd_mixer_selem_channel_id_t channel, long *value);
	int (*get_dB)(snd_mixer_elem_t *elem, int dir, snd_mixer_selem_channel_id_t channel, long *value);
	int (*set_volume)(snd_mixer_elem_t *elem, int dir, snd_mixer_selem_channel_id_t channel, long value);
	int (*set_dB)(snd_mixer_elem_t *elem, int dir, snd_mixer_selem_channel_id_t channel, long value, int xdir);
	int (*get_switch)(snd_mixer_elem_t *elem, int dir, snd_mixer_selem_channel_id_t channel, int *value);
	int (*set_switch)(snd_mixer_elem_t *elem, int dir, snd_mixer_selem_channel_id_t channel, int value);
	int (*enum_item_name)(snd_mixer_elem_t *elem, unsigned int item, size_t maxlen, char *buf);
	int (*get_enum_item)(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, unsigned int *itemp);
	int (*set_enum_item)(snd_mixer_elem_t *elem, snd_mixer_selem_channel_id_t channel, unsigned int item);
};

int snd_mixer_selem_compare(const snd_mixer_elem_t *c1, const snd_mixer_elem_t *c2);

int snd_mixer_sbasic_info(const snd_mixer_class_t *class, sm_class_basic_t *info);
void *snd_mixer_sbasic_get_private(const snd_mixer_class_t *class);
void snd_mixer_sbasic_set_private(const snd_mixer_class_t *class, void *private_data);
void snd_mixer_sbasic_set_private_free(const snd_mixer_class_t *class, void (*private_free)(snd_mixer_class_t *class));

/** \} */

#ifdef __cplusplus
}
#endif

#endif /* __ALSA_MIXER_ABST_H */

