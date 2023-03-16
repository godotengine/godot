/**
 * \file include/control.h
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

#ifndef __ALSA_CONTROL_H
#define __ALSA_CONTROL_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  \defgroup Control Control Interface
 *  The control interface.
 *  See \ref control page for more details.
 *  \{
 */

/** dlsym version for interface entry callback */
#define SND_CONTROL_DLSYM_VERSION	_dlsym_control_001

/** IEC958 structure */
typedef struct snd_aes_iec958 {
	unsigned char status[24];	/**< AES/IEC958 channel status bits */
	unsigned char subcode[147];	/**< AES/IEC958 subcode bits */
	unsigned char pad;		/**< nothing */
	unsigned char dig_subframe[4];	/**< AES/IEC958 subframe bits */
} snd_aes_iec958_t;

/** CTL card info container */
typedef struct _snd_ctl_card_info snd_ctl_card_info_t;

/** CTL element identifier container */
typedef struct _snd_ctl_elem_id snd_ctl_elem_id_t;

/** CTL element identifier list container */
typedef struct _snd_ctl_elem_list snd_ctl_elem_list_t;

/** CTL element info container */
typedef struct _snd_ctl_elem_info snd_ctl_elem_info_t;

/** CTL element value container */
typedef struct _snd_ctl_elem_value snd_ctl_elem_value_t;

/** CTL event container */
typedef struct _snd_ctl_event snd_ctl_event_t;

/** CTL element type */
typedef enum _snd_ctl_elem_type {
	/** Invalid type */
	SND_CTL_ELEM_TYPE_NONE = 0,
	/** Boolean contents */
	SND_CTL_ELEM_TYPE_BOOLEAN,
	/** Integer contents */
	SND_CTL_ELEM_TYPE_INTEGER,
	/** Enumerated contents */
	SND_CTL_ELEM_TYPE_ENUMERATED,
	/** Bytes contents */
	SND_CTL_ELEM_TYPE_BYTES,
	/** IEC958 (S/PDIF) setting content */
	SND_CTL_ELEM_TYPE_IEC958,
	/** 64-bit integer contents */
	SND_CTL_ELEM_TYPE_INTEGER64,
	SND_CTL_ELEM_TYPE_LAST = SND_CTL_ELEM_TYPE_INTEGER64
} snd_ctl_elem_type_t;

/** CTL related interface */
typedef enum _snd_ctl_elem_iface {
	/** Card level */
	SND_CTL_ELEM_IFACE_CARD = 0,
	/** Hardware dependent device */
	SND_CTL_ELEM_IFACE_HWDEP,
	/** Mixer */
	SND_CTL_ELEM_IFACE_MIXER,
	/** PCM */
	SND_CTL_ELEM_IFACE_PCM,
	/** RawMidi */
	SND_CTL_ELEM_IFACE_RAWMIDI,
	/** Timer */
	SND_CTL_ELEM_IFACE_TIMER,
	/** Sequencer */
	SND_CTL_ELEM_IFACE_SEQUENCER,
	SND_CTL_ELEM_IFACE_LAST = SND_CTL_ELEM_IFACE_SEQUENCER
} snd_ctl_elem_iface_t;

/** Event class */
typedef enum _snd_ctl_event_type {
	/** Elements related event */
	SND_CTL_EVENT_ELEM = 0,
	SND_CTL_EVENT_LAST = SND_CTL_EVENT_ELEM
}snd_ctl_event_type_t;

/** Element has been removed (Warning: test this first and if set don't
  * test the other masks) \hideinitializer */
#define SND_CTL_EVENT_MASK_REMOVE 	(~0U)
/** Element value has been changed \hideinitializer */
#define SND_CTL_EVENT_MASK_VALUE	(1<<0)
/** Element info has been changed \hideinitializer */
#define SND_CTL_EVENT_MASK_INFO		(1<<1)
/** Element has been added \hideinitializer */
#define SND_CTL_EVENT_MASK_ADD		(1<<2)
/** Element's TLV value has been changed \hideinitializer */
#define SND_CTL_EVENT_MASK_TLV		(1<<3)

/** CTL name helper */
#define SND_CTL_NAME_NONE				""
/** CTL name helper */
#define SND_CTL_NAME_PLAYBACK				"Playback "
/** CTL name helper */
#define SND_CTL_NAME_CAPTURE				"Capture "

/** CTL name helper */
#define SND_CTL_NAME_IEC958_NONE			""
/** CTL name helper */
#define SND_CTL_NAME_IEC958_SWITCH			"Switch"
/** CTL name helper */
#define SND_CTL_NAME_IEC958_VOLUME			"Volume"
/** CTL name helper */
#define SND_CTL_NAME_IEC958_DEFAULT			"Default"
/** CTL name helper */
#define SND_CTL_NAME_IEC958_MASK			"Mask"
/** CTL name helper */
#define SND_CTL_NAME_IEC958_CON_MASK			"Con Mask"
/** CTL name helper */
#define SND_CTL_NAME_IEC958_PRO_MASK			"Pro Mask"
/** CTL name helper */
#define SND_CTL_NAME_IEC958_PCM_STREAM			"PCM Stream"
/** Element name for IEC958 (S/PDIF) */
#define SND_CTL_NAME_IEC958(expl,direction,what)	"IEC958 " expl SND_CTL_NAME_##direction SND_CTL_NAME_IEC958_##what

/** Mask for the major Power State identifier */
#define SND_CTL_POWER_MASK		0xff00
/** ACPI/PCI Power State D0 */
#define SND_CTL_POWER_D0          	0x0000
/** ACPI/PCI Power State D1 */
#define SND_CTL_POWER_D1     	     	0x0100
/** ACPI/PCI Power State D2 */
#define SND_CTL_POWER_D2 	        0x0200
/** ACPI/PCI Power State D3 */
#define SND_CTL_POWER_D3         	0x0300
/** ACPI/PCI Power State D3hot */
#define SND_CTL_POWER_D3hot		(SND_CTL_POWER_D3|0x0000)
/** ACPI/PCI Power State D3cold */
#define SND_CTL_POWER_D3cold	      	(SND_CTL_POWER_D3|0x0001)

/** TLV type - Container */
#define SND_CTL_TLVT_CONTAINER		0x0000
/** TLV type - basic dB scale */
#define SND_CTL_TLVT_DB_SCALE		0x0001
/** TLV type - linear volume */
#define SND_CTL_TLVT_DB_LINEAR		0x0002
/** TLV type - dB range container */
#define SND_CTL_TLVT_DB_RANGE		0x0003
/** TLV type - dB scale specified by min/max values */
#define SND_CTL_TLVT_DB_MINMAX		0x0004
/** TLV type - dB scale specified by min/max values (with mute) */
#define SND_CTL_TLVT_DB_MINMAX_MUTE	0x0005

/** Mute state */
#define SND_CTL_TLV_DB_GAIN_MUTE	-9999999

/** TLV type - fixed channel map positions */
#define SND_CTL_TLVT_CHMAP_FIXED	0x00101
/** TLV type - freely swappable channel map positions */
#define SND_CTL_TLVT_CHMAP_VAR		0x00102
/** TLV type - pair-wise swappable channel map positions */
#define SND_CTL_TLVT_CHMAP_PAIRED	0x00103

/** CTL type */
typedef enum _snd_ctl_type {
	/** Kernel level CTL */
	SND_CTL_TYPE_HW,
	/** Shared memory client CTL */
	SND_CTL_TYPE_SHM,
	/** INET client CTL (not yet implemented) */
	SND_CTL_TYPE_INET,
	/** External control plugin */
	SND_CTL_TYPE_EXT
} snd_ctl_type_t;

/** Non blocking mode (flag for open mode) \hideinitializer */
#define SND_CTL_NONBLOCK		0x0001

/** Async notification (flag for open mode) \hideinitializer */
#define SND_CTL_ASYNC			0x0002

/** Read only (flag for open mode) \hideinitializer */
#define SND_CTL_READONLY		0x0004

/** CTL handle */
typedef struct _snd_ctl snd_ctl_t;

/** Don't destroy the ctl handle when close */
#define SND_SCTL_NOFREE			0x0001

/** SCTL type */
typedef struct _snd_sctl snd_sctl_t;

int snd_card_load(int card);
int snd_card_next(int *card);
int snd_card_get_index(const char *name);
int snd_card_get_name(int card, char **name);
int snd_card_get_longname(int card, char **name);

int snd_device_name_hint(int card, const char *iface, void ***hints);
int snd_device_name_free_hint(void **hints);
char *snd_device_name_get_hint(const void *hint, const char *id);

int snd_ctl_open(snd_ctl_t **ctl, const char *name, int mode);
int snd_ctl_open_lconf(snd_ctl_t **ctl, const char *name, int mode, snd_config_t *lconf);
int snd_ctl_open_fallback(snd_ctl_t **ctl, snd_config_t *root, const char *name, const char *orig_name, int mode);
int snd_ctl_close(snd_ctl_t *ctl);
int snd_ctl_nonblock(snd_ctl_t *ctl, int nonblock);
static __inline__ int snd_ctl_abort(snd_ctl_t *ctl) { return snd_ctl_nonblock(ctl, 2); }
int snd_async_add_ctl_handler(snd_async_handler_t **handler, snd_ctl_t *ctl, 
			      snd_async_callback_t callback, void *private_data);
snd_ctl_t *snd_async_handler_get_ctl(snd_async_handler_t *handler);
int snd_ctl_poll_descriptors_count(snd_ctl_t *ctl);
int snd_ctl_poll_descriptors(snd_ctl_t *ctl, struct pollfd *pfds, unsigned int space);
int snd_ctl_poll_descriptors_revents(snd_ctl_t *ctl, struct pollfd *pfds, unsigned int nfds, unsigned short *revents);
int snd_ctl_subscribe_events(snd_ctl_t *ctl, int subscribe);
int snd_ctl_card_info(snd_ctl_t *ctl, snd_ctl_card_info_t *info);
int snd_ctl_elem_list(snd_ctl_t *ctl, snd_ctl_elem_list_t *list);
int snd_ctl_elem_info(snd_ctl_t *ctl, snd_ctl_elem_info_t *info);
int snd_ctl_elem_read(snd_ctl_t *ctl, snd_ctl_elem_value_t *data);
int snd_ctl_elem_write(snd_ctl_t *ctl, snd_ctl_elem_value_t *data);
int snd_ctl_elem_lock(snd_ctl_t *ctl, snd_ctl_elem_id_t *id);
int snd_ctl_elem_unlock(snd_ctl_t *ctl, snd_ctl_elem_id_t *id);
int snd_ctl_elem_tlv_read(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id,
			  unsigned int *tlv, unsigned int tlv_size);
int snd_ctl_elem_tlv_write(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id,
			   const unsigned int *tlv);
int snd_ctl_elem_tlv_command(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id,
			     const unsigned int *tlv);
#ifdef __ALSA_HWDEP_H
int snd_ctl_hwdep_next_device(snd_ctl_t *ctl, int * device);
int snd_ctl_hwdep_info(snd_ctl_t *ctl, snd_hwdep_info_t * info);
#endif
#ifdef __ALSA_PCM_H
int snd_ctl_pcm_next_device(snd_ctl_t *ctl, int *device);
int snd_ctl_pcm_info(snd_ctl_t *ctl, snd_pcm_info_t * info);
int snd_ctl_pcm_prefer_subdevice(snd_ctl_t *ctl, int subdev);
#endif
#ifdef __ALSA_RAWMIDI_H
int snd_ctl_rawmidi_next_device(snd_ctl_t *ctl, int * device);
int snd_ctl_rawmidi_info(snd_ctl_t *ctl, snd_rawmidi_info_t * info);
int snd_ctl_rawmidi_prefer_subdevice(snd_ctl_t *ctl, int subdev);
#endif
int snd_ctl_set_power_state(snd_ctl_t *ctl, unsigned int state);
int snd_ctl_get_power_state(snd_ctl_t *ctl, unsigned int *state);

int snd_ctl_read(snd_ctl_t *ctl, snd_ctl_event_t *event);
int snd_ctl_wait(snd_ctl_t *ctl, int timeout);
const char *snd_ctl_name(snd_ctl_t *ctl);
snd_ctl_type_t snd_ctl_type(snd_ctl_t *ctl);

const char *snd_ctl_elem_type_name(snd_ctl_elem_type_t type);
const char *snd_ctl_elem_iface_name(snd_ctl_elem_iface_t iface);
const char *snd_ctl_event_type_name(snd_ctl_event_type_t type);

unsigned int snd_ctl_event_elem_get_mask(const snd_ctl_event_t *obj);
unsigned int snd_ctl_event_elem_get_numid(const snd_ctl_event_t *obj);
void snd_ctl_event_elem_get_id(const snd_ctl_event_t *obj, snd_ctl_elem_id_t *ptr);
snd_ctl_elem_iface_t snd_ctl_event_elem_get_interface(const snd_ctl_event_t *obj);
unsigned int snd_ctl_event_elem_get_device(const snd_ctl_event_t *obj);
unsigned int snd_ctl_event_elem_get_subdevice(const snd_ctl_event_t *obj);
const char *snd_ctl_event_elem_get_name(const snd_ctl_event_t *obj);
unsigned int snd_ctl_event_elem_get_index(const snd_ctl_event_t *obj);

int snd_ctl_elem_list_alloc_space(snd_ctl_elem_list_t *obj, unsigned int entries);
void snd_ctl_elem_list_free_space(snd_ctl_elem_list_t *obj);

char *snd_ctl_ascii_elem_id_get(snd_ctl_elem_id_t *id);
int snd_ctl_ascii_elem_id_parse(snd_ctl_elem_id_t *dst, const char *str);
int snd_ctl_ascii_value_parse(snd_ctl_t *handle,
			      snd_ctl_elem_value_t *dst,
			      snd_ctl_elem_info_t *info,
			      const char *value);

size_t snd_ctl_elem_id_sizeof(void);
/** \hideinitializer
 * \brief allocate an invalid #snd_ctl_elem_id_t using standard alloca
 * \param ptr returned pointer
 */
#define snd_ctl_elem_id_alloca(ptr) __snd_alloca(ptr, snd_ctl_elem_id)
int snd_ctl_elem_id_malloc(snd_ctl_elem_id_t **ptr);
void snd_ctl_elem_id_free(snd_ctl_elem_id_t *obj);
void snd_ctl_elem_id_clear(snd_ctl_elem_id_t *obj);
void snd_ctl_elem_id_copy(snd_ctl_elem_id_t *dst, const snd_ctl_elem_id_t *src);
unsigned int snd_ctl_elem_id_get_numid(const snd_ctl_elem_id_t *obj);
snd_ctl_elem_iface_t snd_ctl_elem_id_get_interface(const snd_ctl_elem_id_t *obj);
unsigned int snd_ctl_elem_id_get_device(const snd_ctl_elem_id_t *obj);
unsigned int snd_ctl_elem_id_get_subdevice(const snd_ctl_elem_id_t *obj);
const char *snd_ctl_elem_id_get_name(const snd_ctl_elem_id_t *obj);
unsigned int snd_ctl_elem_id_get_index(const snd_ctl_elem_id_t *obj);
void snd_ctl_elem_id_set_numid(snd_ctl_elem_id_t *obj, unsigned int val);
void snd_ctl_elem_id_set_interface(snd_ctl_elem_id_t *obj, snd_ctl_elem_iface_t val);
void snd_ctl_elem_id_set_device(snd_ctl_elem_id_t *obj, unsigned int val);
void snd_ctl_elem_id_set_subdevice(snd_ctl_elem_id_t *obj, unsigned int val);
void snd_ctl_elem_id_set_name(snd_ctl_elem_id_t *obj, const char *val);
void snd_ctl_elem_id_set_index(snd_ctl_elem_id_t *obj, unsigned int val);

size_t snd_ctl_card_info_sizeof(void);
/** \hideinitializer
 * \brief allocate an invalid #snd_ctl_card_info_t using standard alloca
 * \param ptr returned pointer
 */
#define snd_ctl_card_info_alloca(ptr) __snd_alloca(ptr, snd_ctl_card_info)
int snd_ctl_card_info_malloc(snd_ctl_card_info_t **ptr);
void snd_ctl_card_info_free(snd_ctl_card_info_t *obj);
void snd_ctl_card_info_clear(snd_ctl_card_info_t *obj);
void snd_ctl_card_info_copy(snd_ctl_card_info_t *dst, const snd_ctl_card_info_t *src);
int snd_ctl_card_info_get_card(const snd_ctl_card_info_t *obj);
const char *snd_ctl_card_info_get_id(const snd_ctl_card_info_t *obj);
const char *snd_ctl_card_info_get_driver(const snd_ctl_card_info_t *obj);
const char *snd_ctl_card_info_get_name(const snd_ctl_card_info_t *obj);
const char *snd_ctl_card_info_get_longname(const snd_ctl_card_info_t *obj);
const char *snd_ctl_card_info_get_mixername(const snd_ctl_card_info_t *obj);
const char *snd_ctl_card_info_get_components(const snd_ctl_card_info_t *obj);

size_t snd_ctl_event_sizeof(void);
/** \hideinitializer
 * \brief allocate an invalid #snd_ctl_event_t using standard alloca
 * \param ptr returned pointer
 */
#define snd_ctl_event_alloca(ptr) __snd_alloca(ptr, snd_ctl_event)
int snd_ctl_event_malloc(snd_ctl_event_t **ptr);
void snd_ctl_event_free(snd_ctl_event_t *obj);
void snd_ctl_event_clear(snd_ctl_event_t *obj);
void snd_ctl_event_copy(snd_ctl_event_t *dst, const snd_ctl_event_t *src);
snd_ctl_event_type_t snd_ctl_event_get_type(const snd_ctl_event_t *obj);

size_t snd_ctl_elem_list_sizeof(void);
/** \hideinitializer
 * \brief allocate an invalid #snd_ctl_elem_list_t using standard alloca
 * \param ptr returned pointer
 */
#define snd_ctl_elem_list_alloca(ptr) __snd_alloca(ptr, snd_ctl_elem_list)
int snd_ctl_elem_list_malloc(snd_ctl_elem_list_t **ptr);
void snd_ctl_elem_list_free(snd_ctl_elem_list_t *obj);
void snd_ctl_elem_list_clear(snd_ctl_elem_list_t *obj);
void snd_ctl_elem_list_copy(snd_ctl_elem_list_t *dst, const snd_ctl_elem_list_t *src);
void snd_ctl_elem_list_set_offset(snd_ctl_elem_list_t *obj, unsigned int val);
unsigned int snd_ctl_elem_list_get_used(const snd_ctl_elem_list_t *obj);
unsigned int snd_ctl_elem_list_get_count(const snd_ctl_elem_list_t *obj);
void snd_ctl_elem_list_get_id(const snd_ctl_elem_list_t *obj, unsigned int idx, snd_ctl_elem_id_t *ptr);
unsigned int snd_ctl_elem_list_get_numid(const snd_ctl_elem_list_t *obj, unsigned int idx);
snd_ctl_elem_iface_t snd_ctl_elem_list_get_interface(const snd_ctl_elem_list_t *obj, unsigned int idx);
unsigned int snd_ctl_elem_list_get_device(const snd_ctl_elem_list_t *obj, unsigned int idx);
unsigned int snd_ctl_elem_list_get_subdevice(const snd_ctl_elem_list_t *obj, unsigned int idx);
const char *snd_ctl_elem_list_get_name(const snd_ctl_elem_list_t *obj, unsigned int idx);
unsigned int snd_ctl_elem_list_get_index(const snd_ctl_elem_list_t *obj, unsigned int idx);

size_t snd_ctl_elem_info_sizeof(void);
/** \hideinitializer
 * \brief allocate an invalid #snd_ctl_elem_info_t using standard alloca
 * \param ptr returned pointer
 */
#define snd_ctl_elem_info_alloca(ptr) __snd_alloca(ptr, snd_ctl_elem_info)
int snd_ctl_elem_info_malloc(snd_ctl_elem_info_t **ptr);
void snd_ctl_elem_info_free(snd_ctl_elem_info_t *obj);
void snd_ctl_elem_info_clear(snd_ctl_elem_info_t *obj);
void snd_ctl_elem_info_copy(snd_ctl_elem_info_t *dst, const snd_ctl_elem_info_t *src);
snd_ctl_elem_type_t snd_ctl_elem_info_get_type(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_readable(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_writable(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_volatile(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_inactive(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_locked(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_tlv_readable(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_tlv_writable(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_tlv_commandable(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_owner(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_is_user(const snd_ctl_elem_info_t *obj);
pid_t snd_ctl_elem_info_get_owner(const snd_ctl_elem_info_t *obj);
unsigned int snd_ctl_elem_info_get_count(const snd_ctl_elem_info_t *obj);
long snd_ctl_elem_info_get_min(const snd_ctl_elem_info_t *obj);
long snd_ctl_elem_info_get_max(const snd_ctl_elem_info_t *obj);
long snd_ctl_elem_info_get_step(const snd_ctl_elem_info_t *obj);
long long snd_ctl_elem_info_get_min64(const snd_ctl_elem_info_t *obj);
long long snd_ctl_elem_info_get_max64(const snd_ctl_elem_info_t *obj);
long long snd_ctl_elem_info_get_step64(const snd_ctl_elem_info_t *obj);
unsigned int snd_ctl_elem_info_get_items(const snd_ctl_elem_info_t *obj);
void snd_ctl_elem_info_set_item(snd_ctl_elem_info_t *obj, unsigned int val);
const char *snd_ctl_elem_info_get_item_name(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_get_dimensions(const snd_ctl_elem_info_t *obj);
int snd_ctl_elem_info_get_dimension(const snd_ctl_elem_info_t *obj, unsigned int idx);
int snd_ctl_elem_info_set_dimension(snd_ctl_elem_info_t *info,
				    const int dimension[4]);
void snd_ctl_elem_info_get_id(const snd_ctl_elem_info_t *obj, snd_ctl_elem_id_t *ptr);
unsigned int snd_ctl_elem_info_get_numid(const snd_ctl_elem_info_t *obj);
snd_ctl_elem_iface_t snd_ctl_elem_info_get_interface(const snd_ctl_elem_info_t *obj);
unsigned int snd_ctl_elem_info_get_device(const snd_ctl_elem_info_t *obj);
unsigned int snd_ctl_elem_info_get_subdevice(const snd_ctl_elem_info_t *obj);
const char *snd_ctl_elem_info_get_name(const snd_ctl_elem_info_t *obj);
unsigned int snd_ctl_elem_info_get_index(const snd_ctl_elem_info_t *obj);
void snd_ctl_elem_info_set_id(snd_ctl_elem_info_t *obj, const snd_ctl_elem_id_t *ptr);
void snd_ctl_elem_info_set_numid(snd_ctl_elem_info_t *obj, unsigned int val);
void snd_ctl_elem_info_set_interface(snd_ctl_elem_info_t *obj, snd_ctl_elem_iface_t val);
void snd_ctl_elem_info_set_device(snd_ctl_elem_info_t *obj, unsigned int val);
void snd_ctl_elem_info_set_subdevice(snd_ctl_elem_info_t *obj, unsigned int val);
void snd_ctl_elem_info_set_name(snd_ctl_elem_info_t *obj, const char *val);
void snd_ctl_elem_info_set_index(snd_ctl_elem_info_t *obj, unsigned int val);

int snd_ctl_add_integer_elem_set(snd_ctl_t *ctl, snd_ctl_elem_info_t *info,
				 unsigned int element_count,
				 unsigned int member_count,
				 long min, long max, long step);
int snd_ctl_add_integer64_elem_set(snd_ctl_t *ctl, snd_ctl_elem_info_t *info,
				   unsigned int element_count,
				   unsigned int member_count,
				   long long min, long long max,
				   long long step);
int snd_ctl_add_boolean_elem_set(snd_ctl_t *ctl, snd_ctl_elem_info_t *info,
				 unsigned int element_count,
				 unsigned int member_count);
int snd_ctl_add_enumerated_elem_set(snd_ctl_t *ctl, snd_ctl_elem_info_t *info,
				    unsigned int element_count,
				    unsigned int member_count,
				    unsigned int items,
				    const char *const labels[]);
int snd_ctl_add_bytes_elem_set(snd_ctl_t *ctl, snd_ctl_elem_info_t *info,
			       unsigned int element_count,
			       unsigned int member_count);

int snd_ctl_elem_add_integer(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id, unsigned int count, long imin, long imax, long istep);
int snd_ctl_elem_add_integer64(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id, unsigned int count, long long imin, long long imax, long long istep);
int snd_ctl_elem_add_boolean(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id, unsigned int count);
int snd_ctl_elem_add_enumerated(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id, unsigned int count, unsigned int items, const char *const names[]);
int snd_ctl_elem_add_iec958(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id);
int snd_ctl_elem_remove(snd_ctl_t *ctl, snd_ctl_elem_id_t *id);

size_t snd_ctl_elem_value_sizeof(void);
/** \hideinitializer
 * \brief allocate an invalid #snd_ctl_elem_value_t using standard alloca
 * \param ptr returned pointer
 */
#define snd_ctl_elem_value_alloca(ptr) __snd_alloca(ptr, snd_ctl_elem_value)
int snd_ctl_elem_value_malloc(snd_ctl_elem_value_t **ptr);
void snd_ctl_elem_value_free(snd_ctl_elem_value_t *obj);
void snd_ctl_elem_value_clear(snd_ctl_elem_value_t *obj);
void snd_ctl_elem_value_copy(snd_ctl_elem_value_t *dst, const snd_ctl_elem_value_t *src);
int snd_ctl_elem_value_compare(snd_ctl_elem_value_t *left, const snd_ctl_elem_value_t *right);
void snd_ctl_elem_value_get_id(const snd_ctl_elem_value_t *obj, snd_ctl_elem_id_t *ptr);
unsigned int snd_ctl_elem_value_get_numid(const snd_ctl_elem_value_t *obj);
snd_ctl_elem_iface_t snd_ctl_elem_value_get_interface(const snd_ctl_elem_value_t *obj);
unsigned int snd_ctl_elem_value_get_device(const snd_ctl_elem_value_t *obj);
unsigned int snd_ctl_elem_value_get_subdevice(const snd_ctl_elem_value_t *obj);
const char *snd_ctl_elem_value_get_name(const snd_ctl_elem_value_t *obj);
unsigned int snd_ctl_elem_value_get_index(const snd_ctl_elem_value_t *obj);
void snd_ctl_elem_value_set_id(snd_ctl_elem_value_t *obj, const snd_ctl_elem_id_t *ptr);
void snd_ctl_elem_value_set_numid(snd_ctl_elem_value_t *obj, unsigned int val);
void snd_ctl_elem_value_set_interface(snd_ctl_elem_value_t *obj, snd_ctl_elem_iface_t val);
void snd_ctl_elem_value_set_device(snd_ctl_elem_value_t *obj, unsigned int val);
void snd_ctl_elem_value_set_subdevice(snd_ctl_elem_value_t *obj, unsigned int val);
void snd_ctl_elem_value_set_name(snd_ctl_elem_value_t *obj, const char *val);
void snd_ctl_elem_value_set_index(snd_ctl_elem_value_t *obj, unsigned int val);
int snd_ctl_elem_value_get_boolean(const snd_ctl_elem_value_t *obj, unsigned int idx);
long snd_ctl_elem_value_get_integer(const snd_ctl_elem_value_t *obj, unsigned int idx);
long long snd_ctl_elem_value_get_integer64(const snd_ctl_elem_value_t *obj, unsigned int idx);
unsigned int snd_ctl_elem_value_get_enumerated(const snd_ctl_elem_value_t *obj, unsigned int idx);
unsigned char snd_ctl_elem_value_get_byte(const snd_ctl_elem_value_t *obj, unsigned int idx);
void snd_ctl_elem_value_set_boolean(snd_ctl_elem_value_t *obj, unsigned int idx, long val);
void snd_ctl_elem_value_set_integer(snd_ctl_elem_value_t *obj, unsigned int idx, long val);
void snd_ctl_elem_value_set_integer64(snd_ctl_elem_value_t *obj, unsigned int idx, long long val);
void snd_ctl_elem_value_set_enumerated(snd_ctl_elem_value_t *obj, unsigned int idx, unsigned int val);
void snd_ctl_elem_value_set_byte(snd_ctl_elem_value_t *obj, unsigned int idx, unsigned char val);
void snd_ctl_elem_set_bytes(snd_ctl_elem_value_t *obj, void *data, size_t size);
const void * snd_ctl_elem_value_get_bytes(const snd_ctl_elem_value_t *obj);
void snd_ctl_elem_value_get_iec958(const snd_ctl_elem_value_t *obj, snd_aes_iec958_t *ptr);
void snd_ctl_elem_value_set_iec958(snd_ctl_elem_value_t *obj, const snd_aes_iec958_t *ptr);

int snd_tlv_parse_dB_info(unsigned int *tlv, unsigned int tlv_size,
			  unsigned int **db_tlvp);
int snd_tlv_get_dB_range(unsigned int *tlv, long rangemin, long rangemax,
			 long *min, long *max);
int snd_tlv_convert_to_dB(unsigned int *tlv, long rangemin, long rangemax,
			  long volume, long *db_gain);
int snd_tlv_convert_from_dB(unsigned int *tlv, long rangemin, long rangemax,
			    long db_gain, long *value, int xdir);
int snd_ctl_get_dB_range(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id,
			 long *min, long *max);
int snd_ctl_convert_to_dB(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id,
			  long volume, long *db_gain);
int snd_ctl_convert_from_dB(snd_ctl_t *ctl, const snd_ctl_elem_id_t *id,
			    long db_gain, long *value, int xdir);

/**
 *  \defgroup HControl High level Control Interface
 *  \ingroup Control
 *  The high level control interface.
 *  See \ref hcontrol page for more details.
 *  \{
 */

/** HCTL element handle */
typedef struct _snd_hctl_elem snd_hctl_elem_t;

/** HCTL handle */
typedef struct _snd_hctl snd_hctl_t;

/**
 * \brief Compare function for sorting HCTL elements
 * \param e1 First element
 * \param e2 Second element
 * \return -1 if e1 < e2, 0 if e1 == e2, 1 if e1 > e2
 */
typedef int (*snd_hctl_compare_t)(const snd_hctl_elem_t *e1,
				  const snd_hctl_elem_t *e2);
int snd_hctl_compare_fast(const snd_hctl_elem_t *c1,
			  const snd_hctl_elem_t *c2);
/** 
 * \brief HCTL callback function
 * \param hctl HCTL handle
 * \param mask event mask
 * \param elem related HCTL element (if any)
 * \return 0 on success otherwise a negative error code
 */
typedef int (*snd_hctl_callback_t)(snd_hctl_t *hctl,
				   unsigned int mask,
				   snd_hctl_elem_t *elem);
/** 
 * \brief HCTL element callback function
 * \param elem HCTL element
 * \param mask event mask
 * \return 0 on success otherwise a negative error code
 */
typedef int (*snd_hctl_elem_callback_t)(snd_hctl_elem_t *elem,
					unsigned int mask);

int snd_hctl_open(snd_hctl_t **hctl, const char *name, int mode);
int snd_hctl_open_ctl(snd_hctl_t **hctlp, snd_ctl_t *ctl);
int snd_hctl_close(snd_hctl_t *hctl);
int snd_hctl_nonblock(snd_hctl_t *hctl, int nonblock);
static __inline__ int snd_hctl_abort(snd_hctl_t *hctl) { return snd_hctl_nonblock(hctl, 2); }
int snd_hctl_poll_descriptors_count(snd_hctl_t *hctl);
int snd_hctl_poll_descriptors(snd_hctl_t *hctl, struct pollfd *pfds, unsigned int space);
int snd_hctl_poll_descriptors_revents(snd_hctl_t *ctl, struct pollfd *pfds, unsigned int nfds, unsigned short *revents);
unsigned int snd_hctl_get_count(snd_hctl_t *hctl);
int snd_hctl_set_compare(snd_hctl_t *hctl, snd_hctl_compare_t hsort);
snd_hctl_elem_t *snd_hctl_first_elem(snd_hctl_t *hctl);
snd_hctl_elem_t *snd_hctl_last_elem(snd_hctl_t *hctl);
snd_hctl_elem_t *snd_hctl_find_elem(snd_hctl_t *hctl, const snd_ctl_elem_id_t *id);
void snd_hctl_set_callback(snd_hctl_t *hctl, snd_hctl_callback_t callback);
void snd_hctl_set_callback_private(snd_hctl_t *hctl, void *data);
void *snd_hctl_get_callback_private(snd_hctl_t *hctl);
int snd_hctl_load(snd_hctl_t *hctl);
int snd_hctl_free(snd_hctl_t *hctl);
int snd_hctl_handle_events(snd_hctl_t *hctl);
const char *snd_hctl_name(snd_hctl_t *hctl);
int snd_hctl_wait(snd_hctl_t *hctl, int timeout);
snd_ctl_t *snd_hctl_ctl(snd_hctl_t *hctl);

snd_hctl_elem_t *snd_hctl_elem_next(snd_hctl_elem_t *elem);
snd_hctl_elem_t *snd_hctl_elem_prev(snd_hctl_elem_t *elem);
int snd_hctl_elem_info(snd_hctl_elem_t *elem, snd_ctl_elem_info_t * info);
int snd_hctl_elem_read(snd_hctl_elem_t *elem, snd_ctl_elem_value_t * value);
int snd_hctl_elem_write(snd_hctl_elem_t *elem, snd_ctl_elem_value_t * value);
int snd_hctl_elem_tlv_read(snd_hctl_elem_t *elem, unsigned int *tlv, unsigned int tlv_size);
int snd_hctl_elem_tlv_write(snd_hctl_elem_t *elem, const unsigned int *tlv);
int snd_hctl_elem_tlv_command(snd_hctl_elem_t *elem, const unsigned int *tlv);

snd_hctl_t *snd_hctl_elem_get_hctl(snd_hctl_elem_t *elem);

void snd_hctl_elem_get_id(const snd_hctl_elem_t *obj, snd_ctl_elem_id_t *ptr);
unsigned int snd_hctl_elem_get_numid(const snd_hctl_elem_t *obj);
snd_ctl_elem_iface_t snd_hctl_elem_get_interface(const snd_hctl_elem_t *obj);
unsigned int snd_hctl_elem_get_device(const snd_hctl_elem_t *obj);
unsigned int snd_hctl_elem_get_subdevice(const snd_hctl_elem_t *obj);
const char *snd_hctl_elem_get_name(const snd_hctl_elem_t *obj);
unsigned int snd_hctl_elem_get_index(const snd_hctl_elem_t *obj);
void snd_hctl_elem_set_callback(snd_hctl_elem_t *obj, snd_hctl_elem_callback_t val);
void * snd_hctl_elem_get_callback_private(const snd_hctl_elem_t *obj);
void snd_hctl_elem_set_callback_private(snd_hctl_elem_t *obj, void * val);

/** \} */

/** \} */

/**
 *  \defgroup SControl Setup Control Interface
 *  \ingroup Control
 *  The setup control interface - set or modify control elements from a configuration file.
 *  \{
 */

int snd_sctl_build(snd_sctl_t **ctl, snd_ctl_t *handle, snd_config_t *config,
		   snd_config_t *private_data, int mode);
int snd_sctl_free(snd_sctl_t *handle);
int snd_sctl_install(snd_sctl_t *handle);
int snd_sctl_remove(snd_sctl_t *handle);

/** \} */

#ifdef __cplusplus
}
#endif

#endif /* __ALSA_CONTROL_H */
