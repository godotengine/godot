/**
 * \file include/hwdep.h
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

#ifndef __ALSA_HWDEP_H
#define __ALSA_HWDEP_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  \defgroup HwDep Hardware Dependant Interface
 *  The Hardware Dependant Interface.
 *  \{
 */

/** dlsym version for interface entry callback */
#define SND_HWDEP_DLSYM_VERSION		_dlsym_hwdep_001

/** HwDep information container */
typedef struct _snd_hwdep_info snd_hwdep_info_t;

/** HwDep DSP status container */
typedef struct _snd_hwdep_dsp_status snd_hwdep_dsp_status_t;

/** HwDep DSP image container */
typedef struct _snd_hwdep_dsp_image snd_hwdep_dsp_image_t;

/** HwDep interface */
typedef enum _snd_hwdep_iface {
	SND_HWDEP_IFACE_OPL2 = 0,	/**< OPL2 raw driver */
	SND_HWDEP_IFACE_OPL3,		/**< OPL3 raw driver */
	SND_HWDEP_IFACE_OPL4,		/**< OPL4 raw driver */
	SND_HWDEP_IFACE_SB16CSP,	/**< SB16CSP driver */
	SND_HWDEP_IFACE_EMU10K1,	/**< EMU10K1 driver */
	SND_HWDEP_IFACE_YSS225,		/**< YSS225 driver */
	SND_HWDEP_IFACE_ICS2115,	/**< ICS2115 driver */
	SND_HWDEP_IFACE_SSCAPE,		/**< Ensoniq SoundScape ISA card (MC68EC000) */
	SND_HWDEP_IFACE_VX,		/**< Digigram VX cards */
	SND_HWDEP_IFACE_MIXART,		/**< Digigram miXart cards */
	SND_HWDEP_IFACE_USX2Y,		/**< Tascam US122, US224 & US428 usb */
	SND_HWDEP_IFACE_EMUX_WAVETABLE,	/**< EmuX wavetable */
	SND_HWDEP_IFACE_BLUETOOTH,	/**< Bluetooth audio */
	SND_HWDEP_IFACE_USX2Y_PCM,	/**< Tascam US122, US224 & US428 raw USB PCM */
	SND_HWDEP_IFACE_PCXHR,		/**< Digigram PCXHR */
	SND_HWDEP_IFACE_SB_RC,		/**< SB Extigy/Audigy2NX remote control */
	SND_HWDEP_IFACE_HDA,		/**< HD-audio */
	SND_HWDEP_IFACE_USB_STREAM,	/**< direct access to usb stream */
	SND_HWDEP_IFACE_FW_DICE,	/**< TC DICE FireWire device */
	SND_HWDEP_IFACE_FW_FIREWORKS,	/**< Echo Audio Fireworks based device */
	SND_HWDEP_IFACE_FW_BEBOB,	/**< BridgeCo BeBoB based device */
	SND_HWDEP_IFACE_FW_OXFW,	/**< Oxford OXFW970/971 based device */
	SND_HWDEP_IFACE_FW_DIGI00X,	/* Digidesign Digi 002/003 family */
	SND_HWDEP_IFACE_FW_TASCAM,	/* TASCAM FireWire series */

	SND_HWDEP_IFACE_LAST = SND_HWDEP_IFACE_FW_TASCAM	/**< last known hwdep interface */
} snd_hwdep_iface_t;

/** open for reading */
#define SND_HWDEP_OPEN_READ		(O_RDONLY)
/** open for writing */
#define SND_HWDEP_OPEN_WRITE		(O_WRONLY)
/** open for reading and writing */
#define SND_HWDEP_OPEN_DUPLEX		(O_RDWR)
/** open mode flag: open in nonblock mode */
#define SND_HWDEP_OPEN_NONBLOCK		(O_NONBLOCK)

/** HwDep handle type */
typedef enum _snd_hwdep_type {
	/** Kernel level HwDep */
	SND_HWDEP_TYPE_HW,
	/** Shared memory client HwDep (not yet implemented) */
	SND_HWDEP_TYPE_SHM,
	/** INET client HwDep (not yet implemented) */
	SND_HWDEP_TYPE_INET
} snd_hwdep_type_t;

/** HwDep handle */
typedef struct _snd_hwdep snd_hwdep_t;

int snd_hwdep_open(snd_hwdep_t **hwdep, const char *name, int mode);
int snd_hwdep_close(snd_hwdep_t *hwdep);
int snd_hwdep_poll_descriptors(snd_hwdep_t *hwdep, struct pollfd *pfds, unsigned int space);
int snd_hwdep_poll_descriptors_count(snd_hwdep_t *hwdep);
int snd_hwdep_poll_descriptors_revents(snd_hwdep_t *hwdep, struct pollfd *pfds, unsigned int nfds, unsigned short *revents);
int snd_hwdep_nonblock(snd_hwdep_t *hwdep, int nonblock);
int snd_hwdep_info(snd_hwdep_t *hwdep, snd_hwdep_info_t * info);
int snd_hwdep_dsp_status(snd_hwdep_t *hwdep, snd_hwdep_dsp_status_t *status);
int snd_hwdep_dsp_load(snd_hwdep_t *hwdep, snd_hwdep_dsp_image_t *block);
int snd_hwdep_ioctl(snd_hwdep_t *hwdep, unsigned int request, void * arg);
ssize_t snd_hwdep_write(snd_hwdep_t *hwdep, const void *buffer, size_t size);
ssize_t snd_hwdep_read(snd_hwdep_t *hwdep, void *buffer, size_t size);

size_t snd_hwdep_info_sizeof(void);
/** allocate #snd_hwdep_info_t container on stack */
#define snd_hwdep_info_alloca(ptr) __snd_alloca(ptr, snd_hwdep_info)
int snd_hwdep_info_malloc(snd_hwdep_info_t **ptr);
void snd_hwdep_info_free(snd_hwdep_info_t *obj);
void snd_hwdep_info_copy(snd_hwdep_info_t *dst, const snd_hwdep_info_t *src);

unsigned int snd_hwdep_info_get_device(const snd_hwdep_info_t *obj);
int snd_hwdep_info_get_card(const snd_hwdep_info_t *obj);
const char *snd_hwdep_info_get_id(const snd_hwdep_info_t *obj);
const char *snd_hwdep_info_get_name(const snd_hwdep_info_t *obj);
snd_hwdep_iface_t snd_hwdep_info_get_iface(const snd_hwdep_info_t *obj);
void snd_hwdep_info_set_device(snd_hwdep_info_t *obj, unsigned int val);

size_t snd_hwdep_dsp_status_sizeof(void);
/** allocate #snd_hwdep_dsp_status_t container on stack */
#define snd_hwdep_dsp_status_alloca(ptr) __snd_alloca(ptr, snd_hwdep_dsp_status)
int snd_hwdep_dsp_status_malloc(snd_hwdep_dsp_status_t **ptr);
void snd_hwdep_dsp_status_free(snd_hwdep_dsp_status_t *obj);
void snd_hwdep_dsp_status_copy(snd_hwdep_dsp_status_t *dst, const snd_hwdep_dsp_status_t *src);

unsigned int snd_hwdep_dsp_status_get_version(const snd_hwdep_dsp_status_t *obj);
const char *snd_hwdep_dsp_status_get_id(const snd_hwdep_dsp_status_t *obj);
unsigned int snd_hwdep_dsp_status_get_num_dsps(const snd_hwdep_dsp_status_t *obj);
unsigned int snd_hwdep_dsp_status_get_dsp_loaded(const snd_hwdep_dsp_status_t *obj);
unsigned int snd_hwdep_dsp_status_get_chip_ready(const snd_hwdep_dsp_status_t *obj);

size_t snd_hwdep_dsp_image_sizeof(void);
/** allocate #snd_hwdep_dsp_image_t container on stack */
#define snd_hwdep_dsp_image_alloca(ptr) __snd_alloca(ptr, snd_hwdep_dsp_image)
int snd_hwdep_dsp_image_malloc(snd_hwdep_dsp_image_t **ptr);
void snd_hwdep_dsp_image_free(snd_hwdep_dsp_image_t *obj);
void snd_hwdep_dsp_image_copy(snd_hwdep_dsp_image_t *dst, const snd_hwdep_dsp_image_t *src);

unsigned int snd_hwdep_dsp_image_get_index(const snd_hwdep_dsp_image_t *obj);
const char *snd_hwdep_dsp_image_get_name(const snd_hwdep_dsp_image_t *obj);
const void *snd_hwdep_dsp_image_get_image(const snd_hwdep_dsp_image_t *obj);
size_t snd_hwdep_dsp_image_get_length(const snd_hwdep_dsp_image_t *obj);

void snd_hwdep_dsp_image_set_index(snd_hwdep_dsp_image_t *obj, unsigned int _index);
void snd_hwdep_dsp_image_set_name(snd_hwdep_dsp_image_t *obj, const char *name);
void snd_hwdep_dsp_image_set_image(snd_hwdep_dsp_image_t *obj, void *buffer);
void snd_hwdep_dsp_image_set_length(snd_hwdep_dsp_image_t *obj, size_t length);

/** \} */

#ifdef __cplusplus
}
#endif

#endif /* __ALSA_HWDEP_H */

