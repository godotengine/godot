/**
 * \file include/control_external.h
 * \brief External control plugin SDK
 * \author Takashi Iwai <tiwai@suse.de>
 * \date 2005
 *
 * External control plugin SDK.
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
#ifndef __ALSA_CONTROL_EXTERNAL_H
#define __ALSA_CONTROL_EXTERNAL_H

#include "control.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  \defgroup CtlPlugin_SDK External Control Plugin SDK
 *  \{
 */

/**
 * Define the object entry for external control plugins
 */
#define SND_CTL_PLUGIN_ENTRY(name) _snd_ctl_##name##_open

/**
 * Define the symbols of the given control plugin with versions
 */
#define SND_CTL_PLUGIN_SYMBOL(name) SND_DLSYM_BUILD_VERSION(SND_CTL_PLUGIN_ENTRY(name), SND_CONTROL_DLSYM_VERSION);

/**
 * Define the control plugin
 */
#define SND_CTL_PLUGIN_DEFINE_FUNC(plugin) \
int SND_CTL_PLUGIN_ENTRY(plugin) (snd_ctl_t **handlep, const char *name,\
				  snd_config_t *root, snd_config_t *conf, int mode)

/** External control plugin handle */
typedef struct snd_ctl_ext snd_ctl_ext_t;
/** Callback table of control ext */
typedef struct snd_ctl_ext_callback snd_ctl_ext_callback_t;
/** Key to access a control pointer */
typedef unsigned long snd_ctl_ext_key_t;
#ifdef DOC_HIDDEN
/* redefine typedef's for stupid doxygen */
typedef snd_ctl_ext snd_ctl_ext_t;
typedef snd_ctl_ext_callback snd_ctl_ext_callback_t;
#endif
/** Callback to handle TLV commands. */
typedef int (snd_ctl_ext_tlv_rw_t)(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key, int op_flag, unsigned int numid,
				   unsigned int *tlv, unsigned int tlv_size);

/*
 * Protocol version
 */
#define SND_CTL_EXT_VERSION_MAJOR	1	/**< Protocol major version */
#define SND_CTL_EXT_VERSION_MINOR	0	/**< Protocol minor version */
#define SND_CTL_EXT_VERSION_TINY	1	/**< Protocol tiny version */
/**
 * external plugin protocol version
 */
#define SND_CTL_EXT_VERSION		((SND_CTL_EXT_VERSION_MAJOR<<16) |\
					 (SND_CTL_EXT_VERSION_MINOR<<8) |\
					 (SND_CTL_EXT_VERSION_TINY))

/** Handle of control ext */
struct snd_ctl_ext {
	/**
	 * protocol version; #SND_CTL_EXT_VERSION must be filled here
	 * before calling #snd_ctl_ext_create()
	 */
	unsigned int version;
	/**
	 * Index of this card; must be filled before calling #snd_ctl_ext_create()
	 */
	int card_idx;
	/**
	 * ID string of this card; must be filled before calling #snd_ctl_ext_create()
	 */
	char id[16];
	/**
	 * Driver name of this card; must be filled before calling #snd_ctl_ext_create()
	 */
	char driver[16];
	/**
	 * short name of this card; must be filled before calling #snd_ctl_ext_create()
	 */
	char name[32];
	/**
	 * Long name of this card; must be filled before calling #snd_ctl_ext_create()
	 */
	char longname[80];
	/**
	 * Mixer name of this card; must be filled before calling #snd_ctl_ext_create()
	 */
	char mixername[80];
	/**
	 * poll descriptor
	 */
	int poll_fd;

	/**
	 * callbacks of this plugin; must be filled before calling #snd_pcm_ioplug_create()
	 */
	const snd_ctl_ext_callback_t *callback;
	/**
	 * private data, which can be used freely in the driver callbacks
	 */
	void *private_data;
	/**
	 * control handle filled by #snd_ctl_ext_create()
	 */
	snd_ctl_t *handle;

	int nonblock;			/**< non-block mode; read-only */
	int subscribed;			/**< events subscribed; read-only */

	/**
	 * optional TLV data for the control (since protocol 1.0.1)
	 */
	union {
		snd_ctl_ext_tlv_rw_t *c;
		const unsigned int *p;
	} tlv;
};

/** Callback table of ext. */
struct snd_ctl_ext_callback {
	/**
	 * close the control handle; optional
	 */
	void (*close)(snd_ctl_ext_t *ext);
	/**
	 * return the total number of elements; required
	 */
	int (*elem_count)(snd_ctl_ext_t *ext);
	/**
	 * return the element id of the given offset (array index); required
	 */
	int (*elem_list)(snd_ctl_ext_t *ext, unsigned int offset, snd_ctl_elem_id_t *id);
	/**
	 * convert the element id to a search key; required
	 */
	snd_ctl_ext_key_t (*find_elem)(snd_ctl_ext_t *ext, const snd_ctl_elem_id_t *id);
	/**
	 * the destructor of the key; optional
	 */
	void (*free_key)(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key);
	/**
	 * get the attribute of the element; required
	 */
	int (*get_attribute)(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key,
			     int *type, unsigned int *acc, unsigned int *count);
	/**
	 * get the element information of integer type
	 */
	int (*get_integer_info)(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key,
				long *imin, long *imax, long *istep);
	/**
	 * get the element information of integer64 type
	 */
	int (*get_integer64_info)(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key,
				  int64_t *imin, int64_t *imax, int64_t *istep);
	/**
	 * get the element information of enumerated type
	 */
	int (*get_enumerated_info)(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key, unsigned int *items);
	/**
	 * get the name of the enumerated item
	 */
	int (*get_enumerated_name)(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key, unsigned int item,
				   char *name, size_t name_max_len);
	/**
	 * read the current values of integer type
	 */
	int (*read_integer)(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key, long *value);
	/**
	 * read the current values of integer64 type
	 */
	int (*read_integer64)(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key, int64_t *value);
	/**
	 * read the current values of enumerated type
	 */
	int (*read_enumerated)(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key, unsigned int *items);
	/**
	 * read the current values of bytes type
	 */
	int (*read_bytes)(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key, unsigned char *data,
			  size_t max_bytes);
	/**
	 * read the current values of iec958 type
	 */
	int (*read_iec958)(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key, snd_aes_iec958_t *iec958);
	/**
	 * update the current values of integer type with the given values
	 */
	int (*write_integer)(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key, long *value);
	/**
	 * update the current values of integer64 type with the given values
	 */
	int (*write_integer64)(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key, int64_t *value);
	/**
	 * update the current values of enumerated type with the given values
	 */
	int (*write_enumerated)(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key, unsigned int *items);
	/**
	 * update the current values of bytes type with the given values
	 */
	int (*write_bytes)(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key, unsigned char *data,
			   size_t max_bytes);
	/**
	 * update the current values of iec958 type with the given values
	 */
	int (*write_iec958)(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key, snd_aes_iec958_t *iec958);
	/**
	 * subscribe/unsubscribe the event notification; optional
	 */
	void (*subscribe_events)(snd_ctl_ext_t *ext, int subscribe);
	/**
	 * read a pending notification event; optional
	 */
	int (*read_event)(snd_ctl_ext_t *ext, snd_ctl_elem_id_t *id, unsigned int *event_mask);
	/**
	 * return the number of poll descriptors; optional
	 */
	int (*poll_descriptors_count)(snd_ctl_ext_t *ext);
	/**
	 * fill the poll descriptors; optional
	 */
	int (*poll_descriptors)(snd_ctl_ext_t *ext, struct pollfd *pfds, unsigned int space);
	/**
	 * mangle the revents of poll descriptors
	 */
	int (*poll_revents)(snd_ctl_ext_t *ext, struct pollfd *pfds, unsigned int nfds, unsigned short *revents);
};

/**
 * The access type bits stored in get_attribute callback
 */
typedef enum snd_ctl_ext_access {
	SND_CTL_EXT_ACCESS_READ = (1<<0),
	SND_CTL_EXT_ACCESS_WRITE = (1<<1),
	SND_CTL_EXT_ACCESS_READWRITE = (3<<0),
	SND_CTL_EXT_ACCESS_VOLATILE = (1<<2),
	SND_CTL_EXT_ACCESS_TLV_READ = (1<<4),
	SND_CTL_EXT_ACCESS_TLV_WRITE = (1<<5),
	SND_CTL_EXT_ACCESS_TLV_READWRITE = (3<<4),
	SND_CTL_EXT_ACCESS_TLV_COMMAND = (1<<6),
	SND_CTL_EXT_ACCESS_INACTIVE = (1<<8),
	SND_CTL_EXT_ACCESS_TLV_CALLBACK = (1<<28),
} snd_ctl_ext_access_t;

/**
 * find_elem callback returns this if no matching control element is found
 */
#define SND_CTL_EXT_KEY_NOT_FOUND	(snd_ctl_ext_key_t)(-1)

int snd_ctl_ext_create(snd_ctl_ext_t *ext, const char *name, int mode);
int snd_ctl_ext_delete(snd_ctl_ext_t *ext);

/** \} */

#ifdef __cplusplus
}
#endif

#endif /* __ALSA_CONTROL_EXTERNAL_H */
