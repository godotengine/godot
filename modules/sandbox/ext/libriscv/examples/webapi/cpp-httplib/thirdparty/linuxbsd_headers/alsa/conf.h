/**
 * \file include/conf.h
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

#ifndef __ALSA_CONF_H
#define __ALSA_CONF_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  \defgroup Config Configuration Interface
 *  The configuration functions and types allow you to read, enumerate,
 *  modify and write the contents of ALSA configuration files.
 *  \{
 */

/** \brief \c dlsym version for the config evaluate callback. */
#define SND_CONFIG_DLSYM_VERSION_EVALUATE	_dlsym_config_evaluate_001
/** \brief \c dlsym version for the config hook callback. */
#define SND_CONFIG_DLSYM_VERSION_HOOK		_dlsym_config_hook_001

/** \brief Configuration node type. */
typedef enum _snd_config_type {
	/** Integer number. */
        SND_CONFIG_TYPE_INTEGER,
	/** 64-bit integer number. */
        SND_CONFIG_TYPE_INTEGER64,
	/** Real number. */
        SND_CONFIG_TYPE_REAL,
	/** Character string. */
        SND_CONFIG_TYPE_STRING,
        /** Pointer (runtime only, cannot be saved). */
        SND_CONFIG_TYPE_POINTER,
	/** Compound node. */
	SND_CONFIG_TYPE_COMPOUND = 1024
} snd_config_type_t;

/**
 * \brief Internal structure for a configuration node object.
 *
 * The ALSA library uses a pointer to this structure as a handle to a
 * configuration node. Applications don't access its contents directly.
 */
typedef struct _snd_config snd_config_t;
/**
 * \brief Type for a configuration compound iterator.
 *
 * The ALSA library uses this pointer type as a handle to a configuration
 * compound iterator. Applications don't directly access the contents of
 * the structure pointed to by this type.
 */
typedef struct _snd_config_iterator *snd_config_iterator_t;
/**
 * \brief Internal structure for a configuration private update object.
 *
 * The ALSA library uses this structure to save private update information.
 */
typedef struct _snd_config_update snd_config_update_t;

extern snd_config_t *snd_config;

int snd_config_top(snd_config_t **config);

int snd_config_load(snd_config_t *config, snd_input_t *in);
int snd_config_load_override(snd_config_t *config, snd_input_t *in);
int snd_config_save(snd_config_t *config, snd_output_t *out);
int snd_config_update(void);
int snd_config_update_r(snd_config_t **top, snd_config_update_t **update, const char *path);
int snd_config_update_free(snd_config_update_t *update);
int snd_config_update_free_global(void);

int snd_config_update_ref(snd_config_t **top);
void snd_config_ref(snd_config_t *top);
void snd_config_unref(snd_config_t *top);

int snd_config_search(snd_config_t *config, const char *key,
		      snd_config_t **result);
int snd_config_searchv(snd_config_t *config, 
		       snd_config_t **result, ...);
int snd_config_search_definition(snd_config_t *config,
				 const char *base, const char *key,
				 snd_config_t **result);

int snd_config_expand(snd_config_t *config, snd_config_t *root,
		      const char *args, snd_config_t *private_data,
		      snd_config_t **result);
int snd_config_evaluate(snd_config_t *config, snd_config_t *root,
			snd_config_t *private_data, snd_config_t **result);

int snd_config_add(snd_config_t *config, snd_config_t *leaf);
int snd_config_delete(snd_config_t *config);
int snd_config_delete_compound_members(const snd_config_t *config);
int snd_config_copy(snd_config_t **dst, snd_config_t *src);

int snd_config_make(snd_config_t **config, const char *key,
		    snd_config_type_t type);
int snd_config_make_integer(snd_config_t **config, const char *key);
int snd_config_make_integer64(snd_config_t **config, const char *key);
int snd_config_make_real(snd_config_t **config, const char *key);
int snd_config_make_string(snd_config_t **config, const char *key);
int snd_config_make_pointer(snd_config_t **config, const char *key);
int snd_config_make_compound(snd_config_t **config, const char *key, int join);

int snd_config_imake_integer(snd_config_t **config, const char *key, const long value);
int snd_config_imake_integer64(snd_config_t **config, const char *key, const long long value);
int snd_config_imake_real(snd_config_t **config, const char *key, const double value);
int snd_config_imake_string(snd_config_t **config, const char *key, const char *ascii);
int snd_config_imake_safe_string(snd_config_t **config, const char *key, const char *ascii);
int snd_config_imake_pointer(snd_config_t **config, const char *key, const void *ptr);

snd_config_type_t snd_config_get_type(const snd_config_t *config);

int snd_config_set_id(snd_config_t *config, const char *id);
int snd_config_set_integer(snd_config_t *config, long value);
int snd_config_set_integer64(snd_config_t *config, long long value);
int snd_config_set_real(snd_config_t *config, double value);
int snd_config_set_string(snd_config_t *config, const char *value);
int snd_config_set_ascii(snd_config_t *config, const char *ascii);
int snd_config_set_pointer(snd_config_t *config, const void *ptr);
int snd_config_get_id(const snd_config_t *config, const char **value);
int snd_config_get_integer(const snd_config_t *config, long *value);
int snd_config_get_integer64(const snd_config_t *config, long long *value);
int snd_config_get_real(const snd_config_t *config, double *value);
int snd_config_get_ireal(const snd_config_t *config, double *value);
int snd_config_get_string(const snd_config_t *config, const char **value);
int snd_config_get_ascii(const snd_config_t *config, char **value);
int snd_config_get_pointer(const snd_config_t *config, const void **value);
int snd_config_test_id(const snd_config_t *config, const char *id);

snd_config_iterator_t snd_config_iterator_first(const snd_config_t *node);
snd_config_iterator_t snd_config_iterator_next(const snd_config_iterator_t iterator);
snd_config_iterator_t snd_config_iterator_end(const snd_config_t *node);
snd_config_t *snd_config_iterator_entry(const snd_config_iterator_t iterator);

/**
 * \brief Helper macro to iterate over the children of a compound node.
 * \param[in,out] pos Iterator variable for the current node.
 * \param[in,out] next Temporary iterator variable for the next node.
 * \param[in] node Handle to the compound configuration node to iterate over.
 *
 * Use this macro like a \c for statement, e.g.:
 * \code
 * snd_config_iterator_t pos, next;
 * snd_config_for_each(pos, next, node) {
 *     snd_config_t *entry = snd_config_iterator_entry(pos);
 *     ...
 * }
 * \endcode
 *
 * This macro allows deleting or removing the current node.
 */
#define snd_config_for_each(pos, next, node) \
	for (pos = snd_config_iterator_first(node), next = snd_config_iterator_next(pos); pos != snd_config_iterator_end(node); pos = next, next = snd_config_iterator_next(pos))

/* Misc functions */

int snd_config_get_bool_ascii(const char *ascii);
int snd_config_get_bool(const snd_config_t *conf);
int snd_config_get_ctl_iface_ascii(const char *ascii);
int snd_config_get_ctl_iface(const snd_config_t *conf);

/* Names functions */

/**
 * Device-name list element
 */
typedef struct snd_devname snd_devname_t;

/**
 * Device-name list element (definition)
 */
struct snd_devname {
	char *name;	/**< Device name string */
	char *comment;	/**< Comments */
	snd_devname_t *next;	/**< Next pointer */
};

int snd_names_list(const char *iface, snd_devname_t **list);
void snd_names_list_free(snd_devname_t *list);

/** \} */

#ifdef __cplusplus
}
#endif

#endif /* __ALSA_CONF_H */
