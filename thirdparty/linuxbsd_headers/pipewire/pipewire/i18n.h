/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2021 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_I18N_H
#define PIPEWIRE_I18N_H

#ifdef __cplusplus
extern "C" {
#endif

/** \defgroup pw_gettext Internationalization
 * Gettext interface
 */

/**
 * \addtogroup pw_gettext
 * \{
 */
#include <spa/support/i18n.h>

SPA_FORMAT_ARG_FUNC(1) const char *pw_gettext(const char *msgid);
SPA_FORMAT_ARG_FUNC(1) const char *pw_ngettext(const char *msgid, const char *msgid_plural, unsigned long int n);

#define _(String)	(pw_gettext(String))
#define N_(String)	(String)

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_I18N_H */
