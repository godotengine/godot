/*
 * libusb strerror code
 * Copyright © 2013 Hans de Goede <hdegoede@redhat.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <config.h>

#include <locale.h>
#include <stdlib.h>
#include <string.h>
#if defined(HAVE_STRINGS_H)
#include <strings.h>
#endif

#include "libusbi.h"

#if defined(_MSC_VER)
#define strncasecmp _strnicmp
#endif

static size_t usbi_locale = 0;

/** \ingroup libusb_misc
 * How to add a new \ref libusb_strerror() translation:
 * <ol>
 * <li> Download the latest \c strerror.c from:<br>
 *      https://raw.github.com/libusb/libusb/master/libusb/sterror.c </li>
 * <li> Open the file in an UTF-8 capable editor </li>
 * <li> Add the 2 letter <a href="http://en.wikipedia.org/wiki/List_of_ISO_639-1_codes">ISO 639-1</a>
 *      code for your locale at the end of \c usbi_locale_supported[]<br>
 *    Eg. for Chinese, you would add "zh" so that:
 *    \code... usbi_locale_supported[] = { "en", "nl", "fr" };\endcode
 *    becomes:
 *    \code... usbi_locale_supported[] = { "en", "nl", "fr", "zh" };\endcode </li>
 * <li> Copy the <tt>{ / * English (en) * / ... }</tt> section and add it at the end of \c usbi_localized_errors<br>
 *    Eg. for Chinese, the last section of \c usbi_localized_errors could look like:
 *    \code
 *     }, { / * Chinese (zh) * /
 *         "Success",
 *         ...
 *         "Other error",
 *     }
 * };\endcode </li>
 * <li> Translate each of the English messages from the section you copied into your language </li>
 * <li> Save the file (in UTF-8 format) and send it to \c libusb-devel\@lists.sourceforge.net </li>
 * </ol>
 */

static const char* usbi_locale_supported[] = { "en", "nl", "fr", "ru" };
static const char* usbi_localized_errors[ARRAYSIZE(usbi_locale_supported)][LIBUSB_ERROR_COUNT] = {
	{ /* English (en) */
		"Success",
		"Input/Output Error",
		"Invalid parameter",
		"Access denied (insufficient permissions)",
		"No such device (it may have been disconnected)",
		"Entity not found",
		"Resource busy",
		"Operation timed out",
		"Overflow",
		"Pipe error",
		"System call interrupted (perhaps due to signal)",
		"Insufficient memory",
		"Operation not supported or unimplemented on this platform",
		"Other error",
	}, { /* Dutch (nl) */
		"Gelukt",
		"Invoer-/uitvoerfout",
		"Ongeldig argument",
		"Toegang geweigerd (onvoldoende toegangsrechten)",
		"Apparaat bestaat niet (verbinding met apparaat verbroken?)",
		"Niet gevonden",
		"Apparaat of hulpbron is bezig",
		"Bewerking verlopen",
		"Waarde is te groot",
		"Gebroken pijp",
		"Onderbroken systeemaanroep",
		"Onvoldoende geheugen beschikbaar",
		"Bewerking wordt niet ondersteund",
		"Andere fout",
	}, { /* French (fr) */
		"Succès",
		"Erreur d'entrée/sortie",
		"Paramètre invalide",
		"Accès refusé (permissions insuffisantes)",
		"Périphérique introuvable (peut-être déconnecté)",
		"Elément introuvable",
		"Resource déjà occupée",
		"Operation expirée",
		"Débordement",
		"Erreur de pipe",
		"Appel système abandonné (peut-être à cause d’un signal)",
		"Mémoire insuffisante",
		"Opération non supportée or non implémentée sur cette plateforme",
		"Autre erreur",
	}, { /* Russian (ru) */
		"Успех",
		"Ошибка ввода/вывода",
		"Неверный параметр",
		"Доступ запрещён (не хватает прав)",
		"Устройство отсутствует (возможно, оно было отсоединено)",
		"Элемент не найден",
		"Ресурс занят",
		"Истекло время ожидания операции",
		"Переполнение",
		"Ошибка канала",
		"Системный вызов прерван (возможно, сигналом)",
		"Память исчерпана",
		"Операция не поддерживается данной платформой",
		"Неизвестная ошибка"
	}
};

/** \ingroup libusb_misc
 * Set the language, and only the language, not the encoding! used for
 * translatable libusb messages.
 *
 * This takes a locale string in the default setlocale format: lang[-region]
 * or lang[_country_region][.codeset]. Only the lang part of the string is
 * used, and only 2 letter ISO 639-1 codes are accepted for it, such as "de".
 * The optional region, country_region or codeset parts are ignored. This
 * means that functions which return translatable strings will NOT honor the
 * specified encoding. 
 * All strings returned are encoded as UTF-8 strings.
 *
 * If libusb_setlocale() is not called, all messages will be in English.
 *
 * The following functions return translatable strings: libusb_strerror().
 * Note that the libusb log messages controlled through libusb_set_debug()
 * are not translated, they are always in English.
 *
 * For POSIX UTF-8 environments if you want libusb to follow the standard
 * locale settings, call libusb_setlocale(setlocale(LC_MESSAGES, NULL)),
 * after your app has done its locale setup.
 *
 * \param locale locale-string in the form of lang[_country_region][.codeset]
 * or lang[-region], where lang is a 2 letter ISO 639-1 code
 * \returns LIBUSB_SUCCESS on success
 * \returns LIBUSB_ERROR_INVALID_PARAM if the locale doesn't meet the requirements
 * \returns LIBUSB_ERROR_NOT_FOUND if the requested language is not supported
 * \returns a LIBUSB_ERROR code on other errors
 */

int API_EXPORTED libusb_setlocale(const char *locale)
{
	size_t i;

	if ( (locale == NULL) || (strlen(locale) < 2)
	  || ((strlen(locale) > 2) && (locale[2] != '-') && (locale[2] != '_') && (locale[2] != '.')) )
		return LIBUSB_ERROR_INVALID_PARAM;

	for (i=0; i<ARRAYSIZE(usbi_locale_supported); i++) {
		if (strncasecmp(usbi_locale_supported[i], locale, 2) == 0)
			break;
	}
	if (i >= ARRAYSIZE(usbi_locale_supported)) {
		return LIBUSB_ERROR_NOT_FOUND;
	}

	usbi_locale = i;

	return LIBUSB_SUCCESS;
}

/** \ingroup libusb_misc
 * Returns a constant string with a short description of the given error code,
 * this description is intended for displaying to the end user and will be in
 * the language set by libusb_setlocale().
 *
 * The returned string is encoded in UTF-8.
 *
 * The messages always start with a capital letter and end without any dot.
 * The caller must not free() the returned string.
 *
 * \param errcode the error code whose description is desired
 * \returns a short description of the error code in UTF-8 encoding
 */
DEFAULT_VISIBILITY const char* LIBUSB_CALL libusb_strerror(enum libusb_error errcode)
{
	int errcode_index = -errcode;

	if ((errcode_index < 0) || (errcode_index >= LIBUSB_ERROR_COUNT)) {
		/* "Other Error", which should always be our last message, is returned */
		errcode_index = LIBUSB_ERROR_COUNT - 1;
	}

	return usbi_localized_errors[usbi_locale][errcode_index];
}
