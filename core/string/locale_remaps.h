/**************************************************************************/
/*  locale_remaps.h                                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

// Windows has some weird locale identifiers which do not honor the ISO 639-1
// standardized nomenclature. Whenever those don't conflict with existing ISO
// identifiers, we override them.
//
// Reference:
// - https://msdn.microsoft.com/en-us/library/windows/desktop/ms693062(v=vs.85).aspx

static const char *locale_renames[][2] = {
	{ "in", "id" }, //  Indonesian
	{ "iw", "he" }, //  Hebrew
	{ "no", "nb" }, //  Norwegian Bokmål
	{ "C", "en" }, // Locale is not set, fallback to English.
	{ nullptr, nullptr }
};

// Additional script information to preferred scripts.
// Language code, script code, default country, supported countries.
// Reference:
// - https://lh.2xlibre.net/locales/
// - https://www.localeplanet.com/icu/index.html
// - https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-lcid/70feba9f-294e-491e-b6eb-56532684c37f

static const char *locale_scripts[][4] = {
	{ "az", "Latn", "", "AZ" },
	{ "az", "Arab", "", "IR" },
	{ "bs", "Latn", "", "BA" },
	{ "ff", "Latn", "", "BF,CM,GH,GM,GN,GW,LR,MR,NE,NG,SL,SN" },
	{ "pa", "Arab", "PK", "PK" },
	{ "pa", "Guru", "IN", "IN" },
	{ "sd", "Arab", "PK", "PK" },
	{ "sd", "Deva", "IN", "IN" },
	{ "shi", "Tfng", "", "MA" },
	{ "sr", "Cyrl", "", "BA,RS,XK" },
	{ "sr", "Latn", "", "ME" },
	{ "uz", "Latn", "", "UZ" },
	{ "uz", "Arab", "AF", "AF" },
	{ "vai", "Vaii", "", "LR" },
	{ "yue", "Hans", "CN", "CN" },
	{ "yue", "Hant", "HK", "HK" },
	{ "zh", "Hans", "CN", "CN,SG" },
	{ "zh", "Hant", "TW", "HK,MO,TW" },
	{ nullptr, nullptr, nullptr, nullptr }
};

// Additional mapping for outdated, temporary or exceptionally reserved country codes.
// Reference:
// - https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
// - https://www.iso.org/obp/ui/#search/code/

static const char *country_renames[][2] = {
	{ "BU", "MM" }, // Burma, name changed to Myanmar.
	{ "KV", "XK" }, // Kosovo (temporary FIPS code to European Commission code), no official ISO code assigned.
	{ "XKK", "XK" },
	{ "TP", "TL" }, // East Timor, name changed to Timor-Leste.
	{ "UK", "GB" }, // United Kingdom, exceptionally reserved code.
	{ nullptr, nullptr }
};

// Additional regional variants.
// Variant name, supported languages.

static const char *locale_variants[][2] = {
	{ "valencia", "ca" },
	{ "iqtelif", "tt" },
	{ "saaho", "aa" },
	{ "tradnl", "es" },
	{ nullptr, nullptr },
};
