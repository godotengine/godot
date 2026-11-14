/**************************************************************************/
/*  xkb_convert_dead_keys.cpp                                             */
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

#include "xkb_convert_dead_keys.h"

#include "xkbcommon/xkbcommon-keysyms.h"
#include "xkbcommon/xkbcommon.h"

xkb_keysym_t xkb_convert_if_dead_key(xkb_keysym_t p_keysym) {
	switch (p_keysym) {
		case XKB_KEY_dead_grave: // not tested
			p_keysym = XKB_KEY_grave;
			break;
		case XKB_KEY_dead_acute:
			p_keysym = XKB_KEY_acute;
			break;
		case XKB_KEY_dead_circumflex:
			p_keysym = XKB_KEY_asciicircum;
			break;

		// from here not yet tested
		case XKB_KEY_dead_tilde: /*XKB_KEY_dead_perispomeni*/
			p_keysym = XKB_KEY_asciitilde;
			break;
		case XKB_KEY_dead_macron:
			p_keysym = XKB_KEY_macron;
			break;
		case XKB_KEY_dead_breve:
			p_keysym = XKB_KEY_breve;
			break;
		case XKB_KEY_dead_abovedot:
			p_keysym = XKB_KEY_abovedot;
			break;
		case XKB_KEY_dead_diaeresis:
			p_keysym = XKB_KEY_diaeresis;
			break;
		case XKB_KEY_dead_abovering:
			// haven't found non dead equivalent
			break;
		case XKB_KEY_dead_doubleacute:
			p_keysym = XKB_KEY_doubleacute;
			break;
		case XKB_KEY_dead_caron:
			p_keysym = XKB_KEY_caron;
			break;
		case XKB_KEY_dead_cedilla:
			p_keysym = XKB_KEY_cedilla;
			break;
		case XKB_KEY_dead_ogonek:
			p_keysym = XKB_KEY_ogonek;
			break;
		case XKB_KEY_dead_iota:
			p_keysym = XKB_KEY_Greek_iota;
			break;
		case XKB_KEY_dead_voiced_sound:
			p_keysym = XKB_KEY_voicedsound;
			break;
		case XKB_KEY_dead_semivoiced_sound:
			p_keysym = XKB_KEY_semivoicedsound;
			break;
		case XKB_KEY_dead_belowdot:
			//
			break;
		case XKB_KEY_dead_hook:
			//
			break;
		case XKB_KEY_dead_horn:
			//
			break;
		case XKB_KEY_dead_stroke:
			//
			break;
		case XKB_KEY_dead_abovecomma: /*XKB_KEY_dead_psili*/
			//
			break;
		case XKB_KEY_dead_abovereversedcomma: /*XKB_KEY_dead_dasia*/
			//
			break;
		case XKB_KEY_dead_doublegrave:
			//
			break;
		case XKB_KEY_dead_belowring:
			//
			break;
		case XKB_KEY_dead_belowmacron:
			//
			break;
		case XKB_KEY_dead_belowcircumflex:
			//
			break;
		case XKB_KEY_dead_belowtilde:
			//
			break;
		case XKB_KEY_dead_belowbreve:
			//
			break;
		case XKB_KEY_dead_belowdiaeresis:
			//
			break;
		case XKB_KEY_dead_invertedbreve:
			//
			break;
		case XKB_KEY_dead_belowcomma:
			//
			break;
		case XKB_KEY_dead_currency:
			p_keysym = XKB_KEY_currency;
			break;

		/* extra dead elements for German T3 layout */
		case XKB_KEY_dead_lowline:
			//xkb_keysym = XKB_KEY_underscore;
			//xkb_keysym = XKB_KEY_underbar;
			break;
		case XKB_KEY_dead_aboveverticalline:
			//
			break;
		case XKB_KEY_dead_belowverticalline:
			//
			break;
		case XKB_KEY_dead_longsolidusoverlay:
			//
			break;

		/* dead vowels for universal syllable entry */
		case XKB_KEY_dead_a:
			p_keysym = XKB_KEY_a;
			break;
		case XKB_KEY_dead_A:
			p_keysym = XKB_KEY_A;
			break;
		case XKB_KEY_dead_e:
			p_keysym = XKB_KEY_e;
			break;
		case XKB_KEY_dead_E:
			p_keysym = XKB_KEY_E;
			break;
		case XKB_KEY_dead_i:
			p_keysym = XKB_KEY_i;
			break;
		case XKB_KEY_dead_I:
			p_keysym = XKB_KEY_I;
			break;
		case XKB_KEY_dead_o:
			p_keysym = XKB_KEY_o;
			break;
		case XKB_KEY_dead_O:
			p_keysym = XKB_KEY_O;
			break;
		case XKB_KEY_dead_u:
			p_keysym = XKB_KEY_u;
			break;
		case XKB_KEY_dead_U:
			p_keysym = XKB_KEY_U;
			break;
		case XKB_KEY_dead_small_schwa:
			//
			break;
		case XKB_KEY_dead_capital_schwa:
			//
			break;
		case XKB_KEY_dead_greek:
			//
			break;
	}

	return p_keysym;
}
