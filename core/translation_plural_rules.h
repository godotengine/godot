/*************************************************************************/
/*  translation_plural_rules.h                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef TRANSLATION_PLURAL_RULES_H
#define TRANSLATION_PLURAL_RULES_H

#include "core/hash_map.h"
#include "core/translation.h"
#include "core/ustring.h"

class TranslationPluralRules {
public:
	static int get_plural_index(const String &p_locale, const int p_n) {
		if (plural_rule_mapping.has(p_locale)) {
			return (*plural_rule_mapping[p_locale].rule)(p_n);
		}

		String lang_code = TranslationServer::get_language_code(p_locale);
		if (plural_rule_mapping.has(lang_code)) {
			return (*plural_rule_mapping[lang_code].rule)(p_n);
		}

		ERR_FAIL_V_MSG(-1, "Cannot find locale \"" + p_locale + "\" in TranslationPluralRules. Please report this bug.");
	}

	static int get_plural_forms(const String &p_locale) {
		if (plural_rule_mapping.has(p_locale)) {
			return plural_rule_mapping[p_locale].forms;
		}

		String lang_code = TranslationServer::get_language_code(p_locale);
		if (plural_rule_mapping.has(lang_code)) {
			return plural_rule_mapping[lang_code].forms;
		}

		ERR_FAIL_V_MSG(-1, "Cannot find locale \"" + p_locale + "\" in TranslationPluralRules. Please report this bug.");
	}

private:
	typedef int (*PluralRule)(const int);

	struct PluralData {
		int forms;
		PluralRule rule;
	};

	static HashMap<String, PluralData> plural_rule_mapping;

	static const HashMap<String, PluralData> build_mapping() {
		HashMap<String, PluralData> temp;

		////////////////////////////////////////////////////////////////////////////////////////////
		// The rest of the file is generated via plural_rules_builder.py (DO NOT MODIFY THIS LINE).

		temp.set("af", { 2, &rule1 });
		temp.set("asa", { 2, &rule1 });
		temp.set("ast", { 2, &rule1 });
		temp.set("az", { 2, &rule1 });
		temp.set("bem", { 2, &rule1 });
		temp.set("bez", { 2, &rule1 });
		temp.set("bg", { 2, &rule1 });
		temp.set("brx", { 2, &rule1 });
		temp.set("ca", { 2, &rule1 });
		temp.set("ce", { 2, &rule1 });
		temp.set("cgg", { 2, &rule1 });
		temp.set("chr", { 2, &rule1 });
		temp.set("ckb", { 2, &rule1 });
		temp.set("da", { 2, &rule1 });
		temp.set("de", { 2, &rule1 });
		temp.set("dv", { 2, &rule1 });
		temp.set("ee", { 2, &rule1 });
		temp.set("el", { 2, &rule1 });
		temp.set("en", { 2, &rule1 });
		temp.set("eo", { 2, &rule1 });
		temp.set("es", { 2, &rule1 });
		temp.set("et", { 2, &rule1 });
		temp.set("eu", { 2, &rule1 });
		temp.set("fi", { 2, &rule1 });
		temp.set("fo", { 2, &rule1 });
		temp.set("fur", { 2, &rule1 });
		temp.set("fy", { 2, &rule1 });
		temp.set("gl", { 2, &rule1 });
		temp.set("gsw", { 2, &rule1 });
		temp.set("ha", { 2, &rule1 });
		temp.set("haw", { 2, &rule1 });
		temp.set("hu", { 2, &rule1 });
		temp.set("io", { 2, &rule1 });
		temp.set("it", { 2, &rule1 });
		temp.set("jgo", { 2, &rule1 });
		temp.set("ji", { 2, &rule1 });
		temp.set("jmc", { 2, &rule1 });
		temp.set("ka", { 2, &rule1 });
		temp.set("kaj", { 2, &rule1 });
		temp.set("kcg", { 2, &rule1 });
		temp.set("kk", { 2, &rule1 });
		temp.set("kkj", { 2, &rule1 });
		temp.set("kl", { 2, &rule1 });
		temp.set("ks", { 2, &rule1 });
		temp.set("ksb", { 2, &rule1 });
		temp.set("ku", { 2, &rule1 });
		temp.set("ky", { 2, &rule1 });
		temp.set("lb", { 2, &rule1 });
		temp.set("lg", { 2, &rule1 });
		temp.set("mas", { 2, &rule1 });
		temp.set("mgo", { 2, &rule1 });
		temp.set("ml", { 2, &rule1 });
		temp.set("mn", { 2, &rule1 });
		temp.set("nah", { 2, &rule1 });
		temp.set("nb", { 2, &rule1 });
		temp.set("nd", { 2, &rule1 });
		temp.set("ne", { 2, &rule1 });
		temp.set("nl", { 2, &rule1 });
		temp.set("nn", { 2, &rule1 });
		temp.set("nnh", { 2, &rule1 });
		temp.set("no", { 2, &rule1 });
		temp.set("nr", { 2, &rule1 });
		temp.set("ny", { 2, &rule1 });
		temp.set("nyn", { 2, &rule1 });
		temp.set("om", { 2, &rule1 });
		temp.set("or", { 2, &rule1 });
		temp.set("os", { 2, &rule1 });
		temp.set("pap", { 2, &rule1 });
		temp.set("ps", { 2, &rule1 });
		temp.set("pt_PT", { 2, &rule1 });
		temp.set("rm", { 2, &rule1 });
		temp.set("rof", { 2, &rule1 });
		temp.set("rwk", { 2, &rule1 });
		temp.set("saq", { 2, &rule1 });
		temp.set("scn", { 2, &rule1 });
		temp.set("sd", { 2, &rule1 });
		temp.set("sdh", { 2, &rule1 });
		temp.set("seh", { 2, &rule1 });
		temp.set("sn", { 2, &rule1 });
		temp.set("so", { 2, &rule1 });
		temp.set("sq", { 2, &rule1 });
		temp.set("ss", { 2, &rule1 });
		temp.set("ssy", { 2, &rule1 });
		temp.set("st", { 2, &rule1 });
		temp.set("sv", { 2, &rule1 });
		temp.set("sw", { 2, &rule1 });
		temp.set("syr", { 2, &rule1 });
		temp.set("ta", { 2, &rule1 });
		temp.set("te", { 2, &rule1 });
		temp.set("teo", { 2, &rule1 });
		temp.set("tig", { 2, &rule1 });
		temp.set("tk", { 2, &rule1 });
		temp.set("tn", { 2, &rule1 });
		temp.set("tr", { 2, &rule1 });
		temp.set("ts", { 2, &rule1 });
		temp.set("ug", { 2, &rule1 });
		temp.set("ur", { 2, &rule1 });
		temp.set("uz", { 2, &rule1 });
		temp.set("ve", { 2, &rule1 });
		temp.set("vo", { 2, &rule1 });
		temp.set("vun", { 2, &rule1 });
		temp.set("wae", { 2, &rule1 });
		temp.set("xh", { 2, &rule1 });
		temp.set("xog", { 2, &rule1 });
		temp.set("yi", { 2, &rule1 });
		temp.set("ak", { 2, &rule2 });
		temp.set("bh", { 2, &rule2 });
		temp.set("ff", { 2, &rule2 });
		temp.set("fr", { 2, &rule2 });
		temp.set("guw", { 2, &rule2 });
		temp.set("hy", { 2, &rule2 });
		temp.set("kab", { 2, &rule2 });
		temp.set("ln", { 2, &rule2 });
		temp.set("mg", { 2, &rule2 });
		temp.set("nso", { 2, &rule2 });
		temp.set("pa", { 2, &rule2 });
		temp.set("pt", { 2, &rule2 });
		temp.set("si", { 2, &rule2 });
		temp.set("ti", { 2, &rule2 });
		temp.set("wa", { 2, &rule2 });
		temp.set("am", { 2, &rule3 });
		temp.set("as", { 2, &rule3 });
		temp.set("bn", { 2, &rule3 });
		temp.set("fa", { 2, &rule3 });
		temp.set("gu", { 2, &rule3 });
		temp.set("hi", { 2, &rule3 });
		temp.set("kn", { 2, &rule3 });
		temp.set("mr", { 2, &rule3 });
		temp.set("zu", { 2, &rule3 });
		temp.set("ar", { 6, &rule4 });
		temp.set("ars", { 6, &rule4 });
		temp.set("be", { 3, &rule5 });
		temp.set("bs", { 3, &rule5 });
		temp.set("hr", { 3, &rule5 });
		temp.set("ru", { 3, &rule5 });
		temp.set("sh", { 3, &rule5 });
		temp.set("sr", { 3, &rule5 });
		temp.set("uk", { 3, &rule5 });
		temp.set("bm", { 1, &rule6 });
		temp.set("bo", { 1, &rule6 });
		temp.set("dz", { 1, &rule6 });
		temp.set("id", { 1, &rule6 });
		temp.set("ig", { 1, &rule6 });
		temp.set("ii", { 1, &rule6 });
		temp.set("in", { 1, &rule6 });
		temp.set("ja", { 1, &rule6 });
		temp.set("jbo", { 1, &rule6 });
		temp.set("jv", { 1, &rule6 });
		temp.set("jw", { 1, &rule6 });
		temp.set("kde", { 1, &rule6 });
		temp.set("kea", { 1, &rule6 });
		temp.set("km", { 1, &rule6 });
		temp.set("ko", { 1, &rule6 });
		temp.set("lkt", { 1, &rule6 });
		temp.set("lo", { 1, &rule6 });
		temp.set("ms", { 1, &rule6 });
		temp.set("my", { 1, &rule6 });
		temp.set("nqo", { 1, &rule6 });
		temp.set("root", { 1, &rule6 });
		temp.set("sah", { 1, &rule6 });
		temp.set("ses", { 1, &rule6 });
		temp.set("sg", { 1, &rule6 });
		temp.set("th", { 1, &rule6 });
		temp.set("to", { 1, &rule6 });
		temp.set("vi", { 1, &rule6 });
		temp.set("wo", { 1, &rule6 });
		temp.set("yo", { 1, &rule6 });
		temp.set("yue", { 1, &rule6 });
		temp.set("zh", { 1, &rule6 });
		temp.set("br", { 5, &rule7 });
		temp.set("cs", { 3, &rule8 });
		temp.set("sk", { 3, &rule8 });
		temp.set("cy", { 6, &rule9 });
		temp.set("dsb", { 4, &rule10 });
		temp.set("hsb", { 4, &rule10 });
		temp.set("sl", { 4, &rule10 });
		temp.set("fil", { 2, &rule11 });
		temp.set("tl", { 2, &rule11 });
		temp.set("ga", { 5, &rule12 });
		temp.set("gd", { 4, &rule13 });
		temp.set("gv", { 4, &rule14 });
		temp.set("he", { 4, &rule15 });
		temp.set("iw", { 4, &rule15 });
		temp.set("is", { 2, &rule16 });
		temp.set("mk", { 2, &rule16 });
		temp.set("iu", { 3, &rule17 });
		temp.set("kw", { 3, &rule17 });
		temp.set("naq", { 3, &rule17 });
		temp.set("se", { 3, &rule17 });
		temp.set("sma", { 3, &rule17 });
		temp.set("smi", { 3, &rule17 });
		temp.set("smj", { 3, &rule17 });
		temp.set("smn", { 3, &rule17 });
		temp.set("sms", { 3, &rule17 });
		temp.set("ksh", { 3, &rule18 });
		temp.set("lag", { 3, &rule19 });
		temp.set("lt", { 3, &rule20 });
		temp.set("lv", { 3, &rule21 });
		temp.set("prg", { 3, &rule21 });
		temp.set("mo", { 3, &rule22 });
		temp.set("ro", { 3, &rule22 });
		temp.set("mt", { 4, &rule23 });
		temp.set("pl", { 3, &rule24 });
		temp.set("shi", { 3, &rule25 });
		temp.set("tzm", { 2, &rule26 });
		return temp;
	}

	static int rule1(const int p_n) {
		// af, asa, ast, az, bem, bez, bg, brx, ca, ce, cgg, chr, ckb, da, de, dv, ee, el, en, eo, es, et, eu, fi, fo, fur, fy, gl, gsw, ha, haw, hu, io, it, jgo, ji, jmc, ka, kaj, kcg, kk, kkj, kl, ks, ksb, ku, ky, lb, lg, mas, mgo, ml, mn, nah, nb, nd, ne, nl, nn, nnh, no, nr, ny, nyn, om, or, os, pap, ps, pt_PT, rm, rof, rwk, saq, scn, sd, sdh, seh, sn, so, sq, ss, ssy, st, sv, sw, syr, ta, te, teo, tig, tk, tn, tr, ts, ug, ur, uz, ve, vo, vun, wae, xh, xog, yi.
		return (p_n != 1);
	}

	static int rule2(const int p_n) {
		// ak, bh, ff, fr, guw, hy, kab, ln, mg, nso, pa, pt, si, ti, wa.
		return (p_n > 1);
	}

	static int rule3(const int p_n) {
		// am, as, bn, fa, gu, hi, kn, mr, zu.
		return (p_n == 0 || p_n == 1);
	}

	static int rule4(const int p_n) {
		// ar, ars.
		return (p_n == 0 ? 0 : p_n == 1 ? 1 : p_n == 2 ? 2 : p_n % 100 >= 3 && p_n % 100 <= 10 ? 3 : p_n % 100 >= 11 && p_n % 100 <= 99 ? 4 : 5);
	}

	static int rule5(const int p_n) {
		// be, bs, hr, ru, sh, sr, uk.
		return (p_n % 10 == 1 && p_n % 100 != 11 ? 0 : p_n % 10 >= 2 && p_n % 10 <= 4 && (p_n % 100 < 12 || p_n % 100 > 14) ? 1 : 2);
	}

	static int rule6(const int p_n) {
		// bm, bo, dz, id, ig, ii, in, ja, jbo, jv, jw, kde, kea, km, ko, lkt, lo, ms, my, nqo, root, sah, ses, sg, th, to, vi, wo, yo, yue, zh.
		return 0;
	}

	static int rule7(const int p_n) {
		// br.
		return (p_n % 10 == 1 && p_n % 100 != 11 && p_n % 100 != 71 && p_n % 100 != 91 ? 0 : p_n % 10 == 2 && p_n % 100 != 12 && p_n % 100 != 72 && p_n % 100 != 92 ? 1 : ((p_n % 10 >= 3 && p_n % 10 <= 4) || p_n % 10 == 9) && (p_n % 100 < 10 || p_n % 100 > 19) && (p_n % 100 < 70 || p_n % 100 > 79) && (p_n % 100 < 90 || p_n % 100 > 99) ? 2 : p_n != 0 && p_n % 1000000 == 0 ? 3 : 4);
	}

	static int rule8(const int p_n) {
		// cs, sk.
		return (p_n == 1 ? 0 : p_n >= 2 && p_n <= 4 ? 1 : 2);
	}

	static int rule9(const int p_n) {
		// cy.
		return (p_n == 0 ? 0 : p_n == 1 ? 1 : p_n == 2 ? 2 : p_n == 3 ? 3 : p_n == 6 ? 4 : 5);
	}

	static int rule10(const int p_n) {
		// dsb, hsb, sl.
		return (p_n % 100 == 1 ? 0 : p_n % 100 == 2 ? 1 : p_n % 100 >= 3 && p_n % 100 <= 4 ? 2 : 3);
	}

	static int rule11(const int p_n) {
		// fil, tl.
		return (p_n == 1 || p_n == 2 || p_n == 3 || (p_n % 10 != 4 && p_n % 10 != 6 && p_n % 10 != 9));
	}

	static int rule12(const int p_n) {
		// ga.
		return (p_n == 1 ? 0 : p_n == 2 ? 1 : p_n >= 3 && p_n <= 6 ? 2 : p_n >= 7 && p_n <= 10 ? 3 : 4);
	}

	static int rule13(const int p_n) {
		// gd.
		return (p_n == 1 || p_n == 11 ? 0 : p_n == 2 || p_n == 12 ? 1 : (p_n >= 3 && p_n <= 10) || (p_n >= 13 && p_n <= 19) ? 2 : 3);
	}

	static int rule14(const int p_n) {
		// gv.
		return (p_n % 10 == 1 ? 0 : p_n % 10 == 2 ? 1 : p_n % 100 == 0 || p_n % 100 == 20 || p_n % 100 == 40 || p_n % 100 == 60 || p_n % 100 == 80 ? 2 : 3);
	}

	static int rule15(const int p_n) {
		// he, iw.
		return (p_n == 1 ? 0 : p_n == 2 ? 1 : p_n > 10 && p_n % 10 == 0 ? 2 : 3);
	}

	static int rule16(const int p_n) {
		// is, mk.
		return (p_n % 10 == 1 && p_n % 100 != 11);
	}

	static int rule17(const int p_n) {
		// iu, kw, naq, se, sma, smi, smj, smn, sms.
		return (p_n == 1 ? 0 : p_n == 2 ? 1 : 2);
	}

	static int rule18(const int p_n) {
		// ksh.
		return (p_n == 0 ? 0 : p_n == 1 ? 1 : 2);
	}

	static int rule19(const int p_n) {
		// lag.
		return (p_n == 0 ? 0 : (p_n == 0 || p_n == 1) && p_n != 0 ? 1 : 2);
	}

	static int rule20(const int p_n) {
		// lt.
		return (p_n % 10 == 1 && (p_n % 100 < 11 || p_n % 100 > 19) ? 0 : p_n % 10 >= 2 && p_n % 10 <= 9 && (p_n % 100 < 11 || p_n % 100 > 19) ? 1 : 2);
	}

	static int rule21(const int p_n) {
		// lv, prg.
		return (p_n % 10 == 0 || (p_n % 100 >= 11 && p_n % 100 <= 19) ? 0 : p_n % 10 == 1 && p_n % 100 != 11 ? 1 : 2);
	}

	static int rule22(const int p_n) {
		// mo, ro.
		return (p_n == 1 ? 0 : p_n == 0 || (p_n != 1 && p_n % 100 >= 1 && p_n % 100 <= 19) ? 1 : 2);
	}

	static int rule23(const int p_n) {
		// mt.
		return (p_n == 1 ? 0 : p_n == 0 || (p_n % 100 >= 2 && p_n % 100 <= 10) ? 1 : p_n % 100 >= 11 && p_n % 100 <= 19 ? 2 : 3);
	}

	static int rule24(const int p_n) {
		// pl.
		return (p_n == 1 ? 0 : p_n % 10 >= 2 && p_n % 10 <= 4 && (p_n % 100 < 12 || p_n % 100 > 14) ? 1 : 2);
	}

	static int rule25(const int p_n) {
		// shi.
		return (p_n == 0 || p_n == 1 ? 0 : p_n >= 2 && p_n <= 10 ? 1 : 2);
	}

	static int rule26(const int p_n) {
		// tzm.
		return (p_n <= 1 || (p_n >= 11 && p_n <= 99));
	}
};

#endif // TRANSLATION_PLURAL_RULES_H
