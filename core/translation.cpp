/*************************************************************************/
/*  translation.cpp                                                      */
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

#include "translation.h"

#include "core/io/resource_loader.h"
#include "core/os/os.h"
#include "core/project_settings.h"

// ISO 639-1 language codes, with the addition of glibc locales with their
// regional identifiers. This list must match the language names (in English)
// of locale_names.
//
// References:
// - https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
// - https://lh.2xlibre.net/locales/

#ifdef DEBUG_TRANSLATION
void Translation::print_translation_map() {
	Error err;
	FileAccess *file = FileAccess::open("translation_map_print_test.txt", FileAccess::WRITE, &err);
	if (err != OK) {
		ERR_PRINT("Failed to open translation_map_print_test.txt");
		return;
	}

	file->store_line("NPlural : " + String::num_int64(this->get_plural_forms()));
	file->store_line("Plural rule : " + this->get_plural_rule());
	file->store_line("");

	List<StringName> context_l;
	translation_map.get_key_list(&context_l);
	for (auto E = context_l.front(); E; E = E->next()) {
		StringName ctx = E->get();
		file->store_line(" ===== Context: " + String::utf8(String(ctx).utf8()) + " ===== ");
		const HashMap<StringName, Vector<StringName>> &inner_map = translation_map[ctx];

		List<StringName> id_l;
		inner_map.get_key_list(&id_l);
		for (auto E2 = id_l.front(); E2; E2 = E2->next()) {
			StringName id = E2->get();
			file->store_line("msgid: " + String::utf8(String(id).utf8()));
			for (int i = 0; i < inner_map[id].size(); i++) {
				file->store_line("msgstr[" + String::num_int64(i) + "]: " + String::utf8(String(inner_map[id][i]).utf8()));
			}
			file->store_line("");
		}
	}
	file->close();
}
#endif

static const char *locale_list[] = {
	"aa", //  Afar
	"aa_DJ", //  Afar (Djibouti)
	"aa_ER", //  Afar (Eritrea)
	"aa_ET", //  Afar (Ethiopia)
	"af", //  Afrikaans
	"af_ZA", //  Afrikaans (South Africa)
	"agr_PE", //  Aguaruna (Peru)
	"ak_GH", //  Akan (Ghana)
	"am_ET", //  Amharic (Ethiopia)
	"an_ES", //  Aragonese (Spain)
	"anp_IN", //  Angika (India)
	"ar", //  Arabic
	"ar_AE", //  Arabic (United Arab Emirates)
	"ar_BH", //  Arabic (Bahrain)
	"ar_DZ", //  Arabic (Algeria)
	"ar_EG", //  Arabic (Egypt)
	"ar_IN", //  Arabic (India)
	"ar_IQ", //  Arabic (Iraq)
	"ar_JO", //  Arabic (Jordan)
	"ar_KW", //  Arabic (Kuwait)
	"ar_LB", //  Arabic (Lebanon)
	"ar_LY", //  Arabic (Libya)
	"ar_MA", //  Arabic (Morocco)
	"ar_OM", //  Arabic (Oman)
	"ar_QA", //  Arabic (Qatar)
	"ar_SA", //  Arabic (Saudi Arabia)
	"ar_SD", //  Arabic (Sudan)
	"ar_SS", //  Arabic (South Soudan)
	"ar_SY", //  Arabic (Syria)
	"ar_TN", //  Arabic (Tunisia)
	"ar_YE", //  Arabic (Yemen)
	"as_IN", //  Assamese (India)
	"ast_ES", //  Asturian (Spain)
	"ayc_PE", //  Southern Aymara (Peru)
	"ay_PE", //  Aymara (Peru)
	"az_AZ", //  Azerbaijani (Azerbaijan)
	"be", //  Belarusian
	"be_BY", //  Belarusian (Belarus)
	"bem_ZM", //  Bemba (Zambia)
	"ber_DZ", //  Berber languages (Algeria)
	"ber_MA", //  Berber languages (Morocco)
	"bg", //  Bulgarian
	"bg_BG", //  Bulgarian (Bulgaria)
	"bhb_IN", //  Bhili (India)
	"bho_IN", //  Bhojpuri (India)
	"bi_TV", //  Bislama (Tuvalu)
	"bn", //  Bengali
	"bn_BD", //  Bengali (Bangladesh)
	"bn_IN", //  Bengali (India)
	"bo", //  Tibetan
	"bo_CN", //  Tibetan (China)
	"bo_IN", //  Tibetan (India)
	"br_FR", //  Breton (France)
	"brx_IN", //  Bodo (India)
	"bs_BA", //  Bosnian (Bosnia and Herzegovina)
	"byn_ER", //  Bilin (Eritrea)
	"ca", //  Catalan
	"ca_AD", //  Catalan (Andorra)
	"ca_ES", //  Catalan (Spain)
	"ca_FR", //  Catalan (France)
	"ca_IT", //  Catalan (Italy)
	"ce_RU", //  Chechen (Russia)
	"chr_US", //  Cherokee (United States)
	"cmn_TW", //  Mandarin Chinese (Taiwan)
	"crh_UA", //  Crimean Tatar (Ukraine)
	"csb_PL", //  Kashubian (Poland)
	"cs", //  Czech
	"cs_CZ", //  Czech (Czech Republic)
	"cv_RU", //  Chuvash (Russia)
	"cy_GB", //  Welsh (United Kingdom)
	"da", //  Danish
	"da_DK", //  Danish (Denmark)
	"de", //  German
	"de_AT", //  German (Austria)
	"de_BE", //  German (Belgium)
	"de_CH", //  German (Switzerland)
	"de_DE", //  German (Germany)
	"de_IT", //  German (Italy)
	"de_LU", //  German (Luxembourg)
	"doi_IN", //  Dogri (India)
	"dv_MV", //  Dhivehi (Maldives)
	"dz_BT", //  Dzongkha (Bhutan)
	"el", //  Greek
	"el_CY", //  Greek (Cyprus)
	"el_GR", //  Greek (Greece)
	"en", //  English
	"en_AG", //  English (Antigua and Barbuda)
	"en_AU", //  English (Australia)
	"en_BW", //  English (Botswana)
	"en_CA", //  English (Canada)
	"en_DK", //  English (Denmark)
	"en_GB", //  English (United Kingdom)
	"en_HK", //  English (Hong Kong)
	"en_IE", //  English (Ireland)
	"en_IL", //  English (Israel)
	"en_IN", //  English (India)
	"en_NG", //  English (Nigeria)
	"en_NZ", //  English (New Zealand)
	"en_PH", //  English (Philippines)
	"en_SG", //  English (Singapore)
	"en_US", //  English (United States)
	"en_ZA", //  English (South Africa)
	"en_ZM", //  English (Zambia)
	"en_ZW", //  English (Zimbabwe)
	"eo", //  Esperanto
	"es", //  Spanish
	"es_AR", //  Spanish (Argentina)
	"es_BO", //  Spanish (Bolivia)
	"es_CL", //  Spanish (Chile)
	"es_CO", //  Spanish (Colombia)
	"es_CR", //  Spanish (Costa Rica)
	"es_CU", //  Spanish (Cuba)
	"es_DO", //  Spanish (Dominican Republic)
	"es_EC", //  Spanish (Ecuador)
	"es_ES", //  Spanish (Spain)
	"es_GT", //  Spanish (Guatemala)
	"es_HN", //  Spanish (Honduras)
	"es_MX", //  Spanish (Mexico)
	"es_NI", //  Spanish (Nicaragua)
	"es_PA", //  Spanish (Panama)
	"es_PE", //  Spanish (Peru)
	"es_PR", //  Spanish (Puerto Rico)
	"es_PY", //  Spanish (Paraguay)
	"es_SV", //  Spanish (El Salvador)
	"es_US", //  Spanish (United States)
	"es_UY", //  Spanish (Uruguay)
	"es_VE", //  Spanish (Venezuela)
	"et", //  Estonian
	"et_EE", //  Estonian (Estonia)
	"eu", //  Basque
	"eu_ES", //  Basque (Spain)
	"fa", //  Persian
	"fa_IR", //  Persian (Iran)
	"ff_SN", //  Fulah (Senegal)
	"fi", //  Finnish
	"fi_FI", //  Finnish (Finland)
	"fil", //  Filipino
	"fil_PH", //  Filipino (Philippines)
	"fo_FO", //  Faroese (Faroe Islands)
	"fr", //  French
	"fr_BE", //  French (Belgium)
	"fr_CA", //  French (Canada)
	"fr_CH", //  French (Switzerland)
	"fr_FR", //  French (France)
	"fr_LU", //  French (Luxembourg)
	"fur_IT", //  Friulian (Italy)
	"fy_DE", //  Western Frisian (Germany)
	"fy_NL", //  Western Frisian (Netherlands)
	"ga", //  Irish
	"ga_IE", //  Irish (Ireland)
	"gd_GB", //  Scottish Gaelic (United Kingdom)
	"gez_ER", //  Geez (Eritrea)
	"gez_ET", //  Geez (Ethiopia)
	"gl_ES", //  Galician (Spain)
	"gu_IN", //  Gujarati (India)
	"gv_GB", //  Manx (United Kingdom)
	"hak_TW", //  Hakka Chinese (Taiwan)
	"ha_NG", //  Hausa (Nigeria)
	"he", //  Hebrew
	"he_IL", //  Hebrew (Israel)
	"hi", //  Hindi
	"hi_IN", //  Hindi (India)
	"hne_IN", //  Chhattisgarhi (India)
	"hr", //  Croatian
	"hr_HR", //  Croatian (Croatia)
	"hsb_DE", //  Upper Sorbian (Germany)
	"ht_HT", //  Haitian (Haiti)
	"hu", //  Hungarian
	"hu_HU", //  Hungarian (Hungary)
	"hus_MX", //  Huastec (Mexico)
	"hy_AM", //  Armenian (Armenia)
	"ia_FR", //  Interlingua (France)
	"id", //  Indonesian
	"id_ID", //  Indonesian (Indonesia)
	"ig_NG", //  Igbo (Nigeria)
	"ik_CA", //  Inupiaq (Canada)
	"is", //  Icelandic
	"is_IS", //  Icelandic (Iceland)
	"it", //  Italian
	"it_CH", //  Italian (Switzerland)
	"it_IT", //  Italian (Italy)
	"iu_CA", //  Inuktitut (Canada)
	"ja", //  Japanese
	"ja_JP", //  Japanese (Japan)
	"kab_DZ", //  Kabyle (Algeria)
	"ka", //  Georgian
	"ka_GE", //  Georgian (Georgia)
	"kk_KZ", //  Kazakh (Kazakhstan)
	"kl_GL", //  Kalaallisut (Greenland)
	"km_KH", //  Central Khmer (Cambodia)
	"kn_IN", //  Kannada (India)
	"kok_IN", //  Konkani (India)
	"ko", //  Korean
	"ko_KR", //  Korean (South Korea)
	"ks_IN", //  Kashmiri (India)
	"ku", //  Kurdish
	"ku_TR", //  Kurdish (Turkey)
	"kw_GB", //  Cornish (United Kingdom)
	"ky_KG", //  Kirghiz (Kyrgyzstan)
	"lb_LU", //  Luxembourgish (Luxembourg)
	"lg_UG", //  Ganda (Uganda)
	"li_BE", //  Limburgan (Belgium)
	"li_NL", //  Limburgan (Netherlands)
	"lij_IT", //  Ligurian (Italy)
	"ln_CD", //  Lingala (Congo)
	"lo_LA", //  Lao (Laos)
	"lt", //  Lithuanian
	"lt_LT", //  Lithuanian (Lithuania)
	"lv", //  Latvian
	"lv_LV", //  Latvian (Latvia)
	"lzh_TW", //  Literary Chinese (Taiwan)
	"mag_IN", //  Magahi (India)
	"mai_IN", //  Maithili (India)
	"mg_MG", //  Malagasy (Madagascar)
	"mh_MH", //  Marshallese (Marshall Islands)
	"mhr_RU", //  Eastern Mari (Russia)
	"mi", //  Māori
	"mi_NZ", //  Māori (New Zealand)
	"miq_NI", //  Mískito (Nicaragua)
	"mk", //  Macedonian
	"mk_MK", //  Macedonian (Macedonia)
	"ml", //  Malayalam
	"ml_IN", //  Malayalam (India)
	"mni_IN", //  Manipuri (India)
	"mn_MN", //  Mongolian (Mongolia)
	"mr_IN", //  Marathi (India)
	"ms", //  Malay
	"ms_MY", //  Malay (Malaysia)
	"mt", //  Maltese
	"mt_MT", //  Maltese (Malta)
	"my_MM", //  Burmese (Myanmar)
	"myv_RU", //  Erzya (Russia)
	"nah_MX", //  Nahuatl languages (Mexico)
	"nan_TW", //  Min Nan Chinese (Taiwan)
	"nb", //  Norwegian Bokmål
	"nb_NO", //  Norwegian Bokmål (Norway)
	"nds_DE", //  Low German (Germany)
	"nds_NL", //  Low German (Netherlands)
	"ne_NP", //  Nepali (Nepal)
	"nhn_MX", //  Central Nahuatl (Mexico)
	"niu_NU", //  Niuean (Niue)
	"niu_NZ", //  Niuean (New Zealand)
	"nl", //  Dutch
	"nl_AW", //  Dutch (Aruba)
	"nl_BE", //  Dutch (Belgium)
	"nl_NL", //  Dutch (Netherlands)
	"nn", //  Norwegian Nynorsk
	"nn_NO", //  Norwegian Nynorsk (Norway)
	"nr_ZA", //  South Ndebele (South Africa)
	"nso_ZA", //  Pedi (South Africa)
	"oc_FR", //  Occitan (France)
	"om", //  Oromo
	"om_ET", //  Oromo (Ethiopia)
	"om_KE", //  Oromo (Kenya)
	"or_IN", //  Oriya (India)
	"os_RU", //  Ossetian (Russia)
	"pa_IN", //  Panjabi (India)
	"pap", //  Papiamento
	"pap_AN", //  Papiamento (Netherlands Antilles)
	"pap_AW", //  Papiamento (Aruba)
	"pap_CW", //  Papiamento (Curaçao)
	"pa_PK", //  Panjabi (Pakistan)
	"pl", //  Polish
	"pl_PL", //  Polish (Poland)
	"pr", //  Pirate
	"ps_AF", //  Pushto (Afghanistan)
	"pt", //  Portuguese
	"pt_BR", //  Portuguese (Brazil)
	"pt_PT", //  Portuguese (Portugal)
	"quy_PE", //  Ayacucho Quechua (Peru)
	"quz_PE", //  Cusco Quechua (Peru)
	"raj_IN", //  Rajasthani (India)
	"ro", //  Romanian
	"ro_RO", //  Romanian (Romania)
	"ru", //  Russian
	"ru_RU", //  Russian (Russia)
	"ru_UA", //  Russian (Ukraine)
	"rw_RW", //  Kinyarwanda (Rwanda)
	"sa_IN", //  Sanskrit (India)
	"sat_IN", //  Santali (India)
	"sc_IT", //  Sardinian (Italy)
	"sco", //  Scots
	"sd_IN", //  Sindhi (India)
	"se_NO", //  Northern Sami (Norway)
	"sgs_LT", //  Samogitian (Lithuania)
	"shs_CA", //  Shuswap (Canada)
	"sid_ET", //  Sidamo (Ethiopia)
	"si", //  Sinhala
	"si_LK", //  Sinhala (Sri Lanka)
	"sk", //  Slovak
	"sk_SK", //  Slovak (Slovakia)
	"sl", //  Slovenian
	"sl_SI", //  Slovenian (Slovenia)
	"so", //  Somali
	"so_DJ", //  Somali (Djibouti)
	"so_ET", //  Somali (Ethiopia)
	"so_KE", //  Somali (Kenya)
	"so_SO", //  Somali (Somalia)
	"son_ML", //  Songhai languages (Mali)
	"sq", //  Albanian
	"sq_AL", //  Albanian (Albania)
	"sq_KV", //  Albanian (Kosovo)
	"sq_MK", //  Albanian (Macedonia)
	"sr", //  Serbian
	"sr_Cyrl", //  Serbian (Cyrillic)
	"sr_Latn", //  Serbian (Latin)
	"sr_ME", //  Serbian (Montenegro)
	"sr_RS", //  Serbian (Serbia)
	"ss_ZA", //  Swati (South Africa)
	"st_ZA", //  Southern Sotho (South Africa)
	"sv", //  Swedish
	"sv_FI", //  Swedish (Finland)
	"sv_SE", //  Swedish (Sweden)
	"sw_KE", //  Swahili (Kenya)
	"sw_TZ", //  Swahili (Tanzania)
	"szl_PL", //  Silesian (Poland)
	"ta", //  Tamil
	"ta_IN", //  Tamil (India)
	"ta_LK", //  Tamil (Sri Lanka)
	"tcy_IN", //  Tulu (India)
	"te", //  Telugu
	"te_IN", //  Telugu (India)
	"tg_TJ", //  Tajik (Tajikistan)
	"the_NP", //  Chitwania Tharu (Nepal)
	"th", //  Thai
	"th_TH", //  Thai (Thailand)
	"ti", //  Tigrinya
	"ti_ER", //  Tigrinya (Eritrea)
	"ti_ET", //  Tigrinya (Ethiopia)
	"tig_ER", //  Tigre (Eritrea)
	"tk_TM", //  Turkmen (Turkmenistan)
	"tl_PH", //  Tagalog (Philippines)
	"tn_ZA", //  Tswana (South Africa)
	"tr", //  Turkish
	"tr_CY", //  Turkish (Cyprus)
	"tr_TR", //  Turkish (Turkey)
	"ts_ZA", //  Tsonga (South Africa)
	"tt_RU", //  Tatar (Russia)
	"ug_CN", //  Uighur (China)
	"uk", //  Ukrainian
	"uk_UA", //  Ukrainian (Ukraine)
	"unm_US", //  Unami (United States)
	"ur", //  Urdu
	"ur_IN", //  Urdu (India)
	"ur_PK", //  Urdu (Pakistan)
	"uz", //  Uzbek
	"uz_UZ", //  Uzbek (Uzbekistan)
	"ve_ZA", //  Venda (South Africa)
	"vi", //  Vietnamese
	"vi_VN", //  Vietnamese (Vietnam)
	"wa_BE", //  Walloon (Belgium)
	"wae_CH", //  Walser (Switzerland)
	"wal_ET", //  Wolaytta (Ethiopia)
	"wo_SN", //  Wolof (Senegal)
	"xh_ZA", //  Xhosa (South Africa)
	"yi_US", //  Yiddish (United States)
	"yo_NG", //  Yoruba (Nigeria)
	"yue_HK", //  Yue Chinese (Hong Kong)
	"zh", //  Chinese
	"zh_CN", //  Chinese (China)
	"zh_HK", //  Chinese (Hong Kong)
	"zh_SG", //  Chinese (Singapore)
	"zh_TW", //  Chinese (Taiwan)
	"zu_ZA", //  Zulu (South Africa)
	nullptr
};

static const char *locale_names[] = {
	"Afar",
	"Afar (Djibouti)",
	"Afar (Eritrea)",
	"Afar (Ethiopia)",
	"Afrikaans",
	"Afrikaans (South Africa)",
	"Aguaruna (Peru)",
	"Akan (Ghana)",
	"Amharic (Ethiopia)",
	"Aragonese (Spain)",
	"Angika (India)",
	"Arabic",
	"Arabic (United Arab Emirates)",
	"Arabic (Bahrain)",
	"Arabic (Algeria)",
	"Arabic (Egypt)",
	"Arabic (India)",
	"Arabic (Iraq)",
	"Arabic (Jordan)",
	"Arabic (Kuwait)",
	"Arabic (Lebanon)",
	"Arabic (Libya)",
	"Arabic (Morocco)",
	"Arabic (Oman)",
	"Arabic (Qatar)",
	"Arabic (Saudi Arabia)",
	"Arabic (Sudan)",
	"Arabic (South Soudan)",
	"Arabic (Syria)",
	"Arabic (Tunisia)",
	"Arabic (Yemen)",
	"Assamese (India)",
	"Asturian (Spain)",
	"Southern Aymara (Peru)",
	"Aymara (Peru)",
	"Azerbaijani (Azerbaijan)",
	"Belarusian",
	"Belarusian (Belarus)",
	"Bemba (Zambia)",
	"Berber languages (Algeria)",
	"Berber languages (Morocco)",
	"Bulgarian",
	"Bulgarian (Bulgaria)",
	"Bhili (India)",
	"Bhojpuri (India)",
	"Bislama (Tuvalu)",
	"Bengali",
	"Bengali (Bangladesh)",
	"Bengali (India)",
	"Tibetan",
	"Tibetan (China)",
	"Tibetan (India)",
	"Breton (France)",
	"Bodo (India)",
	"Bosnian (Bosnia and Herzegovina)",
	"Bilin (Eritrea)",
	"Catalan",
	"Catalan (Andorra)",
	"Catalan (Spain)",
	"Catalan (France)",
	"Catalan (Italy)",
	"Chechen (Russia)",
	"Cherokee (United States)",
	"Mandarin Chinese (Taiwan)",
	"Crimean Tatar (Ukraine)",
	"Kashubian (Poland)",
	"Czech",
	"Czech (Czech Republic)",
	"Chuvash (Russia)",
	"Welsh (United Kingdom)",
	"Danish",
	"Danish (Denmark)",
	"German",
	"German (Austria)",
	"German (Belgium)",
	"German (Switzerland)",
	"German (Germany)",
	"German (Italy)",
	"German (Luxembourg)",
	"Dogri (India)",
	"Dhivehi (Maldives)",
	"Dzongkha (Bhutan)",
	"Greek",
	"Greek (Cyprus)",
	"Greek (Greece)",
	"English",
	"English (Antigua and Barbuda)",
	"English (Australia)",
	"English (Botswana)",
	"English (Canada)",
	"English (Denmark)",
	"English (United Kingdom)",
	"English (Hong Kong)",
	"English (Ireland)",
	"English (Israel)",
	"English (India)",
	"English (Nigeria)",
	"English (New Zealand)",
	"English (Philippines)",
	"English (Singapore)",
	"English (United States)",
	"English (South Africa)",
	"English (Zambia)",
	"English (Zimbabwe)",
	"Esperanto",
	"Spanish",
	"Spanish (Argentina)",
	"Spanish (Bolivia)",
	"Spanish (Chile)",
	"Spanish (Colombia)",
	"Spanish (Costa Rica)",
	"Spanish (Cuba)",
	"Spanish (Dominican Republic)",
	"Spanish (Ecuador)",
	"Spanish (Spain)",
	"Spanish (Guatemala)",
	"Spanish (Honduras)",
	"Spanish (Mexico)",
	"Spanish (Nicaragua)",
	"Spanish (Panama)",
	"Spanish (Peru)",
	"Spanish (Puerto Rico)",
	"Spanish (Paraguay)",
	"Spanish (El Salvador)",
	"Spanish (United States)",
	"Spanish (Uruguay)",
	"Spanish (Venezuela)",
	"Estonian",
	"Estonian (Estonia)",
	"Basque",
	"Basque (Spain)",
	"Persian",
	"Persian (Iran)",
	"Fulah (Senegal)",
	"Finnish",
	"Finnish (Finland)",
	"Filipino",
	"Filipino (Philippines)",
	"Faroese (Faroe Islands)",
	"French",
	"French (Belgium)",
	"French (Canada)",
	"French (Switzerland)",
	"French (France)",
	"French (Luxembourg)",
	"Friulian (Italy)",
	"Western Frisian (Germany)",
	"Western Frisian (Netherlands)",
	"Irish",
	"Irish (Ireland)",
	"Scottish Gaelic (United Kingdom)",
	"Geez (Eritrea)",
	"Geez (Ethiopia)",
	"Galician (Spain)",
	"Gujarati (India)",
	"Manx (United Kingdom)",
	"Hakka Chinese (Taiwan)",
	"Hausa (Nigeria)",
	"Hebrew",
	"Hebrew (Israel)",
	"Hindi",
	"Hindi (India)",
	"Chhattisgarhi (India)",
	"Croatian",
	"Croatian (Croatia)",
	"Upper Sorbian (Germany)",
	"Haitian (Haiti)",
	"Hungarian",
	"Hungarian (Hungary)",
	"Huastec (Mexico)",
	"Armenian (Armenia)",
	"Interlingua (France)",
	"Indonesian",
	"Indonesian (Indonesia)",
	"Igbo (Nigeria)",
	"Inupiaq (Canada)",
	"Icelandic",
	"Icelandic (Iceland)",
	"Italian",
	"Italian (Switzerland)",
	"Italian (Italy)",
	"Inuktitut (Canada)",
	"Japanese",
	"Japanese (Japan)",
	"Kabyle (Algeria)",
	"Georgian",
	"Georgian (Georgia)",
	"Kazakh (Kazakhstan)",
	"Kalaallisut (Greenland)",
	"Central Khmer (Cambodia)",
	"Kannada (India)",
	"Konkani (India)",
	"Korean",
	"Korean (South Korea)",
	"Kashmiri (India)",
	"Kurdish",
	"Kurdish (Turkey)",
	"Cornish (United Kingdom)",
	"Kirghiz (Kyrgyzstan)",
	"Luxembourgish (Luxembourg)",
	"Ganda (Uganda)",
	"Limburgan (Belgium)",
	"Limburgan (Netherlands)",
	"Ligurian (Italy)",
	"Lingala (Congo)",
	"Lao (Laos)",
	"Lithuanian",
	"Lithuanian (Lithuania)",
	"Latvian",
	"Latvian (Latvia)",
	"Literary Chinese (Taiwan)",
	"Magahi (India)",
	"Maithili (India)",
	"Malagasy (Madagascar)",
	"Marshallese (Marshall Islands)",
	"Eastern Mari (Russia)",
	"Māori",
	"Māori (New Zealand)",
	"Mískito (Nicaragua)",
	"Macedonian",
	"Macedonian (Macedonia)",
	"Malayalam",
	"Malayalam (India)",
	"Manipuri (India)",
	"Mongolian (Mongolia)",
	"Marathi (India)",
	"Malay",
	"Malay (Malaysia)",
	"Maltese",
	"Maltese (Malta)",
	"Burmese (Myanmar)",
	"Erzya (Russia)",
	"Nahuatl languages (Mexico)",
	"Min Nan Chinese (Taiwan)",
	"Norwegian Bokmål",
	"Norwegian Bokmål (Norway)",
	"Low German (Germany)",
	"Low German (Netherlands)",
	"Nepali (Nepal)",
	"Central Nahuatl (Mexico)",
	"Niuean (Niue)",
	"Niuean (New Zealand)",
	"Dutch",
	"Dutch (Aruba)",
	"Dutch (Belgium)",
	"Dutch (Netherlands)",
	"Norwegian Nynorsk",
	"Norwegian Nynorsk (Norway)",
	"South Ndebele (South Africa)",
	"Pedi (South Africa)",
	"Occitan (France)",
	"Oromo",
	"Oromo (Ethiopia)",
	"Oromo (Kenya)",
	"Oriya (India)",
	"Ossetian (Russia)",
	"Panjabi (India)",
	"Papiamento",
	"Papiamento (Netherlands Antilles)",
	"Papiamento (Aruba)",
	"Papiamento (Curaçao)",
	"Panjabi (Pakistan)",
	"Polish",
	"Polish (Poland)",
	"Pirate",
	"Pushto (Afghanistan)",
	"Portuguese",
	"Portuguese (Brazil)",
	"Portuguese (Portugal)",
	"Ayacucho Quechua (Peru)",
	"Cusco Quechua (Peru)",
	"Rajasthani (India)",
	"Romanian",
	"Romanian (Romania)",
	"Russian",
	"Russian (Russia)",
	"Russian (Ukraine)",
	"Kinyarwanda (Rwanda)",
	"Sanskrit (India)",
	"Santali (India)",
	"Sardinian (Italy)",
	"Scots (Scotland)",
	"Sindhi (India)",
	"Northern Sami (Norway)",
	"Samogitian (Lithuania)",
	"Shuswap (Canada)",
	"Sidamo (Ethiopia)",
	"Sinhala",
	"Sinhala (Sri Lanka)",
	"Slovak",
	"Slovak (Slovakia)",
	"Slovenian",
	"Slovenian (Slovenia)",
	"Somali",
	"Somali (Djibouti)",
	"Somali (Ethiopia)",
	"Somali (Kenya)",
	"Somali (Somalia)",
	"Songhai languages (Mali)",
	"Albanian",
	"Albanian (Albania)",
	"Albanian (Kosovo)",
	"Albanian (Macedonia)",
	"Serbian",
	"Serbian (Cyrillic)",
	"Serbian (Latin)",
	"Serbian (Montenegro)",
	"Serbian (Serbia)",
	"Swati (South Africa)",
	"Southern Sotho (South Africa)",
	"Swedish",
	"Swedish (Finland)",
	"Swedish (Sweden)",
	"Swahili (Kenya)",
	"Swahili (Tanzania)",
	"Silesian (Poland)",
	"Tamil",
	"Tamil (India)",
	"Tamil (Sri Lanka)",
	"Tulu (India)",
	"Telugu",
	"Telugu (India)",
	"Tajik (Tajikistan)",
	"Chitwania Tharu (Nepal)",
	"Thai",
	"Thai (Thailand)",
	"Tigrinya",
	"Tigrinya (Eritrea)",
	"Tigrinya (Ethiopia)",
	"Tigre (Eritrea)",
	"Turkmen (Turkmenistan)",
	"Tagalog (Philippines)",
	"Tswana (South Africa)",
	"Turkish",
	"Turkish (Cyprus)",
	"Turkish (Turkey)",
	"Tsonga (South Africa)",
	"Tatar (Russia)",
	"Uighur (China)",
	"Ukrainian",
	"Ukrainian (Ukraine)",
	"Unami (United States)",
	"Urdu",
	"Urdu (India)",
	"Urdu (Pakistan)",
	"Uzbek",
	"Uzbek (Uzbekistan)",
	"Venda (South Africa)",
	"Vietnamese",
	"Vietnamese (Vietnam)",
	"Walloon (Belgium)",
	"Walser (Switzerland)",
	"Wolaytta (Ethiopia)",
	"Wolof (Senegal)",
	"Xhosa (South Africa)",
	"Yiddish (United States)",
	"Yoruba (Nigeria)",
	"Yue Chinese (Hong Kong)",
	"Chinese",
	"Chinese (China)",
	"Chinese (Hong Kong)",
	"Chinese (Singapore)",
	"Chinese (Taiwan)",
	"Zulu (South Africa)",
	nullptr
};

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
	{ nullptr, nullptr }
};

///////////////////////////////////////////////

Dictionary Translation::_get_messages() const {
	// Return translation_map as a Dictionary.

	Dictionary d;

	List<StringName> context_l;
	translation_map.get_key_list(&context_l);
	for (auto E = context_l.front(); E; E = E->next()) {
		StringName ctx = E->get();
		const HashMap<StringName, Vector<StringName>> &id_str_map = translation_map[ctx];

		Dictionary d2;
		List<StringName> id_l;
		id_str_map.get_key_list(&id_l);
		// Save list of id and strs associated with a context in a temporary dictionary.
		for (auto E2 = id_l.front(); E2; E2 = E2->next()) {
			StringName id = E2->get();
			d2[id] = id_str_map[id];
		}

		d[ctx] = d2;
	}

	return d;
}

void Translation::_set_messages(const Dictionary &p_messages) {
	// Construct translation_map from a Dictionary.

	List<Variant> context_l;
	p_messages.get_key_list(&context_l);
	for (auto E = context_l.front(); E; E = E->next()) {
		StringName ctx = E->get();
		const Dictionary &id_str_map = p_messages[ctx];

		HashMap<StringName, Vector<StringName>> temp_map;
		List<Variant> id_l;
		id_str_map.get_key_list(&id_l);
		for (auto E2 = id_l.front(); E2; E2 = E2->next()) {
			StringName id = E2->get();
			temp_map[id] = id_str_map[id];
		}

		translation_map[ctx] = temp_map;
	}
}

Vector<String> Translation::_get_message_list() const {
	////This one I'm really not sure what the use case of this function is. So I just follow what it does before.
	// Return all keys in translation_map.

	List<StringName> msgs;
	get_message_list(&msgs);

	Vector<String> v;
	for (auto E = msgs.front(); E; E = E->next()) {
		v.push_back(E->get());
	}

	return v;
}

int Translation::_get_plural_index(int p_n) const {
	// Apply plural rule to a p_n passed in, and get a number between [0;number of plural forms)

	Ref<Expression> expr;
	expr.instance();

	Vector<String> input_name;
	input_name.push_back("n");

	Array input_val;
	input_val.push_back(p_n);

	int result = _get_plural_index(plural_rule, input_name, input_val, expr);
	ERR_FAIL_COND_V_MSG(result < 0, 0, "_get_plural_index() returns a negative number after evaluating a plural rule expression.");

	return result;
}

int Translation::_get_plural_index(const String &p_plural_rule, const Vector<String> &p_input_name, const Array &p_input_value, Ref<Expression> &r_expr) const {
	// Evaluate recursively until we find the first condition that is true.
	// Some examples of p_plural_rule passed in can have the form:
	// "n==0 ? 0 : n==1 ? 1 : n==2 ? 2 : n%100>=3 && n%100<=10 ? 3 : n%100>=11 && n%100<=99 ? 4 : 5" (Arabic)
	// "n >= 2" (French)
	// "n != 1" (English)

	// Parse expression.
	int first_ques_mark = p_plural_rule.find("?");
	String equi_test = p_plural_rule.substr(0, first_ques_mark);
	Error err = r_expr->parse(equi_test, p_input_name);
	ERR_FAIL_COND_V_MSG(err != OK, p_input_value[0], "Cannot parse expression. Error: " + r_expr->get_error_text());

	// Evaluate expression.
	Variant result = r_expr->execute(p_input_value);
	ERR_FAIL_COND_V_MSG(r_expr->has_execute_failed(), p_input_value[0], "Cannot evaluate expression.");

	// Base case of recursion. Variant result will either map to a bool or an integer, in both cases returning it will give the correct plural index.
	if (first_ques_mark == -1) {
		return result;
	}

	if (bool(result)) {
		return p_plural_rule.substr(first_ques_mark + 1, p_plural_rule.find(":") - (first_ques_mark + 1)).to_int();
	}

	String after_colon = p_plural_rule.substr(p_plural_rule.find(":") + 1, p_plural_rule.length());
	return _get_plural_index(after_colon, p_input_name, p_input_value, r_expr);
}

void Translation::set_locale(const String &p_locale) {
	String univ_locale = TranslationServer::standardize_locale(p_locale);

	if (!TranslationServer::is_locale_valid(univ_locale)) {
		String trimmed_locale = TranslationServer::get_language_code(univ_locale);

		ERR_FAIL_COND_MSG(!TranslationServer::is_locale_valid(trimmed_locale), "Invalid locale: " + trimmed_locale + ".");

		locale = trimmed_locale;
	} else {
		locale = univ_locale;
	}

	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_TRANSLATION_CHANGED);
	}
}

void Translation::set_plural_rule(const String &p_plural_rule) {
	// Set plural_forms and plural_rule.
	// p_plural_rule passed in has the form "Plural-Forms: nplurals=2; plural=(n >= 2);".

	int first_semi_col = p_plural_rule.find(";");
	plural_forms = p_plural_rule.substr(p_plural_rule.find("=") + 1, first_semi_col - (p_plural_rule.find("=") + 1)).to_int();

	int expression_start = p_plural_rule.find("=", first_semi_col) + 1;
	int second_semi_col = p_plural_rule.rfind(";");
	plural_rule = p_plural_rule.substr(expression_start, second_semi_col - expression_start);
	// Strip away '(' and ')' to ease evaluating the expression later on.
	plural_rule = plural_rule.replacen("(", "");
	plural_rule = plural_rule.replacen(")", "");
}

void Translation::add_message(const StringName &p_src_text, const StringName &p_xlated_text, const StringName &p_context) {
	HashMap<StringName, Vector<StringName>> &map_id_str = translation_map[p_context];

	if (map_id_str.has(p_src_text)) {
		WARN_PRINT("Double translations for \"" + String(p_src_text) + "\" under the same context \"" + String(p_context) + "\" for locale \"" + get_locale() + "\".\nThere should only be one unique translation for a given string under the same context.");
		map_id_str[p_src_text].set(0, p_xlated_text);
	} else {
		map_id_str[p_src_text].push_back(p_xlated_text);
	}
}

void Translation::add_plural_message(const StringName &p_src_text, const Vector<String> &p_plural_texts, const StringName &p_context) {
	ERR_FAIL_COND_MSG(p_plural_texts.size() != plural_forms, "Trying to add plural texts that don't match the required number of plural forms for locale \"" + get_locale() + "\"");

	HashMap<StringName, Vector<StringName>> &map_id_str = translation_map[p_context];

	if (map_id_str.has(p_src_text)) {
		WARN_PRINT("Double translations for \"" + p_src_text + "\" under the same context \"" + p_context + "\" for locale " + get_locale() + ".\nThere should only be one unique translation for a given string under the same context.");
		map_id_str[p_src_text].clear();
	}

	for (int i = 0; i < p_plural_texts.size(); i++) {
		map_id_str[p_src_text].push_back(p_plural_texts[i]);
	}
}

int Translation::get_plural_forms() const {
	return plural_forms;
}

String Translation::get_plural_rule() const {
	return plural_rule;
}

StringName Translation::get_message(const StringName &p_src_text, const StringName &p_context) const {
	if (!translation_map.has(p_context) || !translation_map[p_context].has(p_src_text)) {
		return StringName();
	}
	ERR_FAIL_COND_V_MSG(translation_map[p_context][p_src_text].empty(), StringName(), "Source text \"" + String(p_src_text) + "\" is registered but doesn't have a translation. Please check add_message() or add_plural_message() to make sure a translation is always added.");

	return translation_map[p_context][p_src_text][0];
}

StringName Translation::get_plural_message(const StringName &p_src_text, const StringName &p_plural_text, int p_n, const StringName &p_context) const {
	ERR_FAIL_COND_V_MSG(p_n < 0, p_src_text, "N passed into translation to get a plural message should not be negative. For negative numbers, use singular translation please. Search \"gettext PO Plural Forms\" online for the documentation on translating negative numbers.");

	if (!translation_map.has(p_context) || !translation_map[p_context].has(p_src_text)) {
		return StringName();
	}
	ERR_FAIL_COND_V_MSG(translation_map[p_context][p_src_text].empty(), StringName(), "Source text \"" + String(p_src_text) + "\" is registered but doesn't have a translation. Please check add_message() or add_plural_message() to make sure a translation is always added.");

	// Return based on English plural rule if locale's plural rule is not registered (normally due to missing or invalid "Plural-Forms" in PO file header).
	if (plural_forms <= 0) {
		if (p_n == 1) {
			return p_src_text;
		} else {
			return p_plural_text;
		}
	}

	return translation_map[p_context][p_src_text][_get_plural_index(p_n)];
}

void Translation::erase_message(const StringName &p_src_text, const StringName &p_context) {
	if (!translation_map.has(p_context)) {
		return;
	}

	translation_map[p_context].erase(p_src_text);
}

void Translation::get_message_list(List<StringName> *r_messages) const {
	////This is the function that PHashTranslation uses to get the list of msgid.
	////Right now I just return the msgid list under "" context, and make no changes to PHashTranslation at all.
	////So PHashTranslation will be functioning like last time, it will not handle context and plurals translation.

	// Return all the keys of translation_map under "" context.

	List<StringName> context_l;
	translation_map.get_key_list(&context_l);

	for (auto E = context_l.front(); E; E = E->next()) {
		if (String(E->get()) != "") {
			continue;
		}

		List<StringName> msgid_l;
		translation_map[E->get()].get_key_list(&msgid_l);

		for (auto E2 = msgid_l.front(); E2; E2 = E2->next()) {
			r_messages->push_back(E2->get());
		}
	}
}

int Translation::get_message_count() const {
	List<StringName> context_l;
	translation_map.get_key_list(&context_l);

	int count = 0;
	for (auto E = context_l.front(); E; E = E->next()) {
		count += translation_map[E->get()].size();
	}
	return count;
}

void Translation::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_locale", "locale"), &Translation::set_locale);
	ClassDB::bind_method(D_METHOD("get_locale"), &Translation::get_locale);
	ClassDB::bind_method(D_METHOD("add_message", "src_message", "xlated_message", "context"), &Translation::add_message, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("add_plural_message", "src_message", "xlated_messages", "context"), &Translation::add_plural_message, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_message", "src_message", "context"), &Translation::get_message, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_plural_message", "src_message", "src_plural_message", "n", "context"), &Translation::get_plural_message, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("erase_message", "src_message", "context"), &Translation::erase_message, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_message_list"), &Translation::_get_message_list);
	ClassDB::bind_method(D_METHOD("get_message_count"), &Translation::get_message_count);
	ClassDB::bind_method(D_METHOD("get_plural_forms"), &Translation::get_plural_forms);
	ClassDB::bind_method(D_METHOD("get_plural_rule"), &Translation::get_plural_rule);
	ClassDB::bind_method(D_METHOD("_set_messages"), &Translation::_set_messages);
	ClassDB::bind_method(D_METHOD("_get_messages"), &Translation::_get_messages);

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "messages", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "_set_messages", "_get_messages");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "locale"), "set_locale", "get_locale");
}

///////////////////////////////////////////////

bool TranslationServer::is_locale_valid(const String &p_locale) {
	const char **ptr = locale_list;

	while (*ptr) {
		if (*ptr == p_locale) {
			return true;
		}
		ptr++;
	}

	return false;
}

String TranslationServer::standardize_locale(const String &p_locale) {
	// Replaces '-' with '_' for macOS Sierra-style locales
	String univ_locale = p_locale.replace("-", "_");

	// Handles known non-ISO locale names used e.g. on Windows
	int idx = 0;
	while (locale_renames[idx][0] != nullptr) {
		if (locale_renames[idx][0] == univ_locale) {
			univ_locale = locale_renames[idx][1];
			break;
		}
		idx++;
	}

	return univ_locale;
}

String TranslationServer::get_language_code(const String &p_locale) {
	ERR_FAIL_COND_V_MSG(p_locale.length() < 2, p_locale, "Invalid locale '" + p_locale + "'.");
	// Most language codes are two letters, but some are three,
	// so we have to look for a regional code separator ('_' or '-')
	// to extract the left part.
	// For example we get 'nah_MX' as input and should return 'nah'.
	int split = p_locale.find("_");
	if (split == -1) {
		split = p_locale.find("-");
	}
	if (split == -1) { // No separator, so the locale is already only a language code.
		return p_locale;
	}
	return p_locale.left(split);
}

void TranslationServer::set_locale(const String &p_locale) {
	String univ_locale = standardize_locale(p_locale);

	if (!is_locale_valid(univ_locale)) {
		String trimmed_locale = get_language_code(univ_locale);
		print_verbose(vformat("Unsupported locale '%s', falling back to '%s'.", p_locale, trimmed_locale));

		if (!is_locale_valid(trimmed_locale)) {
			ERR_PRINT(vformat("Unsupported locale '%s', falling back to 'en'.", trimmed_locale));
			locale = "en";
		} else {
			locale = trimmed_locale;
		}
	} else {
		locale = univ_locale;
	}

	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_TRANSLATION_CHANGED);
	}

	ResourceLoader::reload_translation_remaps();
}

String TranslationServer::get_locale() const {
	return locale;
}

String TranslationServer::get_locale_name(const String &p_locale) const {
	if (!locale_name_map.has(p_locale)) {
		return String();
	}
	return locale_name_map[p_locale];
}

Array TranslationServer::get_loaded_locales() const {
	Array locales;
	for (const Set<Ref<Translation>>::Element *E = translations.front(); E; E = E->next()) {
		const Ref<Translation> &t = E->get();
		ERR_FAIL_COND_V(t.is_null(), Array());
		String l = t->get_locale();

		locales.push_back(l);
	}

	return locales;
}

Vector<String> TranslationServer::get_all_locales() {
	Vector<String> locales;

	const char **ptr = locale_list;

	while (*ptr) {
		locales.push_back(*ptr);
		ptr++;
	}

	return locales;
}

Vector<String> TranslationServer::get_all_locale_names() {
	Vector<String> locales;

	const char **ptr = locale_names;

	while (*ptr) {
		locales.push_back(String::utf8(*ptr));
		ptr++;
	}

	return locales;
}

void TranslationServer::add_translation(const Ref<Translation> &p_translation) {
	translations.insert(p_translation);
}

void TranslationServer::remove_translation(const Ref<Translation> &p_translation) {
	translations.erase(p_translation);
}

void TranslationServer::clear() {
	translations.clear();
}

StringName TranslationServer::translate(const StringName &p_message, const StringName &p_context) const {
	// Match given message against the translation catalog for the project locale.

	if (!enabled) {
		return p_message;
	}

	ERR_FAIL_COND_V_MSG(locale.length() < 2, p_message, "Could not translate message as configured locale '" + locale + "' is invalid.");

	StringName res = _get_message_from_translations(p_message, p_context, locale);

	if (!res && fallback.length() >= 2) {
		res = _get_message_from_translations(p_message, p_context, fallback);
	}

	if (!res) {
		return p_message;
	}

	return res;
}

StringName TranslationServer::translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context) const {
	if (!enabled) {
		if (p_n == 1) {
			return p_message;
		} else {
			return p_message_plural;
		}
	}

	ERR_FAIL_COND_V_MSG(locale.length() < 2, p_message, "Could not translate message as configured locale '" + locale + "' is invalid.");

	StringName res = _get_message_from_translations(p_message, p_context, locale, p_message_plural, p_n);

	if (!res && fallback.length() >= 2) {
		res = _get_message_from_translations(p_message, p_context, fallback, p_message_plural, p_n);
	}

	if (!res) {
		if (p_n == 1) {
			return p_message;
		} else {
			return p_message_plural;
		}
	}

	return res;
}

StringName TranslationServer::_get_message_from_translations(const StringName &p_message, const StringName &p_context, const String &p_locale, const String &p_message_plural, int p_n) const {
	// Locale can be of the form 'll_CC', i.e. language code and regional code,
	// e.g. 'en_US', 'en_GB', etc. It might also be simply 'll', e.g. 'en'.
	// To find the relevant translation, we look for those with locale starting
	// with the language code, and then if any is an exact match for the long
	// form. If not found, we fall back to a near match (another locale with
	// same language code).

	// Note: ResourceLoader::_path_remap reproduces this locale near matching
	// logic, so be sure to propagate changes there when changing things here.

	StringName res;
	String lang = get_language_code(p_locale);
	bool near_match = false;

	for (const Set<Ref<Translation>>::Element *E = translations.front(); E; E = E->next()) {
		const Ref<Translation> &t = E->get();
		ERR_FAIL_COND_V(t.is_null(), p_message);
		String l = t->get_locale();

		bool exact_match = (l == p_locale);
		if (!exact_match) {
			if (near_match) {
				continue; // Only near-match once, but keep looking for exact matches.
			}
			if (get_language_code(l) != lang) {
				continue; // Language code does not match.
			}
		}

		StringName r;
		if (p_n == -1) {
			r = t->get_message(p_message, p_context);
		} else {
			r = t->get_plural_message(p_message, p_message_plural, p_n, p_context);
		}

		if (!r) {
			continue;
		}
		res = r;

		if (exact_match) {
			break;
		} else {
			near_match = true;
		}
	}

	return res;
}

TranslationServer *TranslationServer::singleton = nullptr;

bool TranslationServer::_load_translations(const String &p_from) {
	if (ProjectSettings::get_singleton()->has_setting(p_from)) {
		Vector<String> translations = ProjectSettings::get_singleton()->get(p_from);

		int tcount = translations.size();

		if (tcount) {
			const String *r = translations.ptr();

			for (int i = 0; i < tcount; i++) {
				Ref<Translation> tr = ResourceLoader::load(r[i]);
				if (tr.is_valid()) {
					add_translation(tr);
				}
			}
		}
		return true;
	}

	return false;
}

void TranslationServer::setup() {
	String test = GLOBAL_DEF("locale/test", "");
	test = test.strip_edges();
	if (test != "") {
		set_locale(test);
	} else {
		set_locale(OS::get_singleton()->get_locale());
	}
	fallback = GLOBAL_DEF("locale/fallback", "en");
#ifdef TOOLS_ENABLED
	{
		String options = "";
		int idx = 0;
		while (locale_list[idx]) {
			if (idx > 0) {
				options += ",";
			}
			options += locale_list[idx];
			idx++;
		}
		ProjectSettings::get_singleton()->set_custom_property_info("locale/fallback", PropertyInfo(Variant::STRING, "locale/fallback", PROPERTY_HINT_ENUM, options));
	}
#endif
}

void TranslationServer::set_tool_translation(const Ref<Translation> &p_translation) {
	tool_translation = p_translation;
}

StringName TranslationServer::tool_translate(const StringName &p_message, const StringName &p_context) const {
	if (tool_translation.is_valid()) {
		StringName r = tool_translation->get_message(p_message, p_context);
		if (r) {
			return r;
		}
	}
	return p_message;
}

StringName TranslationServer::tool_translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context) const {
	if (tool_translation.is_valid()) {
		StringName r = tool_translation->get_plural_message(p_message, p_message_plural, p_n, p_context);
		if (r) {
			return r;
		}
	}

	if (p_n == 1) {
		return p_message;
	} else {
		return p_message_plural;
	}
}

void TranslationServer::set_doc_translation(const Ref<Translation> &p_translation) {
	doc_translation = p_translation;
}

StringName TranslationServer::doc_translate(const StringName &p_message, const StringName &p_context) const {
	if (doc_translation.is_valid()) {
		StringName r = doc_translation->get_message(p_message, p_context);
		if (r) {
			return r;
		}
	}
	return p_message;
}

StringName TranslationServer::doc_translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context) const {
	if (doc_translation.is_valid()) {
		StringName r = doc_translation->get_plural_message(p_message, p_message_plural, p_n, p_context);
		if (r) {
			return r;
		}
	}

	if (p_n == 1) {
		return p_message;
	} else {
		return p_message_plural;
	}
}

void TranslationServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_locale", "locale"), &TranslationServer::set_locale);
	ClassDB::bind_method(D_METHOD("get_locale"), &TranslationServer::get_locale);

	ClassDB::bind_method(D_METHOD("get_locale_name", "locale"), &TranslationServer::get_locale_name);

	ClassDB::bind_method(D_METHOD("translate", "message"), &TranslationServer::translate);
	ClassDB::bind_method(D_METHOD("translate_plural", "message", "plural_message", "n", "context"), &TranslationServer::translate_plural, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("add_translation", "translation"), &TranslationServer::add_translation);
	ClassDB::bind_method(D_METHOD("remove_translation", "translation"), &TranslationServer::remove_translation);

	ClassDB::bind_method(D_METHOD("clear"), &TranslationServer::clear);

	ClassDB::bind_method(D_METHOD("get_loaded_locales"), &TranslationServer::get_loaded_locales);
}

void TranslationServer::load_translations() {
	String locale = get_locale();
	_load_translations("locale/translations"); //all
	_load_translations("locale/translations_" + locale.substr(0, 2));

	if (locale.substr(0, 2) != locale) {
		_load_translations("locale/translations_" + locale);
	}
}

TranslationServer::TranslationServer() {
	singleton = this;

	for (int i = 0; locale_list[i]; ++i) {
		locale_name_map.insert(locale_list[i], String::utf8(locale_names[i]));
	}
}
