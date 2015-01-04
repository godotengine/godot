/*************************************************************************/
/*  translation.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "globals.h"
#include "io/resource_loader.h"
#include "os/os.h"

static const char* locale_list[]={
"ar", //  Arabic
"ar_AE", //  Arabic (United Arab Emirates)
"ar_BH", //  Arabic (Bahrain)
"ar_DZ", //  Arabic (Algeria)
"ar_EG", //  Arabic (Egypt)
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
"ar_SY", //  Arabic (Syria)
"ar_TN", //  Arabic (Tunisia)
"ar_YE", //  Arabic (Yemen)
"be", //  Belarusian
"be_BY", //  Belarusian (Belarus)
"bg", //  Bulgarian
"bg_BG", //  Bulgarian (Bulgaria)
"ca", //  Catalan
"ca_ES", //  Catalan (Spain)
"cs", //  Czech
"cs_CZ", //  Czech (Czech Republic)
"da", //  Danish
"da_DK", //  Danish (Denmark)
"de", //  German
"de_AT", //  German (Austria)
"de_CH", //  German (Switzerland)
"de_DE", //  German (Germany)
"de_LU", //  German (Luxembourg)
"el", //  Greek
"el_CY", //  Greek (Cyprus)
"el_GR", //  Greek (Greece)
"en", //  English
"en_AU", //  English (Australia)
"en_CA", //  English (Canada)
"en_GB", //  English (United Kingdom)
"en_IE", //  English (Ireland)
"en_IN", //  English (India)
"en_MT", //  English (Malta)
"en_NZ", //  English (New Zealand)
"en_PH", //  English (Philippines)
"en_SG", //  English (Singapore)
"en_US", //  English (United States)
"en_ZA", //  English (South Africa)
"es", //  Spanish
"es_AR", //  Spanish (Argentina)
"es_BO", //  Spanish (Bolivia)
"es_CL", //  Spanish (Chile)
"es_CO", //  Spanish (Colombia)
"es_CR", //  Spanish (Costa Rica)
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
"fi", //  Finnish
"fi_FI", //  Finnish (Finland)
"fr", //  French
"fr_BE", //  French (Belgium)
"fr_CA", //  French (Canada)
"fr_CH", //  French (Switzerland)
"fr_FR", //  French (France)
"fr_LU", //  French (Luxembourg)
"ga", //  Irish
"ga_IE", //  Irish (Ireland)
"hi", //  Hindi (India)
"hi_IN", //  Hindi (India)
"hr", //  Croatian
"hr_HR", //  Croatian (Croatia)
"hu", //  Hungarian
"hu_HU", //  Hungarian (Hungary)
"in", //  Indonesian
"in_ID", //  Indonesian (Indonesia)
"is", //  Icelandic
"is_IS", //  Icelandic (Iceland)
"it", //  Italian
"it_CH", //  Italian (Switzerland)
"it_IT", //  Italian (Italy)
"iw", //  Hebrew
"iw_IL", //  Hebrew (Israel)
"ja", //  Japanese
"ja_JP", //  Japanese (Japan)
"ja_JP_JP", //  Japanese (Japan,JP)
"ko", //  Korean
"ko_KR", //  Korean (South Korea)
"lt", //  Lithuanian
"lt_LT", //  Lithuanian (Lithuania)
"lv", //  Latvian
"lv_LV", //  Latvian (Latvia)
"mk", //  Macedonian
"mk_MK", //  Macedonian (Macedonia)
"ms", //  Malay
"ms_MY", //  Malay (Malaysia)
"mt", //  Maltese
"mt_MT", //  Maltese (Malta)
"nl", //  Dutch
"nl_BE", //  Dutch (Belgium)
"nl_NL", //  Dutch (Netherlands)
"no", //  Norwegian
"no_NO", //  Norwegian (Norway)
"no_NO_NY", //  Norwegian (Norway,Nynorsk)
"pl", //  Polish
"pl_PL", //  Polish (Poland)
"pt", //  Portuguese
"pt_BR", //  Portuguese (Brazil)
"pt_PT", //  Portuguese (Portugal)
"ro", //  Romanian
"ro_RO", //  Romanian (Romania)
"ru", //  Russian
"ru_RU", //  Russian (Russia)
"sk", //  Slovak
"sk_SK", //  Slovak (Slovakia)
"sl", //  Slovenian
"sl_SI", //  Slovenian (Slovenia)
"sq", //  Albanian
"sq_AL", //  Albanian (Albania)
"sr", //  Serbian
"sr_BA", //  Serbian (Bosnia and Herzegovina)
"sr_CS", //  Serbian (Serbia and Montenegro)
"sr_ME", //  Serbian (Montenegro)
"sr_RS", //  Serbian (Serbia)
"sv", //  Swedish
"sv_SE", //  Swedish (Sweden)
"th", //  Thai
"th_TH", //  Thai (Thailand)
"th_TH_TH", //  Thai (Thailand,TH)
"tr", //  Turkish
"tr_TR", //  Turkish (Turkey)
"uk", //  Ukrainian
"uk_UA", //  Ukrainian (Ukraine)
"vi", //  Vietnamese
"vi_VN", //  Vietnamese (Vietnam)
"zh", //  Chinese
"zh_CN", //  Chinese (China)
"zh_HK", //  Chinese (Hong Kong)
"zh_SG", //  Chinese (Singapore)
"zh_TW", //  Chinese (Taiwan)
0
};

static const char* locale_names[]={
"Arabic",
"Arabic (United Arab Emirates)",
"Arabic (Bahrain)",
"Arabic (Algeria)",
"Arabic (Egypt)",
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
"Arabic (Syria)",
"Arabic (Tunisia)",
"Arabic (Yemen)",
"Belarusian",
"Belarusian (Belarus)",
"Bulgarian",
"Bulgarian (Bulgaria)",
"Catalan",
"Catalan (Spain)",
"Czech",
"Czech (Czech Republic)",
"Danish",
"Danish (Denmark)",
"German",
"German (Austria)",
"German (Switzerland)",
"German (Germany)",
"German (Luxembourg)",
"Greek",
"Greek (Cyprus)",
"Greek (Greece)",
"English",
"English (Australia)",
"English (Canada)",
"English (United Kingdom)",
"English (Ireland)",
"English (India)",
"English (Malta)",
"English (New Zealand)",
"English (Philippines)",
"English (Singapore)",
"English (United States)",
"English (South Africa)",
"Spanish",
"Spanish (Argentina)",
"Spanish (Bolivia)",
"Spanish (Chile)",
"Spanish (Colombia)",
"Spanish (Costa Rica)",
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
"Finnish",
"Finnish (Finland)",
"French",
"French (Belgium)",
"French (Canada)",
"French (Switzerland)",
"French (France)",
"French (Luxembourg)",
"Irish",
"Irish (Ireland)",
"Hindi (India)",
"Hindi (India)",
"Croatian",
"Croatian (Croatia)",
"Hungarian",
"Hungarian (Hungary)",
"Indonesian",
"Indonesian (Indonesia)",
"Icelandic",
"Icelandic (Iceland)",
"Italian",
"Italian (Switzerland)",
"Italian (Italy)",
"Hebrew",
"Hebrew (Israel)",
"Japanese",
"Japanese (Japan)",
"Japanese (Japan JP)",
"Korean",
"Korean (South Korea)",
"Lithuanian",
"Lithuanian (Lithuania)",
"Latvian",
"Latvian (Latvia)",
"Macedonian",
"Macedonian (Macedonia)",
"Malay",
"Malay (Malaysia)",
"Maltese",
"Maltese (Malta)",
"Dutch",
"Dutch (Belgium)",
"Dutch (Netherlands)",
"Norwegian",
"Norwegian (Norway)",
"Norwegian (Norway Nynorsk)",
"Polish",
"Polish (Poland)",
"Portuguese",
"Portuguese (Brazil)",
"Portuguese (Portugal)",
"Romanian",
"Romanian (Romania)",
"Russian",
"Russian (Russia)",
"Slovak",
"Slovak (Slovakia)",
"Slovenian",
"Slovenian (Slovenia)",
"Albanian",
"Albanian (Albania)",
"Serbian",
"Serbian (Bosnia and Herzegovina)",
"Serbian (Serbia and Montenegro)",
"Serbian (Montenegro)",
"Serbian (Serbia)",
"Swedish",
"Swedish (Sweden)",
"Thai",
"Thai (Thailand)",
"Thai (Thailand TH)",
"Turkish",
"Turkish (Turkey)",
"Ukrainian",
"Ukrainian (Ukraine)",
"Vietnamese",
"Vietnamese (Vietnam)",
"Chinese",
"Chinese (China)",
"Chinese (Hong Kong)",
"Chinese (Singapore)",
"Chinese (Taiwan)",
0
};


Vector<String> TranslationServer::get_all_locales() {

	Vector<String> locales;

	const char **ptr=locale_list;

	while (*ptr) {
		locales.push_back(*ptr);
		ptr++;
	}

	return locales;

}

Vector<String> TranslationServer::get_all_locale_names(){

	Vector<String> locales;

	const char **ptr=locale_names;

	while (*ptr) {
		locales.push_back(*ptr);
		ptr++;
	}

	return locales;

}


static bool is_valid_locale(const String& p_locale) {

	const char **ptr=locale_list;

	while (*ptr) {
		if (p_locale==*ptr)
			return true;
		ptr++;
	}

	return false;
}

DVector<String> Translation::_get_messages() const {

	DVector<String> msgs;
	msgs.resize(translation_map.size()*2);
	int idx=0;
	for (const Map<StringName, StringName>::Element *E=translation_map.front();E;E=E->next()) {

		msgs.set(idx+0,E->key());
		msgs.set(idx+1,E->get());
		idx+=2;
	}

	return msgs;
}

DVector<String> Translation::_get_message_list() const {

	DVector<String> msgs;
	msgs.resize(translation_map.size());
	int idx=0;
	for (const Map<StringName, StringName>::Element *E=translation_map.front();E;E=E->next()) {

		msgs.set(idx,E->key());
		idx+=1;
	}

	return msgs;

}

void Translation::_set_messages(const DVector<String>& p_messages){

	int msg_count=p_messages.size();
	ERR_FAIL_COND(msg_count%2);

	DVector<String>::Read r = p_messages.read();

	for(int i=0;i<msg_count;i+=2) {

		add_message( r[i+0], r[i+1] );
	}

}


void Translation::set_locale(const String& p_locale) {

	ERR_EXPLAIN("Invalid Locale: "+p_locale);
	ERR_FAIL_COND(!is_valid_locale(p_locale));
	locale=p_locale;
}

void Translation::add_message( const StringName& p_src_text, const StringName& p_xlated_text ) {

	translation_map[p_src_text]=p_xlated_text;

}
StringName Translation::get_message(const StringName& p_src_text) const {

	const Map<StringName, StringName>::Element *E=translation_map.find(p_src_text);
	if (!E)
		return StringName();

	return E->get();
}

void Translation::erase_message(const StringName& p_src_text) {

	translation_map.erase(p_src_text);
}

void Translation::get_message_list(List<StringName> *r_messages) const {

	for (const Map<StringName, StringName>::Element *E=translation_map.front();E;E=E->next()) {

		r_messages->push_back(E->key());
	}

}

int Translation::get_message_count() const {

	return translation_map.size();
};


void Translation::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_locale","locale"),&Translation::set_locale);
	ObjectTypeDB::bind_method(_MD("get_locale"),&Translation::get_locale);
	ObjectTypeDB::bind_method(_MD("add_message","src_message","xlated_message"),&Translation::add_message);
	ObjectTypeDB::bind_method(_MD("get_message","src_message"),&Translation::get_message);
	ObjectTypeDB::bind_method(_MD("erase_message","src_message"),&Translation::erase_message);
	ObjectTypeDB::bind_method(_MD("get_message_list"),&Translation::_get_message_list);
	ObjectTypeDB::bind_method(_MD("get_message_count"),&Translation::get_message_count);
	ObjectTypeDB::bind_method(_MD("_set_messages"),&Translation::_set_messages);
	ObjectTypeDB::bind_method(_MD("_get_messages"),&Translation::_get_messages);

	ADD_PROPERTY( PropertyInfo(Variant::STRING_ARRAY,"messages",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR), _SCS("_set_messages"), _SCS("_get_messages") );
	ADD_PROPERTY( PropertyInfo(Variant::STRING,"locale"), _SCS("set_locale"), _SCS("get_locale") );
}

Translation::Translation() {

	locale="en";
}



///////////////////////////////////////////////


void TranslationServer::set_locale(const String& p_locale) {

	ERR_EXPLAIN("Invalid Locale: "+p_locale);
	ERR_FAIL_COND(!is_valid_locale(p_locale));
	locale=p_locale;
}

String TranslationServer::get_locale() const {

	return locale;

}

void TranslationServer::add_translation(const Ref<Translation> &p_translation) {

	translations.insert(p_translation);

}
void TranslationServer::remove_translation(const Ref<Translation> &p_translation) {

	translations.erase(p_translation);
}

void TranslationServer::clear() {

	translations.clear();
};

StringName TranslationServer::translate(const StringName& p_message) const {

	//translate using locale

	if (!enabled)
		return p_message;

	StringName res;
	bool near_match=false;
	const CharType *lptr=&locale[0];


	for (const Set< Ref<Translation> >::Element *E=translations.front();E;E=E->next()) {

		const Ref<Translation>& t = E->get();
		String l = t->get_locale();
		if (lptr[0]!=l[0] || lptr[1]!=l[1])
			continue; // locale not match

		//near match
		bool match = (l!=locale);

		if (near_match && !match)
			continue; //only near-match once

		StringName r=t->get_message(p_message);


		if (!r)
			continue;

		res=r;

		if (match)
			break;
		else
			near_match=true;

	}

	if (!res) {
		//try again with fallback
		if (fallback.length()>=2) {

			const CharType *fptr=&fallback[0];
			bool near_match=false;
			for (const Set< Ref<Translation> >::Element *E=translations.front();E;E=E->next()) {

				const Ref<Translation>& t = E->get();
				String l = t->get_locale();
				if (fptr[0]!=l[0] || fptr[1]!=l[1])
					continue; // locale not match

				//near match
				bool match = (l!=fallback);

				if (near_match && !match)
					continue; //only near-match once

				StringName r=t->get_message(p_message);

				if (!r)
					continue;

				res=r;

				if (match)
					break;
				else
					near_match=true;

			}
		}
	}


	if (!res)
		return p_message;

	return res;
}

TranslationServer *TranslationServer::singleton=NULL;

bool TranslationServer::_load_translations(const String& p_from) {

	if (Globals::get_singleton()->has(p_from)) {
		DVector<String> translations=Globals::get_singleton()->get(p_from);

		int tcount=translations.size();

		if (tcount) {
			DVector<String>::Read r = translations.read();

			for(int i=0;i<tcount;i++) {

				print_line( "Loading translation from " + r[i] );
				Ref<Translation> tr = ResourceLoader::load(r[i]);
				if (tr.is_valid())
					add_translation(tr);
			}
		}
		return true;
	}

	return false;
}

void TranslationServer::setup() {

	String test = GLOBAL_DEF("locale/test","");
	test=test.strip_edges();
	if (test!="")
		set_locale( test );
	else
		set_locale( OS::get_singleton()->get_locale() );
	fallback = GLOBAL_DEF("locale/fallback","en");
#ifdef TOOLS_ENABLED

	{
		String options="";
		int idx=0;
		while(locale_list[idx]) {
			if (idx>0)
				options+=", ";
			options+=locale_list[idx];
			idx++;
		}
		Globals::get_singleton()->set_custom_property_info("locale/fallback",PropertyInfo(Variant::STRING,"locale/fallback",PROPERTY_HINT_ENUM,options));
	}
#endif
	//load translations

}

void TranslationServer::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_locale","locale"),&TranslationServer::set_locale);
	ObjectTypeDB::bind_method(_MD("get_locale"),&TranslationServer::get_locale);

	ObjectTypeDB::bind_method(_MD("translate"),&TranslationServer::translate);

	ObjectTypeDB::bind_method(_MD("add_translation"),&TranslationServer::add_translation);
	ObjectTypeDB::bind_method(_MD("remove_translation"),&TranslationServer::remove_translation);

	ObjectTypeDB::bind_method(_MD("clear"),&TranslationServer::clear);

}

void TranslationServer::load_translations() {

	String locale = get_locale();
	bool found = _load_translations("locale/translations"); //all

	if (_load_translations("locale/translations_"+locale.substr(0,2)))
		found=true;
	if ( locale.substr(0,2) != locale ) {
		if (_load_translations("locale/translations_"+locale))
			found=true;
	}


}

TranslationServer::TranslationServer() {


	singleton=this;
	locale="en";
	enabled=true;

}
