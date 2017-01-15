/*************************************************************************/
/*  path_remap.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "path_remap.h"
#include "globals.h"
#include "os/os.h"
#include "translation.h"
PathRemap* PathRemap::singleton=NULL;

PathRemap* PathRemap::get_singleton() {

	return singleton;
}

void PathRemap::add_remap(const String& p_from, const String& p_to,const String& p_locale) {

	if (!remap.has(p_from)) {
		remap[p_from]=RemapData();
	}

	if (p_locale==String())
		remap[p_from].always=p_to;
	else
		remap[p_from].locale[p_locale]=p_to;
}


String PathRemap::get_remap(const String& p_from) const {

	const RemapData *ptr=remap.getptr(p_from);
	if (!ptr) {
		if (OS::get_singleton()->is_stdout_verbose())
			print_line("remap failed: "+p_from);
		return p_from;
	} else {

		const RemapData *ptr2=NULL;

		String locale = TranslationServer::get_singleton()->get_locale();

		if (ptr->locale.has(locale)) {
			if (OS::get_singleton()->is_stdout_verbose())
				print_line("remap found: "+p_from+" -> "+ptr->locale[locale]);

			ptr2=remap.getptr(ptr->locale[locale]);

			if (ptr2 && ptr2->always!=String()) //may have atlas or export remap too
				return ptr2->always;
			else
				return ptr->locale[locale];
		}

		int p = locale.find("_");
		if (p!=-1) {
			locale=locale.substr(0,p);
			if (ptr->locale.has(locale)) {
				if (OS::get_singleton()->is_stdout_verbose())
					print_line("remap found: "+p_from+" -> "+ptr->locale[locale]);

				ptr2=remap.getptr(ptr->locale[locale]);

				if (ptr2 && ptr2->always!=String()) //may have atlas or export remap too
					return ptr2->always;
				else
					return ptr->locale[locale];

			}
		}

		if (ptr->always!=String()) {
			if (OS::get_singleton()->is_stdout_verbose()) {
				print_line("remap found: "+p_from+" -> "+ptr->always);
			}
			return ptr->always;
		}

		if (OS::get_singleton()->is_stdout_verbose())
			print_line("remap failed: "+p_from);

		return p_from;
	}
}
bool PathRemap::has_remap(const String& p_from) const{

	return remap.has(p_from);
}

void PathRemap::erase_remap(const String& p_from){

	ERR_FAIL_COND(!remap.has(p_from));
	remap.erase(p_from);
}

void PathRemap::clear_remaps() {

	remap.clear();
}

void PathRemap::load_remaps() {

	// default remaps first
	PoolVector<String> remaps;
	if (GlobalConfig::get_singleton()->has("remap/all")) {
		remaps = GlobalConfig::get_singleton()->get("remap/all");
	}

	{
		int rlen = remaps.size();

		ERR_FAIL_COND( rlen%2 );
		PoolVector<String>::Read r = remaps.read();
		for(int i=0;i<rlen/2;i++) {

			String from = r[i*2+0];
			String to = r[i*2+1];
			add_remap(from,to);
		}
	}


	// platform remaps second, so override
	if (GlobalConfig::get_singleton()->has("remap/"+OS::get_singleton()->get_name())) {
		remaps = GlobalConfig::get_singleton()->get("remap/"+OS::get_singleton()->get_name());
	} else {
		remaps.resize(0);
	}
	//remaps = Globals::get_singleton()->get("remap/PSP");
	{
		int rlen = remaps.size();

		ERR_FAIL_COND( rlen%2 );
		PoolVector<String>::Read r = remaps.read();
		for(int i=0;i<rlen/2;i++) {

			String from = r[i*2+0];
			String to = r[i*2+1];
			//print_line("add remap: "+from+" -> "+to);
			add_remap(from,to);
		}
	}


	//locale based remaps

	if (GlobalConfig::get_singleton()->has("locale/translation_remaps")) {

		Dictionary remaps = GlobalConfig::get_singleton()->get("locale/translation_remaps");
		List<Variant> rk;
		remaps.get_key_list(&rk);
		for(List<Variant>::Element *E=rk.front();E;E=E->next()) {

			String source = E->get();
			PoolStringArray sa = remaps[E->get()];
			int sas = sa.size();
			PoolStringArray::Read r = sa.read();

			for(int i=0;i<sas;i++) {

				String s = r[i];
				int qp = s.find_last(":");
				if (qp!=-1) {
					String path = s.substr(0,qp);
					String locale = s.substr(qp+1,s.length());
					add_remap(source,path,locale);
				}
			}
		}

	}

}

void PathRemap::_bind_methods() {

	ClassDB::bind_method(_MD("add_remap","from","to","locale"),&PathRemap::add_remap,DEFVAL(String()));
	ClassDB::bind_method(_MD("has_remap","path"),&PathRemap::has_remap);
	ClassDB::bind_method(_MD("get_remap","path"),&PathRemap::get_remap);
	ClassDB::bind_method(_MD("erase_remap","path"),&PathRemap::erase_remap);
	ClassDB::bind_method(_MD("clear_remaps"),&PathRemap::clear_remaps);
}

PathRemap::PathRemap() {

	singleton=this;
}
