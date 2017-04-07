/*************************************************************************/
/*  translation.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef TRANSLATION_H
#define TRANSLATION_H

#include "resource.h"

class Translation : public Resource {

	GDCLASS(Translation, Resource);
	OBJ_SAVE_TYPE(Translation);
	RES_BASE_EXTENSION("xl");

	String locale;
	Map<StringName, StringName> translation_map;

	PoolVector<String> _get_message_list() const;

	PoolVector<String> _get_messages() const;
	void _set_messages(const PoolVector<String> &p_messages);

protected:
	static void _bind_methods();

public:
	void set_locale(const String &p_locale);
	_FORCE_INLINE_ String get_locale() const { return locale; }

	void add_message(const StringName &p_src_text, const StringName &p_xlated_text);
	virtual StringName get_message(const StringName &p_src_text) const; //overridable for other implementations
	void erase_message(const StringName &p_src_text);

	void get_message_list(List<StringName> *r_messages) const;
	int get_message_count() const;

	Translation();
};

class TranslationServer : public Object {

	GDCLASS(TranslationServer, Object);

	String locale;
	String fallback;

	Set<Ref<Translation> > translations;
	Ref<Translation> tool_translation;

	bool enabled;

	static TranslationServer *singleton;
	bool _load_translations(const String &p_from);

	static void _bind_methods();

public:
	_FORCE_INLINE_ static TranslationServer *get_singleton() { return singleton; }

	//yes, portuguese is supported!

	void set_enabled(bool p_enabled) { enabled = p_enabled; }
	_FORCE_INLINE_ bool is_enabled() const { return enabled; }

	void set_locale(const String &p_locale);
	String get_locale() const;

	void add_translation(const Ref<Translation> &p_translation);
	void remove_translation(const Ref<Translation> &p_translation);

	StringName translate(const StringName &p_message) const;

	static Vector<String> get_all_locales();
	static Vector<String> get_all_locale_names();
	static bool is_locale_valid(const String &p_locale);

	void set_tool_translation(const Ref<Translation> &p_translation);
	StringName tool_translate(const StringName &p_message) const;

	void setup();

	void clear();

	void load_translations();

	TranslationServer();
};

#endif // TRANSLATION_H
