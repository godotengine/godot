/**************************************************************************/
/*  translation.h                                                         */
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

#include "core/io/resource.h"
#include "core/object/gdvirtual.gen.inc"

class PluralRules;

class Translation : public Resource {
	GDCLASS(Translation, Resource);
	OBJ_SAVE_TYPE(Translation);
	RES_BASE_EXTENSION("translation");

public:
	struct MessageKey {
		StringName msgctxt;
		StringName msgid;

		// Required to use this struct as a key in HashMap.
		static uint32_t hash(const MessageKey &p_key) {
			uint32_t h = hash_murmur3_one_32(HashMapHasherDefault::hash(p_key.msgctxt));
			return hash_fmix32(hash_murmur3_one_32(HashMapHasherDefault::hash(p_key.msgid), h));
		}
		bool operator==(const MessageKey &p_key) const {
			return msgctxt == p_key.msgctxt && msgid == p_key.msgid;
		}
	};

private:
	String locale = "en";

	HashMap<MessageKey, Vector<StringName>, MessageKey> translation_map;

	mutable PluralRules *plural_rules_cache = nullptr;
	String plural_rules_override;

	virtual Vector<String> _get_message_list() const;

	// For data storage.
	virtual Dictionary _get_messages() const;
	virtual void _set_messages(const Dictionary &p_messages);

protected:
	static void _bind_methods();

	PluralRules *_get_plural_rules() const;

	GDVIRTUAL2RC(StringName, _get_message, StringName, StringName);
	GDVIRTUAL4RC(StringName, _get_plural_message, StringName, StringName, int, StringName);

public:
	void set_locale(const String &p_locale);
	_FORCE_INLINE_ String get_locale() const { return locale; }

	virtual void add_message(const StringName &p_src_text, const StringName &p_xlated_text, const StringName &p_context = "");
	virtual void add_plural_message(const StringName &p_src_text, const Vector<String> &p_plural_xlated_texts, const StringName &p_context = "");
	virtual StringName get_message(const StringName &p_src_text, const StringName &p_context = "") const; //overridable for other implementations
	virtual StringName get_plural_message(const StringName &p_src_text, const StringName &p_plural_text, int p_n, const StringName &p_context = "") const;
	virtual void erase_message(const StringName &p_src_text, const StringName &p_context = "");
	virtual void get_message_list(List<StringName> *r_messages) const;
	virtual int get_message_count() const;
	virtual Vector<String> get_translated_message_list() const;

	void set_plural_rules_override(const String &p_rules);
	String get_plural_rules_override() const;

	// This method is not exposed to scripting intentionally. It is only used by TranslationLoaderPO and tests.
	int get_nplurals() const;

	~Translation();
};
