/*************************************************************************/
/*  gdscript.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "gdscript.h"

#include <stdint.h>

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/core_constants.h"
#include "core/core_string_names.h"
#include "core/io/file_access.h"
#include "core/io/file_access_encrypted.h"
#include "core/os/os.h"
#include "gdscript_analyzer.h"
#include "gdscript_cache.h"
#include "gdscript_compiler.h"
#include "gdscript_parser.h"
#include "gdscript_warning.h"

#ifdef TESTS_ENABLED
#include "tests/gdscript_test_runner.h"
#endif

#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#endif

///////////////////////////

GDScriptNativeClass::GDScriptNativeClass(const StringName &p_name) {
	name = p_name;
}

bool GDScriptNativeClass::_get(const StringName &p_name, Variant &r_ret) const {
	bool ok;
	int v = ClassDB::get_integer_constant(name, p_name, &ok);

	if (ok) {
		r_ret = v;
		return true;
	} else {
		return false;
	}
}

void GDScriptNativeClass::_bind_methods() {
	ClassDB::bind_method(D_METHOD("new"), &GDScriptNativeClass::_new);
}

Variant GDScriptNativeClass::_new() {
	Object *o = instantiate();
	ERR_FAIL_COND_V_MSG(!o, Variant(), "Class type: '" + String(name) + "' is not instantiable.");

	RefCounted *rc = Object::cast_to<RefCounted>(o);
	if (rc) {
		return REF(rc);
	} else {
		return o;
	}
}

Object *GDScriptNativeClass::instantiate() {
	return ClassDB::instantiate(name);
}

GDScriptFunction *GDScript::_super_constructor(GDScript *p_script) {
	if (p_script->initializer) {
		return p_script->initializer;
	} else {
		GDScript *base = p_script->_base;
		if (base != nullptr) {
			return _super_constructor(base);
		} else {
			return nullptr;
		}
	}
}

void GDScript::_super_implicit_constructor(GDScript *p_script, GDScriptInstance *p_instance, Callable::CallError &r_error) {
	GDScript *base = p_script->_base;
	if (base != nullptr) {
		_super_implicit_constructor(base, p_instance, r_error);
		if (r_error.error != Callable::CallError::CALL_OK) {
			return;
		}
	}
	p_script->implicit_initializer->call(p_instance, nullptr, 0, r_error);
}

GDScriptInstance *GDScript::_create_instance(const Variant **p_args, int p_argcount, Object *p_owner, bool p_is_ref_counted, Callable::CallError &r_error) {
	/* STEP 1, CREATE */

	GDScriptInstance *instance = memnew(GDScriptInstance);
	instance->base_ref_counted = p_is_ref_counted;
	instance->members.resize(member_indices.size());
	instance->script = Ref<GDScript>(this);
	instance->owner = p_owner;
	instance->owner_id = p_owner->get_instance_id();
#ifdef DEBUG_ENABLED
	//needed for hot reloading
	for (const KeyValue<StringName, MemberInfo> &E : member_indices) {
		instance->member_indices_cache[E.key] = E.value.index;
	}
#endif
	instance->owner->set_script_instance(instance);

	/* STEP 2, INITIALIZE AND CONSTRUCT */
	{
		MutexLock lock(GDScriptLanguage::singleton->lock);
		instances.insert(instance->owner);
	}

	_super_implicit_constructor(this, instance, r_error);
	if (r_error.error != Callable::CallError::CALL_OK) {
		instance->script = Ref<GDScript>();
		instance->owner->set_script_instance(nullptr);
		{
			MutexLock lock(GDScriptLanguage::singleton->lock);
			instances.erase(p_owner);
		}
		ERR_FAIL_V_MSG(nullptr, "Error constructing a GDScriptInstance.");
	}

	if (p_argcount < 0) {
		return instance;
	}

	initializer = _super_constructor(this);
	if (initializer != nullptr) {
		initializer->call(instance, p_args, p_argcount, r_error);
		if (r_error.error != Callable::CallError::CALL_OK) {
			instance->script = Ref<GDScript>();
			instance->owner->set_script_instance(nullptr);
			{
				MutexLock lock(GDScriptLanguage::singleton->lock);
				instances.erase(p_owner);
			}
			ERR_FAIL_V_MSG(nullptr, "Error constructing a GDScriptInstance.");
		}
	}
	//@TODO make thread safe
	return instance;
}

Variant GDScript::_new(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	/* STEP 1, CREATE */

	if (!valid) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
		return Variant();
	}

	r_error.error = Callable::CallError::CALL_OK;
	REF ref;
	Object *owner = nullptr;

	GDScript *_baseptr = this;
	while (_baseptr->_base) {
		_baseptr = _baseptr->_base;
	}

	ERR_FAIL_COND_V(_baseptr->native.is_null(), Variant());
	if (_baseptr->native.ptr()) {
		owner = _baseptr->native->instantiate();
	} else {
		owner = memnew(RefCounted); //by default, no base means use reference
	}
	ERR_FAIL_COND_V_MSG(!owner, Variant(), "Can't inherit from a virtual class.");

	RefCounted *r = Object::cast_to<RefCounted>(owner);
	if (r) {
		ref = REF(r);
	}

	GDScriptInstance *instance = _create_instance(p_args, p_argcount, owner, r != nullptr, r_error);
	if (!instance) {
		if (ref.is_null()) {
			memdelete(owner); //no owner, sorry
		}
		return Variant();
	}

	if (ref.is_valid()) {
		return ref;
	} else {
		return owner;
	}
}

bool GDScript::can_instantiate() const {
#ifdef TOOLS_ENABLED
	return valid && (tool || ScriptServer::is_scripting_enabled());
#else
	return valid;
#endif
}

Ref<Script> GDScript::get_base_script() const {
	if (_base) {
		return Ref<GDScript>(_base);
	} else {
		return Ref<Script>();
	}
}

StringName GDScript::get_instance_base_type() const {
	if (native.is_valid()) {
		return native->get_name();
	}
	if (base.is_valid() && base->is_valid()) {
		return base->get_instance_base_type();
	}
	return StringName();
}

struct _GDScriptMemberSort {
	int index = 0;
	StringName name;
	_FORCE_INLINE_ bool operator<(const _GDScriptMemberSort &p_member) const { return index < p_member.index; }
};

#ifdef TOOLS_ENABLED

void GDScript::_placeholder_erased(PlaceHolderScriptInstance *p_placeholder) {
	placeholders.erase(p_placeholder);
}
#endif

void GDScript::_get_script_method_list(List<MethodInfo> *r_list, bool p_include_base) const {
	const GDScript *current = this;
	while (current) {
		for (const KeyValue<StringName, GDScriptFunction *> &E : current->member_functions) {
			GDScriptFunction *func = E.value;
			MethodInfo mi;
			mi.name = E.key;
			for (int i = 0; i < func->get_argument_count(); i++) {
				PropertyInfo arginfo = func->get_argument_type(i);
#ifdef TOOLS_ENABLED
				arginfo.name = func->get_argument_name(i);
#endif
				mi.arguments.push_back(arginfo);
			}

			mi.return_val = func->get_return_type();
			r_list->push_back(mi);
		}
		if (!p_include_base) {
			return;
		}

		current = current->_base;
	}
}

void GDScript::get_script_method_list(List<MethodInfo> *r_list) const {
	_get_script_method_list(r_list, true);
}

void GDScript::_get_script_property_list(List<PropertyInfo> *r_list, bool p_include_base) const {
	const GDScript *sptr = this;
	List<PropertyInfo> props;

	while (sptr) {
		Vector<_GDScriptMemberSort> msort;
		for (const KeyValue<StringName, PropertyInfo> &E : sptr->member_info) {
			_GDScriptMemberSort ms;
			ERR_CONTINUE(!sptr->member_indices.has(E.key));
			ms.index = sptr->member_indices[E.key].index;
			ms.name = E.key;
			msort.push_back(ms);
		}

		msort.sort();
		msort.reverse();
		for (int i = 0; i < msort.size(); i++) {
			props.push_front(sptr->member_info[msort[i].name]);
		}
		if (!p_include_base) {
			break;
		}

		sptr = sptr->_base;
	}

	for (const PropertyInfo &E : props) {
		r_list->push_back(E);
	}
}

void GDScript::get_script_property_list(List<PropertyInfo> *r_list) const {
	_get_script_property_list(r_list, true);
}

bool GDScript::has_method(const StringName &p_method) const {
	return member_functions.has(p_method);
}

MethodInfo GDScript::get_method_info(const StringName &p_method) const {
	const Map<StringName, GDScriptFunction *>::Element *E = member_functions.find(p_method);
	if (!E) {
		return MethodInfo();
	}

	GDScriptFunction *func = E->get();
	MethodInfo mi;
	mi.name = E->key();
	for (int i = 0; i < func->get_argument_count(); i++) {
		mi.arguments.push_back(func->get_argument_type(i));
	}

	mi.return_val = func->get_return_type();
	return mi;
}

bool GDScript::get_property_default_value(const StringName &p_property, Variant &r_value) const {
#ifdef TOOLS_ENABLED

	const Map<StringName, Variant>::Element *E = member_default_values_cache.find(p_property);
	if (E) {
		r_value = E->get();
		return true;
	}

	if (base_cache.is_valid()) {
		return base_cache->get_property_default_value(p_property, r_value);
	}
#endif
	return false;
}

ScriptInstance *GDScript::instance_create(Object *p_this) {
	GDScript *top = this;
	while (top->_base) {
		top = top->_base;
	}

	if (top->native.is_valid()) {
		if (!ClassDB::is_parent_class(p_this->get_class_name(), top->native->get_name())) {
			if (EngineDebugger::is_active()) {
				GDScriptLanguage::get_singleton()->debug_break_parse(_get_debug_path(), 1, "Script inherits from native type '" + String(top->native->get_name()) + "', so it can't be instantiated in object of type: '" + p_this->get_class() + "'");
			}
			ERR_FAIL_V_MSG(nullptr, "Script inherits from native type '" + String(top->native->get_name()) + "', so it can't be instantiated in object of type '" + p_this->get_class() + "'" + ".");
		}
	}

	Callable::CallError unchecked_error;
	return _create_instance(nullptr, 0, p_this, Object::cast_to<RefCounted>(p_this) != nullptr, unchecked_error);
}

PlaceHolderScriptInstance *GDScript::placeholder_instance_create(Object *p_this) {
#ifdef TOOLS_ENABLED
	PlaceHolderScriptInstance *si = memnew(PlaceHolderScriptInstance(GDScriptLanguage::get_singleton(), Ref<Script>(this), p_this));
	placeholders.insert(si);
	_update_exports(nullptr, false, si);
	return si;
#else
	return nullptr;
#endif
}

bool GDScript::instance_has(const Object *p_this) const {
	MutexLock lock(GDScriptLanguage::singleton->lock);

	return instances.has((Object *)p_this);
}

bool GDScript::has_source_code() const {
	return !source.is_empty();
}

String GDScript::get_source_code() const {
	return source;
}

void GDScript::set_source_code(const String &p_code) {
	if (source == p_code) {
		return;
	}
	source = p_code;
#ifdef TOOLS_ENABLED
	source_changed_cache = true;
#endif
}

#ifdef TOOLS_ENABLED
void GDScript::_update_exports_values(Map<StringName, Variant> &values, List<PropertyInfo> &propnames) {
	if (base_cache.is_valid()) {
		base_cache->_update_exports_values(values, propnames);
	}

	for (const KeyValue<StringName, Variant> &E : member_default_values_cache) {
		values[E.key] = E.value;
	}

	for (const PropertyInfo &E : members_cache) {
		propnames.push_back(E);
	}
}

void GDScript::_add_doc(const DocData::ClassDoc &p_inner_class) {
	if (_owner) {
		_owner->_add_doc(p_inner_class);
	} else {
		for (int i = 0; i < docs.size(); i++) {
			if (docs[i].name == p_inner_class.name) {
				docs.remove_at(i);
				break;
			}
		}
		docs.append(p_inner_class);
	}
}

void GDScript::_clear_doc() {
	docs.clear();
	doc = DocData::ClassDoc();
}

void GDScript::_update_doc() {
	_clear_doc();

	doc.script_path = "\"" + get_path().get_slice("://", 1) + "\"";
	if (!name.is_empty()) {
		doc.name = name;
	} else {
		doc.name = doc.script_path;
	}

	if (_owner) {
		doc.name = _owner->doc.name + "." + doc.name;
		doc.script_path = doc.script_path + "." + doc.name;
	}

	doc.is_script_doc = true;

	if (base.is_valid() && base->is_valid()) {
		if (!base->doc.name.is_empty()) {
			doc.inherits = base->doc.name;
		} else {
			doc.inherits = base->get_instance_base_type();
		}
	} else if (native.is_valid()) {
		doc.inherits = native->get_name();
	}

	doc.brief_description = doc_brief_description;
	doc.description = doc_description;
	doc.tutorials = doc_tutorials;

	for (const KeyValue<String, DocData::EnumDoc> &E : doc_enums) {
		if (!E.value.description.is_empty()) {
			doc.enums[E.key] = E.value.description;
		}
	}

	List<MethodInfo> methods;
	_get_script_method_list(&methods, false);
	for (int i = 0; i < methods.size(); i++) {
		// Ignore internal methods.
		if (methods[i].name[0] == '@') {
			continue;
		}

		DocData::MethodDoc method_doc;
		const String &class_name = methods[i].name;
		if (member_functions.has(class_name)) {
			GDScriptFunction *fn = member_functions[class_name];

			// Change class name if return type is script reference.
			GDScriptDataType return_type = fn->get_return_type();
			if (return_type.kind == GDScriptDataType::GDSCRIPT) {
				methods[i].return_val.class_name = _get_gdscript_reference_class_name(Object::cast_to<GDScript>(return_type.script_type));
			}

			// Change class name if argument is script reference.
			for (int j = 0; j < fn->get_argument_count(); j++) {
				GDScriptDataType arg_type = fn->get_argument_type(j);
				if (arg_type.kind == GDScriptDataType::GDSCRIPT) {
					methods[i].arguments[j].class_name = _get_gdscript_reference_class_name(Object::cast_to<GDScript>(arg_type.script_type));
				}
			}
		}
		if (doc_functions.has(methods[i].name)) {
			DocData::method_doc_from_methodinfo(method_doc, methods[i], doc_functions[methods[i].name]);
		} else {
			DocData::method_doc_from_methodinfo(method_doc, methods[i], String());
		}
		doc.methods.push_back(method_doc);
	}

	List<PropertyInfo> props;
	_get_script_property_list(&props, false);
	for (int i = 0; i < props.size(); i++) {
		ScriptMemberInfo scr_member_info;
		scr_member_info.propinfo = props[i];
		scr_member_info.propinfo.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
		if (member_indices.has(props[i].name)) {
			const MemberInfo &mi = member_indices[props[i].name];
			scr_member_info.setter = mi.setter;
			scr_member_info.getter = mi.getter;
			if (mi.data_type.kind == GDScriptDataType::GDSCRIPT) {
				scr_member_info.propinfo.class_name = _get_gdscript_reference_class_name(
						Object::cast_to<GDScript>(mi.data_type.script_type));
			}
		}
		if (member_default_values.has(props[i].name)) {
			scr_member_info.has_default_value = true;
			scr_member_info.default_value = member_default_values[props[i].name];
		}
		if (doc_variables.has(props[i].name)) {
			scr_member_info.doc_string = doc_variables[props[i].name];
		}

		DocData::PropertyDoc prop_doc;
		DocData::property_doc_from_scriptmemberinfo(prop_doc, scr_member_info);
		doc.properties.push_back(prop_doc);
	}

	List<MethodInfo> signals;
	_get_script_signal_list(&signals, false);
	for (int i = 0; i < signals.size(); i++) {
		DocData::MethodDoc signal_doc;
		if (doc_signals.has(signals[i].name)) {
			DocData::signal_doc_from_methodinfo(signal_doc, signals[i], doc_signals[signals[i].name]);
		} else {
			DocData::signal_doc_from_methodinfo(signal_doc, signals[i], String());
		}
		doc.signals.push_back(signal_doc);
	}

	for (const KeyValue<StringName, Variant> &E : constants) {
		if (subclasses.has(E.key)) {
			continue;
		}

		// Enums.
		bool is_enum = false;
		if (E.value.get_type() == Variant::DICTIONARY) {
			if (doc_enums.has(E.key)) {
				is_enum = true;
				for (int i = 0; i < doc_enums[E.key].values.size(); i++) {
					doc_enums[E.key].values.write[i].enumeration = E.key;
					doc.constants.push_back(doc_enums[E.key].values[i]);
				}
			}
		}
		if (!is_enum && doc_enums.has("@unnamed_enums")) {
			for (int i = 0; i < doc_enums["@unnamed_enums"].values.size(); i++) {
				if (E.key == doc_enums["@unnamed_enums"].values[i].name) {
					is_enum = true;
					DocData::ConstantDoc constant_doc;
					constant_doc.enumeration = "@unnamed_enums";
					DocData::constant_doc_from_variant(constant_doc, E.key, E.value, doc_enums["@unnamed_enums"].values[i].description);
					doc.constants.push_back(constant_doc);
					break;
				}
			}
		}
		if (!is_enum) {
			DocData::ConstantDoc constant_doc;
			String doc_description;
			if (doc_constants.has(E.key)) {
				doc_description = doc_constants[E.key];
			}
			DocData::constant_doc_from_variant(constant_doc, E.key, E.value, doc_description);
			doc.constants.push_back(constant_doc);
		}
	}

	for (KeyValue<StringName, Ref<GDScript>> &E : subclasses) {
		E.value->_update_doc();
	}

	_add_doc(doc);
}
#endif

bool GDScript::_update_exports(bool *r_err, bool p_recursive_call, PlaceHolderScriptInstance *p_instance_to_update) {
#ifdef TOOLS_ENABLED

	static Vector<GDScript *> base_caches;
	if (!p_recursive_call) {
		base_caches.clear();
	}
	base_caches.append(this);

	bool changed = false;

	if (source_changed_cache) {
		source_changed_cache = false;
		changed = true;

		String basedir = path;

		if (basedir.is_empty()) {
			basedir = get_path();
		}

		if (!basedir.is_empty()) {
			basedir = basedir.get_base_dir();
		}

		GDScriptParser parser;
		GDScriptAnalyzer analyzer(&parser);
		Error err = parser.parse(source, path, false);

		if (err == OK && analyzer.analyze() == OK) {
			const GDScriptParser::ClassNode *c = parser.get_tree();

			if (base_cache.is_valid()) {
				base_cache->inheriters_cache.erase(get_instance_id());
				base_cache = Ref<GDScript>();
			}

			if (c->extends_used) {
				String path = "";
				if (String(c->extends_path) != "" && String(c->extends_path) != get_path()) {
					path = c->extends_path;
					if (path.is_relative_path()) {
						String base = get_path();
						if (base.is_empty() || base.is_relative_path()) {
							ERR_PRINT(("Could not resolve relative path for parent class: " + path).utf8().get_data());
						} else {
							path = base.get_base_dir().plus_file(path);
						}
					}
				} else if (c->extends.size() != 0) {
					const StringName &base = c->extends[0];

					if (ScriptServer::is_global_class(base)) {
						path = ScriptServer::get_global_class_path(base);
					}
				}

				if (!path.is_empty()) {
					if (path != get_path()) {
						Ref<GDScript> bf = ResourceLoader::load(path);

						if (bf.is_valid()) {
							base_cache = bf;
							bf->inheriters_cache.insert(get_instance_id());
						}
					} else {
						ERR_PRINT(("Path extending itself in  " + path).utf8().get_data());
					}
				}
			}

			members_cache.clear();
			member_default_values_cache.clear();
			_signals.clear();

			for (int i = 0; i < c->members.size(); i++) {
				const GDScriptParser::ClassNode::Member &member = c->members[i];

				switch (member.type) {
					case GDScriptParser::ClassNode::Member::VARIABLE: {
						if (!member.variable->exported) {
							continue;
						}

						members_cache.push_back(member.variable->export_info);
						Variant default_value;
						if (member.variable->initializer && member.variable->initializer->is_constant) {
							default_value = member.variable->initializer->reduced_value;
						}
						member_default_values_cache[member.variable->identifier->name] = default_value;
					} break;
					case GDScriptParser::ClassNode::Member::SIGNAL: {
						// TODO: Cache this in parser to avoid loops like this.
						Vector<StringName> parameters_names;
						parameters_names.resize(member.signal->parameters.size());
						for (int j = 0; j < member.signal->parameters.size(); j++) {
							parameters_names.write[j] = member.signal->parameters[j]->identifier->name;
						}
						_signals[member.signal->identifier->name] = parameters_names;
					} break;
					default:
						break; // Nothing.
				}
			}
		} else {
			placeholder_fallback_enabled = true;
			return false;
		}
	} else if (placeholder_fallback_enabled) {
		return false;
	}

	placeholder_fallback_enabled = false;

	if (base_cache.is_valid() && base_cache->is_valid()) {
		for (int i = 0; i < base_caches.size(); i++) {
			if (base_caches[i] == base_cache.ptr()) {
				if (r_err) {
					*r_err = true;
				}
				valid = false; // to show error in the editor
				base_cache->valid = false;
				base_cache->inheriters_cache.clear(); // to prevent future stackoverflows
				base_cache.unref();
				base.unref();
				_base = nullptr;
				ERR_FAIL_V_MSG(false, "Cyclic inheritance in script class.");
			}
		}
		if (base_cache->_update_exports(r_err, true)) {
			if (r_err && *r_err) {
				return false;
			}
			changed = true;
		}
	}

	if ((changed || p_instance_to_update) && placeholders.size()) { //hm :(

		// update placeholders if any
		Map<StringName, Variant> values;
		List<PropertyInfo> propnames;
		_update_exports_values(values, propnames);

		if (changed) {
			for (Set<PlaceHolderScriptInstance *>::Element *E = placeholders.front(); E; E = E->next()) {
				E->get()->update(propnames, values);
			}
		} else {
			p_instance_to_update->update(propnames, values);
		}
	}

	return changed;

#else
	return false;
#endif
}

void GDScript::update_exports() {
#ifdef TOOLS_ENABLED

	bool cyclic_error = false;
	_update_exports(&cyclic_error);
	if (cyclic_error) {
		return;
	}

	Set<ObjectID> copy = inheriters_cache; //might get modified

	for (Set<ObjectID>::Element *E = copy.front(); E; E = E->next()) {
		Object *id = ObjectDB::get_instance(E->get());
		GDScript *s = Object::cast_to<GDScript>(id);
		if (!s) {
			continue;
		}
		s->update_exports();
	}

#endif
}

void GDScript::_set_subclass_path(Ref<GDScript> &p_sc, const String &p_path) {
	p_sc->path = p_path;
	for (KeyValue<StringName, Ref<GDScript>> &E : p_sc->subclasses) {
		_set_subclass_path(E.value, p_path);
	}
}

String GDScript::_get_debug_path() const {
	if (is_built_in() && !get_name().is_empty()) {
		return get_name() + " (" + get_path().get_slice("::", 0) + ")";
	} else {
		return get_path();
	}
}

Error GDScript::reload(bool p_keep_state) {
	bool has_instances;
	{
		MutexLock lock(GDScriptLanguage::singleton->lock);

		has_instances = instances.size();
	}

	ERR_FAIL_COND_V(!p_keep_state && has_instances, ERR_ALREADY_IN_USE);

	String basedir = path;

	if (basedir.is_empty()) {
		basedir = get_path();
	}

	if (!basedir.is_empty()) {
		basedir = basedir.get_base_dir();
	}

// Loading a template, don't parse.
#ifdef TOOLS_ENABLED
	if (basedir.begins_with(EditorSettings::get_singleton()->get_project_script_templates_dir())) {
		return OK;
	}
#else
	if (source.find("_BASE_") != -1) {
		return OK;
	}
#endif

	{
		String source_path = path;
		if (source_path.is_empty()) {
			source_path = get_path();
		}
		if (!source_path.is_empty()) {
			MutexLock lock(GDScriptCache::singleton->lock);
			if (!GDScriptCache::singleton->shallow_gdscript_cache.has(source_path)) {
				GDScriptCache::singleton->shallow_gdscript_cache[source_path] = this;
			}
		}
	}

	valid = false;
	GDScriptParser parser;
	Error err = parser.parse(source, path, false);
	if (err) {
		if (EngineDebugger::is_active()) {
			GDScriptLanguage::get_singleton()->debug_break_parse(_get_debug_path(), parser.get_errors().front()->get().line, "Parser Error: " + parser.get_errors().front()->get().message);
		}
		// TODO: Show all error messages.
		_err_print_error("GDScript::reload", path.is_empty() ? "built-in" : (const char *)path.utf8().get_data(), parser.get_errors().front()->get().line, ("Parse Error: " + parser.get_errors().front()->get().message).utf8().get_data(), false, ERR_HANDLER_SCRIPT);
		ERR_FAIL_V(ERR_PARSE_ERROR);
	}

	GDScriptAnalyzer analyzer(&parser);
	err = analyzer.analyze();

	if (err) {
		if (EngineDebugger::is_active()) {
			GDScriptLanguage::get_singleton()->debug_break_parse(_get_debug_path(), parser.get_errors().front()->get().line, "Parser Error: " + parser.get_errors().front()->get().message);
		}

		const List<GDScriptParser::ParserError>::Element *e = parser.get_errors().front();
		while (e != nullptr) {
			_err_print_error("GDScript::reload", path.is_empty() ? "built-in" : (const char *)path.utf8().get_data(), e->get().line, ("Parse Error: " + e->get().message).utf8().get_data(), false, ERR_HANDLER_SCRIPT);
			e = e->next();
		}
		ERR_FAIL_V(ERR_PARSE_ERROR);
	}

	bool can_run = ScriptServer::is_scripting_enabled() || parser.is_tool();

	GDScriptCompiler compiler;
	err = compiler.compile(&parser, this, p_keep_state);

#ifdef TOOLS_ENABLED
	_update_doc();
#endif

	if (err) {
		if (can_run) {
			if (EngineDebugger::is_active()) {
				GDScriptLanguage::get_singleton()->debug_break_parse(_get_debug_path(), compiler.get_error_line(), "Parser Error: " + compiler.get_error());
			}
			_err_print_error("GDScript::reload", path.is_empty() ? "built-in" : (const char *)path.utf8().get_data(), compiler.get_error_line(), ("Compile Error: " + compiler.get_error()).utf8().get_data(), false, ERR_HANDLER_SCRIPT);
			ERR_FAIL_V(ERR_COMPILATION_FAILED);
		} else {
			return err;
		}
	}
#ifdef DEBUG_ENABLED
	for (const GDScriptWarning &warning : parser.get_warnings()) {
		if (EngineDebugger::is_active()) {
			Vector<ScriptLanguage::StackInfo> si;
			EngineDebugger::get_script_debugger()->send_error("", get_path(), warning.start_line, warning.get_name(), warning.get_message(), false, ERR_HANDLER_WARNING, si);
		}
	}
#endif

	valid = true;

	for (KeyValue<StringName, Ref<GDScript>> &E : subclasses) {
		_set_subclass_path(E.value, path);
	}

	_init_rpc_methods_properties();

	return OK;
}

ScriptLanguage *GDScript::get_language() const {
	return GDScriptLanguage::get_singleton();
}

void GDScript::get_constants(Map<StringName, Variant> *p_constants) {
	if (p_constants) {
		for (const KeyValue<StringName, Variant> &E : constants) {
			(*p_constants)[E.key] = E.value;
		}
	}
}

void GDScript::get_members(Set<StringName> *p_members) {
	if (p_members) {
		for (Set<StringName>::Element *E = members.front(); E; E = E->next()) {
			p_members->insert(E->get());
		}
	}
}

const Vector<Multiplayer::RPCConfig> GDScript::get_rpc_methods() const {
	return rpc_functions;
}

Variant GDScript::call(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	GDScript *top = this;
	while (top) {
		Map<StringName, GDScriptFunction *>::Element *E = top->member_functions.find(p_method);
		if (E) {
			ERR_FAIL_COND_V_MSG(!E->get()->is_static(), Variant(), "Can't call non-static function '" + String(p_method) + "' in script.");

			return E->get()->call(nullptr, p_args, p_argcount, r_error);
		}
		top = top->_base;
	}

	//none found, regular

	return Script::call(p_method, p_args, p_argcount, r_error);
}

bool GDScript::_get(const StringName &p_name, Variant &r_ret) const {
	{
		const GDScript *top = this;
		while (top) {
			{
				const Map<StringName, Variant>::Element *E = top->constants.find(p_name);
				if (E) {
					r_ret = E->get();
					return true;
				}
			}

			{
				const Map<StringName, Ref<GDScript>>::Element *E = subclasses.find(p_name);
				if (E) {
					r_ret = E->get();
					return true;
				}
			}
			top = top->_base;
		}

		if (p_name == GDScriptLanguage::get_singleton()->strings._script_source) {
			r_ret = get_source_code();
			return true;
		}
	}

	return false;
}

bool GDScript::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == GDScriptLanguage::get_singleton()->strings._script_source) {
		set_source_code(p_value);
		reload();
	} else {
		return false;
	}

	return true;
}

void GDScript::_get_property_list(List<PropertyInfo> *p_properties) const {
	p_properties->push_back(PropertyInfo(Variant::STRING, "script/source", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
}

void GDScript::_bind_methods() {
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "new", &GDScript::_new, MethodInfo("new"));

	ClassDB::bind_method(D_METHOD("get_as_byte_code"), &GDScript::get_as_byte_code);
}

Vector<uint8_t> GDScript::get_as_byte_code() const {
	return Vector<uint8_t>();
};

// TODO: Fully remove this. There's not this kind of "bytecode" anymore.
Error GDScript::load_byte_code(const String &p_path) {
	return ERR_COMPILATION_FAILED;
}

Error GDScript::load_source_code(const String &p_path) {
	Vector<uint8_t> sourcef;
	Error err;
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ, &err);
	if (err) {
		ERR_FAIL_COND_V(err, err);
	}

	uint64_t len = f->get_length();
	sourcef.resize(len + 1);
	uint8_t *w = sourcef.ptrw();
	uint64_t r = f->get_buffer(w, len);
	f->close();
	memdelete(f);
	ERR_FAIL_COND_V(r != len, ERR_CANT_OPEN);
	w[len] = 0;

	String s;
	if (s.parse_utf8((const char *)w)) {
		ERR_FAIL_V_MSG(ERR_INVALID_DATA, "Script '" + p_path + "' contains invalid unicode (UTF-8), so it was not loaded. Please ensure that scripts are saved in valid UTF-8 unicode.");
	}

	source = s;
#ifdef TOOLS_ENABLED
	source_changed_cache = true;
#endif
	path = p_path;
	return OK;
}

const Map<StringName, GDScriptFunction *> &GDScript::debug_get_member_functions() const {
	return member_functions;
}

StringName GDScript::debug_get_member_by_index(int p_idx) const {
	for (const KeyValue<StringName, MemberInfo> &E : member_indices) {
		if (E.value.index == p_idx) {
			return E.key;
		}
	}

	return "<error>";
}

Ref<GDScript> GDScript::get_base() const {
	return base;
}

bool GDScript::inherits_script(const Ref<Script> &p_script) const {
	Ref<GDScript> gd = p_script;
	if (gd.is_null()) {
		return false;
	}

	const GDScript *s = this;

	while (s) {
		if (s == p_script.ptr()) {
			return true;
		}
		s = s->_base;
	}

	return false;
}

bool GDScript::has_script_signal(const StringName &p_signal) const {
	if (_signals.has(p_signal)) {
		return true;
	}
	if (base.is_valid()) {
		return base->has_script_signal(p_signal);
	}
#ifdef TOOLS_ENABLED
	else if (base_cache.is_valid()) {
		return base_cache->has_script_signal(p_signal);
	}
#endif
	return false;
}

void GDScript::_get_script_signal_list(List<MethodInfo> *r_list, bool p_include_base) const {
	for (const KeyValue<StringName, Vector<StringName>> &E : _signals) {
		MethodInfo mi;
		mi.name = E.key;
		for (int i = 0; i < E.value.size(); i++) {
			PropertyInfo arg;
			arg.name = E.value[i];
			mi.arguments.push_back(arg);
		}
		r_list->push_back(mi);
	}

	if (!p_include_base) {
		return;
	}

	if (base.is_valid()) {
		base->get_script_signal_list(r_list);
	}
#ifdef TOOLS_ENABLED
	else if (base_cache.is_valid()) {
		base_cache->get_script_signal_list(r_list);
	}

#endif
}

void GDScript::get_script_signal_list(List<MethodInfo> *r_signals) const {
	_get_script_signal_list(r_signals, true);
}

String GDScript::_get_gdscript_reference_class_name(const GDScript *p_gdscript) {
	ERR_FAIL_NULL_V(p_gdscript, String());

	String class_name;
	while (p_gdscript) {
		if (class_name.is_empty()) {
			class_name = p_gdscript->get_script_class_name();
		} else {
			class_name = p_gdscript->get_script_class_name() + "." + class_name;
		}
		p_gdscript = p_gdscript->_owner;
	}
	return class_name;
}

GDScript::GDScript() :
		script_list(this) {
#ifdef DEBUG_ENABLED
	{
		MutexLock lock(GDScriptLanguage::get_singleton()->lock);

		GDScriptLanguage::get_singleton()->script_list.add(&script_list);
	}
#endif
}

void GDScript::_save_orphaned_subclasses() {
	struct ClassRefWithName {
		ObjectID id;
		String fully_qualified_name;
	};
	Vector<ClassRefWithName> weak_subclasses;
	// collect subclasses ObjectID and name
	for (KeyValue<StringName, Ref<GDScript>> &E : subclasses) {
		E.value->_owner = nullptr; //bye, you are no longer owned cause I died
		ClassRefWithName subclass;
		subclass.id = E.value->get_instance_id();
		subclass.fully_qualified_name = E.value->fully_qualified_name;
		weak_subclasses.push_back(subclass);
	}

	// clear subclasses to allow unused subclasses to be deleted
	subclasses.clear();
	// subclasses are also held by constants, clear those as well
	constants.clear();

	// keep orphan subclass only for subclasses that are still in use
	for (int i = 0; i < weak_subclasses.size(); i++) {
		ClassRefWithName subclass = weak_subclasses[i];
		Object *obj = ObjectDB::get_instance(subclass.id);
		if (!obj) {
			continue;
		}
		// subclass is not released
		GDScriptLanguage::get_singleton()->add_orphan_subclass(subclass.fully_qualified_name, subclass.id);
	}
}

void GDScript::_init_rpc_methods_properties() {
	// Copy the base rpc methods so we don't mask their IDs.
	rpc_functions.clear();
	if (base.is_valid()) {
		rpc_functions = base->rpc_functions;
	}

	GDScript *cscript = this;
	Map<StringName, Ref<GDScript>>::Element *sub_E = subclasses.front();
	while (cscript) {
		// RPC Methods
		for (KeyValue<StringName, GDScriptFunction *> &E : cscript->member_functions) {
			Multiplayer::RPCConfig config = E.value->get_rpc_config();
			if (config.rpc_mode != Multiplayer::RPC_MODE_DISABLED) {
				config.name = E.value->get_name();
				if (rpc_functions.find(config) == -1) {
					rpc_functions.push_back(config);
				}
			}
		}

		if (cscript != this) {
			sub_E = sub_E->next();
		}

		if (sub_E) {
			cscript = sub_E->get().ptr();
		} else {
			cscript = nullptr;
		}
	}

	// Sort so we are 100% that they are always the same.
	rpc_functions.sort_custom<Multiplayer::SortRPCConfig>();
}

GDScript::~GDScript() {
	{
		MutexLock lock(GDScriptLanguage::get_singleton()->lock);

		while (SelfList<GDScriptFunctionState> *E = pending_func_states.first()) {
			// Order matters since clearing the stack may already cause
			// the GDSCriptFunctionState to be destroyed and thus removed from the list.
			pending_func_states.remove(E);
			E->self()->_clear_stack();
		}
	}

	for (const KeyValue<StringName, GDScriptFunction *> &E : member_functions) {
		memdelete(E.value);
	}

	if (GDScriptCache::singleton) { // Cache may have been already destroyed at engine shutdown.
		GDScriptCache::remove_script(get_path());
	}

	_save_orphaned_subclasses();

#ifdef TOOLS_ENABLED
	// Clearing inner class doc, script doc only cleared when the script source deleted.
	if (_owner) {
		_clear_doc();
	}
#endif

#ifdef DEBUG_ENABLED
	{
		MutexLock lock(GDScriptLanguage::get_singleton()->lock);

		GDScriptLanguage::get_singleton()->script_list.remove(&script_list);
	}
#endif
}

//////////////////////////////
//         INSTANCE         //
//////////////////////////////

bool GDScriptInstance::set(const StringName &p_name, const Variant &p_value) {
	//member
	{
		const Map<StringName, GDScript::MemberInfo>::Element *E = script->member_indices.find(p_name);
		if (E) {
			const GDScript::MemberInfo *member = &E->get();
			if (member->setter) {
				const Variant *val = &p_value;
				Callable::CallError err;
				call(member->setter, &val, 1, err);
				if (err.error == Callable::CallError::CALL_OK) {
					return true; //function exists, call was successful
				} else {
					return false;
				}
			} else {
				if (member->data_type.has_type) {
					if (member->data_type.builtin_type == Variant::ARRAY && member->data_type.has_container_element_type()) {
						// Typed array.
						if (p_value.get_type() == Variant::ARRAY) {
							return VariantInternal::get_array(&members.write[member->index])->typed_assign(p_value);
						} else {
							return false;
						}
					} else if (!member->data_type.is_type(p_value)) {
						// Try conversion
						Callable::CallError ce;
						const Variant *value = &p_value;
						Variant converted;
						Variant::construct(member->data_type.builtin_type, converted, &value, 1, ce);
						if (ce.error == Callable::CallError::CALL_OK) {
							members.write[member->index] = converted;
							return true;
						} else {
							return false;
						}
					}
				}
				members.write[member->index] = p_value;
			}
			return true;
		}
	}

	GDScript *sptr = script.ptr();
	while (sptr) {
		Map<StringName, GDScriptFunction *>::Element *E = sptr->member_functions.find(GDScriptLanguage::get_singleton()->strings._set);
		if (E) {
			Variant name = p_name;
			const Variant *args[2] = { &name, &p_value };

			Callable::CallError err;
			Variant ret = E->get()->call(this, (const Variant **)args, 2, err);
			if (err.error == Callable::CallError::CALL_OK && ret.get_type() == Variant::BOOL && ret.operator bool()) {
				return true;
			}
		}
		sptr = sptr->_base;
	}

	return false;
}

bool GDScriptInstance::get(const StringName &p_name, Variant &r_ret) const {
	const GDScript *sptr = script.ptr();
	while (sptr) {
		{
			const Map<StringName, GDScript::MemberInfo>::Element *E = script->member_indices.find(p_name);
			if (E) {
				if (E->get().getter) {
					Callable::CallError err;
					r_ret = const_cast<GDScriptInstance *>(this)->call(E->get().getter, nullptr, 0, err);
					if (err.error == Callable::CallError::CALL_OK) {
						return true;
					}
				}
				r_ret = members[E->get().index];
				return true; //index found
			}
		}

		{
			const GDScript *sl = sptr;
			while (sl) {
				const Map<StringName, Variant>::Element *E = sl->constants.find(p_name);
				if (E) {
					r_ret = E->get();
					return true; //index found
				}
				sl = sl->_base;
			}
		}

		{
			// Signals.
			const GDScript *sl = sptr;
			while (sl) {
				const Map<StringName, Vector<StringName>>::Element *E = sl->_signals.find(p_name);
				if (E) {
					r_ret = Signal(this->owner, E->key());
					return true; //index found
				}
				sl = sl->_base;
			}
		}

		{
			// Methods.
			const GDScript *sl = sptr;
			while (sl) {
				const Map<StringName, GDScriptFunction *>::Element *E = sl->member_functions.find(p_name);
				if (E) {
					r_ret = Callable(this->owner, E->key());
					return true; //index found
				}
				sl = sl->_base;
			}
		}

		{
			const Map<StringName, GDScriptFunction *>::Element *E = sptr->member_functions.find(GDScriptLanguage::get_singleton()->strings._get);
			if (E) {
				Variant name = p_name;
				const Variant *args[1] = { &name };

				Callable::CallError err;
				Variant ret = const_cast<GDScriptFunction *>(E->get())->call(const_cast<GDScriptInstance *>(this), (const Variant **)args, 1, err);
				if (err.error == Callable::CallError::CALL_OK && ret.get_type() != Variant::NIL) {
					r_ret = ret;
					return true;
				}
			}
		}
		sptr = sptr->_base;
	}

	return false;
}

Variant::Type GDScriptInstance::get_property_type(const StringName &p_name, bool *r_is_valid) const {
	const GDScript *sptr = script.ptr();
	while (sptr) {
		if (sptr->member_info.has(p_name)) {
			if (r_is_valid) {
				*r_is_valid = true;
			}
			return sptr->member_info[p_name].type;
		}
		sptr = sptr->_base;
	}

	if (r_is_valid) {
		*r_is_valid = false;
	}
	return Variant::NIL;
}

void GDScriptInstance::get_property_list(List<PropertyInfo> *p_properties) const {
	// exported members, not done yet!

	const GDScript *sptr = script.ptr();
	List<PropertyInfo> props;

	while (sptr) {
		const Map<StringName, GDScriptFunction *>::Element *E = sptr->member_functions.find(GDScriptLanguage::get_singleton()->strings._get_property_list);
		if (E) {
			Callable::CallError err;
			Variant ret = const_cast<GDScriptFunction *>(E->get())->call(const_cast<GDScriptInstance *>(this), nullptr, 0, err);
			if (err.error == Callable::CallError::CALL_OK) {
				ERR_FAIL_COND_MSG(ret.get_type() != Variant::ARRAY, "Wrong type for _get_property_list, must be an array of dictionaries.");

				Array arr = ret;
				for (int i = 0; i < arr.size(); i++) {
					Dictionary d = arr[i];
					ERR_CONTINUE(!d.has("name"));
					ERR_CONTINUE(!d.has("type"));
					PropertyInfo pinfo;
					pinfo.type = Variant::Type(d["type"].operator int());
					ERR_CONTINUE(pinfo.type < 0 || pinfo.type >= Variant::VARIANT_MAX);
					pinfo.name = d["name"];
					ERR_CONTINUE(pinfo.name.is_empty());
					if (d.has("hint")) {
						pinfo.hint = PropertyHint(d["hint"].operator int());
					}
					if (d.has("hint_string")) {
						pinfo.hint_string = d["hint_string"];
					}
					if (d.has("usage")) {
						pinfo.usage = d["usage"];
					}

					props.push_back(pinfo);
				}
			}
		}

		//instance a fake script for editing the values

		Vector<_GDScriptMemberSort> msort;
		for (const KeyValue<StringName, PropertyInfo> &F : sptr->member_info) {
			_GDScriptMemberSort ms;
			ERR_CONTINUE(!sptr->member_indices.has(F.key));
			ms.index = sptr->member_indices[F.key].index;
			ms.name = F.key;
			msort.push_back(ms);
		}

		msort.sort();
		msort.reverse();
		for (int i = 0; i < msort.size(); i++) {
			props.push_front(sptr->member_info[msort[i].name]);
		}

		sptr = sptr->_base;
	}

	for (const PropertyInfo &E : props) {
		p_properties->push_back(E);
	}
}

void GDScriptInstance::get_method_list(List<MethodInfo> *p_list) const {
	const GDScript *sptr = script.ptr();
	while (sptr) {
		for (const KeyValue<StringName, GDScriptFunction *> &E : sptr->member_functions) {
			MethodInfo mi;
			mi.name = E.key;
			mi.flags |= METHOD_FLAG_FROM_SCRIPT;
			for (int i = 0; i < E.value->get_argument_count(); i++) {
				mi.arguments.push_back(PropertyInfo(Variant::NIL, "arg" + itos(i)));
			}
			p_list->push_back(mi);
		}
		sptr = sptr->_base;
	}
}

bool GDScriptInstance::has_method(const StringName &p_method) const {
	const GDScript *sptr = script.ptr();
	while (sptr) {
		const Map<StringName, GDScriptFunction *>::Element *E = sptr->member_functions.find(p_method);
		if (E) {
			return true;
		}
		sptr = sptr->_base;
	}

	return false;
}

Variant GDScriptInstance::call(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	GDScript *sptr = script.ptr();
	while (sptr) {
		Map<StringName, GDScriptFunction *>::Element *E = sptr->member_functions.find(p_method);
		if (E) {
			return E->get()->call(this, p_args, p_argcount, r_error);
		}
		sptr = sptr->_base;
	}
	r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
	return Variant();
}

void GDScriptInstance::notification(int p_notification) {
	//notification is not virtual, it gets called at ALL levels just like in C.
	Variant value = p_notification;
	const Variant *args[1] = { &value };

	GDScript *sptr = script.ptr();
	while (sptr) {
		Map<StringName, GDScriptFunction *>::Element *E = sptr->member_functions.find(GDScriptLanguage::get_singleton()->strings._notification);
		if (E) {
			Callable::CallError err;
			E->get()->call(this, args, 1, err);
			if (err.error != Callable::CallError::CALL_OK) {
				//print error about notification call
			}
		}
		sptr = sptr->_base;
	}
}

String GDScriptInstance::to_string(bool *r_valid) {
	if (has_method(CoreStringNames::get_singleton()->_to_string)) {
		Callable::CallError ce;
		Variant ret = call(CoreStringNames::get_singleton()->_to_string, nullptr, 0, ce);
		if (ce.error == Callable::CallError::CALL_OK) {
			if (ret.get_type() != Variant::STRING) {
				if (r_valid) {
					*r_valid = false;
				}
				ERR_FAIL_V_MSG(String(), "Wrong type for " + CoreStringNames::get_singleton()->_to_string + ", must be a String.");
			}
			if (r_valid) {
				*r_valid = true;
			}
			return ret.operator String();
		}
	}
	if (r_valid) {
		*r_valid = false;
	}
	return String();
}

Ref<Script> GDScriptInstance::get_script() const {
	return script;
}

ScriptLanguage *GDScriptInstance::get_language() {
	return GDScriptLanguage::get_singleton();
}

const Vector<Multiplayer::RPCConfig> GDScriptInstance::get_rpc_methods() const {
	return script->get_rpc_methods();
}

void GDScriptInstance::reload_members() {
#ifdef DEBUG_ENABLED

	members.resize(script->member_indices.size()); //resize

	Vector<Variant> new_members;
	new_members.resize(script->member_indices.size());

	//pass the values to the new indices
	for (KeyValue<StringName, GDScript::MemberInfo> &E : script->member_indices) {
		if (member_indices_cache.has(E.key)) {
			Variant value = members[member_indices_cache[E.key]];
			new_members.write[E.value.index] = value;
		}
	}

	//apply
	members = new_members;

	//pass the values to the new indices
	member_indices_cache.clear();
	for (const KeyValue<StringName, GDScript::MemberInfo> &E : script->member_indices) {
		member_indices_cache[E.key] = E.value.index;
	}

#endif
}

GDScriptInstance::GDScriptInstance() {
	owner = nullptr;
	base_ref_counted = false;
}

GDScriptInstance::~GDScriptInstance() {
	MutexLock lock(GDScriptLanguage::get_singleton()->lock);

	while (SelfList<GDScriptFunctionState> *E = pending_func_states.first()) {
		// Order matters since clearing the stack may already cause
		// the GDSCriptFunctionState to be destroyed and thus removed from the list.
		pending_func_states.remove(E);
		E->self()->_clear_stack();
	}

	if (script.is_valid() && owner) {
		script->instances.erase(owner);
	}
}

/************* SCRIPT LANGUAGE **************/

GDScriptLanguage *GDScriptLanguage::singleton = nullptr;

String GDScriptLanguage::get_name() const {
	return "GDScript";
}

/* LANGUAGE FUNCTIONS */

void GDScriptLanguage::_add_global(const StringName &p_name, const Variant &p_value) {
	if (globals.has(p_name)) {
		//overwrite existing
		global_array.write[globals[p_name]] = p_value;
		return;
	}
	globals[p_name] = global_array.size();
	global_array.push_back(p_value);
	_global_array = global_array.ptrw();
}

void GDScriptLanguage::add_global_constant(const StringName &p_variable, const Variant &p_value) {
	_add_global(p_variable, p_value);
}

void GDScriptLanguage::add_named_global_constant(const StringName &p_name, const Variant &p_value) {
	named_globals[p_name] = p_value;
}

void GDScriptLanguage::remove_named_global_constant(const StringName &p_name) {
	ERR_FAIL_COND(!named_globals.has(p_name));
	named_globals.erase(p_name);
}

void GDScriptLanguage::init() {
	//populate global constants
	int gcc = CoreConstants::get_global_constant_count();
	for (int i = 0; i < gcc; i++) {
		_add_global(StaticCString::create(CoreConstants::get_global_constant_name(i)), CoreConstants::get_global_constant_value(i));
	}

	_add_global(StaticCString::create("PI"), Math_PI);
	_add_global(StaticCString::create("TAU"), Math_TAU);
	_add_global(StaticCString::create("INF"), INFINITY);
	_add_global(StaticCString::create("NAN"), NAN);

	//populate native classes

	List<StringName> class_list;
	ClassDB::get_class_list(&class_list);
	for (const StringName &n : class_list) {
		if (globals.has(n)) {
			continue;
		}
		Ref<GDScriptNativeClass> nc = memnew(GDScriptNativeClass(n));
		_add_global(n, nc);
	}

	//populate singletons

	List<Engine::Singleton> singletons;
	Engine::get_singleton()->get_singletons(&singletons);
	for (const Engine::Singleton &E : singletons) {
		_add_global(E.name, E.ptr);
	}

#ifdef TESTS_ENABLED
	GDScriptTests::GDScriptTestRunner::handle_cmdline();
#endif
}

String GDScriptLanguage::get_type() const {
	return "GDScript";
}

String GDScriptLanguage::get_extension() const {
	return "gd";
}

Error GDScriptLanguage::execute_file(const String &p_path) {
	// ??
	return OK;
}

void GDScriptLanguage::finish() {
}

void GDScriptLanguage::profiling_start() {
#ifdef DEBUG_ENABLED
	MutexLock lock(this->lock);

	SelfList<GDScriptFunction> *elem = function_list.first();
	while (elem) {
		elem->self()->profile.call_count = 0;
		elem->self()->profile.self_time = 0;
		elem->self()->profile.total_time = 0;
		elem->self()->profile.frame_call_count = 0;
		elem->self()->profile.frame_self_time = 0;
		elem->self()->profile.frame_total_time = 0;
		elem->self()->profile.last_frame_call_count = 0;
		elem->self()->profile.last_frame_self_time = 0;
		elem->self()->profile.last_frame_total_time = 0;
		elem = elem->next();
	}

	profiling = true;
#endif
}

void GDScriptLanguage::profiling_stop() {
#ifdef DEBUG_ENABLED
	MutexLock lock(this->lock);

	profiling = false;
#endif
}

int GDScriptLanguage::profiling_get_accumulated_data(ProfilingInfo *p_info_arr, int p_info_max) {
	int current = 0;
#ifdef DEBUG_ENABLED

	MutexLock lock(this->lock);

	SelfList<GDScriptFunction> *elem = function_list.first();
	while (elem) {
		if (current >= p_info_max) {
			break;
		}
		p_info_arr[current].call_count = elem->self()->profile.call_count;
		p_info_arr[current].self_time = elem->self()->profile.self_time;
		p_info_arr[current].total_time = elem->self()->profile.total_time;
		p_info_arr[current].signature = elem->self()->profile.signature;
		elem = elem->next();
		current++;
	}
#endif

	return current;
}

int GDScriptLanguage::profiling_get_frame_data(ProfilingInfo *p_info_arr, int p_info_max) {
	int current = 0;

#ifdef DEBUG_ENABLED
	MutexLock lock(this->lock);

	SelfList<GDScriptFunction> *elem = function_list.first();
	while (elem) {
		if (current >= p_info_max) {
			break;
		}
		if (elem->self()->profile.last_frame_call_count > 0) {
			p_info_arr[current].call_count = elem->self()->profile.last_frame_call_count;
			p_info_arr[current].self_time = elem->self()->profile.last_frame_self_time;
			p_info_arr[current].total_time = elem->self()->profile.last_frame_total_time;
			p_info_arr[current].signature = elem->self()->profile.signature;
			current++;
		}
		elem = elem->next();
	}
#endif

	return current;
}

struct GDScriptDepSort {
	//must support sorting so inheritance works properly (parent must be reloaded first)
	bool operator()(const Ref<GDScript> &A, const Ref<GDScript> &B) const {
		if (A == B) {
			return false; //shouldn't happen but..
		}
		const GDScript *I = B->get_base().ptr();
		while (I) {
			if (I == A.ptr()) {
				// A is a base of B
				return true;
			}

			I = I->get_base().ptr();
		}

		return false; //not a base
	}
};

void GDScriptLanguage::reload_all_scripts() {
#ifdef DEBUG_ENABLED
	print_verbose("GDScript: Reloading all scripts");
	List<Ref<GDScript>> scripts;
	{
		MutexLock lock(this->lock);

		SelfList<GDScript> *elem = script_list.first();
		while (elem) {
			if (elem->self()->get_path().is_resource_file()) {
				print_verbose("GDScript: Found: " + elem->self()->get_path());
				scripts.push_back(Ref<GDScript>(elem->self())); //cast to gdscript to avoid being erased by accident
			}
			elem = elem->next();
		}
	}

	//as scripts are going to be reloaded, must proceed without locking here

	scripts.sort_custom<GDScriptDepSort>(); //update in inheritance dependency order

	for (Ref<GDScript> &script : scripts) {
		print_verbose("GDScript: Reloading: " + script->get_path());
		script->load_source_code(script->get_path());
		script->reload(true);
	}
#endif
}

void GDScriptLanguage::reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload) {
#ifdef DEBUG_ENABLED

	List<Ref<GDScript>> scripts;
	{
		MutexLock lock(this->lock);

		SelfList<GDScript> *elem = script_list.first();
		while (elem) {
			if (elem->self()->get_path().is_resource_file()) {
				scripts.push_back(Ref<GDScript>(elem->self())); //cast to gdscript to avoid being erased by accident
			}
			elem = elem->next();
		}
	}

	//when someone asks you why dynamically typed languages are easier to write....

	Map<Ref<GDScript>, Map<ObjectID, List<Pair<StringName, Variant>>>> to_reload;

	//as scripts are going to be reloaded, must proceed without locking here

	scripts.sort_custom<GDScriptDepSort>(); //update in inheritance dependency order

	for (Ref<GDScript> &script : scripts) {
		bool reload = script == p_script || to_reload.has(script->get_base());

		if (!reload) {
			continue;
		}

		to_reload.insert(script, Map<ObjectID, List<Pair<StringName, Variant>>>());

		if (!p_soft_reload) {
			//save state and remove script from instances
			Map<ObjectID, List<Pair<StringName, Variant>>> &map = to_reload[script];

			while (script->instances.front()) {
				Object *obj = script->instances.front()->get();
				//save instance info
				List<Pair<StringName, Variant>> state;
				if (obj->get_script_instance()) {
					obj->get_script_instance()->get_property_state(state);
					map[obj->get_instance_id()] = state;
					obj->set_script(Variant());
				}
			}

//same thing for placeholders
#ifdef TOOLS_ENABLED

			while (script->placeholders.size()) {
				Object *obj = script->placeholders.front()->get()->get_owner();

				//save instance info
				if (obj->get_script_instance()) {
					map.insert(obj->get_instance_id(), List<Pair<StringName, Variant>>());
					List<Pair<StringName, Variant>> &state = map[obj->get_instance_id()];
					obj->get_script_instance()->get_property_state(state);
					obj->set_script(Variant());
				} else {
					// no instance found. Let's remove it so we don't loop forever
					script->placeholders.erase(script->placeholders.front()->get());
				}
			}

#endif

			for (const KeyValue<ObjectID, List<Pair<StringName, Variant>>> &F : script->pending_reload_state) {
				map[F.key] = F.value; //pending to reload, use this one instead
			}
		}
	}

	for (KeyValue<Ref<GDScript>, Map<ObjectID, List<Pair<StringName, Variant>>>> &E : to_reload) {
		Ref<GDScript> scr = E.key;
		scr->reload(p_soft_reload);

		//restore state if saved
		for (KeyValue<ObjectID, List<Pair<StringName, Variant>>> &F : E.value) {
			List<Pair<StringName, Variant>> &saved_state = F.value;

			Object *obj = ObjectDB::get_instance(F.key);
			if (!obj) {
				continue;
			}

			if (!p_soft_reload) {
				//clear it just in case (may be a pending reload state)
				obj->set_script(Variant());
			}
			obj->set_script(scr);

			ScriptInstance *script_instance = obj->get_script_instance();

			if (!script_instance) {
				//failed, save reload state for next time if not saved
				if (!scr->pending_reload_state.has(obj->get_instance_id())) {
					scr->pending_reload_state[obj->get_instance_id()] = saved_state;
				}
				continue;
			}

			if (script_instance->is_placeholder() && scr->is_placeholder_fallback_enabled()) {
				PlaceHolderScriptInstance *placeholder = static_cast<PlaceHolderScriptInstance *>(script_instance);
				for (List<Pair<StringName, Variant>>::Element *G = saved_state.front(); G; G = G->next()) {
					placeholder->property_set_fallback(G->get().first, G->get().second);
				}
			} else {
				for (List<Pair<StringName, Variant>>::Element *G = saved_state.front(); G; G = G->next()) {
					script_instance->set(G->get().first, G->get().second);
				}
			}

			scr->pending_reload_state.erase(obj->get_instance_id()); //as it reloaded, remove pending state
		}

		//if instance states were saved, set them!
	}

#endif
}

void GDScriptLanguage::frame() {
	calls = 0;

#ifdef DEBUG_ENABLED
	if (profiling) {
		MutexLock lock(this->lock);

		SelfList<GDScriptFunction> *elem = function_list.first();
		while (elem) {
			elem->self()->profile.last_frame_call_count = elem->self()->profile.frame_call_count;
			elem->self()->profile.last_frame_self_time = elem->self()->profile.frame_self_time;
			elem->self()->profile.last_frame_total_time = elem->self()->profile.frame_total_time;
			elem->self()->profile.frame_call_count = 0;
			elem->self()->profile.frame_self_time = 0;
			elem->self()->profile.frame_total_time = 0;
			elem = elem->next();
		}
	}

#endif
}

/* EDITOR FUNCTIONS */
void GDScriptLanguage::get_reserved_words(List<String> *p_words) const {
	// TODO: Add annotations here?
	static const char *_reserved_words[] = {
		// operators
		"and",
		"in",
		"not",
		"or",
		// types and values
		"false",
		"float",
		"int",
		"bool",
		"null",
		"PI",
		"TAU",
		"INF",
		"NAN",
		"self",
		"true",
		"void",
		// functions
		"as",
		"assert",
		"await",
		"breakpoint",
		"class",
		"class_name",
		"extends",
		"is",
		"func",
		"preload",
		"signal",
		"super",
		"trait",
		"yield",
		// var
		"const",
		"enum",
		"static",
		"var",
		// control flow
		"break",
		"continue",
		"if",
		"elif",
		"else",
		"for",
		"pass",
		"return",
		"match",
		"while",
		nullptr
	};

	const char **w = _reserved_words;

	while (*w) {
		p_words->push_back(*w);
		w++;
	}

	List<StringName> functions;
	GDScriptUtilityFunctions::get_function_list(&functions);

	for (const StringName &E : functions) {
		p_words->push_back(String(E));
	}
}

bool GDScriptLanguage::is_control_flow_keyword(String p_keyword) const {
	return p_keyword == "break" ||
			p_keyword == "continue" ||
			p_keyword == "elif" ||
			p_keyword == "else" ||
			p_keyword == "if" ||
			p_keyword == "for" ||
			p_keyword == "match" ||
			p_keyword == "pass" ||
			p_keyword == "return" ||
			p_keyword == "while";
}

bool GDScriptLanguage::handles_global_class_type(const String &p_type) const {
	return p_type == "GDScript";
}

String GDScriptLanguage::get_global_class_name(const String &p_path, String *r_base_type, String *r_icon_path) const {
	Vector<uint8_t> sourcef;
	Error err;
	FileAccessRef f = FileAccess::open(p_path, FileAccess::READ, &err);
	if (err) {
		return String();
	}

	String source = f->get_as_utf8_string();

	GDScriptParser parser;
	err = parser.parse(source, p_path, false);

	// TODO: Simplify this code by using the analyzer to get full inheritance.
	if (err == OK) {
		const GDScriptParser::ClassNode *c = parser.get_tree();
		if (r_icon_path) {
			if (c->icon_path.is_empty() || c->icon_path.is_absolute_path()) {
				*r_icon_path = c->icon_path;
			} else if (c->icon_path.is_relative_path()) {
				*r_icon_path = p_path.get_base_dir().plus_file(c->icon_path).simplify_path();
			}
		}
		if (r_base_type) {
			const GDScriptParser::ClassNode *subclass = c;
			String path = p_path;
			GDScriptParser subparser;
			while (subclass) {
				if (subclass->extends_used) {
					if (!subclass->extends_path.is_empty()) {
						if (subclass->extends.size() == 0) {
							get_global_class_name(subclass->extends_path, r_base_type);
							subclass = nullptr;
							break;
						} else {
							Vector<StringName> extend_classes = subclass->extends;

							FileAccessRef subfile = FileAccess::open(subclass->extends_path, FileAccess::READ);
							if (!subfile) {
								break;
							}
							String subsource = subfile->get_as_utf8_string();

							if (subsource.is_empty()) {
								break;
							}
							String subpath = subclass->extends_path;
							if (subpath.is_relative_path()) {
								subpath = path.get_base_dir().plus_file(subpath).simplify_path();
							}

							if (OK != subparser.parse(subsource, subpath, false)) {
								break;
							}
							path = subpath;
							subclass = subparser.get_tree();

							while (extend_classes.size() > 0) {
								bool found = false;
								for (int i = 0; i < subclass->members.size(); i++) {
									if (subclass->members[i].type != GDScriptParser::ClassNode::Member::CLASS) {
										continue;
									}

									const GDScriptParser::ClassNode *inner_class = subclass->members[i].m_class;
									if (inner_class->identifier->name == extend_classes[0]) {
										extend_classes.remove_at(0);
										found = true;
										subclass = inner_class;
										break;
									}
								}
								if (!found) {
									subclass = nullptr;
									break;
								}
							}
						}
					} else if (subclass->extends.size() == 1) {
						*r_base_type = subclass->extends[0];
						subclass = nullptr;
					} else {
						break;
					}
				} else {
					*r_base_type = "RefCounted";
					subclass = nullptr;
				}
			}
		}
		return c->identifier != nullptr ? String(c->identifier->name) : String();
	}

	return String();
}

GDScriptLanguage::GDScriptLanguage() {
	calls = 0;
	ERR_FAIL_COND(singleton);
	singleton = this;
	strings._init = StaticCString::create("_init");
	strings._notification = StaticCString::create("_notification");
	strings._set = StaticCString::create("_set");
	strings._get = StaticCString::create("_get");
	strings._get_property_list = StaticCString::create("_get_property_list");
	strings._script_source = StaticCString::create("script/source");
	_debug_parse_err_line = -1;
	_debug_parse_err_file = "";

	profiling = false;
	script_frame_time = 0;

	_debug_call_stack_pos = 0;
	int dmcs = GLOBAL_DEF("debug/settings/gdscript/max_call_stack", 1024);
	ProjectSettings::get_singleton()->set_custom_property_info("debug/settings/gdscript/max_call_stack", PropertyInfo(Variant::INT, "debug/settings/gdscript/max_call_stack", PROPERTY_HINT_RANGE, "1024,4096,1,or_greater")); //minimum is 1024

	if (EngineDebugger::is_active()) {
		//debugging enabled!

		_debug_max_call_stack = dmcs;
		_call_stack = memnew_arr(CallLevel, _debug_max_call_stack + 1);

	} else {
		_debug_max_call_stack = 0;
		_call_stack = nullptr;
	}

#ifdef DEBUG_ENABLED
	GLOBAL_DEF("debug/gdscript/warnings/enable", true);
	GLOBAL_DEF("debug/gdscript/warnings/treat_warnings_as_errors", false);
	GLOBAL_DEF("debug/gdscript/warnings/exclude_addons", true);
	GLOBAL_DEF("debug/gdscript/completion/autocomplete_setters_and_getters", false);
	for (int i = 0; i < (int)GDScriptWarning::WARNING_MAX; i++) {
		String warning = GDScriptWarning::get_name_from_code((GDScriptWarning::Code)i).to_lower();
		bool default_enabled = !warning.begins_with("unsafe_");
		GLOBAL_DEF("debug/gdscript/warnings/" + warning, default_enabled);
	}
#endif // DEBUG_ENABLED
}

GDScriptLanguage::~GDScriptLanguage() {
	if (_call_stack) {
		memdelete_arr(_call_stack);
	}

	// Clear dependencies between scripts, to ensure cyclic references are broken (to avoid leaks at exit).
	SelfList<GDScript> *s = script_list.first();
	while (s) {
		GDScript *script = s->self();
		// This ensures the current script is not released before we can check what's the next one
		// in the list (we can't get the next upfront because we don't know if the reference breaking
		// will cause it -or any other after it, for that matter- to be released so the next one
		// is not the same as before).
		script->reference();

		for (KeyValue<StringName, GDScriptFunction *> &E : script->member_functions) {
			GDScriptFunction *func = E.value;
			for (int i = 0; i < func->argument_types.size(); i++) {
				func->argument_types.write[i].script_type_ref = Ref<Script>();
			}
			func->return_type.script_type_ref = Ref<Script>();
		}
		for (KeyValue<StringName, GDScript::MemberInfo> &E : script->member_indices) {
			E.value.data_type.script_type_ref = Ref<Script>();
		}

		s = s->next();
		script->unreference();
	}

	singleton = nullptr;
}

void GDScriptLanguage::add_orphan_subclass(const String &p_qualified_name, const ObjectID &p_subclass) {
	orphan_subclasses[p_qualified_name] = p_subclass;
}

Ref<GDScript> GDScriptLanguage::get_orphan_subclass(const String &p_qualified_name) {
	Map<String, ObjectID>::Element *orphan_subclass_element = orphan_subclasses.find(p_qualified_name);
	if (!orphan_subclass_element) {
		return Ref<GDScript>();
	}
	ObjectID orphan_subclass = orphan_subclass_element->get();
	Object *obj = ObjectDB::get_instance(orphan_subclass);
	orphan_subclasses.erase(orphan_subclass_element);
	if (!obj) {
		return Ref<GDScript>();
	}
	return Ref<GDScript>(Object::cast_to<GDScript>(obj));
}

/*************** RESOURCE ***************/

RES ResourceFormatLoaderGDScript::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	if (r_error) {
		*r_error = ERR_FILE_CANT_OPEN;
	}

	Error err;
	Ref<GDScript> script = GDScriptCache::get_full_script(p_path, err);

	// TODO: Reintroduce binary and encrypted scripts.

	if (script.is_null()) {
		// Don't fail loading because of parsing error.
		script.instantiate();
	}

	if (r_error) {
		*r_error = OK;
	}

	return script;
}

void ResourceFormatLoaderGDScript::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("gd");
	// TODO: Reintroduce binary and encrypted scripts.
	// p_extensions->push_back("gdc");
	// p_extensions->push_back("gde");
}

bool ResourceFormatLoaderGDScript::handles_type(const String &p_type) const {
	return (p_type == "Script" || p_type == "GDScript");
}

String ResourceFormatLoaderGDScript::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	// TODO: Reintroduce binary and encrypted scripts.
	if (el == "gd" /*|| el == "gdc" || el == "gde"*/) {
		return "GDScript";
	}
	return "";
}

void ResourceFormatLoaderGDScript::get_dependencies(const String &p_path, List<String> *p_dependencies, bool p_add_types) {
	FileAccessRef file = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_MSG(!file, "Cannot open file '" + p_path + "'.");

	String source = file->get_as_utf8_string();
	if (source.is_empty()) {
		return;
	}

	GDScriptParser parser;
	if (OK != parser.parse(source, p_path, false)) {
		return;
	}

	for (const String &E : parser.get_dependencies()) {
		p_dependencies->push_back(E);
	}
}

Error ResourceFormatSaverGDScript::save(const String &p_path, const RES &p_resource, uint32_t p_flags) {
	Ref<GDScript> sqscr = p_resource;
	ERR_FAIL_COND_V(sqscr.is_null(), ERR_INVALID_PARAMETER);

	String source = sqscr->get_source_code();

	Error err;
	FileAccess *file = FileAccess::open(p_path, FileAccess::WRITE, &err);

	ERR_FAIL_COND_V_MSG(err, err, "Cannot save GDScript file '" + p_path + "'.");

	file->store_string(source);
	if (file->get_error() != OK && file->get_error() != ERR_FILE_EOF) {
		memdelete(file);
		return ERR_CANT_CREATE;
	}
	file->close();
	memdelete(file);

	if (ScriptServer::is_reload_scripts_on_save_enabled()) {
		GDScriptLanguage::get_singleton()->reload_tool_script(p_resource, false);
	}

	return OK;
}

void ResourceFormatSaverGDScript::get_recognized_extensions(const RES &p_resource, List<String> *p_extensions) const {
	if (Object::cast_to<GDScript>(*p_resource)) {
		p_extensions->push_back("gd");
	}
}

bool ResourceFormatSaverGDScript::recognize(const RES &p_resource) const {
	return Object::cast_to<GDScript>(*p_resource) != nullptr;
}
