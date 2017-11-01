/*************************************************************************/
/*  csharp_script.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "csharp_script.h"

#include <mono/metadata/threads.h>

#include "os/file_access.h"
#include "os/os.h"
#include "os/thread.h"
#include "project_settings.h"

#ifdef TOOLS_ENABLED
#include "editor/bindings_generator.h"
#include "editor/csharp_project.h"
#include "editor/editor_node.h"
#include "editor/godotsharp_editor.h"
#include "utils/string_utils.h"
#endif

#include "godotsharp_dirs.h"
#include "mono_gd/gd_mono_class.h"
#include "mono_gd/gd_mono_marshal.h"
#include "signal_awaiter_utils.h"

#define CACHED_STRING_NAME(m_var) (CSharpLanguage::get_singleton()->get_string_names().m_var)

#ifdef TOOLS_ENABLED
static bool _create_project_solution_if_needed() {

	String sln_path = GodotSharpDirs::get_project_sln_path();
	String csproj_path = GodotSharpDirs::get_project_csproj_path();

	if (!FileAccess::exists(sln_path) || !FileAccess::exists(csproj_path)) {
		// A solution does not yet exist, create a new one

		CRASH_COND(GodotSharpEditor::get_singleton() == NULL);
		return GodotSharpEditor::get_singleton()->call("_create_project_solution");
	}

	return true;
}
#endif

CSharpLanguage *CSharpLanguage::singleton = NULL;

String CSharpLanguage::get_name() const {

	return "C#";
}

String CSharpLanguage::get_type() const {

	return "CSharpScript";
}

String CSharpLanguage::get_extension() const {

	return "cs";
}

Error CSharpLanguage::execute_file(const String &p_path) {

	// ??
	return OK;
}

#ifdef TOOLS_ENABLED
void gdsharp_editor_init_callback() {

	EditorNode *editor = EditorNode::get_singleton();
	editor->add_child(memnew(GodotSharpEditor(editor)));
}
#endif

void CSharpLanguage::init() {

	gdmono = memnew(GDMono);
	gdmono->initialize();

#ifdef MONO_GLUE_DISABLED
	WARN_PRINT("This binary is built with `mono_glue=no` and cannot be used for scripting");
#endif

#if defined(TOOLS_ENABLED) && defined(DEBUG_METHODS_ENABLED)
	if (gdmono->get_editor_tools_assembly() != NULL) {
		List<String> cmdline_args = OS::get_singleton()->get_cmdline_args();
		BindingsGenerator::handle_cmdline_args(cmdline_args);
	}
#endif

#ifdef TOOLS_ENABLED
	EditorNode::add_init_callback(&gdsharp_editor_init_callback);
#endif
}

void CSharpLanguage::finish() {

	// Release gchandle bindings before finalizing mono runtime
	gchandle_bindings.clear();

	if (gdmono) {
		memdelete(gdmono);
		gdmono = NULL;
	}
}

void CSharpLanguage::get_reserved_words(List<String> *p_words) const {

	static const char *_reserved_words[] = {
		// Reserved keywords
		"abstract",
		"as",
		"base",
		"bool",
		"break",
		"byte",
		"case",
		"catch",
		"char",
		"checked",
		"class",
		"const",
		"continue",
		"decimal",
		"default",
		"delegate",
		"do",
		"double",
		"else",
		"enum",
		"event",
		"explicit",
		"extern",
		"false",
		"finally",
		"fixed",
		"float",
		"for",
		"forech",
		"goto",
		"if",
		"implicit",
		"in",
		"int",
		"interface",
		"internal",
		"is",
		"lock",
		"long",
		"namespace",
		"new",
		"null",
		"object",
		"operator",
		"out",
		"override",
		"params",
		"private",
		"protected",
		"public",
		"readonly",
		"ref",
		"return",
		"sbyte",
		"sealed",
		"short",
		"sizeof",
		"stackalloc",
		"static",
		"string",
		"struct",
		"switch",
		"this",
		"throw",
		"true",
		"try",
		"typeof",
		"uint",
		"ulong",
		"unchecked",
		"unsafe",
		"ushort",
		"using",
		"virtual",
		"volatile",
		"void",
		"while",

		// Contextual keywords. Not reserved words, but I guess we should include
		// them because this seems to be used only for syntax highlighting.
		"add",
		"ascending",
		"by",
		"descending",
		"dynamic",
		"equals",
		"from",
		"get",
		"global",
		"group",
		"in",
		"into",
		"join",
		"let",
		"on",
		"orderby",
		"partial",
		"remove",
		"select",
		"set",
		"value",
		"var",
		"where",
		"yield",
		0
	};

	const char **w = _reserved_words;

	while (*w) {
		p_words->push_back(*w);
		w++;
	}
}

void CSharpLanguage::get_comment_delimiters(List<String> *p_delimiters) const {

	p_delimiters->push_back("//"); // single-line comment
	p_delimiters->push_back("/* */"); // delimited comment
}

void CSharpLanguage::get_string_delimiters(List<String> *p_delimiters) const {

	p_delimiters->push_back("' '"); // character literal
	p_delimiters->push_back("\" \""); // regular string literal
	p_delimiters->push_back("@\" \""); // verbatim string literal
}

Ref<Script> CSharpLanguage::get_template(const String &p_class_name, const String &p_base_class_name) const {

	String script_template = "using " BINDINGS_NAMESPACE ";\n"
							 "using System;\n"
							 "\n"
							 "public class %CLASS_NAME% : %BASE_CLASS_NAME%\n"
							 "{\n"
							 "    // Member variables here, example:\n"
							 "    // private int a = 2;\n"
							 "    // private string b = \"textvar\";\n"
							 "\n"
							 "    public override void _Ready()\n"
							 "    {\n"
							 "        // Called every time the node is added to the scene.\n"
							 "        // Initialization here\n"
							 "        \n"
							 "    }\n"
							 "\n"
							 "//    public override void _Process(float delta)\n"
							 "//    {\n"
							 "//        // Called every frame. Delta is time since last frame.\n"
							 "//        // Update game logic here.\n"
							 "//        \n"
							 "//    }\n"
							 "}\n";

	script_template = script_template.replace("%BASE_CLASS_NAME%", p_base_class_name)
							  .replace("%CLASS_NAME%", p_class_name);

	Ref<CSharpScript> script;
	script.instance();
	script->set_source_code(script_template);
	script->set_name(p_class_name);

	return script;
}

bool CSharpLanguage::is_using_templates() {

	return true;
}

void CSharpLanguage::make_template(const String &p_class_name, const String &p_base_class_name, Ref<Script> &p_script) {

	String src = p_script->get_source_code();
	src = src.replace("%BASE%", p_base_class_name)
				  .replace("%CLASS%", p_class_name)
				  .replace("%TS%", _get_indentation());
	p_script->set_source_code(src);
}

Script *CSharpLanguage::create_script() const {

	return memnew(CSharpScript);
}

bool CSharpLanguage::has_named_classes() const {

	return false;
}

bool CSharpLanguage::supports_builtin_mode() const {

	return false;
}

static String variant_type_to_managed_name(const String &p_var_type_name) {

	if (p_var_type_name.empty())
		return "object";

	if (!ClassDB::class_exists(p_var_type_name)) {
		Variant::Type var_types[] = {
			Variant::BOOL,
			Variant::INT,
			Variant::REAL,
			Variant::STRING,
			Variant::VECTOR2,
			Variant::RECT2,
			Variant::VECTOR3,
			Variant::TRANSFORM2D,
			Variant::PLANE,
			Variant::QUAT,
			Variant::RECT3,
			Variant::BASIS,
			Variant::TRANSFORM,
			Variant::COLOR,
			Variant::NODE_PATH,
			Variant::_RID
		};

		for (int i = 0; i < sizeof(var_types) / sizeof(Variant::Type); i++) {
			if (p_var_type_name == Variant::get_type_name(var_types[i]))
				return p_var_type_name;
		}

		if (p_var_type_name == "String")
			return "string"; // I prefer this one >:[

		// TODO these will be rewritten later into custom containers

		if (p_var_type_name == "Array")
			return "object[]";

		if (p_var_type_name == "Dictionary")
			return "Dictionary<object, object>";

		if (p_var_type_name == "PoolByteArray")
			return "byte[]";
		if (p_var_type_name == "PoolIntArray")
			return "int[]";
		if (p_var_type_name == "PoolRealArray")
			return "float[]";
		if (p_var_type_name == "PoolStringArray")
			return "string[]";
		if (p_var_type_name == "PoolVector2Array")
			return "Vector2[]";
		if (p_var_type_name == "PoolVector3Array")
			return "Vector3[]";
		if (p_var_type_name == "PoolColorArray")
			return "Color[]";

		return "object";
	}

	return p_var_type_name;
}

String CSharpLanguage::make_function(const String &p_class, const String &p_name, const PoolStringArray &p_args) const {
#ifdef TOOLS_ENABLED
	// FIXME
	// - Due to Godot's API limitation this just appends the function to the end of the file
	// - Use fully qualified name if there is ambiguity
	String s = "private void " + p_name + "(";
	for (int i = 0; i < p_args.size(); i++) {
		const String &arg = p_args[i];

		if (i > 0)
			s += ", ";

		s += variant_type_to_managed_name(arg.get_slice(":", 1)) + " " + escape_csharp_keyword(arg.get_slice(":", 0));
	}
	s += ")\n{\n    // Replace with function body\n}\n";

	return s;
#else
	return String();
#endif
}

String CSharpLanguage::_get_indentation() const {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		bool use_space_indentation = EDITOR_DEF("text_editor/indent/type", 0);

		if (use_space_indentation) {
			int indent_size = EDITOR_DEF("text_editor/indent/size", 4);

			String space_indent = "";
			for (int i = 0; i < indent_size; i++) {
				space_indent += " ";
			}
			return space_indent;
		}
	}
#endif
	return "\t";
}

void CSharpLanguage::frame() {

	const Ref<MonoGCHandle> &task_scheduler_handle = GDMonoUtils::mono_cache.task_scheduler_handle;

	if (task_scheduler_handle.is_valid()) {
		MonoObject *task_scheduler = task_scheduler_handle->get_target();

		if (task_scheduler) {
			GDMonoUtils::GodotTaskScheduler_Activate thunk = CACHED_METHOD_THUNK(GodotTaskScheduler, Activate);

			ERR_FAIL_NULL(thunk);

			MonoObject *ex;
			thunk(task_scheduler, &ex);

			if (ex) {
				mono_print_unhandled_exception(ex);
				ERR_FAIL();
			}
		}
	}
}

struct CSharpScriptDepSort {

	// must support sorting so inheritance works properly (parent must be reloaded first)
	bool operator()(const Ref<CSharpScript> &A, const Ref<CSharpScript> &B) const {
		if (A == B)
			return false; // shouldn't happen but..
		GDMonoClass *I = B->base;
		while (I) {
			if (I == A->script_class) {
				// A is a base of B
				return true;
			}

			I = I->get_parent_class();
		}

		return false; // not a base
	}
};

void CSharpLanguage::reload_all_scripts() {

#ifdef DEBUG_ENABLED

#ifndef NO_THREADS
	lock->lock();
#endif

	List<Ref<CSharpScript> > scripts;

	SelfList<CSharpScript> *elem = script_list.first();
	while (elem) {
		if (elem->self()->get_path().is_resource_file()) {
			scripts.push_back(Ref<CSharpScript>(elem->self())); //cast to gdscript to avoid being erased by accident
		}
		elem = elem->next();
	}

#ifndef NO_THREADS
	lock->unlock();
#endif

	//as scripts are going to be reloaded, must proceed without locking here

	scripts.sort_custom<CSharpScriptDepSort>(); //update in inheritance dependency order

	for (List<Ref<CSharpScript> >::Element *E = scripts.front(); E; E = E->next()) {
		E->get()->load_source_code(E->get()->get_path());
		E->get()->reload(true);
	}
#endif
}

void CSharpLanguage::reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload) {

	(void)p_script; // UNUSED

#ifdef TOOLS_ENABLED
	MonoReloadNode::get_singleton()->restart_reload_timer();
	reload_assemblies_if_needed(p_soft_reload);
#endif
}

#ifdef TOOLS_ENABLED
void CSharpLanguage::reload_assemblies_if_needed(bool p_soft_reload) {

	if (gdmono->is_runtime_initialized()) {

		GDMonoAssembly *proj_assembly = gdmono->get_project_assembly();

		String name = ProjectSettings::get_singleton()->get("application/config/name");
		if (name.empty()) {
			name = "UnnamedProject";
		}

		if (proj_assembly) {
			String proj_asm_path = proj_assembly->get_path();

			if (!FileAccess::exists(proj_assembly->get_path())) {
				// Maybe it wasn't loaded from the default path, so check this as well
				proj_asm_path = GodotSharpDirs::get_res_temp_assemblies_dir().plus_file(name);
				if (!FileAccess::exists(proj_asm_path))
					return; // No assembly to load
			}

			if (FileAccess::get_modified_time(proj_asm_path) <= proj_assembly->get_modified_time())
				return; // Already up to date
		} else {
			if (!FileAccess::exists(GodotSharpDirs::get_res_temp_assemblies_dir().plus_file(name)))
				return; // No assembly to load
		}
	}

#ifndef NO_THREADS
	lock->lock();
#endif

	List<Ref<CSharpScript> > scripts;

	SelfList<CSharpScript> *elem = script_list.first();
	while (elem) {
		if (elem->self()->get_path().is_resource_file()) {

			scripts.push_back(Ref<CSharpScript>(elem->self())); //cast to CSharpScript to avoid being erased by accident
		}
		elem = elem->next();
	}

#ifndef NO_THREADS
	lock->unlock();
#endif

	//when someone asks you why dynamically typed languages are easier to write....

	Map<Ref<CSharpScript>, Map<ObjectID, List<Pair<StringName, Variant> > > > to_reload;

	//as scripts are going to be reloaded, must proceed without locking here

	scripts.sort_custom<CSharpScriptDepSort>(); //update in inheritance dependency order

	for (List<Ref<CSharpScript> >::Element *E = scripts.front(); E; E = E->next()) {

		to_reload.insert(E->get(), Map<ObjectID, List<Pair<StringName, Variant> > >());

		if (!p_soft_reload) {

			//save state and remove script from instances
			Map<ObjectID, List<Pair<StringName, Variant> > > &map = to_reload[E->get()];

			while (E->get()->instances.front()) {
				Object *obj = E->get()->instances.front()->get();
				//save instance info
				List<Pair<StringName, Variant> > state;
				if (obj->get_script_instance()) {

					obj->get_script_instance()->get_property_state(state);

					Ref<MonoGCHandle> gchandle = CAST_CSHARP_INSTANCE(obj->get_script_instance())->gchandle;
					if (gchandle.is_valid())
						gchandle->release();

					map[obj->get_instance_id()] = state;
					obj->set_script(RefPtr());
				}
			}

			//same thing for placeholders
			while (E->get()->placeholders.size()) {

				Object *obj = E->get()->placeholders.front()->get()->get_owner();
				//save instance info
				List<Pair<StringName, Variant> > state;
				if (obj->get_script_instance()) {
					obj->get_script_instance()->get_property_state(state);
					map[obj->get_instance_id()] = state;
					obj->set_script(RefPtr());
				}
			}

			for (Map<ObjectID, List<Pair<StringName, Variant> > >::Element *F = E->get()->pending_reload_state.front(); F; F = F->next()) {
				map[F->key()] = F->get(); //pending to reload, use this one instead
			}

			E->get()->_clear();
		}
	}

	if (gdmono->reload_scripts_domain() != OK)
		return;

	for (Map<Ref<CSharpScript>, Map<ObjectID, List<Pair<StringName, Variant> > > >::Element *E = to_reload.front(); E; E = E->next()) {

		Ref<CSharpScript> scr = E->key();
		scr->exports_invalidated = true;
		scr->reload(p_soft_reload);
		scr->update_exports();

		//restore state if saved
		for (Map<ObjectID, List<Pair<StringName, Variant> > >::Element *F = E->get().front(); F; F = F->next()) {

			Object *obj = ObjectDB::get_instance(F->key());
			if (!obj)
				continue;

			if (!p_soft_reload) {
				//clear it just in case (may be a pending reload state)
				obj->set_script(RefPtr());
			}
			obj->set_script(scr.get_ref_ptr());
			if (!obj->get_script_instance()) {
				//failed, save reload state for next time if not saved
				if (!scr->pending_reload_state.has(obj->get_instance_id())) {
					scr->pending_reload_state[obj->get_instance_id()] = F->get();
				}
				continue;
			}

			for (List<Pair<StringName, Variant> >::Element *G = F->get().front(); G; G = G->next()) {
				obj->get_script_instance()->set(G->get().first, G->get().second);
			}

			scr->pending_reload_state.erase(obj->get_instance_id()); //as it reloaded, remove pending state
		}

		//if instance states were saved, set them!
	}

	if (Engine::get_singleton()->is_editor_hint())
		EditorNode::get_singleton()->get_property_editor()->update_tree();
}
#endif

void CSharpLanguage::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("cs");
}

#ifdef TOOLS_ENABLED
Error CSharpLanguage::open_in_external_editor(const Ref<Script> &p_script, int p_line, int p_col) {

	return GodotSharpEditor::get_singleton()->open_in_external_editor(p_script, p_line, p_col);
}

bool CSharpLanguage::overrides_external_editor() {

	return GodotSharpEditor::get_singleton()->overrides_external_editor();
}
#endif

void CSharpLanguage::thread_enter() {

#if 0
	if (mono->is_runtime_initialized()) {
		GDMonoUtils::attach_current_thread();
	}
#endif
}

void CSharpLanguage::thread_exit() {

#if 0
	if (mono->is_runtime_initialized()) {
		GDMonoUtils::detach_current_thread();
	}
#endif
}

bool CSharpLanguage::debug_break_parse(const String &p_file, int p_line, const String &p_error) {

	// Break because of parse error
	if (ScriptDebugger::get_singleton() && Thread::get_caller_id() == Thread::get_main_id()) {
		// TODO
		//_debug_parse_err_line = p_line;
		//_debug_parse_err_file = p_file;
		//_debug_error = p_error;
		ScriptDebugger::get_singleton()->debug(this, false);
		return true;
	} else {
		return false;
	}
}

bool CSharpLanguage::debug_break(const String &p_error, bool p_allow_continue) {

	if (ScriptDebugger::get_singleton() && Thread::get_caller_id() == Thread::get_main_id()) {
		// TODO
		//_debug_parse_err_line = -1;
		//_debug_parse_err_file = "";
		//_debug_error = p_error;
		ScriptDebugger::get_singleton()->debug(this, p_allow_continue);
		return true;
	} else {
		return false;
	}
}

void CSharpLanguage::set_language_index(int p_idx) {

	ERR_FAIL_COND(lang_idx != -1);
	lang_idx = p_idx;
}

CSharpLanguage::CSharpLanguage() {

	ERR_FAIL_COND(singleton);
	singleton = this;

	gdmono = NULL;

#ifdef NO_THREADS
	lock = NULL;
	gchandle_bind_lock = NULL;
#else
	lock = Mutex::create();
	script_bind_lock = Mutex::create();
#endif

	lang_idx = -1;
}

CSharpLanguage::~CSharpLanguage() {

	finish();

	if (lock) {
		memdelete(lock);
		lock = NULL;
	}

	if (script_bind_lock) {
		memdelete(script_bind_lock);
		script_bind_lock = NULL;
	}

	singleton = NULL;
}

void *CSharpLanguage::alloc_instance_binding_data(Object *p_object) {

#ifdef DEBUG_ENABLED
	// I don't trust you
	if (p_object->get_script_instance())
		CRASH_COND(NULL != CAST_CSHARP_INSTANCE(p_object->get_script_instance()));
#endif

	StringName type_name = p_object->get_class_name();

	// ¯\_(ツ)_/¯
	const ClassDB::ClassInfo *classinfo = ClassDB::classes.getptr(type_name);
	while (classinfo && !classinfo->exposed)
		classinfo = classinfo->inherits_ptr;
	ERR_FAIL_NULL_V(classinfo, NULL);
	type_name = classinfo->name;

	GDMonoClass *type_class = GDMonoUtils::type_get_proxy_class(type_name);

	ERR_FAIL_NULL_V(type_class, NULL);

	MonoObject *mono_object = GDMonoUtils::create_managed_for_godot_object(type_class, type_name, p_object);

	ERR_FAIL_NULL_V(mono_object, NULL);

	// Tie managed to unmanaged
	bool strong_handle = true;
	Reference *ref = Object::cast_to<Reference>(p_object);

	if (ref) {
		strong_handle = false;

		// Unsafe refcount increment. The managed instance also counts as a reference.
		// This way if the unmanaged world has no references to our owner
		// but the managed instance is alive, the refcount will be 1 instead of 0.
		// See: _GodotSharp::_dispose_object(Object *p_object)

		ref->reference();
	}

	Ref<MonoGCHandle> gchandle = strong_handle ? MonoGCHandle::create_strong(mono_object) :
												 MonoGCHandle::create_weak(mono_object);

#ifndef NO_THREADS
	script_bind_lock->lock();
#endif

	void *data = (void *)gchandle_bindings.insert(p_object, gchandle);

#ifndef NO_THREADS
	script_bind_lock->unlock();
#endif

	return data;
}

void CSharpLanguage::free_instance_binding_data(void *p_data) {

	if (GDMono::get_singleton() == NULL) {
#ifdef DEBUG_ENABLED
		CRASH_COND(!gchandle_bindings.empty());
#endif
		// Mono runtime finalized, all the gchandle bindings were already released
		return;
	}

#ifndef NO_THREADS
	script_bind_lock->lock();
#endif

	gchandle_bindings.erase((Map<Object *, Ref<MonoGCHandle> >::Element *)p_data);

#ifndef NO_THREADS
	script_bind_lock->unlock();
#endif
}

void CSharpInstance::_ml_call_reversed(GDMonoClass *klass, const StringName &p_method, const Variant **p_args, int p_argcount) {

	GDMonoClass *base = klass->get_parent_class();
	if (base && base != script->native)
		_ml_call_reversed(base, p_method, p_args, p_argcount);

	GDMonoMethod *method = klass->get_method(p_method, p_argcount);

	if (method) {
		method->invoke(get_mono_object(), p_args);
	}
}

CSharpInstance *CSharpInstance::create_for_managed_type(Object *p_owner, CSharpScript *p_script, const Ref<MonoGCHandle> &p_gchandle) {

	CSharpInstance *instance = memnew(CSharpInstance);

	Reference *ref = Object::cast_to<Reference>(p_owner);

	instance->base_ref = ref != NULL;
	instance->script = Ref<CSharpScript>(p_script);
	instance->owner = p_owner;
	instance->gchandle = p_gchandle;

	if (instance->base_ref)
		instance->_reference_owner_unsafe();

	p_script->instances.insert(p_owner);

	return instance;
}

MonoObject *CSharpInstance::get_mono_object() const {
#ifdef DEBUG_ENABLED
	CRASH_COND(gchandle.is_null());
#endif
	return gchandle->get_target();
}

bool CSharpInstance::set(const StringName &p_name, const Variant &p_value) {

	ERR_FAIL_COND_V(!script.is_valid(), false);

	GDMonoClass *top = script->script_class;

	while (top && top != script->native) {
		GDMonoField *field = script->script_class->get_field(p_name);

		if (field) {
			MonoObject *mono_object = get_mono_object();

			ERR_EXPLAIN("Reference has been garbage collected?");
			ERR_FAIL_NULL_V(mono_object, false);

			field->set_value(mono_object, p_value);

			return true;
		}

		top = top->get_parent_class();
	}

	// Call _set

	Variant name = p_name;
	const Variant *args[2] = { &name, &p_value };

	MonoObject *mono_object = get_mono_object();
	top = script->script_class;

	while (top && top != script->native) {
		GDMonoMethod *method = top->get_method(CACHED_STRING_NAME(_set), 2);

		if (method) {
			MonoObject *ret = method->invoke(mono_object, args);

			if (ret && GDMonoMarshal::unbox<MonoBoolean>(ret) == true)
				return true;
		}

		top = top->get_parent_class();
	}

	return false;
}

bool CSharpInstance::get(const StringName &p_name, Variant &r_ret) const {

	ERR_FAIL_COND_V(!script.is_valid(), false);

	GDMonoClass *top = script->script_class;

	while (top && top != script->native) {
		GDMonoField *field = top->get_field(p_name);

		if (field) {
			MonoObject *mono_object = get_mono_object();

			ERR_EXPLAIN("Reference has been garbage collected?");
			ERR_FAIL_NULL_V(mono_object, false);

			MonoObject *value = field->get_value(mono_object);
			r_ret = GDMonoMarshal::mono_object_to_variant(value, field->get_type());
			return true;
		}

		// Call _get

		GDMonoMethod *method = top->get_method(CACHED_STRING_NAME(_get), 1);

		if (method) {
			Variant name = p_name;
			const Variant *args[1] = { &name };

			MonoObject *ret = method->invoke(get_mono_object(), args);

			if (ret) {
				r_ret = GDMonoMarshal::mono_object_to_variant(ret);
				return true;
			}
		}

		top = top->get_parent_class();
	}

	return false;
}

void CSharpInstance::get_property_list(List<PropertyInfo> *p_properties) const {

	for (Map<StringName, PropertyInfo>::Element *E = script->member_info.front(); E; E = E->next()) {
		p_properties->push_back(E->value());
	}
}

Variant::Type CSharpInstance::get_property_type(const StringName &p_name, bool *r_is_valid) const {

	if (script->member_info.has(p_name)) {
		if (r_is_valid)
			*r_is_valid = true;
		return script->member_info[p_name].type;
	}

	if (r_is_valid)
		*r_is_valid = false;

	return Variant::NIL;
}

bool CSharpInstance::has_method(const StringName &p_method) const {

	if (!script.is_valid())
		return false;

	GDMonoClass *top = script->script_class;

	while (top && top != script->native) {
		if (top->has_method(p_method)) {
			return true;
		}

		top = top->get_parent_class();
	}

	return false;
}

Variant CSharpInstance::call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

	MonoObject *mono_object = get_mono_object();

	ERR_EXPLAIN("Reference has been garbage collected?");
	ERR_FAIL_NULL_V(mono_object, Variant());

	if (!script.is_valid())
		return Variant();

	GDMonoClass *top = script->script_class;

	while (top && top != script->native) {
		GDMonoMethod *method = top->get_method(p_method, p_argcount);

		if (method) {
			MonoObject *return_value = method->invoke(mono_object, p_args);

			if (return_value) {
				return GDMonoMarshal::mono_object_to_variant(return_value, method->get_return_type());
			} else {
				return Variant();
			}
		}

		top = top->get_parent_class();
	}

	r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;

	return Variant();
}

void CSharpInstance::call_multilevel(const StringName &p_method, const Variant **p_args, int p_argcount) {

	if (script.is_valid()) {
		MonoObject *mono_object = get_mono_object();

		GDMonoClass *top = script->script_class;

		while (top && top != script->native) {
			GDMonoMethod *method = top->get_method(p_method, p_argcount);

			if (method)
				method->invoke(mono_object, p_args);

			top = top->get_parent_class();
		}
	}
}

void CSharpInstance::call_multilevel_reversed(const StringName &p_method, const Variant **p_args, int p_argcount) {

	if (script.is_valid()) {
		_ml_call_reversed(script->script_class, p_method, p_args, p_argcount);
	}
}

void CSharpInstance::_reference_owner_unsafe() {

#ifdef DEBUG_ENABLED
	CRASH_COND(!base_ref);
#endif

	// Unsafe refcount increment. The managed instance also counts as a reference.
	// This way if the unmanaged world has no references to our owner
	// but the managed instance is alive, the refcount will be 1 instead of 0.
	// See: _unreference_owner_unsafe()

	// May not me referenced yet, so we must use init_ref() instead of reference()
	Object::cast_to<Reference>(owner)->init_ref();
}

void CSharpInstance::_unreference_owner_unsafe() {

#ifdef DEBUG_ENABLED
	CRASH_COND(!base_ref);
#endif

	// Called from CSharpInstance::mono_object_disposed() or ~CSharpInstance()

	// Unsafe refcount decrement. The managed instance also counts as a reference.
	// See: _reference_owner_unsafe()

	if (Object::cast_to<Reference>(owner)->unreference()) {
		memdelete(owner);
		owner = NULL;
	}
}

void CSharpInstance::mono_object_disposed() {

	if (base_ref)
		_unreference_owner_unsafe();
}

void CSharpInstance::refcount_incremented() {

	CRASH_COND(!base_ref);

	Reference *ref_owner = Object::cast_to<Reference>(owner);

	if (ref_owner->reference_get_count() > 1) { // Remember the managed side holds a reference, hence 1 instead of 0 here
		// The reference count was increased after the managed side was the only one referencing our owner.
		// This means the owner is being referenced again by the unmanaged side,
		// so the owner must hold the managed side alive again to avoid it from being GCed.

		// Release the current weak handle and replace it with a strong handle.
		uint32_t strong_gchandle = MonoGCHandle::make_strong_handle(gchandle->get_target());
		gchandle->release();
		gchandle->set_handle(strong_gchandle);
	}
}

bool CSharpInstance::refcount_decremented() {

	CRASH_COND(!base_ref);

	Reference *ref_owner = Object::cast_to<Reference>(owner);

	int refcount = ref_owner->reference_get_count();

	if (refcount == 1) { // Remember the managed side holds a reference, hence 1 instead of 0 here
		// If owner owner is no longer referenced by the unmanaged side,
		// the managed instance takes responsibility of deleting the owner when GCed.

		// Release the current strong handle and replace it with a weak handle.
		uint32_t weak_gchandle = MonoGCHandle::make_weak_handle(gchandle->get_target());
		gchandle->release();
		gchandle->set_handle(weak_gchandle);

		return false;
	}

	ref_dying = (refcount == 0);

	return ref_dying;
}

ScriptInstance::RPCMode CSharpInstance::get_rpc_mode(const StringName &p_method) const {

	GDMonoClass *top = script->script_class;

	while (top && top != script->native) {
		GDMonoMethod *method = top->get_method(p_method);

		if (method) { // TODO should we reject static methods?
			// TODO cache result
			if (method->has_attribute(CACHED_CLASS(RemoteAttribute)))
				return RPC_MODE_REMOTE;
			if (method->has_attribute(CACHED_CLASS(SyncAttribute)))
				return RPC_MODE_SYNC;
			if (method->has_attribute(CACHED_CLASS(MasterAttribute)))
				return RPC_MODE_MASTER;
			if (method->has_attribute(CACHED_CLASS(SlaveAttribute)))
				return RPC_MODE_SLAVE;
		}

		top = top->get_parent_class();
	}

	return RPC_MODE_DISABLED;
}

ScriptInstance::RPCMode CSharpInstance::get_rset_mode(const StringName &p_variable) const {

	GDMonoClass *top = script->script_class;

	while (top && top != script->native) {
		GDMonoField *field = top->get_field(p_variable);

		if (field) { // TODO should we reject static fields?
			// TODO cache result
			if (field->has_attribute(CACHED_CLASS(RemoteAttribute)))
				return RPC_MODE_REMOTE;
			if (field->has_attribute(CACHED_CLASS(SyncAttribute)))
				return RPC_MODE_SYNC;
			if (field->has_attribute(CACHED_CLASS(MasterAttribute)))
				return RPC_MODE_MASTER;
			if (field->has_attribute(CACHED_CLASS(SlaveAttribute)))
				return RPC_MODE_SLAVE;
		}

		top = top->get_parent_class();
	}

	return RPC_MODE_DISABLED;
}

void CSharpInstance::notification(int p_notification) {

	Variant value = p_notification;
	const Variant *args[1] = { &value };

	call_multilevel(CACHED_STRING_NAME(_notification), args, 1);
}

Ref<Script> CSharpInstance::get_script() const {

	return script;
}

ScriptLanguage *CSharpInstance::get_language() {

	return CSharpLanguage::get_singleton();
}

CSharpInstance::CSharpInstance() {

	owner = NULL;
	base_ref = false;
	ref_dying = false;
}

CSharpInstance::~CSharpInstance() {

	if (gchandle.is_valid()) {
		gchandle->release(); // Make sure it's released
	}

	if (base_ref && !ref_dying) { // it may be called from the owner's destructor
#ifdef DEBUG_ENABLED
		CRASH_COND(!owner); // dunno, just in case
#endif
		_unreference_owner_unsafe();
	}

	if (script.is_valid() && owner) {
#ifndef NO_THREADS
		CSharpLanguage::singleton->lock->lock();
#endif

#ifdef DEBUG_ENABLED
		// CSharpInstance must not be created unless it's going to be added to the list for sure
		Set<Object *>::Element *match = script->instances.find(owner);
		CRASH_COND(!match);
		script->instances.erase(match);
#else
		script->instances.erase(owner);
#endif

#ifndef NO_THREADS
		CSharpLanguage::singleton->lock->unlock();
#endif
	}
}

#ifdef TOOLS_ENABLED
void CSharpScript::_placeholder_erased(PlaceHolderScriptInstance *p_placeholder) {

	placeholders.erase(p_placeholder);
}
#endif

#ifdef TOOLS_ENABLED
void CSharpScript::_update_exports_values(Map<StringName, Variant> &values, List<PropertyInfo> &propnames) {

	if (base_cache.is_valid()) {
		base_cache->_update_exports_values(values, propnames);
	}

	for (Map<StringName, Variant>::Element *E = exported_members_defval_cache.front(); E; E = E->next()) {
		values[E->key()] = E->get();
	}

	for (List<PropertyInfo>::Element *E = exported_members_cache.front(); E; E = E->next()) {
		propnames.push_back(E->get());
	}
}
#endif

bool CSharpScript::_update_exports() {

#ifdef TOOLS_ENABLED
	if (!valid)
		return false;

	bool changed = false;

	if (exports_invalidated) {
		exports_invalidated = false;

		changed = true;

		member_info.clear();
		exported_members_cache.clear();
		exported_members_defval_cache.clear();

		// We are creating a temporary new instance of the class here to get the default value
		// TODO Workaround. Should be replaced with IL opcodes analysis

		MonoObject *tmp_object = mono_object_new(SCRIPTS_DOMAIN, script_class->get_raw());

		if (tmp_object) {
			CACHED_FIELD(GodotObject, ptr)->set_value_raw(tmp_object, tmp_object); // FIXME WTF is this workaround

			GDMonoMethod *ctor = script_class->get_method(CACHED_STRING_NAME(dotctor), 0);
			MonoObject *ex = NULL;
			ctor->invoke(tmp_object, NULL, &ex);

			if (ex) {
				ERR_PRINT("Exception thrown from constructor of temporary MonoObject:");
				mono_print_unhandled_exception(ex);
				tmp_object = NULL;
				ERR_FAIL_V(false);
			}
		} else {
			ERR_PRINT("Failed to create temporary MonoObject");
			return false;
		}

		GDMonoClass *top = script_class;

		while (top && top != native) {
			const Vector<GDMonoField *> &fields = top->get_all_fields();

			for (int i = 0; i < fields.size(); i++) {
				GDMonoField *field = fields[i];

				if (field->is_static()) {
					if (field->has_attribute(CACHED_CLASS(ExportAttribute)))
						ERR_PRINTS("Cannot export field because it is static: " + top->get_full_name() + "." + field->get_name());
					continue;
				}

				String name = field->get_name();
				StringName cname = name;

				if (member_info.has(cname))
					continue;

				ManagedType field_type = field->get_type();
				Variant::Type type = GDMonoMarshal::managed_to_variant_type(field_type);

				if (field->has_attribute(CACHED_CLASS(ExportAttribute))) {
					// Field has Export attribute
					MonoObject *attr = field->get_attribute(CACHED_CLASS(ExportAttribute));

					PropertyHint hint;
					String hint_string;

					if (type == Variant::NIL) {
						ERR_PRINTS("Unknown type of exported field: " + top->get_full_name() + "." + field->get_name());
						continue;
					} else if (type == Variant::INT && field_type.type_encoding == MONO_TYPE_VALUETYPE && mono_class_is_enum(field_type.type_class->get_raw())) {
						type = Variant::INT;
						hint = PROPERTY_HINT_ENUM;

						Vector<MonoClassField *> fields = field_type.type_class->get_enum_fields();

						for (int i = 0; i < fields.size(); i++) {
							if (i > 0)
								hint_string += ",";
							hint_string += mono_field_get_name(fields[i]);
						}
					} else if (type == Variant::OBJECT && CACHED_CLASS(GodotReference)->is_assignable_from(field_type.type_class)) {
						hint = PROPERTY_HINT_RESOURCE_TYPE;
						hint_string = NATIVE_GDMONOCLASS_NAME(field_type.type_class);
					} else {
						hint = PropertyHint(CACHED_FIELD(ExportAttribute, hint)->get_int_value(attr));
						hint_string = CACHED_FIELD(ExportAttribute, hint_string)->get_string_value(attr);
					}

					PropertyInfo prop_info = PropertyInfo(type, name, hint, hint_string, PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_SCRIPT_VARIABLE);

					member_info[cname] = prop_info;
					exported_members_cache.push_back(prop_info);

					if (tmp_object) {
						exported_members_defval_cache[cname] = GDMonoMarshal::mono_object_to_variant(field->get_value(tmp_object));
					}
				} else {
					member_info[cname] = PropertyInfo(type, name, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_SCRIPT_VARIABLE);
				}
			}

			top = top->get_parent_class();
		}
	}

	if (placeholders.size()) {
		// Update placeholders if any
		Map<StringName, Variant> values;
		List<PropertyInfo> propnames;
		_update_exports_values(values, propnames);

		for (Set<PlaceHolderScriptInstance *>::Element *E = placeholders.front(); E; E = E->next()) {
			E->get()->update(propnames, values);
		}
	}

	return changed;
#endif
	return false;
}

void CSharpScript::_clear() {

	tool = false;
	valid = false;

	base = NULL;
	native = NULL;
	script_class = NULL;
}

Variant CSharpScript::call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

	GDMonoClass *top = script_class;

	while (top && top != native) {
		GDMonoMethod *method = top->get_method(p_method, p_argcount);

		if (method && method->is_static()) {
			MonoObject *result = method->invoke(NULL, p_args);

			if (result) {
				return GDMonoMarshal::mono_object_to_variant(result, method->get_return_type());
			} else {
				return Variant();
			}
		}

		top = top->get_parent_class();
	}

	// No static method found. Try regular instance calls
	return Script::call(p_method, p_args, p_argcount, r_error);
}

void CSharpScript::_resource_path_changed() {

	String path = get_path();

	if (!path.empty()) {
		name = get_path().get_file().get_basename();
	}
}

bool CSharpScript::_get(const StringName &p_name, Variant &r_ret) const {

	if (p_name == CSharpLanguage::singleton->string_names._script_source) {

		r_ret = get_source_code();
		return true;
	}

	return false;
}

bool CSharpScript::_set(const StringName &p_name, const Variant &p_value) {

	if (p_name == CSharpLanguage::singleton->string_names._script_source) {

		set_source_code(p_value);
		reload();
		return true;
	}

	return false;
}

void CSharpScript::_get_property_list(List<PropertyInfo> *p_properties) const {

	p_properties->push_back(PropertyInfo(Variant::STRING, CSharpLanguage::singleton->string_names._script_source, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
}

void CSharpScript::_bind_methods() {

	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "new", &CSharpScript::_new, MethodInfo(Variant::OBJECT, "new"));
}

Ref<CSharpScript> CSharpScript::create_for_managed_type(GDMonoClass *p_class) {

	// This method should not fail

	CRASH_COND(!p_class);

	Ref<CSharpScript> script = memnew(CSharpScript);

	script->name = p_class->get_name();
	script->script_class = p_class;
	script->native = GDMonoUtils::get_class_native_base(script->script_class);

	CRASH_COND(script->native == NULL);

	GDMonoClass *base = script->script_class->get_parent_class();

	if (base != script->native)
		script->base = base;

#ifdef DEBUG_ENABLED
	// For debug builds, we must fetch from all native base methods as well.
	// Native base methods must be fetched before the current class.
	// Not needed if the script class itself is a native class.

	if (script->script_class != script->native) {
		GDMonoClass *native_top = script->native;
		while (native_top) {
			native_top->fetch_methods_with_godot_api_checks(script->native);

			if (native_top == CACHED_CLASS(GodotObject))
				break;

			native_top = native_top->get_parent_class();
		}
	}
#endif

	script->script_class->fetch_methods_with_godot_api_checks(script->native);

	// Need to fetch method from base classes as well
	GDMonoClass *top = script->script_class;
	while (top && top != script->native) {
		top->fetch_methods_with_godot_api_checks(script->native);
		top = top->get_parent_class();
	}

	return script;
}

bool CSharpScript::can_instance() const {

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {

		if (get_path().find("::") == -1) { // Ignore if built-in script. Can happen if the file is deleted...
			if (_create_project_solution_if_needed()) {
				CSharpProject::add_item(GodotSharpDirs::get_project_csproj_path(),
						"Compile",
						ProjectSettings::get_singleton()->globalize_path(get_path()));
			} else {
				ERR_PRINTS("Cannot add " + get_path() + " to the C# project because it could not be created.");
			}
		}
	}
#endif

	return valid || (!tool && !ScriptServer::is_scripting_enabled());
}

StringName CSharpScript::get_instance_base_type() const {

	if (native)
		return native->get_name();
	else
		return StringName();
}

CSharpInstance *CSharpScript::_create_instance(const Variant **p_args, int p_argcount, Object *p_owner, bool p_isref, Variant::CallError &r_error) {

	/* STEP 1, CREATE */

	CSharpInstance *instance = memnew(CSharpInstance);
	instance->base_ref = p_isref;
	instance->script = Ref<CSharpScript>(this);
	instance->owner = p_owner;
	instance->owner->set_script_instance(instance);

	if (instance->base_ref)
		instance->_reference_owner_unsafe();

	/* STEP 2, INITIALIZE AND CONSTRUCT */

	MonoObject *mono_object = mono_object_new(SCRIPTS_DOMAIN, script_class->get_raw());

	if (!mono_object) {
		instance->script = Ref<CSharpScript>();
		instance->owner->set_script_instance(NULL);
		r_error.error = Variant::CallError::CALL_ERROR_INSTANCE_IS_NULL;
		ERR_EXPLAIN("Failed to allocate memory for the object");
		ERR_FAIL_V(NULL);
	}

#ifndef NO_THREADS
	CSharpLanguage::singleton->lock->lock();
#endif

	instances.insert(instance->owner);

#ifndef NO_THREADS
	CSharpLanguage::singleton->lock->unlock();
#endif

	CACHED_FIELD(GodotObject, ptr)->set_value_raw(mono_object, instance->owner);

	// Construct
	GDMonoMethod *ctor = script_class->get_method(CACHED_STRING_NAME(dotctor), p_argcount);
	ctor->invoke(mono_object, p_args, NULL);

	// Tie managed to unmanaged
	instance->gchandle = MonoGCHandle::create_strong(mono_object);

	/* STEP 3, PARTY */

	//@TODO make thread safe
	return instance;
}

Variant CSharpScript::_new(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

	if (!valid) {
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
		return Variant();
	}

	r_error.error = Variant::CallError::CALL_OK;
	REF ref;
	Object *owner = NULL;

	ERR_FAIL_NULL_V(native, Variant());

	owner = ClassDB::instance(NATIVE_GDMONOCLASS_NAME(native));

	Reference *r = Object::cast_to<Reference>(owner);
	if (r) {
		ref = REF(r);
	}

	CSharpInstance *instance = _create_instance(p_args, p_argcount, owner, r != NULL, r_error);
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

ScriptInstance *CSharpScript::instance_create(Object *p_this) {

	if (!valid)
		return NULL;

	if (!tool && !ScriptServer::is_scripting_enabled()) {
#ifdef TOOLS_ENABLED
		PlaceHolderScriptInstance *si = memnew(PlaceHolderScriptInstance(CSharpLanguage::get_singleton(), Ref<Script>(this), p_this));
		placeholders.insert(si);
		_update_exports();
		return si;
#else
		return NULL;
#endif
	}

	if (native) {
		String native_name = native->get_name();
		if (!ClassDB::is_parent_class(p_this->get_class_name(), native_name)) {
			if (ScriptDebugger::get_singleton()) {
				CSharpLanguage::get_singleton()->debug_break_parse(get_path(), 0, "Script inherits from native type '" + native_name + "', so it can't be instanced in object of type: '" + p_this->get_class() + "'");
			}
			ERR_EXPLAIN("Script inherits from native type '" + native_name + "', so it can't be instanced in object of type: '" + p_this->get_class() + "'");
			ERR_FAIL_V(NULL);
		}
	}

	Variant::CallError unchecked_error;
	return _create_instance(NULL, 0, p_this, Object::cast_to<Reference>(p_this), unchecked_error);
}

bool CSharpScript::instance_has(const Object *p_this) const {

#ifndef NO_THREADS
	CSharpLanguage::singleton->lock->lock();
#endif

	bool ret = instances.has((Object *)p_this);

#ifndef NO_THREADS
	CSharpLanguage::singleton->lock->unlock();
#endif

	return ret;
}

bool CSharpScript::has_source_code() const {

	return !source.empty();
}

String CSharpScript::get_source_code() const {

	return source;
}

void CSharpScript::set_source_code(const String &p_code) {

	if (source == p_code)
		return;
	source = p_code;
#ifdef TOOLS_ENABLED
	source_changed_cache = true;
#endif
}

bool CSharpScript::has_method(const StringName &p_method) const {

	return script_class->has_method(p_method);
}

Error CSharpScript::reload(bool p_keep_state) {

#ifndef NO_THREADS
	CSharpLanguage::singleton->lock->lock();
#endif

	bool has_instances = instances.size();

#ifndef NO_THREADS
	CSharpLanguage::singleton->lock->unlock();
#endif

	ERR_FAIL_COND_V(!p_keep_state && has_instances, ERR_ALREADY_IN_USE);

	GDMonoAssembly *project_assembly = GDMono::get_singleton()->get_project_assembly();

	if (project_assembly) {
		script_class = project_assembly->get_object_derived_class(name);

		if (!script_class) {
			ERR_PRINTS("Cannot find class " + name + " for script " + get_path());
		}
#ifdef DEBUG_ENABLED
		else if (OS::get_singleton()->is_stdout_verbose()) {
			OS::get_singleton()->print(String("Found class " + script_class->get_namespace() + "." +
											  script_class->get_name() + " for script " + get_path() + "\n")
											   .utf8());
		}
#endif

		valid = script_class != NULL;

		if (script_class) {
			tool = script_class->has_attribute(CACHED_CLASS(ToolAttribute));

			native = GDMonoUtils::get_class_native_base(script_class);

			CRASH_COND(native == NULL);

			GDMonoClass *base_class = script_class->get_parent_class();

			if (base_class != native)
				base = base_class;

#ifdef DEBUG_ENABLED
			// For debug builds, we must fetch from all native base methods as well.
			// Native base methods must be fetched before the current class.
			// Not needed if the script class itself is a native class.

			if (script_class != native) {
				GDMonoClass *native_top = native;
				while (native_top) {
					native_top->fetch_methods_with_godot_api_checks(native);

					if (native_top == CACHED_CLASS(GodotObject))
						break;

					native_top = native_top->get_parent_class();
				}
			}
#endif

			script_class->fetch_methods_with_godot_api_checks(native);

			// Need to fetch method from base classes as well
			GDMonoClass *top = script_class;
			while (top && top != native) {
				top->fetch_methods_with_godot_api_checks(native);
				top = top->get_parent_class();
			}
		}

		return OK;
	}

	return ERR_FILE_MISSING_DEPENDENCIES;
}

ScriptLanguage *CSharpScript::get_language() const {

	return CSharpLanguage::get_singleton();
}

bool CSharpScript::get_property_default_value(const StringName &p_property, Variant &r_value) const {

#ifdef TOOLS_ENABLED

	const Map<StringName, Variant>::Element *E = exported_members_defval_cache.find(p_property);
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

void CSharpScript::update_exports() {

#ifdef TOOLS_ENABLED
	_update_exports();
#endif
}

Ref<Script> CSharpScript::get_base_script() const {

	// TODO search in metadata file once we have it, not important any way?
	return Ref<Script>();
}

void CSharpScript::get_script_property_list(List<PropertyInfo> *p_list) const {

	for (Map<StringName, PropertyInfo>::Element *E = member_info.front(); E; E = E->next()) {
		p_list->push_back(E->value());
	}
}

int CSharpScript::get_member_line(const StringName &p_member) const {

	// TODO omnisharp
	return -1;
}

Error CSharpScript::load_source_code(const String &p_path) {

	PoolVector<uint8_t> sourcef;
	Error err;
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ, &err);
	ERR_FAIL_COND_V(err != OK, err);

	int len = f->get_len();
	sourcef.resize(len + 1);
	PoolVector<uint8_t>::Write w = sourcef.write();
	int r = f->get_buffer(w.ptr(), len);
	f->close();
	memdelete(f);
	ERR_FAIL_COND_V(r != len, ERR_CANT_OPEN);
	w[len] = 0;

	String s;
	if (s.parse_utf8((const char *)w.ptr())) {

		ERR_EXPLAIN("Script '" + p_path + "' contains invalid unicode (utf-8), so it was not loaded. Please ensure that scripts are saved in valid utf-8 unicode.");
		ERR_FAIL_V(ERR_INVALID_DATA);
	}

	source = s;

#ifdef TOOLS_ENABLED
	source_changed_cache = true;
#endif

	return OK;
}

StringName CSharpScript::get_script_name() const {

	return name;
}

CSharpScript::CSharpScript()
	: script_list(this) {

	_clear();

#ifdef TOOLS_ENABLED
	source_changed_cache = false;
	exports_invalidated = true;
#endif

	_resource_path_changed();

#ifdef DEBUG_ENABLED

#ifndef NO_THREADS
	CSharpLanguage::get_singleton()->lock->lock();
#endif

	CSharpLanguage::get_singleton()->script_list.add(&script_list);

#ifndef NO_THREADS
	CSharpLanguage::get_singleton()->lock->unlock();
#endif

#endif // DEBUG_ENABLED
}

CSharpScript::~CSharpScript() {

#ifdef DEBUG_ENABLED

#ifndef NO_THREADS
	CSharpLanguage::get_singleton()->lock->lock();
#endif

	CSharpLanguage::get_singleton()->script_list.remove(&script_list);

#ifndef NO_THREADS
	CSharpLanguage::get_singleton()->lock->unlock();
#endif

#endif // DEBUG_ENABLED
}

/*************** RESOURCE ***************/

RES ResourceFormatLoaderCSharpScript::load(const String &p_path, const String &p_original_path, Error *r_error) {

	if (r_error)
		*r_error = ERR_FILE_CANT_OPEN;

	// TODO ignore anything inside bin/ and obj/ in tools builds?

	CSharpScript *script = memnew(CSharpScript);

	Ref<CSharpScript> scriptres(script);

#if defined(DEBUG_ENABLED) || defined(TOOLS_ENABLED)
	Error err = script->load_source_code(p_path);
	ERR_FAIL_COND_V(err != OK, RES());
#endif

	script->set_path(p_original_path);

#ifndef TOOLS_ENABLED

#ifdef DEBUG_ENABLED
	// User is responsible for thread attach/detach
	ERR_EXPLAIN("Thread is not attached");
	CRASH_COND(mono_domain_get() == NULL);
#endif

#else
	if (Engine::get_singleton()->is_editor_hint() && mono_domain_get() == NULL) {

		CRASH_COND(Thread::get_caller_id() == Thread::get_main_id());

		// Thread is not attached, but we will make an exception in this case
		// because this may be called by one of the editor's worker threads.
		// Attach this thread temporarily to reload the script.

		MonoThread *mono_thread = mono_thread_attach(SCRIPTS_DOMAIN);
		CRASH_COND(mono_thread == NULL);
		script->reload();
		mono_thread_detach(mono_thread);

	} else // just reload it normally
#endif
	script->reload();

	if (r_error)
		*r_error = OK;

	return scriptres;
}

void ResourceFormatLoaderCSharpScript::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("cs");
}

bool ResourceFormatLoaderCSharpScript::handles_type(const String &p_type) const {

	return p_type == "Script" || p_type == CSharpLanguage::get_singleton()->get_type();
}

String ResourceFormatLoaderCSharpScript::get_resource_type(const String &p_path) const {

	return p_path.get_extension().to_lower() == "cs" ? CSharpLanguage::get_singleton()->get_type() : "";
}

Error ResourceFormatSaverCSharpScript::save(const String &p_path, const RES &p_resource, uint32_t p_flags) {

	Ref<CSharpScript> sqscr = p_resource;
	ERR_FAIL_COND_V(sqscr.is_null(), ERR_INVALID_PARAMETER);

	String source = sqscr->get_source_code();

#ifdef TOOLS_ENABLED
	if (!FileAccess::exists(p_path)) {
		// The file does not yet exists, let's assume the user just created this script

		if (_create_project_solution_if_needed()) {
			CSharpProject::add_item(GodotSharpDirs::get_project_csproj_path(),
					"Compile",
					ProjectSettings::get_singleton()->globalize_path(p_path));
		} else {
			ERR_PRINTS("Cannot add " + p_path + " to the C# project because it could not be created.");
		}
	}
#endif

	Error err;
	FileAccess *file = FileAccess::open(p_path, FileAccess::WRITE, &err);
	ERR_FAIL_COND_V(err, err);

	file->store_string(source);

	if (file->get_error() != OK && file->get_error() != ERR_FILE_EOF) {
		memdelete(file);
		return ERR_CANT_CREATE;
	}

	file->close();
	memdelete(file);

	if (ScriptServer::is_reload_scripts_on_save_enabled()) {
		CSharpLanguage::get_singleton()->reload_tool_script(p_resource, false);
	}

	return OK;
}

void ResourceFormatSaverCSharpScript::get_recognized_extensions(const RES &p_resource, List<String> *p_extensions) const {

	if (Object::cast_to<CSharpScript>(p_resource.ptr())) {
		p_extensions->push_back("cs");
	}
}

bool ResourceFormatSaverCSharpScript::recognize(const RES &p_resource) const {

	return Object::cast_to<CSharpScript>(p_resource.ptr()) != NULL;
}

CSharpLanguage::StringNameCache::StringNameCache() {

	_signal_callback = StaticCString::create("_signal_callback");
	_set = StaticCString::create("_set");
	_get = StaticCString::create("_get");
	_notification = StaticCString::create("_notification");
	_script_source = StaticCString::create("script/source");
	dotctor = StaticCString::create(".ctor");
}
