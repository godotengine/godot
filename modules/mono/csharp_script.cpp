/*************************************************************************/
/*  csharp_script.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "core/io/json.h"
#include "core/os/file_access.h"
#include "core/os/os.h"
#include "core/os/thread.h"
#include "core/project_settings.h"

#ifdef TOOLS_ENABLED
#include "editor/bindings_generator.h"
#include "editor/csharp_project.h"
#include "editor/editor_node.h"
#include "editor/godotsharp_editor.h"
#endif

#include "godotsharp_dirs.h"
#include "mono_gd/gd_mono_class.h"
#include "mono_gd/gd_mono_marshal.h"
#include "signal_awaiter_utils.h"
#include "utils/macros.h"
#include "utils/string_utils.h"
#include "utils/thread_local.h"

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

#ifndef MONO_GLUE_ENABLED
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

	GLOBAL_DEF("mono/export/include_scripts_content", false);
#endif
}

void CSharpLanguage::finish() {

	finalizing = true;

#ifdef TOOLS_ENABLED
	// Must be here, to avoid StringName leaks
	if (BindingsGenerator::singleton) {
		memdelete(BindingsGenerator::singleton);
		BindingsGenerator::singleton = NULL;
	}
#endif

	// Release gchandle bindings before finalizing mono runtime
	script_bindings.clear();

	if (gdmono) {
		memdelete(gdmono);
		gdmono = NULL;
	}

	finalizing = false;
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
		"foreach",
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
		"void",
		"volatile",
		"while",

		// Contextual keywords. Not reserved words, but I guess we should include
		// them because this seems to be used only for syntax highlighting.
		"add",
		"alias",
		"ascending",
		"async",
		"await",
		"by",
		"descending",
		"dynamic",
		"equals",
		"from",
		"get",
		"global",
		"group",
		"into",
		"join",
		"let",
		"nameof",
		"on",
		"orderby",
		"partial",
		"remove",
		"select",
		"set",
		"value",
		"var",
		"when",
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

static String get_base_class_name(const String &p_base_class_name, const String p_class_name) {

	String base_class = p_base_class_name;
	if (p_class_name == base_class) {
		base_class = "Godot." + base_class;
	}
	return base_class;
}

Ref<Script> CSharpLanguage::get_template(const String &p_class_name, const String &p_base_class_name) const {

	String script_template = "using " BINDINGS_NAMESPACE ";\n"
							 "using System;\n"
							 "\n"
							 "public class %CLASS% : %BASE%\n"
							 "{\n"
							 "    // Declare member variables here. Examples:\n"
							 "    // private int a = 2;\n"
							 "    // private string b = \"text\";\n"
							 "\n"
							 "    // Called when the node enters the scene tree for the first time.\n"
							 "    public override void _Ready()\n"
							 "    {\n"
							 "        \n"
							 "    }\n"
							 "\n"
							 "//  // Called every frame. 'delta' is the elapsed time since the previous frame.\n"
							 "//  public override void _Process(float delta)\n"
							 "//  {\n"
							 "//      \n"
							 "//  }\n"
							 "}\n";

	String base_class_name = get_base_class_name(p_base_class_name, p_class_name);
	script_template = script_template.replace("%BASE%", base_class_name)
							  .replace("%CLASS%", p_class_name);

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
	String base_class_name = get_base_class_name(p_base_class_name, p_class_name);
	src = src.replace("%BASE%", base_class_name)
				  .replace("%CLASS%", p_class_name)
				  .replace("%TS%", _get_indentation());
	p_script->set_source_code(src);
}

String CSharpLanguage::validate_path(const String &p_path) const {

	String class_name = p_path.get_file().get_basename();
	List<String> keywords;
	get_reserved_words(&keywords);
	if (keywords.find(class_name)) {
		return TTR("Class name can't be a reserved keyword");
	}
	return "";
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

#ifdef TOOLS_ENABLED
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
			Variant::AABB,
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

String CSharpLanguage::make_function(const String &, const String &p_name, const PoolStringArray &p_args) const {
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
	s += ")\n{\n    // Replace with function body.\n}\n";

	return s;
}
#else
String CSharpLanguage::make_function(const String &, const String &, const PoolStringArray &) const {
	return String();
}
#endif

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

Vector<ScriptLanguage::StackInfo> CSharpLanguage::debug_get_current_stack_info() {

#ifdef DEBUG_ENABLED
	_TLS_RECURSION_GUARD_V_(Vector<StackInfo>());

	if (!gdmono->is_runtime_initialized() || !GDMono::get_singleton()->get_core_api_assembly() || !GDMonoUtils::mono_cache.corlib_cache_updated)
		return Vector<StackInfo>();

	MonoObject *stack_trace = mono_object_new(mono_domain_get(), CACHED_CLASS(System_Diagnostics_StackTrace)->get_mono_ptr());

	MonoBoolean need_file_info = true;
	void *ctor_args[1] = { &need_file_info };

	CACHED_METHOD(System_Diagnostics_StackTrace, ctor_bool)->invoke_raw(stack_trace, ctor_args);

	Vector<StackInfo> si;
	si = stack_trace_get_info(stack_trace);

	return si;
#else
	return Vector<StackInfo>();
#endif
}

#ifdef DEBUG_ENABLED
Vector<ScriptLanguage::StackInfo> CSharpLanguage::stack_trace_get_info(MonoObject *p_stack_trace) {

	_TLS_RECURSION_GUARD_V_(Vector<StackInfo>());

	MonoException *exc = NULL;

	GDMonoUtils::StackTrace_GetFrames st_get_frames = CACHED_METHOD_THUNK(System_Diagnostics_StackTrace, GetFrames);
	MonoArray *frames = st_get_frames(p_stack_trace, (MonoObject **)&exc);

	if (exc) {
		GDMonoUtils::debug_print_unhandled_exception(exc);
		return Vector<StackInfo>();
	}

	int frame_count = mono_array_length(frames);

	if (frame_count <= 0)
		return Vector<StackInfo>();

	GDMonoUtils::DebugUtils_StackFrameInfo get_sf_info = CACHED_METHOD_THUNK(DebuggingUtils, GetStackFrameInfo);

	Vector<StackInfo> si;
	si.resize(frame_count);

	for (int i = 0; i < frame_count; i++) {
		StackInfo &sif = si.write[i];
		MonoObject *frame = mono_array_get(frames, MonoObject *, i);

		MonoString *file_name;
		int file_line_num;
		MonoString *method_decl;
		get_sf_info(frame, &file_name, &file_line_num, &method_decl, (MonoObject **)&exc);

		if (exc) {
			GDMonoUtils::debug_print_unhandled_exception(exc);
			return Vector<StackInfo>();
		}

		// TODO
		// what if the StackFrame method is null (method_decl is empty). should we skip this frame?
		// can reproduce with a MissingMethodException on internal calls

		sif.file = GDMonoMarshal::mono_string_to_godot(file_name);
		sif.line = file_line_num;
		sif.func = GDMonoMarshal::mono_string_to_godot(method_decl);
	}

	return si;
}
#endif

void CSharpLanguage::frame() {

	if (gdmono && gdmono->is_runtime_initialized() && gdmono->get_core_api_assembly() != NULL) {
		const Ref<MonoGCHandle> &task_scheduler_handle = GDMonoUtils::mono_cache.task_scheduler_handle;

		if (task_scheduler_handle.is_valid()) {
			MonoObject *task_scheduler = task_scheduler_handle->get_target();

			if (task_scheduler) {
				GDMonoUtils::GodotTaskScheduler_Activate thunk = CACHED_METHOD_THUNK(GodotTaskScheduler, Activate);

				MonoException *exc = NULL;
				thunk(task_scheduler, (MonoObject **)&exc);

				if (exc) {
					GDMonoUtils::debug_unhandled_exception(exc);
					_UNREACHABLE_();
				}
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

	if (!gdmono->is_runtime_initialized())
		return;

	GDMonoAssembly *proj_assembly = gdmono->get_project_assembly();

	String name = ProjectSettings::get_singleton()->get("application/config/name");
	if (name.empty()) {
		name = "UnnamedProject";
	}

	name += ".dll";

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

	if (!gdmono->get_core_api_assembly() && gdmono->metadata_is_api_assembly_invalidated(APIAssembly::API_CORE))
		return; // The core API assembly to load is invalidated

	if (!gdmono->get_editor_api_assembly() && gdmono->metadata_is_api_assembly_invalidated(APIAssembly::API_EDITOR))
		return; // The editor API assembly to load is invalidated

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
				} else {
					// no instance found. Let's remove it so we don't loop forever
					E->get()->placeholders.erase(E->get()->placeholders.front()->get());
				}
			}

			for (Map<ObjectID, List<Pair<StringName, Variant> > >::Element *F = E->get()->pending_reload_state.front(); F; F = F->next()) {
				map[F->key()] = F->get(); //pending to reload, use this one instead
			}

			E->get()->_clear();
		}
	}

	if (gdmono->reload_scripts_domain() != OK) {
		// Failed to reload the scripts domain
		// Make sure to add the scripts back to their owners before returning
		for (Map<Ref<CSharpScript>, Map<ObjectID, List<Pair<StringName, Variant> > > >::Element *E = to_reload.front(); E; E = E->next()) {
			Ref<CSharpScript> scr = E->key();
			for (Map<ObjectID, List<Pair<StringName, Variant> > >::Element *F = E->get().front(); F; F = F->next()) {
				Object *obj = ObjectDB::get_instance(F->key());
				if (!obj)
					continue;
				obj->set_script(scr.get_ref_ptr());
				// Save reload state for next time if not saved
				if (!scr->pending_reload_state.has(obj->get_instance_id())) {
					scr->pending_reload_state[obj->get_instance_id()] = F->get();
				}
			}
		}
		return;
	}

	for (Map<Ref<CSharpScript>, Map<ObjectID, List<Pair<StringName, Variant> > > >::Element *E = to_reload.front(); E; E = E->next()) {

		Ref<CSharpScript> scr = E->key();
		scr->exports_invalidated = true;
		scr->signals_invalidated = true;
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

			if (scr->valid && scr->is_tool() && obj->get_script_instance()->is_placeholder()) {
				// Script instance was a placeholder, but now the script was built successfully and is a tool script.
				// We have to replace the placeholder with an actual C# script instance.
				scr->placeholders.erase(static_cast<PlaceHolderScriptInstance *>(obj->get_script_instance()));
				ScriptInstance *script_instance = scr->instance_create(obj);
				obj->set_script_instance(script_instance); // Not necessary as it's already done in instance_create, but just in case...
			}

			for (List<Pair<StringName, Variant> >::Element *G = F->get().front(); G; G = G->next()) {
				obj->get_script_instance()->set(G->get().first, G->get().second);
			}

			scr->pending_reload_state.erase(obj->get_instance_id()); //as it reloaded, remove pending state
		}

		//if instance states were saved, set them!
	}

	if (Engine::get_singleton()->is_editor_hint()) {
		EditorNode::get_singleton()->get_inspector()->update_tree();
		NodeDock::singleton->update_lists();
	}
}
#endif

void CSharpLanguage::project_assembly_loaded() {

	scripts_metadata.clear();

	String scripts_metadata_filename = "scripts_metadata.";

#ifdef TOOLS_ENABLED
	scripts_metadata_filename += Engine::get_singleton()->is_editor_hint() ? "editor" : "editor_player";
#else
#ifdef DEBUG_ENABLED
	scripts_metadata_filename += "debug";
#else
	scripts_metadata_filename += "release";
#endif
#endif

	String scripts_metadata_path = GodotSharpDirs::get_res_metadata_dir().plus_file(scripts_metadata_filename);

	if (FileAccess::exists(scripts_metadata_path)) {
		String old_json;

		Error ferr = read_all_file_utf8(scripts_metadata_path, old_json);
		ERR_FAIL_COND(ferr != OK);

		Variant old_dict_var;
		String err_str;
		int err_line;
		Error json_err = JSON::parse(old_json, old_dict_var, err_str, err_line);
		if (json_err != OK) {
			ERR_PRINTS("Failed to parse metadata file: '" + err_str + "' (" + String::num_int64(err_line) + ")");
			return;
		}

		scripts_metadata = old_dict_var.operator Dictionary();

		print_verbose("Successfully loaded scripts metadata");
	} else {
		if (!Engine::get_singleton()->is_editor_hint()) {
			ERR_PRINT("Missing scripts metadata file");
		}
	}
}

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

void CSharpLanguage::release_script_gchandle(Ref<MonoGCHandle> &p_gchandle) {

	if (!p_gchandle->is_released()) { // Do not locking unnecessarily
#ifndef NO_THREADS
		get_singleton()->script_gchandle_release_lock->lock();
#endif

		p_gchandle->release();

#ifndef NO_THREADS
		get_singleton()->script_gchandle_release_lock->unlock();
#endif
	}
}

void CSharpLanguage::release_script_gchandle(MonoObject *p_pinned_expected_obj, Ref<MonoGCHandle> &p_gchandle) {

	uint32_t pinned_gchandle = MonoGCHandle::new_strong_handle_pinned(p_pinned_expected_obj); // we might lock after this, so pin it

	if (!p_gchandle->is_released()) { // Do not locking unnecessarily
#ifndef NO_THREADS
		get_singleton()->script_gchandle_release_lock->lock();
#endif

		MonoObject *target = p_gchandle->get_target();

		// We release the gchandle if it points to the MonoObject* we expect (otherwise it was
		// already released and could have been replaced) or if we can't get its target MonoObject*
		// (which doesn't necessarily mean it was released, and we want it released in order to
		// avoid locking other threads unnecessarily).
		if (target == p_pinned_expected_obj || target == NULL) {
			p_gchandle->release();
		}

#ifndef NO_THREADS
		get_singleton()->script_gchandle_release_lock->unlock();
#endif
	}

	MonoGCHandle::free_handle(pinned_gchandle);
}

CSharpLanguage::CSharpLanguage() {

	ERR_FAIL_COND(singleton);
	singleton = this;

	finalizing = false;

	gdmono = NULL;

#ifdef NO_THREADS
	lock = NULL;
	gchandle_bind_lock = NULL;
	script_gchandle_release_lock = NULL;
#else
	lock = Mutex::create();
	script_bind_lock = Mutex::create();
	script_gchandle_release_lock = Mutex::create();
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

	if (script_gchandle_release_lock) {
		memdelete(script_gchandle_release_lock);
		script_gchandle_release_lock = NULL;
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

	CSharpScriptBinding script_binding;

	script_binding.type_name = type_name;
	script_binding.wrapper_class = type_class; // cache
	script_binding.gchandle = MonoGCHandle::create_strong(mono_object);

#ifndef NO_THREADS
	script_bind_lock->lock();
#endif

	void *data = (void *)script_bindings.insert(p_object, script_binding);

#ifndef NO_THREADS
	script_bind_lock->unlock();
#endif

	// Tie managed to unmanaged
	Reference *ref = Object::cast_to<Reference>(p_object);

	if (ref) {
		// Unsafe refcount increment. The managed instance also counts as a reference.
		// This way if the unmanaged world has no references to our owner
		// but the managed instance is alive, the refcount will be 1 instead of 0.
		// See: godot_icall_Reference_Dtor(MonoObject *p_obj, Object *p_ptr)

		ref->reference();
	}

	return data;
}

void CSharpLanguage::free_instance_binding_data(void *p_data) {

	if (GDMono::get_singleton() == NULL) {
#ifdef DEBUG_ENABLED
		CRASH_COND(!script_bindings.empty());
#endif
		// Mono runtime finalized, all the gchandle bindings were already released
		return;
	}

	if (finalizing)
		return; // inside CSharpLanguage::finish(), all the gchandle bindings are released there

#ifndef NO_THREADS
	script_bind_lock->lock();
#endif

	Map<Object *, CSharpScriptBinding>::Element *data = (Map<Object *, CSharpScriptBinding>::Element *)p_data;

	// Set the native instance field to IntPtr.Zero, if not yet garbage collected
	MonoObject *mono_object = data->value().gchandle->get_target();
	if (mono_object) {
		CACHED_FIELD(GodotObject, ptr)->set_value_raw(mono_object, NULL);
	}

	script_bindings.erase(data);

#ifndef NO_THREADS
	script_bind_lock->unlock();
#endif
}

void CSharpLanguage::refcount_incremented_instance_binding(Object *p_object) {

	Reference *ref_owner = Object::cast_to<Reference>(p_object);

#ifdef DEBUG_ENABLED
	CRASH_COND(!ref_owner);
#endif

	void *data = p_object->get_script_instance_binding(get_language_index());
	if (!data)
		return;
	Ref<MonoGCHandle> &gchandle = ((Map<Object *, CSharpScriptBinding>::Element *)data)->get().gchandle;

	if (ref_owner->reference_get_count() > 1 && gchandle->is_weak()) { // The managed side also holds a reference, hence 1 instead of 0
		// The reference count was increased after the managed side was the only one referencing our owner.
		// This means the owner is being referenced again by the unmanaged side,
		// so the owner must hold the managed side alive again to avoid it from being GCed.

		MonoObject *target = gchandle->get_target();
		if (!target)
			return; // Called after the managed side was collected, so nothing to do here

		// Release the current weak handle and replace it with a strong handle.
		uint32_t strong_gchandle = MonoGCHandle::new_strong_handle(target);
		gchandle->release();
		gchandle->set_handle(strong_gchandle, MonoGCHandle::STRONG_HANDLE);
	}
}

bool CSharpLanguage::refcount_decremented_instance_binding(Object *p_object) {

	Reference *ref_owner = Object::cast_to<Reference>(p_object);

#ifdef DEBUG_ENABLED
	CRASH_COND(!ref_owner);
#endif

	int refcount = ref_owner->reference_get_count();

	void *data = p_object->get_script_instance_binding(get_language_index());
	if (!data)
		return refcount == 0;
	Ref<MonoGCHandle> &gchandle = ((Map<Object *, CSharpScriptBinding>::Element *)data)->get().gchandle;

	if (refcount == 1 && !gchandle->is_weak()) { // The managed side also holds a reference, hence 1 instead of 0
		// If owner owner is no longer referenced by the unmanaged side,
		// the managed instance takes responsibility of deleting the owner when GCed.

		MonoObject *target = gchandle->get_target();
		if (!target)
			return refcount == 0; // Called after the managed side was collected, so nothing to do here

		// Release the current strong handle and replace it with a weak handle.
		uint32_t weak_gchandle = MonoGCHandle::new_weak_handle(target);
		gchandle->release();
		gchandle->set_handle(weak_gchandle, MonoGCHandle::WEAK_HANDLE);

		return false;
	}

	return refcount == 0;
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

	ERR_FAIL_COND_V(gchandle.is_null(), NULL);
	return gchandle->get_target();
}

bool CSharpInstance::set(const StringName &p_name, const Variant &p_value) {

	ERR_FAIL_COND_V(!script.is_valid(), false);

	MonoObject *mono_object = get_mono_object();
	ERR_FAIL_NULL_V(mono_object, false);

	GDMonoClass *top = script->script_class;

	while (top && top != script->native) {
		GDMonoField *field = script->script_class->get_field(p_name);

		if (field) {
			field->set_value_from_variant(mono_object, p_value);
			return true;
		}

		GDMonoProperty *property = script->script_class->get_property(p_name);

		if (property) {
			property->set_value(mono_object, GDMonoMarshal::variant_to_mono_object(p_value, property->get_type()));
			return true;
		}

		top = top->get_parent_class();
	}

	// Call _set

	top = script->script_class;

	while (top && top != script->native) {
		GDMonoMethod *method = top->get_method(CACHED_STRING_NAME(_set), 2);

		if (method) {
			Variant name = p_name;
			const Variant *args[2] = { &name, &p_value };

			MonoObject *ret = method->invoke(mono_object, args);

			if (ret && GDMonoMarshal::unbox<MonoBoolean>(ret))
				return true;

			break;
		}

		top = top->get_parent_class();
	}

	return false;
}

bool CSharpInstance::get(const StringName &p_name, Variant &r_ret) const {

	ERR_FAIL_COND_V(!script.is_valid(), false);

	MonoObject *mono_object = get_mono_object();
	ERR_FAIL_NULL_V(mono_object, false);

	GDMonoClass *top = script->script_class;

	while (top && top != script->native) {
		GDMonoField *field = top->get_field(p_name);

		if (field) {
			MonoObject *value = field->get_value(mono_object);
			r_ret = GDMonoMarshal::mono_object_to_variant(value);
			return true;
		}

		GDMonoProperty *property = top->get_property(p_name);

		if (property) {
			MonoException *exc = NULL;
			MonoObject *value = property->get_value(mono_object, &exc);
			if (exc) {
				r_ret = Variant();
				GDMonoUtils::set_pending_exception(exc);
			} else {
				r_ret = GDMonoMarshal::mono_object_to_variant(value);
			}
			return true;
		}

		top = top->get_parent_class();
	}

	// Call _get

	top = script->script_class;

	while (top && top != script->native) {
		GDMonoMethod *method = top->get_method(CACHED_STRING_NAME(_get), 1);

		if (method) {
			Variant name = p_name;
			const Variant *args[1] = { &name };

			MonoObject *ret = method->invoke(mono_object, args);

			if (ret) {
				r_ret = GDMonoMarshal::mono_object_to_variant(ret);
				return true;
			}

			break;
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
		if (top->has_fetched_method_unknown_params(p_method)) {
			return true;
		}

		top = top->get_parent_class();
	}

	return false;
}

Variant CSharpInstance::call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

	MonoObject *mono_object = get_mono_object();

	if (!mono_object) {
		r_error.error = Variant::CallError::CALL_ERROR_INSTANCE_IS_NULL;
		ERR_FAIL_V(Variant());
	}

	if (!script.is_valid())
		ERR_FAIL_V(Variant());

	GDMonoClass *top = script->script_class;

	while (top && top != script->native) {
		GDMonoMethod *method = top->get_method(p_method, p_argcount);

		if (method) {
			MonoObject *return_value = method->invoke(mono_object, p_args);

			r_error.error = Variant::CallError::CALL_OK;

			if (return_value) {
				return GDMonoMarshal::mono_object_to_variant(return_value);
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

		ERR_FAIL_NULL(mono_object);

		_call_multilevel(mono_object, p_method, p_args, p_argcount);
	}
}

void CSharpInstance::_call_multilevel(MonoObject *p_mono_object, const StringName &p_method, const Variant **p_args, int p_argcount) {

	GDMonoClass *top = script->script_class;

	while (top && top != script->native) {
		GDMonoMethod *method = top->get_method(p_method, p_argcount);

		if (method) {
			method->invoke(p_mono_object, p_args);
			return;
		}

		top = top->get_parent_class();
	}
}

void CSharpInstance::call_multilevel_reversed(const StringName &p_method, const Variant **p_args, int p_argcount) {

	// Sorry, the method is the one that controls the call order

	call_multilevel(p_method, p_args, p_argcount);
}

bool CSharpInstance::_reference_owner_unsafe() {

#ifdef DEBUG_ENABLED
	CRASH_COND(!base_ref);
	CRASH_COND(owner == NULL);
	CRASH_COND(unsafe_referenced); // already referenced
#endif

	// Unsafe refcount increment. The managed instance also counts as a reference.
	// This way if the unmanaged world has no references to our owner
	// but the managed instance is alive, the refcount will be 1 instead of 0.
	// See: _unreference_owner_unsafe()

	// May not me referenced yet, so we must use init_ref() instead of reference()
	bool success = Object::cast_to<Reference>(owner)->init_ref();
	unsafe_referenced = success;
	return success;
}

bool CSharpInstance::_unreference_owner_unsafe() {

#ifdef DEBUG_ENABLED
	CRASH_COND(!base_ref);
	CRASH_COND(owner == NULL);
#endif

	if (!unsafe_referenced)
		return false; // Already unreferenced

	unsafe_referenced = false;

	// Called from CSharpInstance::mono_object_disposed() or ~CSharpInstance()

	// Unsafe refcount decrement. The managed instance also counts as a reference.
	// See: _reference_owner_unsafe()

	bool die = static_cast<Reference *>(owner)->unreference();

	if (die) {
		memdelete(owner);
		owner = NULL;
	}

	return die;
}

MonoObject *CSharpInstance::_internal_new_managed() {
#ifdef DEBUG_ENABLED
	CRASH_COND(!gchandle.is_valid());
#endif

	CSharpLanguage::get_singleton()->release_script_gchandle(gchandle);

	ERR_FAIL_NULL_V(owner, NULL);
	ERR_FAIL_COND_V(script.is_null(), NULL);

	if (base_ref)
		_reference_owner_unsafe();

	MonoObject *mono_object = mono_object_new(SCRIPTS_DOMAIN, script->script_class->get_mono_ptr());

	if (!mono_object) {
		script = Ref<CSharpScript>();
		owner->set_script_instance(NULL);
		ERR_EXPLAIN("Failed to allocate memory for the object");
		ERR_FAIL_V(NULL);
	}

	CACHED_FIELD(GodotObject, ptr)->set_value_raw(mono_object, owner);

	// Construct
	GDMonoMethod *ctor = script->script_class->get_method(CACHED_STRING_NAME(dotctor), 0);
	ctor->invoke_raw(mono_object, NULL);

	// Tie managed to unmanaged
	gchandle = MonoGCHandle::create_strong(mono_object);

	return mono_object;
}

void CSharpInstance::mono_object_disposed(MonoObject *p_obj) {

#ifdef DEBUG_ENABLED
	CRASH_COND(base_ref);
	CRASH_COND(gchandle.is_null());
#endif
	CSharpLanguage::get_singleton()->release_script_gchandle(p_obj, gchandle);
}

void CSharpInstance::mono_object_disposed_baseref(MonoObject *p_obj, bool p_is_finalizer, bool &r_owner_deleted) {

#ifdef DEBUG_ENABLED
	CRASH_COND(!base_ref);
	CRASH_COND(gchandle.is_null());
#endif
	if (_unreference_owner_unsafe()) {
		r_owner_deleted = true;
	} else {
		r_owner_deleted = false;
		CSharpLanguage::get_singleton()->release_script_gchandle(p_obj, gchandle);
		if (p_is_finalizer) {
			// If the native instance is still alive, then it was
			// referenced from another thread before the finalizer could
			// unreference it and delete it, so we want to keep it.
			// GC.ReRegisterForFinalize(this) is not safe because the objects
			// referenced by this could have already been collected.
			// Instead we will create a new managed instance here.
			_internal_new_managed();
		}
	}
}

void CSharpInstance::refcount_incremented() {

#ifdef DEBUG_ENABLED
	CRASH_COND(!base_ref);
	CRASH_COND(owner == NULL);
#endif

	Reference *ref_owner = Object::cast_to<Reference>(owner);

	if (ref_owner->reference_get_count() > 1 && gchandle->is_weak()) { // The managed side also holds a reference, hence 1 instead of 0
		// The reference count was increased after the managed side was the only one referencing our owner.
		// This means the owner is being referenced again by the unmanaged side,
		// so the owner must hold the managed side alive again to avoid it from being GCed.

		// Release the current weak handle and replace it with a strong handle.
		uint32_t strong_gchandle = MonoGCHandle::new_strong_handle(gchandle->get_target());
		gchandle->release();
		gchandle->set_handle(strong_gchandle, MonoGCHandle::STRONG_HANDLE);
	}
}

bool CSharpInstance::refcount_decremented() {

#ifdef DEBUG_ENABLED
	CRASH_COND(!base_ref);
	CRASH_COND(owner == NULL);
#endif

	Reference *ref_owner = Object::cast_to<Reference>(owner);

	int refcount = ref_owner->reference_get_count();

	if (refcount == 1 && !gchandle->is_weak()) { // The managed side also holds a reference, hence 1 instead of 0
		// If owner owner is no longer referenced by the unmanaged side,
		// the managed instance takes responsibility of deleting the owner when GCed.

		// Release the current strong handle and replace it with a weak handle.
		uint32_t weak_gchandle = MonoGCHandle::new_weak_handle(gchandle->get_target());
		gchandle->release();
		gchandle->set_handle(weak_gchandle, MonoGCHandle::WEAK_HANDLE);

		return false;
	}

	ref_dying = (refcount == 0);

	return ref_dying;
}

MultiplayerAPI::RPCMode CSharpInstance::_member_get_rpc_mode(GDMonoClassMember *p_member) const {

	if (p_member->has_attribute(CACHED_CLASS(RemoteAttribute)))
		return MultiplayerAPI::RPC_MODE_REMOTE;
	if (p_member->has_attribute(CACHED_CLASS(MasterAttribute)))
		return MultiplayerAPI::RPC_MODE_MASTER;
	if (p_member->has_attribute(CACHED_CLASS(PuppetAttribute)))
		return MultiplayerAPI::RPC_MODE_PUPPET;
	if (p_member->has_attribute(CACHED_CLASS(SlaveAttribute)))
		return MultiplayerAPI::RPC_MODE_PUPPET;
	if (p_member->has_attribute(CACHED_CLASS(RemoteSyncAttribute)))
		return MultiplayerAPI::RPC_MODE_REMOTESYNC;
	if (p_member->has_attribute(CACHED_CLASS(SyncAttribute)))
		return MultiplayerAPI::RPC_MODE_REMOTESYNC;
	if (p_member->has_attribute(CACHED_CLASS(MasterSyncAttribute)))
		return MultiplayerAPI::RPC_MODE_MASTERSYNC;
	if (p_member->has_attribute(CACHED_CLASS(PuppetSyncAttribute)))
		return MultiplayerAPI::RPC_MODE_PUPPETSYNC;

	return MultiplayerAPI::RPC_MODE_DISABLED;
}

MultiplayerAPI::RPCMode CSharpInstance::get_rpc_mode(const StringName &p_method) const {

	GDMonoClass *top = script->script_class;

	while (top && top != script->native) {
		GDMonoMethod *method = top->get_fetched_method_unknown_params(p_method);

		if (method && !method->is_static())
			return _member_get_rpc_mode(method);

		top = top->get_parent_class();
	}

	return MultiplayerAPI::RPC_MODE_DISABLED;
}

MultiplayerAPI::RPCMode CSharpInstance::get_rset_mode(const StringName &p_variable) const {

	GDMonoClass *top = script->script_class;

	while (top && top != script->native) {
		GDMonoField *field = top->get_field(p_variable);

		if (field && !field->is_static())
			return _member_get_rpc_mode(field);

		GDMonoProperty *property = top->get_property(p_variable);

		if (property && !property->is_static())
			return _member_get_rpc_mode(property);

		top = top->get_parent_class();
	}

	return MultiplayerAPI::RPC_MODE_DISABLED;
}

void CSharpInstance::notification(int p_notification) {

	if (p_notification == Object::NOTIFICATION_PREDELETE) {
		// When NOTIFICATION_PREDELETE is sent, we also take the chance to call Dispose().
		// It's safe to call Dispose() multiple times and NOTIFICATION_PREDELETE is guaranteed
		// to be sent at least once, which happens right before the call to the destructor.

		if (base_ref) {
			// It's not safe to proceed if the owner derives Reference and the refcount reached 0.
			// At this point, Dispose() was already called (manually or from the finalizer) so
			// that's not a problem. The refcount wouldn't have reached 0 otherwise, since the
			// managed side references it and Dispose() needs to be called to release it.
			// However, this means C# Reference scripts can't receive NOTIFICATION_PREDELETE, but
			// this is likely the case with GDScript as well: https://github.com/godotengine/godot/issues/6784
			return;
		}

		_call_notification(p_notification);

		MonoObject *mono_object = get_mono_object();
		ERR_FAIL_NULL(mono_object);

		GDMonoUtils::GodotObject_Dispose thunk = CACHED_METHOD_THUNK(GodotObject, Dispose);

		MonoException *exc = NULL;
		thunk(mono_object, (MonoObject **)&exc);

		if (exc) {
			GDMonoUtils::set_pending_exception(exc);
		}

		return;
	}

	_call_notification(p_notification);
}

void CSharpInstance::_call_notification(int p_notification) {

	MonoObject *mono_object = get_mono_object();
	ERR_FAIL_NULL(mono_object);

	// Custom version of _call_multilevel, optimized for _notification

	uint32_t arg = p_notification;
	void *args[1] = { &arg };
	StringName method_name = CACHED_STRING_NAME(_notification);

	GDMonoClass *top = script->script_class;

	while (top && top != script->native) {
		GDMonoMethod *method = top->get_method(method_name, 1);

		if (method) {
			method->invoke_raw(mono_object, args);
			return;
		}

		top = top->get_parent_class();
	}
}

Ref<Script> CSharpInstance::get_script() const {

	return script;
}

ScriptLanguage *CSharpInstance::get_language() {

	return CSharpLanguage::get_singleton();
}

CSharpInstance::CSharpInstance() :
		owner(NULL),
		base_ref(false),
		ref_dying(false),
		unsafe_referenced(false) {
}

CSharpInstance::~CSharpInstance() {

	if (gchandle.is_valid()) {
		gchandle->release(); // Make sure it's released
	}

	if (base_ref && !ref_dying && owner) { // it may be called from the owner's destructor
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
	if (!valid) {
		for (Set<PlaceHolderScriptInstance *>::Element *E = placeholders.front(); E; E = E->next()) {
			E->get()->set_build_failed(true);
		}
		return false;
	}

	bool changed = false;

	if (exports_invalidated) {
		exports_invalidated = false;

		changed = true;

		member_info.clear();
		exported_members_cache.clear();
		exported_members_defval_cache.clear();

		// Here we create a temporary managed instance of the class to get the initial values

		MonoObject *tmp_object = mono_object_new(SCRIPTS_DOMAIN, script_class->get_mono_ptr());

		if (!tmp_object) {
			ERR_PRINT("Failed to create temporary MonoObject");
			return false;
		}

		uint32_t tmp_pinned_gchandle = MonoGCHandle::new_strong_handle_pinned(tmp_object); // pin it (not sure if needed)

		GDMonoMethod *ctor = script_class->get_method(CACHED_STRING_NAME(dotctor), 0);
		MonoException *ctor_exc = NULL;
		ctor->invoke(tmp_object, NULL, &ctor_exc);

		if (ctor_exc) {
			MonoGCHandle::free_handle(tmp_pinned_gchandle);
			tmp_object = NULL;

			ERR_PRINT("Exception thrown from constructor of temporary MonoObject:");
			GDMonoUtils::debug_print_unhandled_exception(ctor_exc);
			return false;
		}

		GDMonoClass *top = script_class;

		while (top && top != native) {
			PropertyInfo prop_info;
			bool exported;

			const Vector<GDMonoField *> &fields = top->get_all_fields();

			for (int i = fields.size() - 1; i >= 0; i--) {
				GDMonoField *field = fields[i];

				if (_get_member_export(top, field, prop_info, exported)) {
					StringName name = field->get_name();

					if (exported) {
						member_info[name] = prop_info;
						exported_members_cache.push_front(prop_info);

						if (tmp_object) {
							exported_members_defval_cache[name] = GDMonoMarshal::mono_object_to_variant(field->get_value(tmp_object));
						}
					} else {
						member_info[name] = prop_info;
					}
				}
			}

			const Vector<GDMonoProperty *> &properties = top->get_all_properties();

			for (int i = properties.size() - 1; i >= 0; i--) {
				GDMonoProperty *property = properties[i];

				if (_get_member_export(top, property, prop_info, exported)) {
					StringName name = property->get_name();

					if (exported) {
						member_info[name] = prop_info;
						exported_members_cache.push_front(prop_info);

						if (tmp_object) {
							MonoException *exc = NULL;
							MonoObject *ret = property->get_value(tmp_object, &exc);
							if (exc) {
								exported_members_defval_cache[name] = Variant();
								GDMonoUtils::debug_print_unhandled_exception(exc);
							} else {
								exported_members_defval_cache[name] = GDMonoMarshal::mono_object_to_variant(ret);
							}
						}
					} else {
						member_info[name] = prop_info;
					}
				}
			}

			top = top->get_parent_class();
		}

		// Dispose the temporary managed instance

		GDMonoUtils::GodotObject_Dispose thunk = CACHED_METHOD_THUNK(GodotObject, Dispose);

		MonoException *exc = NULL;
		thunk(tmp_object, (MonoObject **)&exc);

		if (exc) {
			ERR_PRINT("Exception thrown from method Dispose() of temporary MonoObject:");
			GDMonoUtils::debug_print_unhandled_exception(exc);
		}

		MonoGCHandle::free_handle(tmp_pinned_gchandle);
		tmp_object = NULL;
	}

	if (placeholders.size()) {
		// Update placeholders if any
		Map<StringName, Variant> values;
		List<PropertyInfo> propnames;
		_update_exports_values(values, propnames);

		for (Set<PlaceHolderScriptInstance *>::Element *E = placeholders.front(); E; E = E->next()) {
			E->get()->set_build_failed(false);
			E->get()->update(propnames, values);
		}
	}

	return changed;
#endif
	return false;
}

void CSharpScript::load_script_signals(GDMonoClass *p_class, GDMonoClass *p_native_class) {

	// no need to load the script's signals more than once
	if (!signals_invalidated) {
		return;
	}

	// make sure this classes signals are empty when loading for the first time
	_signals.clear();

	GDMonoClass *top = p_class;
	while (top && top != p_native_class) {
		const Vector<GDMonoClass *> &delegates = top->get_all_delegates();
		for (int i = delegates.size() - 1; i >= 0; --i) {
			Vector<Argument> parameters;

			GDMonoClass *delegate = delegates[i];

			if (_get_signal(top, delegate, parameters)) {
				_signals[delegate->get_name()] = parameters;
			}
		}

		top = top->get_parent_class();
	}

	signals_invalidated = false;
}

bool CSharpScript::_get_signal(GDMonoClass *p_class, GDMonoClass *p_delegate, Vector<Argument> &params) {
	if (p_delegate->has_attribute(CACHED_CLASS(SignalAttribute))) {
		MonoType *raw_type = p_delegate->get_mono_type();

		if (mono_type_get_type(raw_type) == MONO_TYPE_CLASS) {
			// Arguments are accessibles as arguments of .Invoke method
			GDMonoMethod *invoke = p_delegate->get_method("Invoke", -1);

			Vector<StringName> names;
			Vector<ManagedType> types;
			invoke->get_parameter_names(names);
			invoke->get_parameter_types(types);

			if (names.size() == types.size()) {
				for (int i = 0; i < names.size(); ++i) {
					Argument arg;
					arg.name = names[i];
					arg.type = GDMonoMarshal::managed_to_variant_type(types[i]);

					if (arg.type == Variant::NIL) {
						ERR_PRINTS("Unknown type of signal parameter: " + arg.name + " in " + p_class->get_full_name());
						return false;
					}

					params.push_back(arg);
				}

				return true;
			}
		}
	}

	return false;
}

#ifdef TOOLS_ENABLED
/**
 * Returns false if there was an error, otherwise true.
 * If there was an error, r_prop_info and r_exported are not assigned any value.
 */
bool CSharpScript::_get_member_export(GDMonoClass *p_class, GDMonoClassMember *p_member, PropertyInfo &r_prop_info, bool &r_exported) {

	StringName name = p_member->get_name();

	if (p_member->is_static()) {
		if (p_member->has_attribute(CACHED_CLASS(ExportAttribute)))
			ERR_PRINTS("Cannot export member because it is static: " + p_class->get_full_name() + "." + name.operator String());
		return false;
	}

	if (member_info.has(name))
		return false;

	ManagedType type;

	if (p_member->get_member_type() == GDMonoClassMember::MEMBER_TYPE_FIELD) {
		type = static_cast<GDMonoField *>(p_member)->get_type();
	} else if (p_member->get_member_type() == GDMonoClassMember::MEMBER_TYPE_PROPERTY) {
		type = static_cast<GDMonoProperty *>(p_member)->get_type();
	} else {
		CRASH_NOW();
	}

	Variant::Type variant_type = GDMonoMarshal::managed_to_variant_type(type);

	if (!p_member->has_attribute(CACHED_CLASS(ExportAttribute))) {
		r_prop_info = PropertyInfo(variant_type, name.operator String(), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_SCRIPT_VARIABLE);
		r_exported = false;
		return true;
	}

	if (p_member->get_member_type() == GDMonoClassMember::MEMBER_TYPE_PROPERTY) {
		GDMonoProperty *property = static_cast<GDMonoProperty *>(p_member);
		if (!property->has_getter() || !property->has_setter()) {
			ERR_PRINTS("Cannot export property because it does not provide a getter or a setter: " + p_class->get_full_name() + "." + name.operator String());
			return false;
		}
	}

	MonoObject *attr = p_member->get_attribute(CACHED_CLASS(ExportAttribute));

	PropertyHint hint = PROPERTY_HINT_NONE;
	String hint_string;

	if (variant_type == Variant::NIL) {
		ERR_PRINTS("Unknown type of exported member: " + p_class->get_full_name() + "." + name.operator String());
		return false;
	} else if (variant_type == Variant::INT && type.type_encoding == MONO_TYPE_VALUETYPE && mono_class_is_enum(type.type_class->get_mono_ptr())) {
		variant_type = Variant::INT;
		hint = PROPERTY_HINT_ENUM;

		Vector<MonoClassField *> fields = type.type_class->get_enum_fields();

		MonoType *enum_basetype = mono_class_enum_basetype(type.type_class->get_mono_ptr());

		String name_only_hint_string;

		// True: enum Foo { Bar, Baz, Quux }
		// True: enum Foo { Bar = 0, Baz = 1, Quux = 2 }
		// False: enum Foo { Bar = 0, Baz = 7, Quux = 5 }
		bool uses_default_values = true;

		for (int i = 0; i < fields.size(); i++) {
			MonoClassField *field = fields[i];

			if (i > 0) {
				hint_string += ",";
				name_only_hint_string += ",";
			}

			String enum_field_name = mono_field_get_name(field);
			hint_string += enum_field_name;
			name_only_hint_string += enum_field_name;

			// TODO:
			// Instead of using mono_field_get_value_object, we can do this without boxing. Check the
			// internal mono functions: ves_icall_System_Enum_GetEnumValuesAndNames and the get_enum_field.

			MonoObject *val_obj = mono_field_get_value_object(mono_domain_get(), field, NULL);

			if (val_obj == NULL) {
				ERR_PRINTS("Failed to get '" + enum_field_name + "' constant enum value of exported member: " +
						   p_class->get_full_name() + "." + name.operator String());
				return false;
			}

			bool r_error;
			uint64_t val = GDMonoUtils::unbox_enum_value(val_obj, enum_basetype, r_error);
			if (r_error) {
				ERR_PRINTS("Failed to unbox '" + enum_field_name + "' constant enum value of exported member: " +
						   p_class->get_full_name() + "." + name.operator String());
				return false;
			}

			if (val != i) {
				uses_default_values = false;
			}

			hint_string += ":";
			hint_string += String::num_uint64(val);
		}

		if (uses_default_values) {
			// If we use the format NAME:VAL, that's what the editor displays.
			// That's annoying if the user is not using custom values for the enum constants.
			// This may not be needed in the future if the editor is changed to not display values.
			hint_string = name_only_hint_string;
		}
	} else if (variant_type == Variant::OBJECT && CACHED_CLASS(GodotReference)->is_assignable_from(type.type_class)) {
		hint = PROPERTY_HINT_RESOURCE_TYPE;
		hint_string = NATIVE_GDMONOCLASS_NAME(type.type_class);
	} else {
		hint = PropertyHint(CACHED_FIELD(ExportAttribute, hint)->get_int_value(attr));
		hint_string = CACHED_FIELD(ExportAttribute, hintString)->get_string_value(attr);
	}

	r_prop_info = PropertyInfo(variant_type, name.operator String(), hint, hint_string, PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_SCRIPT_VARIABLE);
	r_exported = true;

	return true;
}
#endif

void CSharpScript::_clear() {

	tool = false;
	valid = false;

	base = NULL;
	native = NULL;
	script_class = NULL;
}

Variant CSharpScript::call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

	if (unlikely(GDMono::get_singleton() == NULL)) {
		// Probably not the best error but eh.
		r_error.error = Variant::CallError::CALL_ERROR_INSTANCE_IS_NULL;
		return Variant();
	}

	GDMonoClass *top = script_class;

	while (top && top != native) {
		GDMonoMethod *method = top->get_method(p_method, p_argcount);

		if (method && method->is_static()) {
			MonoObject *result = method->invoke(NULL, p_args);

			if (result) {
				return GDMonoMarshal::mono_object_to_variant(result);
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

	p_properties->push_back(PropertyInfo(Variant::STRING, CSharpLanguage::singleton->string_names._script_source, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
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

	script->load_script_signals(script->script_class, script->native);

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

#ifdef TOOLS_ENABLED
	bool extra_cond = tool || ScriptServer::is_scripting_enabled();
#else
	bool extra_cond = true;
#endif

	// FIXME Need to think this through better.
	// For tool scripts, this will never fire if the class is not found. That's because we
	// don't know if it's a tool script if we can't find the class to access the attributes.
	if (extra_cond && !script_class) {
		if (GDMono::get_singleton()->get_project_assembly() == NULL) {
			// The project assembly is not loaded
			ERR_EXPLAIN("Cannot instance script because the project assembly is not loaded. Script: " + get_path());
			ERR_FAIL_V(NULL);
		} else {
			// The project assembly is loaded, but the class could not found
			ERR_EXPLAIN("Cannot instance script because the class '" + name + "' could not be found. Script: " + get_path());
			ERR_FAIL_V(NULL);
		}
	}

	return valid && extra_cond;
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

	MonoObject *mono_object = mono_object_new(SCRIPTS_DOMAIN, script_class->get_mono_ptr());

	if (!mono_object) {
		instance->script = Ref<CSharpScript>();
		instance->owner->set_script_instance(NULL);
		r_error.error = Variant::CallError::CALL_ERROR_INSTANCE_IS_NULL;
		ERR_EXPLAIN("Failed to allocate memory for the object");
		ERR_FAIL_V(NULL);
	}

	uint32_t pinned_gchandle = MonoGCHandle::new_strong_handle_pinned(mono_object); // we might lock after this, so pin it

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
	ctor->invoke(mono_object, p_args);

	// Tie managed to unmanaged
	instance->gchandle = MonoGCHandle::create_strong(mono_object);

	/* STEP 3, PARTY */

	MonoGCHandle::free_handle(pinned_gchandle);

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

#ifdef DEBUG_ENABLED
	CRASH_COND(!valid);
#endif

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

PlaceHolderScriptInstance *CSharpScript::placeholder_instance_create(Object *p_this) {

#ifdef TOOLS_ENABLED
	PlaceHolderScriptInstance *si = memnew(PlaceHolderScriptInstance(CSharpLanguage::get_singleton(), Ref<Script>(this), p_this));
	placeholders.insert(si);
	_update_exports();
	return si;
#else
	return NULL;
#endif
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

	if (!script_class)
		return false;

	return script_class->has_fetched_method_unknown_params(p_method);
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
		const Variant *script_metadata_var = CSharpLanguage::get_singleton()->get_scripts_metadata().getptr(get_path());
		if (script_metadata_var) {
			Dictionary script_metadata = script_metadata_var->operator Dictionary()["class"];
			const Variant *namespace_ = script_metadata.getptr("namespace");
			const Variant *class_name = script_metadata.getptr("class_name");
			ERR_FAIL_NULL_V(namespace_, ERR_BUG);
			ERR_FAIL_NULL_V(class_name, ERR_BUG);
			GDMonoClass *klass = project_assembly->get_class(namespace_->operator String(), class_name->operator String());
			if (klass) {
				bool obj_type = CACHED_CLASS(GodotObject)->is_assignable_from(klass);
				ERR_FAIL_COND_V(!obj_type, ERR_BUG);
				script_class = klass;
			}
		} else {
			// Missing script metadata. Fallback to legacy method
			script_class = project_assembly->get_object_derived_class(name);
		}

		valid = script_class != NULL;

		if (script_class) {
#ifdef DEBUG_ENABLED
			print_verbose("Found class " + script_class->get_namespace() + "." + script_class->get_name() + " for script " + get_path());
#endif

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

			load_script_signals(script_class, native);
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

bool CSharpScript::has_script_signal(const StringName &p_signal) const {
	if (_signals.has(p_signal))
		return true;

	return false;
}

void CSharpScript::get_script_signal_list(List<MethodInfo> *r_signals) const {
	for (const Map<StringName, Vector<Argument> >::Element *E = _signals.front(); E; E = E->next()) {
		MethodInfo mi;

		mi.name = E->key();
		for (int i = 0; i < E->get().size(); i++) {
			PropertyInfo arg;
			arg.name = E->get()[i].name;
			mi.arguments.push_back(arg);
		}
		r_signals->push_back(mi);
	}
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

	Error ferr = read_all_file_utf8(p_path, source);
	if (ferr != OK) {
		if (ferr == ERR_INVALID_DATA) {
			ERR_EXPLAIN("Script '" + p_path + "' contains invalid unicode (utf-8), so it was not loaded. Please ensure that scripts are saved in valid utf-8 unicode.");
		}
		ERR_FAIL_V(ferr);
	}

#ifdef TOOLS_ENABLED
	source_changed_cache = true;
#endif

	return OK;
}

StringName CSharpScript::get_script_name() const {

	return name;
}

CSharpScript::CSharpScript() :
		script_list(this) {

	_clear();

#ifdef TOOLS_ENABLED
	source_changed_cache = false;
	exports_invalidated = true;
#endif

	signals_invalidated = true;

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

#endif

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint() && mono_domain_get() == NULL) {

		CRASH_COND(Thread::get_caller_id() == Thread::get_main_id());

		// Thread is not attached, but we will make an exception in this case
		// because this may be called by one of the editor's worker threads.
		// Attach this thread temporarily to reload the script.

		if (SCRIPTS_DOMAIN) {
			MonoThread *mono_thread = mono_thread_attach(SCRIPTS_DOMAIN);
			CRASH_COND(mono_thread == NULL);
			script->reload();
			mono_thread_detach(mono_thread);
		}

	} else { // just reload it normally
#endif
		script->reload();

#ifdef TOOLS_ENABLED
	}
#endif

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
			ERR_PRINTS("Failed to create C# project");
			ERR_PRINTS("Cannot add " + p_path + " to the C# project");
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
