/**************************************************************************/
/*  typescript_runner.cpp                                                 */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#include "typescript_runner.h"

#include "core/os/os.h"
#include "core/io/file_access.h"
#include "core/os/time.h"

TypeScriptRunner::TypeScriptRunner() {
    current_state = EXECUTION_STATE_IDLE;
    typescript_compiler_path = "tsc";
    node_js_path = "node";
    temp_directory = OS::get_singleton()->get_user_data_dir().path_join("ai_agent_temp");
    auto_cleanup_temp_files = true;
    enable_debugging = false;
}

TypeScriptRunner::~TypeScriptRunner() {
    cleanup();
}

void TypeScriptRunner::_bind_methods() {
    // Configuration methods
    ClassDB::bind_method(D_METHOD("set_typescript_compiler_path", "path"), &TypeScriptRunner::set_typescript_compiler_path);
    ClassDB::bind_method(D_METHOD("get_typescript_compiler_path"), &TypeScriptRunner::get_typescript_compiler_path);
    
    ClassDB::bind_method(D_METHOD("set_node_js_path", "path"), &TypeScriptRunner::set_node_js_path);
    ClassDB::bind_method(D_METHOD("get_node_js_path"), &TypeScriptRunner::get_node_js_path);
    
    ClassDB::bind_method(D_METHOD("set_temp_directory", "temp_dir"), &TypeScriptRunner::set_temp_directory);
    ClassDB::bind_method(D_METHOD("get_temp_directory"), &TypeScriptRunner::get_temp_directory);
    
    ClassDB::bind_method(D_METHOD("set_auto_cleanup_temp_files", "auto_cleanup"), &TypeScriptRunner::set_auto_cleanup_temp_files);
    ClassDB::bind_method(D_METHOD("get_auto_cleanup_temp_files"), &TypeScriptRunner::get_auto_cleanup_temp_files);
    
    ClassDB::bind_method(D_METHOD("set_enable_debugging", "enable_debugging"), &TypeScriptRunner::set_enable_debugging);
    ClassDB::bind_method(D_METHOD("get_enable_debugging"), &TypeScriptRunner::get_enable_debugging);

    // Execution methods
    ClassDB::bind_method(D_METHOD("execute_typescript_code", "code", "context_node"), &TypeScriptRunner::execute_typescript_code, DEFVAL(Variant()));
    ClassDB::bind_method(D_METHOD("execute_typescript_file", "file_path", "context_node"), &TypeScriptRunner::execute_typescript_file, DEFVAL(Variant()));
    ClassDB::bind_method(D_METHOD("compile_typescript_to_javascript", "ts_code"), &TypeScriptRunner::compile_typescript_to_javascript);
    
    // State methods
    ClassDB::bind_method(D_METHOD("get_execution_state"), &TypeScriptRunner::get_execution_state);
    ClassDB::bind_method(D_METHOD("get_last_output"), &TypeScriptRunner::get_last_output);
    ClassDB::bind_method(D_METHOD("get_last_error"), &TypeScriptRunner::get_last_error);
    
    // Context methods
    ClassDB::bind_method(D_METHOD("set_execution_context_variable", "name", "value"), &TypeScriptRunner::set_execution_context_variable);
    ClassDB::bind_method(D_METHOD("get_execution_context_variable", "name"), &TypeScriptRunner::get_execution_context_variable);
    ClassDB::bind_method(D_METHOD("clear_execution_context"), &TypeScriptRunner::clear_execution_context);
    ClassDB::bind_method(D_METHOD("get_execution_context"), &TypeScriptRunner::get_execution_context);
    
    // Module methods
    ClassDB::bind_method(D_METHOD("add_module_path", "module_path"), &TypeScriptRunner::add_module_path);
    ClassDB::bind_method(D_METHOD("remove_module_path", "module_path"), &TypeScriptRunner::remove_module_path);
    ClassDB::bind_method(D_METHOD("get_loaded_modules"), &TypeScriptRunner::get_loaded_modules);
    ClassDB::bind_method(D_METHOD("clear_loaded_modules"), &TypeScriptRunner::clear_loaded_modules);
    
    // Utility methods
    ClassDB::bind_method(D_METHOD("is_typescript_available"), &TypeScriptRunner::is_typescript_available);
    ClassDB::bind_method(D_METHOD("is_nodejs_available"), &TypeScriptRunner::is_nodejs_available);
    ClassDB::bind_method(D_METHOD("get_typescript_version"), &TypeScriptRunner::get_typescript_version);
    ClassDB::bind_method(D_METHOD("cleanup"), &TypeScriptRunner::cleanup);
    
    // TypeScript definitions
    ClassDB::bind_method(D_METHOD("get_godot_typescript_definitions"), &TypeScriptRunner::get_godot_typescript_definitions);
    ClassDB::bind_method(D_METHOD("generate_godot_definitions_file", "output_path"), &TypeScriptRunner::generate_godot_definitions_file);

    // Properties
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "typescript_compiler_path"), "set_typescript_compiler_path", "get_typescript_compiler_path");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "node_js_path"), "set_node_js_path", "get_node_js_path");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "temp_directory", PROPERTY_HINT_GLOBAL_DIR), "set_temp_directory", "get_temp_directory");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_cleanup_temp_files"), "set_auto_cleanup_temp_files", "get_auto_cleanup_temp_files");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_debugging"), "set_enable_debugging", "get_enable_debugging");

    // Enums
    BIND_ENUM_CONSTANT(EXECUTION_STATE_IDLE);
    BIND_ENUM_CONSTANT(EXECUTION_STATE_COMPILING);
    BIND_ENUM_CONSTANT(EXECUTION_STATE_RUNNING);
    BIND_ENUM_CONSTANT(EXECUTION_STATE_COMPLETED);
    BIND_ENUM_CONSTANT(EXECUTION_STATE_ERROR);
}

String TypeScriptRunner::_create_temp_file(const String &content, const String &extension) {
    // Ensure temp directory exists
    if (!DirAccess::exists(temp_directory)) {
        DirAccess::create_dir_recursive_absolute(temp_directory);
    }
    
    // Generate unique filename
    String filename = "ai_agent_" + String::num_int64(Time::get_unix_time_from_system()) + "_" + 
                     String::num_int64(Engine::get_singleton()->get_process_frames()) + extension;
    String filepath = temp_directory.path_join(filename);
    
    // Write content to file
    Ref<FileAccess> file = FileAccess::open(filepath, FileAccess::WRITE);
    if (file.is_null()) {
        return "";
    }
    
    file->store_string(content);
    file->close();
    
    return filepath;
}

Error TypeScriptRunner::_compile_typescript(const String &ts_file_path, String &js_file_path) {
    current_state = EXECUTION_STATE_COMPILING;
    
    // Prepare output path
    js_file_path = ts_file_path.get_basename() + ".js";
    
    // Create TypeScript compilation command
    List<String> args;
    args.push_back(ts_file_path);
    args.push_back("--outFile");
    args.push_back(js_file_path);
    args.push_back("--target");
    args.push_back("ES2020");
    args.push_back("--module");
    args.push_back("CommonJS");
    
    if (enable_debugging) {
        args.push_back("--sourceMap");
    }
    
    String output;
    int exit_code = OS::get_singleton()->execute(typescript_compiler_path, args, &output, nullptr, true);
    
    if (exit_code != 0) {
        last_error = "TypeScript compilation failed: " + output;
        current_state = EXECUTION_STATE_ERROR;
        return ERR_COMPILATION_FAILED;
    }
    
    // Check if output file was created
    if (!FileAccess::file_exists(js_file_path)) {
        last_error = "TypeScript compilation succeeded but output file not found";
        current_state = EXECUTION_STATE_ERROR;
        return ERR_FILE_NOT_FOUND;
    }
    
    return OK;
}

Error TypeScriptRunner::_execute_javascript(const String &js_file_path) {
    current_state = EXECUTION_STATE_RUNNING;
    
    List<String> args;
    args.push_back(js_file_path);
    
    String output;
    int exit_code = OS::get_singleton()->execute(node_js_path, args, &output, nullptr, true);
    
    last_output = output;
    
    if (exit_code != 0) {
        last_error = "JavaScript execution failed with exit code " + String::num_int64(exit_code) + ": " + output;
        current_state = EXECUTION_STATE_ERROR;
        return ERR_SCRIPT_FAILED;
    }
    
    current_state = EXECUTION_STATE_COMPLETED;
    return OK;
}

void TypeScriptRunner::_setup_execution_context(Node *context_node) {
    execution_context.clear();
    
    if (context_node) {
        execution_context["context_node"] = context_node;
        execution_context["node_name"] = context_node->get_name();
        execution_context["node_class"] = context_node->get_class();
        
        if (context_node->get_parent()) {
            execution_context["parent_node"] = context_node->get_parent();
        }
        
        if (context_node->is_inside_tree()) {
            execution_context["scene_tree"] = context_node->get_tree();
            if (context_node->get_tree()->current_scene) {
                execution_context["current_scene"] = context_node->get_tree()->current_scene;
            }
        }
    }
}

void TypeScriptRunner::_cleanup_temp_files() {
    if (!auto_cleanup_temp_files || temp_directory.is_empty()) {
        return;
    }
    
    Ref<DirAccess> dir = DirAccess::open(temp_directory);
    if (dir.is_null()) {
        return;
    }
    
    dir->list_dir_begin();
    String file_name = dir->get_next();
    while (!file_name.is_empty()) {
        if (file_name.begins_with("ai_agent_") && (file_name.ends_with(".ts") || file_name.ends_with(".js") || file_name.ends_with(".js.map"))) {
            String file_path = temp_directory.path_join(file_name);
            dir->remove(file_name);
        }
        file_name = dir->get_next();
    }
    dir->list_dir_end();
}

// Configuration methods
void TypeScriptRunner::set_typescript_compiler_path(const String &p_path) {
    typescript_compiler_path = p_path;
}

String TypeScriptRunner::get_typescript_compiler_path() const {
    return typescript_compiler_path;
}

void TypeScriptRunner::set_node_js_path(const String &p_path) {
    node_js_path = p_path;
}

String TypeScriptRunner::get_node_js_path() const {
    return node_js_path;
}

void TypeScriptRunner::set_temp_directory(const String &p_temp_dir) {
    temp_directory = p_temp_dir;
}

String TypeScriptRunner::get_temp_directory() const {
    return temp_directory;
}

void TypeScriptRunner::set_auto_cleanup_temp_files(bool p_auto_cleanup) {
    auto_cleanup_temp_files = p_auto_cleanup;
}

bool TypeScriptRunner::get_auto_cleanup_temp_files() const {
    return auto_cleanup_temp_files;
}

void TypeScriptRunner::set_enable_debugging(bool p_enable_debugging) {
    enable_debugging = p_enable_debugging;
}

bool TypeScriptRunner::get_enable_debugging() const {
    return enable_debugging;
}

// Execution methods
Error TypeScriptRunner::execute_typescript_code(const String &code, Node *context_node) {
    if (code.is_empty()) {
        last_error = "TypeScript code cannot be empty";
        current_state = EXECUTION_STATE_ERROR;
        return ERR_INVALID_PARAMETER;
    }
    
    _setup_execution_context(context_node);
    
    // Create temporary TypeScript file
    String ts_file = _create_temp_file(code, ".ts");
    if (ts_file.is_empty()) {
        last_error = "Failed to create temporary TypeScript file";
        current_state = EXECUTION_STATE_ERROR;
        return ERR_CANT_CREATE;
    }
    
    current_script_path = ts_file;
    
    // Compile TypeScript to JavaScript
    String js_file;
    Error compile_err = _compile_typescript(ts_file, js_file);
    if (compile_err != OK) {
        return compile_err;
    }
    
    // Execute JavaScript
    Error exec_err = _execute_javascript(js_file);
    
    // Cleanup temp files if enabled
    if (auto_cleanup_temp_files) {
        _cleanup_temp_files();
    }
    
    return exec_err;
}

Error TypeScriptRunner::execute_typescript_file(const String &file_path, Node *context_node) {
    if (!FileAccess::file_exists(file_path)) {
        last_error = "TypeScript file not found: " + file_path;
        current_state = EXECUTION_STATE_ERROR;
        return ERR_FILE_NOT_FOUND;
    }
    
    _setup_execution_context(context_node);
    current_script_path = file_path;
    
    // Compile TypeScript to JavaScript
    String js_file;
    Error compile_err = _compile_typescript(file_path, js_file);
    if (compile_err != OK) {
        return compile_err;
    }
    
    // Execute JavaScript
    Error exec_err = _execute_javascript(js_file);
    
    // Cleanup temp files if enabled (only the generated JS file)
    if (auto_cleanup_temp_files && FileAccess::file_exists(js_file)) {
        DirAccess::remove_absolute(js_file);
        String map_file = js_file + ".map";
        if (FileAccess::file_exists(map_file)) {
            DirAccess::remove_absolute(map_file);
        }
    }
    
    return exec_err;
}

Error TypeScriptRunner::compile_typescript_to_javascript(const String &ts_code, String &js_code) {
    if (ts_code.is_empty()) {
        return ERR_INVALID_PARAMETER;
    }
    
    // Create temporary TypeScript file
    String ts_file = _create_temp_file(ts_code, ".ts");
    if (ts_file.is_empty()) {
        return ERR_CANT_CREATE;
    }
    
    // Compile TypeScript
    String js_file;
    Error err = _compile_typescript(ts_file, js_file);
    if (err != OK) {
        return err;
    }
    
    // Read compiled JavaScript
    Ref<FileAccess> file = FileAccess::open(js_file, FileAccess::READ);
    if (file.is_null()) {
        return ERR_CANT_OPEN;
    }
    
    js_code = file->get_as_text();
    file->close();
    
    // Cleanup temp files
    if (auto_cleanup_temp_files) {
        _cleanup_temp_files();
    }
    
    return OK;
}

// State management
TypeScriptRunner::ExecutionState TypeScriptRunner::get_execution_state() const {
    return current_state;
}

String TypeScriptRunner::get_last_output() const {
    return last_output;
}

String TypeScriptRunner::get_last_error() const {
    return last_error;
}

// Context management
void TypeScriptRunner::set_execution_context_variable(const String &name, const Variant &value) {
    execution_context[name] = value;
}

Variant TypeScriptRunner::get_execution_context_variable(const String &name) const {
    return execution_context.get(name, Variant());
}

void TypeScriptRunner::clear_execution_context() {
    execution_context.clear();
}

Dictionary TypeScriptRunner::get_execution_context() const {
    return execution_context;
}

// Module system
void TypeScriptRunner::add_module_path(const String &module_path) {
    if (!loaded_modules.has(module_path)) {
        loaded_modules.push_back(module_path);
    }
}

void TypeScriptRunner::remove_module_path(const String &module_path) {
    loaded_modules.erase(module_path);
}

Array TypeScriptRunner::get_loaded_modules() const {
    return loaded_modules;
}

void TypeScriptRunner::clear_loaded_modules() {
    loaded_modules.clear();
}

// Utilities
bool TypeScriptRunner::is_typescript_available() const {
    List<String> args;
    args.push_back("--version");
    
    String output;
    int exit_code = OS::get_singleton()->execute(typescript_compiler_path, args, &output, nullptr, true);
    
    return exit_code == 0;
}

bool TypeScriptRunner::is_nodejs_available() const {
    List<String> args;
    args.push_back("--version");
    
    String output;
    int exit_code = OS::get_singleton()->execute(node_js_path, args, &output, nullptr, true);
    
    return exit_code == 0;
}

String TypeScriptRunner::get_typescript_version() const {
    List<String> args;
    args.push_back("--version");
    
    String output;
    int exit_code = OS::get_singleton()->execute(typescript_compiler_path, args, &output, nullptr, true);
    
    if (exit_code == 0) {
        return output.strip_edges();
    }
    
    return "TypeScript not available";
}

void TypeScriptRunner::cleanup() {
    if (auto_cleanup_temp_files) {
        _cleanup_temp_files();
    }
    current_state = EXECUTION_STATE_IDLE;
    last_output = "";
    last_error = "";
    current_script_path = "";
}

// Built-in TypeScript definitions
String TypeScriptRunner::get_godot_typescript_definitions() const {
    return R"(
// Basic Godot TypeScript definitions for AI Agents
declare interface GodotAPI {
    print(message: string): void;
    print_rich(message: string): void;
}

declare const Godot: GodotAPI;

declare class Node {
    name: string;
    get_name(): string;
    get_class(): string;
    get_parent(): Node | null;
    get_children(): Node[];
    get_child_count(): number;
    is_inside_tree(): boolean;
    get_tree(): SceneTree | null;
}

declare class SceneTree {
    current_scene: Node | null;
    get_nodes_in_group(group: string): Node[];
}

declare interface AIAgent extends Node {
    remember(key: string, value: any): void;
    recall(key: string): any;
    forget(key: string): void;
    clear_memory(): void;
    get_memory(): {[key: string]: any};
    
    send_message(message: string): void;
    add_action(name: string, callback: Function, description?: string): void;
    remove_action(name: string): void;
}

declare const console: {
    log(...args: any[]): void;
    error(...args: any[]): void;
    warn(...args: any[]): void;
    info(...args: any[]): void;
};
)";
}

Error TypeScriptRunner::generate_godot_definitions_file(const String &output_path) {
    String definitions = get_godot_typescript_definitions();
    
    Ref<FileAccess> file = FileAccess::open(output_path, FileAccess::WRITE);
    if (file.is_null()) {
        return ERR_CANT_CREATE;
    }
    
    file->store_string(definitions);
    file->close();
    
    return OK;
}