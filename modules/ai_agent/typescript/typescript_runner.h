/**************************************************************************/
/*  typescript_runner.h                                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#pragma once

#include "core/object/ref_counted.h"
#include "core/io/file_access.h"
#include "core/variant/variant.h"
#include "scene/main/node.h"

class TypeScriptRunner : public RefCounted {
    GDCLASS(TypeScriptRunner, RefCounted);

public:
    enum ExecutionState {
        EXECUTION_STATE_IDLE,
        EXECUTION_STATE_COMPILING,
        EXECUTION_STATE_RUNNING,
        EXECUTION_STATE_COMPLETED,
        EXECUTION_STATE_ERROR,
    };

private:
    ExecutionState current_state;
    String typescript_compiler_path;
    String temp_directory;
    String current_script_path;
    String last_output;
    String last_error;
    Dictionary execution_context;
    Array loaded_modules;
    
    // Configuration
    bool auto_cleanup_temp_files;
    bool enable_debugging;
    String node_js_path;
    
    String _create_temp_file(const String &content, const String &extension = ".ts");
    Error _compile_typescript(const String &ts_file_path, String &js_file_path);
    Error _execute_javascript(const String &js_file_path);
    void _setup_execution_context(Node *context_node = nullptr);
    void _cleanup_temp_files();

protected:
    static void _bind_methods();

public:
    TypeScriptRunner();
    ~TypeScriptRunner();

    // Configuration
    void set_typescript_compiler_path(const String &p_path);
    String get_typescript_compiler_path() const;
    
    void set_node_js_path(const String &p_path);
    String get_node_js_path() const;
    
    void set_temp_directory(const String &p_temp_dir);
    String get_temp_directory() const;
    
    void set_auto_cleanup_temp_files(bool p_auto_cleanup);
    bool get_auto_cleanup_temp_files() const;
    
    void set_enable_debugging(bool p_enable_debugging);
    bool get_enable_debugging() const;

    // Execution methods
    Error execute_typescript_code(const String &code, Node *context_node = nullptr);
    Error execute_typescript_file(const String &file_path, Node *context_node = nullptr);
    Error compile_typescript_to_javascript(const String &ts_code, String &js_code);
    
    // State management
    ExecutionState get_execution_state() const;
    String get_last_output() const;
    String get_last_error() const;
    
    // Context management
    void set_execution_context_variable(const String &name, const Variant &value);
    Variant get_execution_context_variable(const String &name) const;
    void clear_execution_context();
    Dictionary get_execution_context() const;
    
    // Module system
    void add_module_path(const String &module_path);
    void remove_module_path(const String &module_path);
    Array get_loaded_modules() const;
    void clear_loaded_modules();
    
    // Utilities
    bool is_typescript_available() const;
    bool is_nodejs_available() const;
    String get_typescript_version() const;
    void cleanup();

    // Built-in TypeScript definitions for Godot
    String get_godot_typescript_definitions() const;
    Error generate_godot_definitions_file(const String &output_path);
};

VARIANT_ENUM_CAST(TypeScriptRunner::ExecutionState)