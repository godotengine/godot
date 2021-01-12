_module_gdscript_srcs = files([
    'gdscript.cpp',
    'gdscript_analyzer.cpp',
    'gdscript_byte_codegen.cpp',
    'gdscript_cache.cpp',
    'gdscript_compiler.cpp',
    'gdscript_disassembler.cpp',
    'gdscript_editor.cpp',
    'gdscript_function.cpp',
    'gdscript_parser.cpp',
    'gdscript_tokenizer.cpp',
    'gdscript_utility_functions.cpp',
    'gdscript_vm.cpp',
    'gdscript_warning.cpp',
    'register_types.cpp',
])

_module_gdscript_cpp_args = []
_module_gdscript_deps = [DEP_GODOT_ALL]

# do we have tools?
if get_option('tools')
    _module_gdscript_srcs += files([
        'editor/gdscript_highlighter.cpp',
        'editor/gdscript_translation_parser_plugin.cpp',
    ])

    if 'jsonrpc' in MODULES_ENABLED and 'websocket' in MODULES_ENABLED
        _module_gdscript_srcs += files([
            'language_server/gdscript_extend_parser.cpp',
            'language_server/gdscript_language_protocol.cpp',
            'language_server/gdscript_language_server.cpp',
            'language_server/gdscript_text_document.cpp',
            'language_server/gdscript_workspace.cpp',
        ])
        _module_gdscript_deps += [DEP_MODULE_JSONRPC]
    else
        _module_gdscript_cpp_args += ['-DGDSCRIPT_NO_LSP']
    endif
endif

# do we have tests?
if get_option('tests')
    _module_gamecenter_srcs += files([
        'tests/test_gdscript.cpp',
    ])
    _module_gdscript_cpp_args += ['-DTESTS_ENABLED']
endif

lib_gdscript = library('module_gdscript', _module_gdscript_srcs, cpp_args: _module_gdscript_cpp_args,
    dependencies: _module_gdscript_deps
)

DEP_MODULE_GDSCRIPT = declare_dependency(link_with: lib_gdscript)

MODULE_DEPENDENCIES += DEP_MODULE_GDSCRIPT