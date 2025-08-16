#pragma once
#include <godot_cpp/classes/script_language.hpp>

godot::ScriptLanguage *get_elf_language();

#if defined(__wasm__) || defined(__ios__) || defined(__ANDROID__)
#define EDITORLESS_PLATFORM
#else
#define PLATFORM_HAS_EDITOR
#endif
