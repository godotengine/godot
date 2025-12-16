# GDScript ELF64 CLI Option Implementation Guide

## Overview
This guide shows how to add `--gdscript-elf64-compile` CLI option to Godot.

## Changes Required in main/main.cpp

### 1. Add variable declaration (around line 3807, near `gdscript_docs_path`)

```cpp
#ifdef MODULE_GDSCRIPT_ENABLED
	String gdscript_docs_path;
	String gdscript_elf64_compile_input;
	String gdscript_elf64_compile_output;
#endif
```

### 2. Add help text (around line 553, after `--lsp-port`)

```cpp
#if defined(MODULE_GDSCRIPT_ENABLED) && !defined(GDSCRIPT_NO_LSP)
	print_help_option("--lsp-port <port>", "Use the specified port for the GDScript Language Server Protocol. Recommended port range [1024, 49151].\n", CLI_OPTION_AVAILABILITY_EDITOR);
#endif // MODULE_GDSCRIPT_ENABLED && !GDSCRIPT_NO_LSP
#if defined(MODULE_GDSCRIPT_ELF_ENABLED)
	print_help_option("--gdscript-elf64-compile <script.gd> [output_dir]", "Compile GDScript functions to ELF64 binaries.\n", CLI_OPTION_AVAILABILITY_EDITOR);
#endif
```

### 3. Add option parsing (around line 3892, after `--gdscript-docs`)

```cpp
#ifdef MODULE_GDSCRIPT_ENABLED
			} else if (E->get() == "--gdscript-docs") {
				gdscript_docs_path = E->next()->get();
#endif
#if defined(MODULE_GDSCRIPT_ELF_ENABLED)
			} else if (E->get() == "--gdscript-elf64-compile") {
				if (E->next()) {
					gdscript_elf64_compile_input = E->next()->get();
					if (E->next()) {
						gdscript_elf64_compile_output = E->next()->get();
					} else {
						gdscript_elf64_compile_output = ".";
					}
				} else {
					OS::get_singleton()->print("Missing <script.gd> argument for --gdscript-elf64-compile.\n");
					goto error;
				}
#endif
```

### 4. Add execution logic (around line 4340, after gdscript-docs handling)

```cpp
#ifdef MODULE_GDSCRIPT_ENABLED
		if (!doc_tool_path.is_empty() && !gdscript_docs_path.is_empty()) {
			// ... existing code ...
		}
#if defined(MODULE_GDSCRIPT_ELF_ENABLED)
		if (!gdscript_elf64_compile_input.is_empty()) {
			Ref<GDScript> script = ResourceLoader::load(gdscript_elf64_compile_input);
			if (script.is_null()) {
				OS::get_singleton()->print("Error: Failed to load script: " + gdscript_elf64_compile_input + "\n");
				goto error;
			}
			
			if (!script->can_compile_to_elf64()) {
				OS::get_singleton()->print("Error: Script cannot be compiled to ELF64\n");
				goto error;
			}
			
			Dictionary elf64_dict = script->compile_all_functions_to_elf64();
			if (elf64_dict.is_empty()) {
				OS::get_singleton()->print("Error: No functions were compiled\n");
				goto error;
			}
			
			Ref<DirAccess> dir = DirAccess::open(gdscript_elf64_compile_output);
			if (dir.is_null()) {
				OS::get_singleton()->print("Error: Cannot open output directory: " + gdscript_elf64_compile_output + "\n");
				goto error;
			}
			
			String base_name = gdscript_elf64_compile_input.get_file().get_basename();
			for (int i = 0; i < elf64_dict.size(); i++) {
				StringName func_name = elf64_dict.get_key_at_index(i);
				PackedByteArray elf_data = elf64_dict[func_name];
				String output_path = gdscript_elf64_compile_output.path_join(base_name + "_" + func_name + ".elf");
				
				Ref<FileAccess> file = FileAccess::open(output_path, FileAccess::WRITE);
				if (file.is_null()) {
					OS::get_singleton()->print("Error: Cannot write to: " + output_path + "\n");
					continue;
				}
				file->store_buffer(elf_data);
				file->close();
				OS::get_singleton()->print("Compiled: " + func_name + " -> " + output_path + " (" + itos(elf_data.size()) + " bytes)\n");
			}
			
			OS::get_singleton()->print("Successfully compiled " + itos(elf64_dict.size()) + " function(s)\n");
			exit_err = OK;
			goto error;
		}
#endif
#endif
```

## Usage

After implementation:
```bash
godot --headless --gdscript-elf64-compile script.gd output_dir/
```

## Recommendation

**Start with Option 1** (the GDScript tool) - it's already working and requires no engine changes. You can always add the CLI option later if needed.
