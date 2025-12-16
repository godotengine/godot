#!/usr/bin/env -S godot --headless --script
# GDScript ELF64 Compiler Tool
# Usage: godot --headless --script tools/gdscript_elf64_compiler.gd <input.gd> [output_dir] [--mode godot|linux|0|1]

@tool
extends SceneTree

func _init():
	var args = OS.get_cmdline_args()
	
	# Find the script name in arguments to get user arguments after it
	var script_name = "gdscript_elf64_compiler.gd"
	var script_index = -1
	for i in range(args.size()):
		if args[i].ends_with(script_name):
			script_index = i
			break
	
	# Parse arguments
	var input_path = ""
	var output_dir = "."
	var mode = 2  # 0 = GODOT_SYSCALL, 1 = LINUX_SYSCALL, 2 = HYBRID (default)
	
	if script_index >= 0:
		var i = script_index + 1
		while i < args.size():
			var arg = args[i]
			if arg == "--mode":
				if i + 1 < args.size():
					var mode_str = args[i + 1].to_lower()
					if mode_str == "godot" or mode_str == "0":
						mode = 0
					elif mode_str == "linux" or mode_str == "1":
						mode = 1
					elif mode_str == "hybrid" or mode_str == "2":
						mode = 2
					else:
						print("Error: Invalid mode '", args[i + 1], "'. Use 'godot'/'0', 'linux'/'1', or 'hybrid'/'2'")
						quit(1)
						return
					i += 2
					continue
			elif not arg.begins_with("--"):
				if input_path.is_empty():
					input_path = arg
				elif output_dir == ".":
					output_dir = arg
			i += 1
	
	if input_path.is_empty():
		print("Usage: godot --headless --script tools/gdscript_elf64_compiler.gd <input.gd> [output_dir] [--mode godot|linux|hybrid|0|1|2]")
		print("  --mode: Compilation mode (default: hybrid)")
		print("    godot or 0: Pure Godot sandbox syscalls (ECALL 500+)")
		print("    linux or 1: Pure Linux syscalls (1-400+)")
		print("    hybrid or 2: Hybrid - Godot syscalls by default, Linux syscalls when needed (recommended)")
		quit(1)
		return
	
	# Load GDScript
	var script = load(input_path) as GDScript
	if script == null:
		print("Error: Failed to load script: ", input_path)
		quit(1)
		return
	
	# Check if script can be compiled
	if not script.can_compile_to_elf64(mode):
		print("Error: Script cannot be compiled to ELF64 in mode ", mode)
		quit(1)
		return
	
	# Compile all functions with specified mode
	var elf64_dict = script.compile_all_functions_to_elf64(mode)
	if elf64_dict.is_empty():
		print("Error: No functions were compiled")
		quit(1)
		return
	
	# Ensure output directory exists
	var dir = DirAccess.open(".")
	if dir == null:
		print("Error: Cannot access current directory")
		quit(1)
		return
	
	# Create output directory if it doesn't exist
	if not dir.dir_exists(output_dir):
		var err = dir.make_dir_recursive(output_dir)
		if err != OK:
			print("Error: Cannot create output directory: ", output_dir)
			quit(1)
			return
	
	# Open output directory for writing
	dir = DirAccess.open(output_dir)
	if dir == null:
		print("Error: Cannot open output directory: ", output_dir)
		quit(1)
		return
	
	# Save each function's ELF64 binary
	var base_name = input_path.get_file().get_basename()
	var success_count = 0
	for func_name in elf64_dict:
		var elf_data = elf64_dict[func_name] as PackedByteArray
		if elf_data.is_empty():
			print("Warning: Function ", func_name, " produced empty ELF64 binary, skipping")
			continue
		
		var output_path = output_dir.path_join(base_name + "_" + func_name + ".elf")
		var file = FileAccess.open(output_path, FileAccess.WRITE)
		if file == null:
			print("Error: Cannot write to: ", output_path)
			continue
		file.store_buffer(elf_data)
		file.close()
		var mode_str = "godot" if mode == 0 else ("linux" if mode == 1 else "hybrid")
		print("Compiled (", mode_str, "): ", func_name, " -> ", output_path, " (", elf_data.size(), " bytes)")
		success_count += 1
	
	if success_count > 0:
		var mode_str = "godot" if mode == 0 else ("linux" if mode == 1 else "hybrid")
		print("Successfully compiled ", success_count, " function(s) in ", mode_str, " mode")
		quit(0)
	else:
		print("Error: No functions were successfully compiled")
		quit(1)
