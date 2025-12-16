#!/usr/bin/env -S godot --headless --script
# GDScript ELF64 Compiler Tool
# Usage: godot --headless --script tools/gdscript_elf64_compiler.gd <input.gd> [output_dir]

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
	
	# Get user arguments (everything after the script)
	var user_args = []
	if script_index >= 0:
		for i in range(script_index + 1, args.size()):
			# Skip Godot flags
			if not args[i].begins_with("--"):
				user_args.append(args[i])
	
	if user_args.size() < 1:
		print("Usage: godot --headless --script tools/gdscript_elf64_compiler.gd <input.gd> [output_dir]")
		quit(1)
		return
	
	var input_path = user_args[0]
	var output_dir = user_args[1] if user_args.size() > 1 else "."
	
	# Load GDScript
	var script = load(input_path) as GDScript
	if script == null:
		print("Error: Failed to load script: ", input_path)
		quit(1)
		return
	
	# Check if script can be compiled
	if not script.can_compile_to_elf64():
		print("Error: Script cannot be compiled to ELF64")
		quit(1)
		return
	
	# Compile all functions
	var elf64_dict = script.compile_all_functions_to_elf64()
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
		print("Compiled: ", func_name, " -> ", output_path, " (", elf_data.size(), " bytes)")
		success_count += 1
	
	if success_count > 0:
		print("Successfully compiled ", success_count, " function(s)")
		quit(0)
	else:
		print("Error: No functions were successfully compiled")
		quit(1)
