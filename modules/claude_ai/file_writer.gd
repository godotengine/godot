@tool
extends RefCounted

## File Writer for Claude AI
## Allows AI-generated code to be written to project files

class_name FileWriter

## Note: class_name registration happens in register_types if needed

## Parse AI response and write files (AI Game Engine - supports all file types)
## Supports multiple formats:
## 1. # File: path/to/file.gd (GDScript)
## 2. # File: path/to/file.tscn (Scene files)
## 3. # File: path/to/file.tres (Resource files)
## 4. ```gdscript:path/to/file.gd
## 5. File: path/to/file.gd
static func parse_and_write_files(response_text: String, project_root: String = "res://") -> Dictionary:
	print("DotAI FileWriter: Starting to parse and write files...")
	print("DotAI FileWriter: Response text length: ", response_text.length())
	
	var result = {
		"success": false,  # Will be set to true if any files are written successfully
		"files_written": [],
		"files_failed": [],
		"messages": [],
		"files_modified": [],
		"files_created": [],
		"error": ""
	}
	
	var files = []
	var current_file = null
	var current_content = ""
	var in_code_block = false
	var code_block_lang = ""
	
	var lines = response_text.split("\n")
	var i = 0
	
	while i < lines.size():
		var line = lines[i]
		
		# Check for file marker patterns
		var file_match = _match_file_marker(line)
		if file_match != null:
			# Save previous file if exists
			if current_file != null and current_content.strip_edges() != "":
				files.append({
					"path": current_file,
					"content": current_content.strip_edges(),
					"is_new": not _file_exists(_resolve_path(project_root, current_file))
				})
			
			# Start new file
			current_file = file_match.strip_edges()
			current_content = ""
			in_code_block = false
			i += 1
			continue
		
		# Check for code block start with file path
		if line.begins_with("```"):
			var lang_and_path = line.trim_prefix("```").strip_edges()
			if ":" in lang_and_path:
				var parts = lang_and_path.split(":", false, 1)
				code_block_lang = parts[0]
				var path_part = parts[1].strip_edges()
				if path_part != "":
					# This is a file path in code block
					if current_file == null:
						current_file = path_part
						current_content = ""
					in_code_block = true
					i += 1
					continue
			else:
				in_code_block = true
				i += 1
				continue
		
		# Check for code block end
		if line.strip_edges() == "```" and in_code_block:
			in_code_block = false
			i += 1
			continue
		
		# Add line to current file content
		if current_file != null:
			if in_code_block or not line.begins_with("```"):
				current_content += line + "\n"
		else:
			# Content before first file marker - might be explanation or code
			# Try to detect if it's actual code
			var stripped_line = line.strip_edges()
			
			# More aggressive code detection
			if stripped_line.begins_with("@tool") or \
			   stripped_line.begins_with("extends") or \
			   stripped_line.begins_with("class_name") or \
			   stripped_line.begins_with("func ") or \
			   stripped_line.begins_with("var ") or \
			   stripped_line.begins_with("const ") or \
			   stripped_line.begins_with("signal ") or \
			   stripped_line.begins_with("[gd_scene") or \
			   stripped_line.begins_with("[ext_resource") or \
			   stripped_line.begins_with("[node"):
				# Looks like code, try to infer filename from class_name or extends
				# We need to look ahead to get more context for inference
				var remaining_text = ""
				for j in range(i, min(i + 50, lines.size())):
					remaining_text += lines[j] + "\n"
				var inferred_filename = _infer_filename_from_code(stripped_line, remaining_text)
				current_file = inferred_filename
				current_content = line + "\n"
				print("DotAI FileWriter: Detected code without file marker, inferred filename: ", inferred_filename)
		
		i += 1
	
	# Save last file
	if current_file != null and current_content.strip_edges() != "":
		files.append({
			"path": current_file,
			"content": current_content.strip_edges(),
			"is_new": not _file_exists(_resolve_path(project_root, current_file))
		})
	
	# If no file markers found, try to extract from code blocks or treat as single file
	if files.size() == 0:
		print("DotAI FileWriter: No file markers found, attempting to extract code...")
		
		# Try to extract code from markdown code blocks first
		var code_regex = RegEx.new()
		code_regex.compile("```(?:gdscript|gd|gdscript)?\\s*\\n([\\s\\S]*?)```")
		var code_match = code_regex.search(response_text)
		if code_match:
			var code_content = code_match.get_string(1).strip_edges()
			print("DotAI FileWriter: Found code block, content length: ", code_content.length())
			if code_content.length() > 10:  # Minimum content check
				var lines = code_content.split("\n")
				var first_line = lines[0] if lines.size() > 0 else ""
				var inferred_path = _infer_filename_from_code(first_line, code_content)
				files.append({
					"path": inferred_path,
					"content": code_content,
					"is_new": true
				})
				print("DotAI FileWriter: Extracted code block to file: ", inferred_path)
		
		# If still no files, check if response contains GDScript code directly
		if files.size() == 0 and response_text.strip_edges() != "":
			# Check if response looks like code (contains GDScript keywords)
			var looks_like_code = false
			var code_keywords = ["extends ", "class_name", "func ", "var ", "const ", "signal ", "@export", "@tool", "@onready"]
			for keyword in code_keywords:
				if keyword in response_text:
					looks_like_code = true
					break
			
			# Also check for scene file markers
			if "[gd_scene" in response_text or "[ext_resource" in response_text or "[node" in response_text:
				looks_like_code = true
			
			if looks_like_code:
				print("DotAI FileWriter: Response contains code keywords, treating as code file...")
				# Remove any markdown formatting or explanations
				var cleaned_text = response_text.strip_edges()
				
				# Try to remove markdown code block markers if present
				if cleaned_text.begins_with("```"):
					var parts = cleaned_text.split("```")
					if parts.size() >= 3:
						cleaned_text = parts[1].strip_edges()
						# Remove language identifier if present
						if "\n" in cleaned_text:
							var first_newline = cleaned_text.find("\n")
							var first_line = cleaned_text.substr(0, first_newline)
							if "gdscript" in first_line.to_lower() or "gd" in first_line.to_lower():
								cleaned_text = cleaned_text.substr(first_newline + 1).strip_edges()
				
				# Remove explanations before code (lines that don't look like code)
				var lines = cleaned_text.split("\n")
				var code_start_idx = 0
				for i in range(lines.size()):
					var line = lines[i].strip_edges()
					if line.begins_with("extends") or \
					   line.begins_with("class_name") or \
					   line.begins_with("@tool") or \
					   line.begins_with("func ") or \
					   line.begins_with("var ") or \
					   line.begins_with("[gd_scene") or \
					   line.begins_with("[ext_resource") or \
					   line.begins_with("[node"):
						code_start_idx = i
						break
				
				if code_start_idx > 0:
					cleaned_text = "\n".join(lines.slice(code_start_idx))
				
				if cleaned_text.length() > 10:  # Minimum content check
					var first_line = lines[code_start_idx] if lines.size() > code_start_idx else ""
					var inferred_path = _infer_filename_from_code(first_line, cleaned_text)
					files.append({
						"path": inferred_path,
						"content": cleaned_text,
						"is_new": true
					})
					print("DotAI FileWriter: Extracted code to file: ", inferred_path, " (", cleaned_text.length(), " chars)")
	
	print("DotAI FileWriter: Found ", files.size(), " file(s) to write")
	
	if files.size() == 0:
		print("DotAI FileWriter: WARNING - No files detected after parsing!")
		print("DotAI FileWriter: Response text length: ", response_text.length())
		print("DotAI FileWriter: Response text preview (first 1000 chars): ", response_text.substr(0, 1000))
		
		# Last resort: if response has any code-like content, save it
		# Be very aggressive - save anything that looks remotely like code
		if response_text.strip_edges().length() > 10:
			var has_code_content = false
			var code_indicators = ["extends", "class_name", "func ", "var ", "const ", "@export", "@tool", "[gd_scene", "[node"]
			for indicator in code_indicators:
				if indicator in response_text:
					has_code_content = true
					break
			
			if has_code_content:
				print("DotAI FileWriter: Detected code content, creating fallback file...")
				var lines = response_text.split("\n")
				var first_line = lines[0] if lines.size() > 0 else ""
				var inferred_path = _infer_filename_from_code(first_line, response_text)
				
				# Clean up the content - remove markdown if present
				var cleaned_content = response_text.strip_edges()
				if cleaned_content.begins_with("```"):
					var parts = cleaned_content.split("```")
					if parts.size() >= 3:
						cleaned_content = parts[1].strip_edges()
						# Remove language identifier
						if "\n" in cleaned_content:
							var first_newline = cleaned_content.find("\n")
							cleaned_content = cleaned_content.substr(first_newline + 1).strip_edges()
				
				files.append({
					"path": inferred_path,
					"content": cleaned_content,
					"is_new": true
				})
				print("DotAI FileWriter: Created fallback file: ", inferred_path)
			else:
				result.error = "No code detected in response. Response may be explanation only."
				result.messages.append("⚠ " + result.error)
				print("DotAI FileWriter: No code indicators found in response")
		else:
			result.error = "Response text is too short or empty"
			result.messages.append("⚠ " + result.error)
	
	# Write files
	for file_data in files:
		var file_path = file_data.path
		
		# Ensure path starts with res://
		if not file_path.begins_with("res://"):
			file_path = _resolve_path(project_root, file_path)
		
		print("DotAI FileWriter: ========================================")
		print("DotAI FileWriter: Processing file: ", file_path)
		print("DotAI FileWriter: Content length: ", file_data.content.length())
		print("DotAI FileWriter: Is new file: ", file_data.get("is_new", true))
		
		var is_new_file = file_data.get("is_new", true)
		var write_result = write_file(file_path, file_data.content)
		
		print("DotAI FileWriter: Write result - Success: ", write_result.success, " Error: ", write_result.error)
		
		if write_result.success:
			# Set success flag if at least one file was written successfully
			result.success = true
			result.files_written.append(file_path)
			if is_new_file:
				result.files_created.append(file_path)
				result.messages.append("✓ Created: " + file_path)
			else:
				result.files_modified.append(file_path)
				result.messages.append("✓ Modified: " + file_path)
		else:
			result.files_failed.append(file_path)
			result.messages.append("✗ Failed: " + file_path + " - " + write_result.error)
	
	# If no files were written and no error was set, set a generic error
	if result.files_written.size() == 0 and result.error == "":
		result.error = "No files were written. Check response format and ensure code is present."
	
	# Log final result
	print("DotAI FileWriter: ========================================")
	print("DotAI FileWriter: Final result - Success: ", result.success)
	print("DotAI FileWriter: Files written: ", result.files_written.size())
	print("DotAI FileWriter: Files created: ", result.files_created.size())
	print("DotAI FileWriter: Files failed: ", result.files_failed.size())
	if result.error != "":
		print("DotAI FileWriter: Error: ", result.error)
	print("DotAI FileWriter: ========================================")
	
	return result

## Infer filename from code content (enhanced for game development)
static func _infer_filename_from_code(first_line: String, full_text: String) -> String:
	# Try to extract class_name first (most reliable)
	var class_name_regex = RegEx.new()
	class_name_regex.compile("class_name\\s+([A-Za-z_][A-Za-z0-9_]*)")
	var class_match = class_name_regex.search(full_text)
	if class_match:
		var class_name = class_match.get_string(1)
		# Convert PascalCase to snake_case for filename
		var filename = class_name.to_snake_case() + ".gd"
		
		# Determine directory based on class name patterns
		var class_lower = class_name.to_lower()
		if "player" in class_lower:
			return "scripts/player/" + filename
		elif "enemy" in class_lower:
			return "scripts/enemies/" + filename
		elif "manager" in class_lower or "game" in class_lower:
			return "scripts/managers/" + filename
		elif "ui" in class_lower or "hud" in class_lower or "menu" in class_lower:
			return "scripts/ui/" + filename
		elif "collectible" in class_lower or "coin" in class_lower or "item" in class_lower:
			return "scripts/collectibles/" + filename
		else:
			return "scripts/" + filename
	
	# Try to infer from extends and content analysis
	var extends_regex = RegEx.new()
	extends_regex.compile("extends\\s+([A-Za-z_][A-Za-z0-9_]*)")
	var extends_match = extends_regex.search(first_line)
	if extends_match:
		var extends_class = extends_match.get_string(1)
		var content_lower = full_text.to_lower()
		
		# Smart inference based on content
		if "player" in content_lower or "movement" in content_lower or "jump" in content_lower:
			if extends_class == "CharacterBody2D" or extends_class == "CharacterBody3D":
				return "scripts/player/player.gd"
		elif "enemy" in content_lower or "patrol" in content_lower or "ai" in content_lower:
			return "scripts/enemies/enemy.gd"
		elif "game" in content_lower and "manager" in content_lower:
			return "scripts/managers/game_manager.gd"
		elif "ui" in content_lower or "hud" in content_lower or "label" in content_lower:
			return "scripts/ui/hud.gd"
		elif "coin" in content_lower or "collectible" in content_lower:
			return "scripts/collectibles/coin.gd"
		
		# Fallback to extends-based inference
		if extends_class == "CharacterBody2D":
			return "scripts/player/player.gd"
		elif extends_class == "CharacterBody3D":
			return "scripts/player/player.gd"
		elif extends_class == "Node2D":
			return "scripts/entity.gd"
		elif extends_class == "Node":
			return "scripts/manager.gd"
		elif extends_class == "Control":
			return "scripts/ui/ui.gd"
		else:
			var filename = extends_class.to_snake_case() + ".gd"
			return "scripts/" + filename
	
	# Default fallback
	return "scripts/generated_script.gd"

## Match various file marker patterns
static func _match_file_marker(line: String) -> String:
	# Pattern 1: # File: path/to/file.gd
	var regex1 = RegEx.new()
	regex1.compile("^#\\s*File:\\s*([^\\n]+)")
	var match1 = regex1.search(line)
	if match1:
		return match1.get_string(1).strip_edges()
	
	# Pattern 2: File: path/to/file.gd (without #)
	var regex2 = RegEx.new()
	regex2.compile("^File:\\s*([^\\n]+)")
	var match2 = regex2.search(line)
	if match2:
		return match2.get_string(1).strip_edges()
	
	return null

## Resolve relative path to absolute project path
static func _resolve_path(project_root: String, relative_path: String) -> String:
	if relative_path.begins_with("res://"):
		return relative_path
	
	var root = project_root.trim_suffix("/")
	if root == "":
		root = "res://"
	
	var path = relative_path.lstrip("/")
	return root + "/" + path

## Check if file exists
static func _file_exists(file_path: String) -> bool:
	return FileAccess.file_exists(file_path)

## Write content to a file with validation and optimization
static func write_file(file_path: String, content: String) -> Dictionary:
	print("DotAI FileWriter: write_file called for: ", file_path, " (content length: ", content.length(), ")")
	
	var result = {"success": false, "error": "", "warnings": [], "optimizations": []}
	
	# Validate content before writing
	if content.is_empty():
		result.error = "Cannot write empty file"
		print("DotAI FileWriter: ERROR - Empty content")
		return result
	
	# Basic syntax validation for GDScript files
	if file_path.ends_with(".gd"):
		var validation_result = _validate_gdscript(content, file_path)
		if not validation_result.valid:
			result.warnings.append_array(validation_result.issues)
		if validation_result.warnings.size() > 0:
			result.warnings.append_array(validation_result.warnings)
	
	# Ensure directory exists
	var dir_path = file_path.get_base_dir()
	print("DotAI FileWriter: Directory path: ", dir_path)
	
	if dir_path != "" and dir_path != "res://":
		var dir = DirAccess.open("res://")
		if dir == null:
			result.error = "Failed to open res:// directory"
			print("DotAI FileWriter: ERROR - Cannot open res:// directory")
			return result
		
		var current_path = "res://"
		var path_parts = dir_path.trim_prefix("res://").split("/")
		print("DotAI FileWriter: Path parts: ", path_parts)
		
		for part in path_parts:
			if part != "":
				current_path += "/" + part
				print("DotAI FileWriter: Checking/creating: ", current_path)
				
				if not dir.dir_exists(current_path):
					var error = dir.make_dir(current_path)
					if error != OK:
						result.error = "Failed to create directory: " + current_path + " (error: " + str(error) + ")"
						print("DotAI FileWriter: ERROR - Failed to create directory: ", current_path, " Error: ", error)
						return result
					else:
						print("DotAI FileWriter: Created directory: ", current_path)
				else:
					print("DotAI FileWriter: Directory exists: ", current_path)
	
	# Optimize content (basic optimizations)
	var optimized_content = _optimize_content(content, file_path)
	if optimized_content != content:
		result.optimizations.append("Applied code optimizations")
	
	# Write file
	print("DotAI FileWriter: Attempting to open file for writing: ", file_path)
	
	# In Godot 4, FileAccess.open() can return null, check error code
	var file = FileAccess.open(file_path, FileAccess.WRITE)
	if file == null:
		var error_code = FileAccess.get_open_error()
		var error_msg = "Failed to open file for writing: " + file_path
		
		# Provide helpful error messages based on error code
		match error_code:
			ERR_FILE_NOT_FOUND:
				error_msg += " (File not found - directory may not exist)"
			ERR_CANT_OPEN:
				error_msg += " (Cannot open - permission denied or file locked)"
			ERR_INVALID_PARAMETER:
				error_msg += " (Invalid path)"
			_:
				error_msg += " (Error code: " + str(error_code) + ")"
		
		result.error = error_msg
		print("DotAI FileWriter: ERROR - Failed to open file: ", file_path)
		print("DotAI FileWriter: Error code: ", error_code)
		print("DotAI FileWriter: File path details - Base dir: ", file_path.get_base_dir(), " File name: ", file_path.get_file())
		print("DotAI FileWriter: Full path: ", file_path)
		return result
	
	print("DotAI FileWriter: File opened successfully, writing content...")
	file.store_string(optimized_content)
	var stored_bytes = file.get_position()
	file.close()
	
	print("DotAI FileWriter: Wrote ", stored_bytes, " bytes to file: ", file_path)
	
	# Verify file was written (give it a moment for file system to update)
	# In Godot, files are written immediately, but verification might need a moment
	result.success = true
	print("DotAI FileWriter: File write completed successfully: ", file_path)
	
	# Optional: Verify file exists (may fail immediately after write due to file system caching)
	# This is informational only - the write succeeded if we got here
	if FileAccess.file_exists(file_path):
		print("DotAI FileWriter: File verified to exist: ", file_path)
	else:
		print("DotAI FileWriter: NOTE - File verification check failed (file may still be written, check file system)")
		# Don't fail here - file might exist but verification is delayed
	
	return result

## Validate GDScript code
static func _validate_gdscript(content: String, file_path: String) -> Dictionary:
	var issues = []
	var warnings = []
	
	# Check for basic syntax issues
	if content.contains("extends") and not content.contains("class_name") and not content.contains("extends Node"):
		# Check if extends is valid
		var extends_regex = RegEx.new()
		extends_regex.compile("extends\\s+([\\w/]+)")
		var match = extends_regex.search(content)
		if match:
			var extends_class = match.get_string(1)
			if extends_class.begins_with("res://") and not FileAccess.file_exists(extends_class):
				warnings.append("Extends path may not exist: " + extends_class)
	
	# Check for common issues
	if content.contains("get_node(\"../\")"):
		warnings.append("Consider using get_parent() instead of get_node(\"../\")")
	
	if content.count("get_node") > 5:
		warnings.append("Consider caching frequently accessed nodes for better performance")
	
	return {"valid": issues.size() == 0, "issues": issues, "warnings": warnings}

## Optimize content
static func _optimize_content(content: String, file_path: String) -> String:
	var optimized = content
	
	# Basic optimizations
	if file_path.ends_with(".gd"):
		# Replace inefficient patterns
		optimized = optimized.replace("get_node(\"../\")", "get_parent()")
		optimized = optimized.replace("get_node(\"./\")", "self")
		
		# Ensure consistent line endings
		optimized = optimized.replace("\r\n", "\n")
		optimized = optimized.replace("\r", "\n")
	
	return optimized

## Extract file paths from AI response
static func extract_file_paths(response_text: String) -> Array:
	var paths = []
	var regex = RegEx.new()
	regex.compile("#\\s*File:\\s*([^\\n]+)")
	
	var search_text = response_text
	while true:
		var result = regex.search(search_text)
		if result == null:
			break
		paths.append(result.get_string(1).strip_edges())
		search_text = search_text.substr(result.get_end())
	
	return paths

