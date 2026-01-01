@tool
extends RefCounted

## Enhanced Codebase Scanner for Claude AI
## Provides Cursor-like codebase awareness with intelligent file analysis

class_name CodebaseScanner

## File metadata structure
class FileMetadata:
	var path: String
	var full_path: String
	var name: String
	var extension: String
	var content: String = ""
	var class_name: String = ""
	var extends_class: String = ""
	var extends_path: String = ""
	var imports: Array[String] = []
	var functions: Array[String] = []
	var signals: Array[String] = []
	var variables: Array[String] = []
	var is_tool: bool = false
	var line_count: int = 0
	var last_modified: int = 0
	
	func _init(p_path: String, p_full_path: String):
		path = p_path
		full_path = p_full_path
		name = p_path.get_file()
		extension = name.get_extension()

## Project structure cache
static var _project_cache: Dictionary = {}
static var _file_metadata_cache: Dictionary = {}
static var _dependency_graph: Dictionary = {}

## Scan and cache the entire project
static func scan_and_cache_project(root_path: String = "res://", force_refresh: bool = false) -> Dictionary:
	if not force_refresh and _project_cache.has(root_path):
		return _project_cache[root_path]
	
	var structure = {
		"files": [],
		"scripts": [],
		"scenes": [],
		"directories": [],
		"metadata": {}
	}
	
	_file_metadata_cache.clear()
	_dependency_graph.clear()
	
	_scan_directory(root_path, structure, "")
	
	# Parse all scripts for metadata
	for script_info in structure.scripts:
		var metadata = _parse_script_metadata(script_info.full_path)
		if metadata:
			_file_metadata_cache[script_info.path] = metadata
			structure.metadata[script_info.path] = metadata
	
	# Build dependency graph
	_build_dependency_graph()
	
	_project_cache[root_path] = structure
	return structure

## Recursively scan directory
static func _scan_directory(path: String, structure: Dictionary, relative_path: String):
	var dir = DirAccess.open(path)
	if dir == null:
		return
	
	dir.list_dir_begin()
	var file_name = dir.get_next()
	
	while file_name != "":
		# Skip hidden files and common ignore patterns
		if file_name.begins_with(".") or file_name in ["node_modules", ".git", ".import"]:
			file_name = dir.get_next()
			continue
		
		if dir.current_is_dir():
			var full_path = path.path_join(file_name)
			var rel_path = relative_path.path_join(file_name) if relative_path != "" else file_name
			structure.directories.append(rel_path)
			_scan_directory(full_path, structure, rel_path)
		else:
			var full_path = path.path_join(file_name)
			var rel_path = relative_path.path_join(file_name) if relative_path != "" else file_name
			
			var file_info = {
				"path": rel_path,
				"full_path": full_path,
				"name": file_name,
				"extension": file_name.get_extension()
			}
			structure.files.append(file_info)
			
			if file_name.ends_with(".gd"):
				structure.scripts.append(file_info)
			elif file_name.ends_with(".tscn") or file_name.ends_with(".scn"):
				structure.scenes.append(file_info)
		
		file_name = dir.get_next()

## Parse GDScript file to extract metadata
static func _parse_script_metadata(file_path: String) -> FileMetadata:
	var file = FileAccess.open(file_path, FileAccess.READ)
	if file == null:
		return null
	
	var content = file.get_as_text()
	file.close()
	
	var rel_path = file_path.trim_prefix("res://")
	var metadata = FileMetadata.new(rel_path, file_path)
	metadata.content = content
	metadata.line_count = content.split("\n").size()
	
	# Parse basic structure using regex (lightweight parsing)
	var lines = content.split("\n")
	var in_class = false
	
	for i in range(min(lines.size(), 100)):  # Parse first 100 lines for metadata
		var line = lines[i].strip_edges()
		
		# Skip empty lines and comments
		if line.is_empty() or line.begins_with("#"):
			continue
		
		# Check for @tool
		if line.begins_with("@tool"):
			metadata.is_tool = true
			continue
		
		# Check for class_name
		if line.begins_with("class_name"):
			var regex = RegEx.new()
			regex.compile("class_name\\s+([A-Za-z_][A-Za-z0-9_]*)")
			var result = regex.search(line)
			if result:
				metadata.class_name = result.get_string(1)
			continue
		
		# Check for extends
		if line.begins_with("extends"):
			var regex = RegEx.new()
			# Match: extends "path/to/file.gd" or extends Node or extends "path/to/file.gd".SomeClass
			regex.compile("extends\\s+(\"([^\"]+)\"|([A-Za-z_][A-Za-z0-9_.]*))")
			var result = regex.search(line)
			if result:
				var match_str = result.get_string(1)
				if match_str.begins_with("\""):
					metadata.extends_path = result.get_string(2)
				else:
					metadata.extends_class = match_str
			continue
		
		# Check for preload/load imports
		if "preload(" in line or ".load(" in line or "load(" in line:
			var regex = RegEx.new()
			regex.compile("(?:preload|load)\\(\\s*\"([^\"]+)\"\\s*\\)")
			var results = regex.search_all(line)
			for r in results:
				var import_path = r.get_string(1)
				if not metadata.imports.has(import_path):
					metadata.imports.append(import_path)
			continue
		
		# Extract function names
		if line.begins_with("func "):
			var regex = RegEx.new()
			regex.compile("func\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\(")
			var result = regex.search(line)
			if result:
				metadata.functions.append(result.get_string(1))
			continue
		
		# Extract signal names
		if line.begins_with("signal "):
			var regex = RegEx.new()
			regex.compile("signal\\s+([A-Za-z_][A-Za-z0-9_]*)")
			var result = regex.search(line)
			if result:
				metadata.signals.append(result.get_string(1))
			continue
		
		# Extract @export variables
		if "@export" in line and "var " in line:
			var regex = RegEx.new()
			regex.compile("var\\s+([A-Za-z_][A-Za-z0-9_]*)")
			var result = regex.search(line)
			if result:
				metadata.variables.append(result.get_string(1))
			continue
	
	return metadata

## Build dependency graph between files
static func _build_dependency_graph():
	_dependency_graph.clear()
	
	for path in _file_metadata_cache:
		var metadata = _file_metadata_cache[path]
		if not _dependency_graph.has(path):
			_dependency_graph[path] = {
				"dependencies": [],  # Files this file depends on
				"dependents": []     # Files that depend on this file
			}
		
		# Add extends path as dependency
		if not metadata.extends_path.is_empty():
			var extends_full_path = _resolve_path(metadata.path, metadata.extends_path)
			if extends_full_path != "":
				_dependency_graph[path].dependencies.append(extends_full_path)
				if not _dependency_graph.has(extends_full_path):
					_dependency_graph[extends_full_path] = {"dependencies": [], "dependents": []}
				_dependency_graph[extends_full_path].dependents.append(path)
		
		# Add imports as dependencies
		for import_path in metadata.imports:
			var import_full_path = _resolve_path(metadata.path, import_path)
			if import_full_path != "":
				_dependency_graph[path].dependencies.append(import_full_path)
				if not _dependency_graph.has(import_full_path):
					_dependency_graph[import_full_path] = {"dependencies": [], "dependents": []}
				_dependency_graph[import_full_path].dependents.append(path)

## Resolve relative path to absolute project path
static func _resolve_path(from_path: String, relative_path: String) -> String:
	if relative_path.begins_with("res://"):
		return relative_path.trim_prefix("res://")
	
	var base_dir = from_path.get_base_dir()
	if base_dir == "":
		base_dir = "."
	
	var resolved = base_dir.path_join(relative_path).simplify_path()
	return resolved

## Get project summary with enhanced structure
static func get_project_summary(root_path: String = "res://") -> String:
	var structure = scan_and_cache_project(root_path)
	var summary = "PROJECT STRUCTURE:\n"
	summary += "==================\n\n"
	
	summary += "Scripts (" + str(structure.scripts.size()) + "):\n"
	for script in structure.scripts.slice(0, 100):
		var metadata = structure.metadata.get(script.path, null)
		if metadata and not metadata.class_name.is_empty():
			summary += "  - " + script.path + " (class: " + metadata.class_name + ")\n"
		else:
			summary += "  - " + script.path + "\n"
	if structure.scripts.size() > 100:
		summary += "  ... and " + str(structure.scripts.size() - 100) + " more\n"
	
	summary += "\nScenes (" + str(structure.scenes.size()) + "):\n"
	for scene in structure.scenes.slice(0, 50):
		summary += "  - " + scene.path + "\n"
	if structure.scenes.size() > 50:
		summary += "  ... and " + str(structure.scenes.size() - 50) + " more\n"
	
	# Show key classes
	var key_classes = []
	for path in structure.metadata:
		var metadata = structure.metadata[path]
		if not metadata.class_name.is_empty():
			key_classes.append(metadata.class_name + " -> " + path)
	
	if key_classes.size() > 0:
		summary += "\nKey Classes:\n"
		for cls in key_classes.slice(0, 30):
			summary += "  - " + cls + "\n"
	
	return summary

## Get relevant files with intelligent scoring (Cursor-like)
static func get_relevant_files(user_prompt: String, root_path: String = "res://", max_files: int = 15) -> Array:
	var structure = scan_and_cache_project(root_path)
	var prompt_lower = user_prompt.to_lower()
	var keywords = _extract_keywords(prompt_lower)
	
	var scored_files = []
	
	# Score all scripts
	for script_path in structure.metadata:
		var metadata = structure.metadata[script_path]
		var score = _calculate_relevance_score(metadata, keywords, prompt_lower)
		
		if score > 0:
			scored_files.append({
				"metadata": metadata,
				"score": score
			})
	
	# Sort by score
	scored_files.sort_custom(func(a, b): return a.score > b.score)
	
	# Get top files and their dependencies
	var selected_files = {}
	var result = []
	
	for item in scored_files.slice(0, max_files):
		var metadata = item.metadata
		selected_files[metadata.path] = true
		
		# Add file content
		if metadata.content.is_empty():
			metadata.content = read_file_content(metadata.full_path)
		
		result.append({
			"path": metadata.path,
			"content": metadata.content,
			"score": item.score,
			"class_name": metadata.class_name,
			"extends": metadata.extends_class if metadata.extends_class != "" else metadata.extends_path
		})
		
		# Add related files (dependencies and dependents)
		if _dependency_graph.has(metadata.path):
			var deps = _dependency_graph[metadata.path]
			
			# Add top dependencies
			for dep_path in deps.dependencies.slice(0, 2):
				if not selected_files.has(dep_path) and structure.metadata.has(dep_path):
					var dep_metadata = structure.metadata[dep_path]
					if dep_metadata.content.is_empty():
						dep_metadata.content = read_file_content(dep_metadata.full_path)
					
					result.append({
						"path": dep_metadata.path,
						"content": dep_metadata.content,
						"score": item.score * 0.5,  # Lower score for dependencies
						"class_name": dep_metadata.class_name,
						"extends": dep_metadata.extends_class if dep_metadata.extends_class != "" else dep_metadata.extends_path,
						"relation": "dependency"
					})
					selected_files[dep_path] = true
					
					if result.size() >= max_files * 2:  # Allow some overflow for context
						break
	
	# Re-sort by score
	result.sort_custom(func(a, b): return a.score > b.score)
	
	return result.slice(0, max_files)

## Extract meaningful keywords from prompt
static func _extract_keywords(text: String) -> Array:
	var words = text.split(" ")
	var keywords = []
	var stop_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "should", "could", "can", "may", "might", "must", "create", "make", "add", "new", "script", "file", "function", "class"]
	
	for word in words:
		word = word.strip_edges().to_lower()
		if word.length() > 2 and not stop_words.has(word):
			keywords.append(word)
	
	return keywords

## Calculate relevance score for a file
static func _calculate_relevance_score(metadata: FileMetadata, keywords: Array, prompt: String) -> float:
	var score = 0.0
	
	# Check class name match
	if not metadata.class_name.is_empty():
		var class_lower = metadata.class_name.to_lower()
		for keyword in keywords:
			if class_lower.contains(keyword) or keyword.contains(class_lower):
				score += 10.0
	
	# Check file path/name match
	var path_lower = metadata.path.to_lower()
	var name_lower = metadata.name.to_lower()
	for keyword in keywords:
		if path_lower.contains(keyword) or name_lower.contains(keyword):
			score += 5.0
		if keyword in name_lower:
			score += 8.0
	
	# Check function names
	for func_name in metadata.functions:
		var func_lower = func_name.to_lower()
		for keyword in keywords:
			if func_lower.contains(keyword) or keyword.contains(func_lower):
				score += 3.0
	
	# Check signal names
	for signal_name in metadata.signals:
		var sig_lower = signal_name.to_lower()
		for keyword in keywords:
			if sig_lower.contains(keyword):
				score += 3.0
	
	# Check content (limited to avoid performance issues)
	if metadata.content.length() < 5000:  # Only for smaller files
		var content_lower = metadata.content.to_lower()
		for keyword in keywords:
			var count = content_lower.count(keyword)
			score += count * 0.5
	
	# Boost score for files that extend common base classes mentioned in prompt
	if not metadata.extends_class.is_empty():
		var extends_lower = metadata.extends_class.to_lower()
		for keyword in keywords:
			if extends_lower.contains(keyword):
				score += 4.0
	
	return score

## Read file content
static func read_file_content(file_path: String) -> String:
	var file = FileAccess.open(file_path, FileAccess.READ)
	if file == null:
		return ""
	
	var content = file.get_as_text()
	file.close()
	return content

## Get file metadata
static func get_file_metadata(file_path: String) -> FileMetadata:
	if _file_metadata_cache.has(file_path):
		return _file_metadata_cache[file_path]
	
	# Try to parse if not cached
	var full_path = "res://" + file_path.trim_prefix("res://")
	var metadata = _parse_script_metadata(full_path)
	if metadata:
		_file_metadata_cache[file_path] = metadata
	return metadata

## Get files that depend on a given file
static func get_dependents(file_path: String) -> Array:
	if not _dependency_graph.has(file_path):
		return []
	return _dependency_graph[file_path].dependents.duplicate()

## Get files that a given file depends on
static func get_dependencies(file_path: String) -> Array:
	if not _dependency_graph.has(file_path):
		return []
	return _dependency_graph[file_path].dependencies.duplicate()

## Clear cache (useful for refresh)
static func clear_cache():
	_project_cache.clear()
	_file_metadata_cache.clear()
	_dependency_graph.clear()