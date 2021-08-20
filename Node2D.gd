extends Node2D


var enum_replace : Array = [["BEFORE", "AFTER"]]

var collected_files : Array = []

var supported_extensions : Array = [".gd"]#, ".tscn", ".tres", ]


func _ready():
	var project_file_check : File = File.new()
	if !project_file_check.file_exists("project.godot"):
		print("Directory doesn't contains any files")
		return

	check_for_files()
	
	while !collected_files.empty():
		var file_name : String = collected_files.pop_back()
		var file : File = File.new()
		if file.open(file_name, File.READ_WRITE) != OK:
			print("Failed to open file \"" + file_name + "\"")
			continue
		var file_content : String = file.get_as_text()
		file_content = enum_replacing(file_content)
		
		# Save to file
		if file.open(file_name, File.WRITE) != OK:
			print("Failed to save data to file " + file_name)
			
		file.store_string(file_content)
	
func enum_replacing(text_to_replace : String) -> String:
	for i in enum_replace:
		text_to_replace = text_to_replace.replace(i[0],i[1])
	return text_to_replace

func check_for_files():
	var directories_to_check : Array = ["res://"]
	var dir = Directory.new()
	while !directories_to_check.empty():
		var path : String = directories_to_check.pop_back()
		if dir.open(path) == OK:
			dir.list_dir_begin()
			var current_dir : String = dir.get_current_dir()
			var file_name : String = dir.get_next()
			while file_name != "":
				if file_name in ["..", ".", "logs", ".import", "Node2D.gd"]: # Node2D is only for testing
					file_name = dir.get_next()
					continue
				if dir.current_is_dir():
					directories_to_check.append(current_dir +  file_name + "/")
				else:
					var proper_extension : bool = false
					for extension in supported_extensions:
						if file_name.ends_with(extension):
							proper_extension = true
							break
							
					if proper_extension:
						collected_files.append(current_dir + file_name)
						
				file_name = dir.get_next()
		else:
			print("An error occurred when trying to access the path.")
