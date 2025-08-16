extends SceneTree

func print_properties(props, thing, print_all_meta=false):
	for i in range(props.size()):
		var prop_name = props[i].name
		var prop_value = thing.get(props[i].name)
		var print_value = str(prop_value)
		if(print_value.length() > 100):
			print_value = print_value.substr(0, 97) + '...'
		elif(print_value == ''):
			print_value = 'EMPTY'

		print(prop_name, ' = ', print_value)
		if(print_all_meta):
			print('  ', props[i])



# debug/gdscript/warnings/native_method_override = 1
func print_project_settings():
	print(ProjectSettings)
	print_properties(ProjectSettings.get_property_list(), ProjectSettings)

func _init():
	print_project_settings()
	quit()