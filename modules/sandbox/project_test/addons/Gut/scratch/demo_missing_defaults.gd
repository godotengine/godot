extends SceneTree

# Created for https://github.com/godotengine/godot/issues/66218
func mixed_example(p1, p2 = {}, p3 = "a", p4 = 1, p5 = []):
	pass


func _init():
	var script = load('res://scratch/demo_missing_defaults.gd')
	var methods  = script.get_script_method_list()
	for m in methods:
		print('-- ', m.name, ' (', m.default_args.size(), ')')
		print(JSON.stringify(m, '  '))

	quit()