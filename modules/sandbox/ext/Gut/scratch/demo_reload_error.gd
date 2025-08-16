# Demo for https://github.com/godotengine/godot/issues/65263
extends SceneTree

func _init():
	var script_source = '' + \
	"func hello_world():\n" + \
	"\tprint('--- hello world ---')"

	print(script_source)

	var DynScript = GDScript.new()
	DynScript.source_code = script_source
	DynScript.resource_path = 'workaround for godot issue #65263'
	print('pre-reload')
	DynScript.reload()
	print('post-reload')

	var inst = DynScript.new()
	inst.hello_world()

	quit()
