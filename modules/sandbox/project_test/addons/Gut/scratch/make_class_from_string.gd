extends SceneTree
# ##############################################################################
# Proof of concept to create a class from a string instead of writing it to a
# file and then loading it.
#
# I hope to use this to create doubles in the future, though it will make
# debugging the generated code a little  harder, it always bothered me that I
# was writing and reading files.  It's slower and wastes cycles on the HD.
# Probably shouldn't bother me that much, but it does..
# ##############################################################################

class SuperPack:
	extends PackedScene
	var _script =  null
	var _scene = null

	func set_script_obj(obj):
		_script = obj

	func instantiate(edit_state=0):
		var inst = _scene.instantiate(edit_state)
		inst.set_script(_script)
		return inst

	func load_scene(path):
		_scene = load(path)


func make_class():
	var text = ""

	text = "class MadeIt:\n" + \
		"\tvar something=\"hello\"\n" + \
		"\tfunc do_something():\n" +\
		"\t\treturn 'did it'"

	return text

func make_node():
	var text = "extends Node2D\n" + \
	"func do_something():\n" + \
			"\treturn 'did it!!'"
	return text


func get_script_for_text(text):
	var script = GDScript.new()
	script.set_source_code(text)
	script.reload()
	return script

func create_node2d():
	var n = Node2D.new()
	n.set_script(get_script_for_text(make_node()))
	print('create node2d = ', n.do_something())
	n.free()

func create_instance():
	var obj = RefCounted.new()
	obj.set_script(get_script_for_text(make_class()))

	var inner_class = obj.MadeIt.new()
	print('create instance  = ', inner_class.do_something())

func create_scene():
	var s2 = SuperPack.new()
	s2.load_scene('res://test/resources/doubler_test_objects/double_me_scene.tscn')
	print(s2._scene._bundled)
	s2.set_script_obj(get_script_for_text(make_node()))

	var inst = s2.instantiate()
	print('create scene = ', inst.do_something())


func _init():
	create_node2d()
	create_instance()
	create_scene()
	quit()
