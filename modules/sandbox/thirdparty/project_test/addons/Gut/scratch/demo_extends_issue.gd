# Script to demo this failing test:
#
# func test_doubled_instances_extend_the_inner_class():
# 	var inst = doubler.double_inner(INNER_CLASSES_PATH, 'InnerA').new()
# 	assert_is(inst, InnerClasses.InnerA)
extends SceneTree

const INNER_CLASSES_PATH = 'res://test/resources/doubler_test_objects/inner_classes.gd'
var InnerClasses = load(INNER_CLASSES_PATH)

class ExtendsInnerClassesInnerAWithPath:
	extends 'res://test/resources/doubler_test_objects/inner_classes.gd'.InnerA

func make_script(script_source):
	var DynScript = GDScript.new()
	DynScript.source_code = script_source
	DynScript.reload()
	return DynScript


# this works
func demo_node2d():
	var script_source = '' + \
	"extends Node2D\n" + \
	"func hello_world():\n" + \
	"\tprint('--- hello world ---')"

	print(script_source)

	var inst = make_script(script_source).new()
	if(inst is Node2D):
		print('pass - yes it is')
	else:
		print('fail - unfortunately it is not')

	inst.free()


func demo_dyn_inner_class():
	print('-- demo_dyn_inner_class')
	var script_source = '' + \
	"extends '" + INNER_CLASSES_PATH + "'.InnerA\n"
	var inst = make_script(script_source).new()

	if(inst is InnerClasses):
		print('fail - is InnerClasses')
	if(inst is InnerClasses.InnerA):
		print('pass - is InnerA')


func demo_inner_extends_full_path():
	print('-- demo_inner_extends_full_path')
	var inst = ExtendsInnerClassesInnerAWithPath.new()
	print(inst.get_a())
	if(inst is InnerClasses):
		print('fail - 2 is InnerClasses')
	if(inst is InnerClasses.InnerA):
		print('pass - 2 is InnerA')


# Currently not demoing anything wrong.
func _init():
	demo_dyn_inner_class()
	demo_inner_extends_full_path()
	demo_node2d()
	quit()
