extends GutTest


var SimpleScene = load("res://test/resources/simple_scene.tscn")
var SimpleSceneScript = load("res://test/resources/simple_scene.gd")

class TheSpawner:
	@export var spawn_resource : SpawnResource

	func spawn():
		var inst = spawn_resource.scene.instantiate()
		inst.foo()
		return inst


class SpawnResource:
	extends Resource

	# The bug is that if this has a type of PackedScene it would error when
	# you tried to use a doubled scene for this value.
	@export var scene : PackedScene = null

	func foo():
		return "bar"



func before_all():
	register_inner_classes(get_script())


func test_are_we_sure_scenes_can_be_doubled():
	var DoubleSimpleScene = double(SimpleScene)
	var inst = DoubleSimpleScene.instantiate()

	assert_not_null(inst)
	assert_has_method(inst, "foo")


func test_can_use_a_double_of_a_scene_as_the_value_of_a_PackedScene_variable():
	var DoubleSimpleScene = double(SimpleScene)
	var res = SpawnResource.new()
	res.scene = DoubleSimpleScene

	var spawner = TheSpawner.new()
	spawner.spawn_resource = res

	var spawned = spawner.spawn()
	assert_not_null(spawned)


func test_can_use_a_double_of_a_scene_as_the_value_of_a_PackedScene_variable2():
	var DoubleSimpleScene = double(SimpleScene)
	var res = SpawnResource.new()
	res.scene = DoubleSimpleScene

	var spawner = TheSpawner.new()
	spawner.spawn_resource = res

	var spawned = spawner.spawn()
	assert_ne(spawned.this_is_different_in_scene, "default")
