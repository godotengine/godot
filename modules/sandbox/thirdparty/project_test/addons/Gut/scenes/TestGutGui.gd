extends Node2D

var GutRunner = load('res://addons/gut/gui/GutRunner.tscn')

var _runner = GutRunner.instantiate()


func _ready():
	# wait a bit for _utils to be happy.
	await get_tree().create_timer(.2).timeout
	add_child(_runner)


func _on_start_run_pressed():
	_runner.run_tests()
