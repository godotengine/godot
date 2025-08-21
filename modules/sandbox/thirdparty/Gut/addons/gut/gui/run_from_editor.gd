# ------------------------------------------------------------------------------
# This is the entry point when running tests from the editor.
#
# This script should conform to, or ignore, the strictest warning settings.
# ------------------------------------------------------------------------------
extends Node2D

var GutLoader : Object

func _init() -> void:
	GutLoader = load("res://addons/gut/gut_loader.gd")


@warning_ignore("unsafe_method_access")
func _ready() -> void:
	var runner : Node = load("res://addons/gut/gui/GutRunner.tscn").instantiate()
	add_child(runner)
	runner.run_from_editor()
	GutLoader.restore_ignore_addons()
