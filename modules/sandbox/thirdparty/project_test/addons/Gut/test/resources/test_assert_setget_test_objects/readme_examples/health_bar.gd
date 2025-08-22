extends Control

const Health = preload("res://test/resources/test_assert_setget_test_objects/readme_examples/health.gd")
var health: Health = null :
	get:
		return health
	set(node):
		health = node

@onready var progress_bar = $ProgressBar
@onready var label = $Label

func update() -> void:
	if health != null:
		label.text = "%s / %s" %[health.current_hp, health.max_hp]
		progress_bar.max_value = health.max_hp
		progress_bar.value = health.current_hp
