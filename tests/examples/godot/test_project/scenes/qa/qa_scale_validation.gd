extends "res://scripts/qa_test_base.gd"
## Scale Validation Test: Ensures non-uniform scaling emits warnings.

@export var expected_warning_substring: String = "non-uniform scale"

var splat_node: Node

func _ready():
	test_name = "Scale Validation"
	test_duration = 2.0
	warmup_frames = 5
	super._ready()

	splat_node = get_node_or_null("SplatNode")

func _on_test_start():
	if splat_node != null:
		splat_node.update_configuration_warnings()

func _on_test_complete():
	if splat_node == null:
		_test_result = false
		_test_message = "SplatNode missing"
		return
	if not splat_node.has_method("get_configuration_warnings"):
		_test_result = false
		_test_message = "get_configuration_warnings not exposed"
		return

	var warnings: PackedStringArray = splat_node.get_configuration_warnings()
	var found = false
	for warning in warnings:
		if String(warning).to_lower().find(expected_warning_substring.to_lower()) != -1:
			found = true
			break

	result_metrics["warnings"] = warnings
	_test_result = found
	_test_message = "Warning found" if found else "Expected warning not found"
