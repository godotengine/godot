@tool
extends ConfirmationDialog

var lod: int = 0
var description: String = ""


func _ready() -> void:
	set_unparent_when_invisible(true)
	about_to_popup.connect(_on_about_to_popup)
	visibility_changed.connect(_on_visibility_changed)
	%LodBox.value_changed.connect(_on_lod_box_value_changed)


func _on_about_to_popup() -> void:
	lod = %LodBox.value


func _on_visibility_changed() -> void:
	# Change text on the autowrap label only when the popup is visible.
	# Works around Godot issue #47005:
	# https://github.com/godotengine/godot/issues/47005
	if visible:
		%DescriptionLabel.text = description


func _on_lod_box_value_changed(p_value: float) -> void:
	lod = %LodBox.value
