extends PanelContainer

signal close_requested

@onready var label: RichTextLabel = %Content

func _on_close_button_pressed() -> void:
	close_requested.emit()

func _get_license_text() -> String:
	var text = tr("This project is powered by Godot Engine, which relies on a number of third-party free and open source libraries, all compatible with the terms of its MIT license. The following is an exhaustive list of all such third-party components with their respective copyright statements and license terms.") + "\n\n"
	text += tr("Components:") + "\n\n"

	var copyright_info = Engine.get_copyright_info()
	for component in copyright_info:
		var name = component.name
		text += "- " + name
		var parts = component.parts
		for part in parts:
			var copyrights = "";
			for copyright in part.copyright:
				copyrights += "\n    © " + copyright
			text += copyrights

			var license = "\n    License: " + part.license + "\n"
			text += license + "\n\n"

	text += tr("Licenses:") + "\n\n"

	var licenses: Dictionary = Engine.get_license_info()
	for name in licenses:
		var body = licenses[name];
		text += "- " + name + "\n"
		text += "    " + body.replace("\n", "\n    ") + "\n\n"

	return text

func _update_license_text() -> void:
	var text = ""
	if is_visible_in_tree():
		text = _get_license_text()
	label.text = text

func _ready() -> void:
	_update_license_text()

func _on_visibility_changed() -> void:
	if !is_node_ready():
		return

	_update_license_text()
