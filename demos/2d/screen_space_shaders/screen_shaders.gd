
extends Control


func _ready():
	for c in get_node("pictures").get_children():
		get_node("picture").add_item("PIC: " + c.get_name())
	for c in get_node("effects").get_children():
		get_node("effect").add_item("FX: " + c.get_name())


func _on_picture_item_selected(ID):
	for c in range(get_node("pictures").get_child_count()):
		if (ID == c):
			get_node("pictures").get_child(c).show()
		else:
			get_node("pictures").get_child(c).hide()


func _on_effect_item_selected(ID):
	for c in range(get_node("effects").get_child_count()):
		if (ID == c):
			get_node("effects").get_child(c).show()
		else:
			get_node("effects").get_child(c).hide()
