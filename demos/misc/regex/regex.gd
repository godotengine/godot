extends VBoxContainer

var regex = RegEx.new()

func update_expression():
	regex.compile(get_node("Expression").get_text())
	update_text()

func update_text():
	var text = get_node("Text").get_text()
	regex.find(text)
	var list = get_node("List")
	for child in list.get_children():
		child.queue_free()
	for res in regex.get_captures():
		var label = Label.new()
		label.set_text(res)
		list.add_child(label)

func _ready():
	get_node("Text").set_text("They asked me \"What's going on \\\"in the manor\\\"?\"")
	update_expression()
