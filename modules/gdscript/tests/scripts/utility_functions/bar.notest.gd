extends Resource

@export var text: String
@export var qux: Resource


func _init(p_text = "", p_qux = null):
	text = p_text
	qux = p_qux
