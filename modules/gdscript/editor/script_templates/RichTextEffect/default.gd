# meta-description: Base template for rich text effects

@tool
# Having a class name is handy for picking the effect in the Inspector.
class_name RichText_CLASS_
extends _BASE_


# To use this effect:
# - Enable BBCode on a RichTextLabel.
# - Register this effect on the label.
# - Use [_CLASS_SNAKE_CASE_ param=2.0]hello[/_CLASS_SNAKE_CASE_] in text.
var bbcode := "_CLASS_SNAKE_CASE_"


func _process_custom_fx(char_fx: CharFXTransform) -> bool:
	var param: float = char_fx.env.get("param", 1.0)
	return true
