# meta-description: Base template for rich text effects

@tool
class_name _CLASS_
extends _BASE_


# To use this effect:
# - Enable BBCode on a RichTextLabel.
# - Register this effect on the label.
# - Use [_CLASS_ param=2.0]hello[/_CLASS_] in text.
var bbcode := "_CLASS_"


func _process_custom_fx(char_fx: CharFXTransform) -> bool:
	var param: float = char_fx.env.get("param", 1.0)
	return true
