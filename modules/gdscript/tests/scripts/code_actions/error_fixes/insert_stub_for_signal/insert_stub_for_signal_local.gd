extends Node

func _ready() -> void:
    var a: Area2D = Area2D.new()
    a.body_shape_entered.connect(new_signal_func)
