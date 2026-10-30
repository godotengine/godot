extends Area2D

func _ready() -> void:
    body_shape_entered.connect(new_signal_func)
