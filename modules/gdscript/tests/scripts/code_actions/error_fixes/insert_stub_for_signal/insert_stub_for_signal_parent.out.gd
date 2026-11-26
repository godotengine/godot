extends Area2D

func _ready() -> void:
    body_shape_entered.connect(new_signal_func)


func new_signal_func(body_rid: RID, body: Node2D, body_shape_index: int, local_shape_index: int) -> void:
    pass
