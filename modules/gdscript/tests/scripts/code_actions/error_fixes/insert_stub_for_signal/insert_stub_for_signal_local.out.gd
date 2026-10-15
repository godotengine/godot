extends Node

func _ready() -> void:
    var a: Area2D = Area2D.new()
    a.body_shape_entered.connect(new_signal_func)


func new_signal_func(body_rid: RID, body: Node2D, body_shape_index: int, local_shape_index: int) -> void:
    pass
