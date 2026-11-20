extends Node

signal my_signal(first_arg: int, second_arg: String)

func _ready() -> void:
    my_signal.connect(new_signal_func)


func new_signal_func(first_arg: int, second_arg: String) -> void:
    pass
