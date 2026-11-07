extends Node

signal my_signal(first_arg: int, second_arg: String)

func _ready() -> void:
    my_signal.connect(new_signal_func)
