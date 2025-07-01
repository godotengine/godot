extends Node

const A = preload("res://completion/class_a.notest.gd")

func _ready() -> void:
    var a := A.new()
    var tween := get_tree().create_tween()
    tween.tween_property(a, ➡)
