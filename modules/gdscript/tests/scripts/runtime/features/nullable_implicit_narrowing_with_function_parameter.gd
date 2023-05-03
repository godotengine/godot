func some_func(position: Vector2) -> void:
    print(position)
func test():
    var nullable_node: Node2D? = null
    if nullable_node != null:
        # Invalid argument for "some_func()" function: argument 1 must not be null, but "Vector2?" is nullable.
        some_func(nullable_node.position)