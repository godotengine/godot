func test():
    var nullable_node: Node2D? = null
    # Potentially unsafe access.
    if is_instance_valid(nullable_node) and nullable_node.name == &"Node":
        print(nullable_node)