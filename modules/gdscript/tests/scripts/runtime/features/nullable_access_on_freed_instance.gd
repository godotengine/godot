func test():
    var node := Node.new()
    node.free()
    print(node?.get_parent())