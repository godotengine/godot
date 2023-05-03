func test():
    var node := Node.new()
    node.free()
    node = node ?? Node.new()
    print(node?.get_parent())
    node.free()