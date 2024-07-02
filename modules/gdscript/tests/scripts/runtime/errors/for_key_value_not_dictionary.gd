func test():
    var a: Dictionary
    for k, v in a: # Runtime type check is not needed.
        pass
    var b: Variant = [1, 2, 3]
    for u, v in b:  # Runtime type check is needed.
        pass
