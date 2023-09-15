func test():
    var {foo = a, bar = b = {foo = 1, bar = 2} # Should error
