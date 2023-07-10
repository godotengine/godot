class A:
    var b: A?
    func banana():
        pass
func test():
    var foo := A.new()
    foo?.b.banana()
