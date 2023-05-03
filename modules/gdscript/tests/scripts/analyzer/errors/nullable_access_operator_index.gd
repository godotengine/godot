class A:
    var b: A?
    var c: String?
    func banana():
        pass
func test():
    var foo := A.new()
    foo.c?[0].begins_with("")
