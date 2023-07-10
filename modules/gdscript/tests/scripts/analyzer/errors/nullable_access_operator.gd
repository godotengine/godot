class A:
    var b: A?
func test():
    var foo := A.new()
    foo?.b.b
