class B:
    var idx := 0
class A:
    var a: B? = B.new()
func test():
    var foo := A.new()
    if foo.a:
        for i in foo.a?.idx:
            print(i)
