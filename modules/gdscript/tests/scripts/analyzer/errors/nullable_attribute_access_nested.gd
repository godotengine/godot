class B:
    var bar

class A:
    var foo: B?

func test():
    var foobar: A = A.new()
    foobar.foo.bar