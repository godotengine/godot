@private

class Foo:
    readonly var a = 1

class Bar extends Foo:
    func bar():
        a = 10

func test():
    pass
