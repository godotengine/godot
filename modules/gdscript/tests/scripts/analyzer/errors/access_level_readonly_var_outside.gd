@private

class Foo:
    readonly var a = 1

func test():
    Foo.new().a = 10
