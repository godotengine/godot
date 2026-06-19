enum B {X, Y, Z}
const C = 1

class InnerClass1:
    const D = 1
    var v: int

    static func f():
        pass

    func g():
        pass


class InnerClass2:
    const E = 1

    static func h():
        pass

    func i():
        pass


func a():
    var instance := InnerClass1.new()
    instance.➡
