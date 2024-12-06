@private

class Foo:
    public var a = 1
    readonly var b = 2
    protected var c = 3
    private var d = 4
    var e = 5

    public var public = 11
    readonly var readonly = 22
    protected var protected = 33
    private var private = 44

    public func foo():
        print(a)
        print(b)
        print(c)
        print(d)
        print(e)
        foo2()
        foo3()
        self.foo2()
        self.foo3()
        a = 10
        b = 20
        c = 30
        d = 40
        e = 50
        Foo3()
        self.Foo3()
        Foo.Foo3()
        public = 111
        readonly = 222
        protected = 333
        private = 444
        print(public)
        print(readonly)
        print(protected)
        print(private)

    protected func foo2():
        print("foo2")

    private func foo3():
        print("foo3")

    protected func bar():
        print("bar")

    public static func Foo():
        print("Foo")

    protected static func Foo2():
        print("Foo2")

    private static func Foo3():
        print("Foo3")

class Bar extends Foo:
    readonly var f = Foo.new()

    public func bar():
        a = 100
        print(a)
        print(b)
        print(c)
        self.a = 1000
        print(self.a)
        print(self.b)
        print(self.c)
        foo2()
        self.foo2()
        super.foo2()
        super()
        Foo()
        Foo2()
        Foo.Foo()
        Foo.Foo2()
        super.Foo()
        super.Foo2()

class Test extends Bar:
    public func test():
        c = 333
        print(c)
        self.c = 3333
        print(self.c)
        self.c = 33333
        foo2()
        self.foo2()
        super.foo2()
        Foo()
        Foo2()
        Foo.Foo()
        Foo.Foo2()
        Bar.Foo()
        Bar.Foo2()
        super.Foo()
        super.Foo2()

func test():
    var t = Bar.new()
    print(t.a)
    print(t.b)
    t.a = 100
    t.foo()
    t.bar()
    t.Foo()
    Foo.Foo()

    var t2 = Test.new()
    t2.test()

    t.f.a = 11111
