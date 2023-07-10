class A:
    var bar: String? = null
    var baz: A? = null
    var barz: A = null
    func get_baz() -> A?:
        return null
func test():
    var foo: A = null
    foo.barz.get_baz().bar