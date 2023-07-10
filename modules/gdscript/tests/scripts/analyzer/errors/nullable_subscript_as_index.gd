class A:
    var index: int?
    func get_nullable_instance() -> A?:
        return A.new()
func test():
    var foo: Array[Array] = [[]]
    var a_instance := A.new()
    foo[0][a_instance.get_nullable_instance()?.index]
