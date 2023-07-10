class A:
    var foo: String = ""
func test():
    var nullable_instance: A? = null
    (nullable_instance as A).foo # Should not error
    (nullable_instance as A?).foo # Should Error