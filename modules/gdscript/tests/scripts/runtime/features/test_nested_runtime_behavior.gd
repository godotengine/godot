# Test runtime behavior of nested containers

func test():
    # Test type checking at runtime
    var typed_nested : Array[Array[int]] = [ [ 1, 2 ], [ 3, 4 ] ]
    Utils.check(typed_nested.get_typed_builtin() == TYPE_ARRAY)
    Utils.check(typed_nested.is_typed() == true)

    # Test element access and modification
    var data : Dictionary[String, Array[int]] = { "nums" : [ 1, 2, 3 ] }
    data["nums"].append(4)
    Utils.check(data["nums"].size() == 4)

    # Test type preservation through operations
    var matrix : Array[Array[int]] = [ [ 1, 2 ], [ 3, 4 ] ]
    var row = matrix[0] # Should maintain Array[int] type
    row.append(5)
    Utils.check(row.size() == 3)

    # Test nested type validation
    var safe_nested : Array[Dictionary[String, int]] = []
    safe_nested.append({ "a" : 1, "b" : 2 })
    Utils.check(safe_nested.size() == 1)

    # Test deep nesting
    var deep : Array[Array[Array[String]]] = [[ [ "deep", "nested" ], [ "very", "deep" ] ]]
    Utils.check(deep[0][0][0] == "deep")

    # Test weak vs strong assignment behavior
    var weak = [[ 1, 2 ]]
    var strong : = [[ 1, 2 ]]
    Utils.check(typeof(weak) == TYPE_ARRAY)
    Utils.check(typeof(strong) == TYPE_ARRAY)

    print("ok")
