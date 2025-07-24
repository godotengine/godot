# Test nested array types with various assignment types

func test():
    # Strong typing with nested arrays
    var nested_int_array : Array[Array[int]] = [ [ 1, 2 ], [ 3, 4 ] ]
    var nested_string_array : Array[Array[String]] = [ [ "a", "b" ], [ "c", "d" ] ]
    Utils.check(nested_int_array[0][0] == 1)
    Utils.check(nested_string_array[0][0] == "a")

    # Inferred typing with nested arrays
    var inferred_nested : = [ [ 1, 2 ], [ 3, 4 ] ]
    var inferred_mixed : = [ [ 1, "two" ], [ 3.0, true ] ]
    Utils.check(inferred_nested[0][0] == 1)
    Utils.check(inferred_mixed[0][1] == "two")

    # Assignment tests
    nested_int_array = [ [ 5, 6 ], [ 7, 8 ] ]
    Utils.check(nested_int_array[0][0] == 5)
    nested_int_array[0] = [ 9, 10 ]
    Utils.check(nested_int_array[0][0] == 9)
    nested_int_array[0][0] = 11
    Utils.check(nested_int_array[0][0] == 11)

    # Triple nesting
    var triple_nested : Array[Array[Array[int]]] = [ [ [ 1, 2 ], [ 3, 4 ] ], [ [ 5, 6 ], [ 7, 8 ] ] ]
    triple_nested[0][0][0] = 99
    Utils.check(triple_nested[0][0][0] == 99)

    # Weak assignment behavior
    var weak_nested = [ [ 1, 2 ], [ 3, 4 ] ]
    Utils.check(typeof(weak_nested) == TYPE_ARRAY)
    weak_nested = [ [ "string", "array" ], [ "should", "work" ] ]
    Utils.check(weak_nested[0][0] == "string")

    # Mixed nested containers
    var array_of_dicts : Array[Dictionary[String, int]] = [ { "a" : 1, "b" : 2 }, { "c" : 3, "d" : 4 } ]
    array_of_dicts[0]["a"] = 10
    Utils.check(array_of_dicts[0]["a"] == 10)

    print("ok")
