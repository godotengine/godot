# Test that nested type mismatches are caught

func test():
    var nested_ints : Array[Array[int]] = [ [ 1, 2 ], [ 3, 4 ] ]

    # This should fail - trying to assign string array to int array
    nested_ints[0] = [ "not", "allowed" ]

    print("should not reach")
