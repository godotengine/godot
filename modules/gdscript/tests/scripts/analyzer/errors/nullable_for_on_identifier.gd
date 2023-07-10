func test():
    var nullable_array: Array? = []
    for element in nullable_array: # Should error
        print(element)