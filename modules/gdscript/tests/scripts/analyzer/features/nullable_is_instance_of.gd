func test():
    var arr: Array? = null
    if is_instance_of(arr, TYPE_ARRAY):
        print(arr)
    else:
        print("arr is null")