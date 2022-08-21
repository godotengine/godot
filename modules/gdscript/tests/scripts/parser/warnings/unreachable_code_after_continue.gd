func test():
    var x := 10
    match x:
        0:
            continue
            print_debug("Error: print after continue")
        _:
            continue
            print_debug("Error: print after continue")
