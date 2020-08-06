var a # No init.
var b = 42 # Init.

func test():
    var c # No init, local.
    var d = 23 # Init, local.

    a = 1
    c = 2

    prints(a, b, c, d)
    print("OK")
