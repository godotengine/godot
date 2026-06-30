func test():
    print("=== One binary operator ===")
    one()
    print("=== Three binary operators ===")
    three()


func one():
    var fail = false

    for i in 2**2:
        var a : bool = i & 0x1 > 0
        var b : bool = i & 0x2 > 0

        var if_branch = false
        var else_branch = false
        if a and b:
            if_branch = true
        else:
            else_branch = true

        fail = fail or if_branch == else_branch or if_branch != (a and b)

    prints("a and b:", "FAIL" if fail else "SUCCESS")


    fail = false
    for i in 2**2:
        var a : bool = i & 0x1 > 0
        var b : bool = i & 0x2 > 0

        var if_branch = false
        var else_branch = false
        if a or b:
            if_branch = true
        else:
            else_branch = true

        fail = fail or if_branch == else_branch or if_branch != (a or b)

    prints("a or b:", "FAIL" if fail else "SUCCESS")

func three():
    var fail = false

    for i in 2**4:
        var a : bool = i & 0x1 > 0
        var b : bool = i & 0x2 > 0
        var c : bool = i & 0x4 > 0
        var d : bool = i & 0x8 > 0

        var if_branch = false
        var else_branch = false
        if (a and b) and (c and d):
            if_branch = true
        else:
            else_branch = true

        fail = fail or if_branch == else_branch or if_branch != ((a and b) and (c and d))

    prints("(a and b) and (c and d):", "FAIL" if fail else "SUCCESS")

    fail = false

    for i in 2**4:
        var a : bool = i & 0x1 > 0
        var b : bool = i & 0x2 > 0
        var c : bool = i & 0x4 > 0
        var d : bool = i & 0x8 > 0

        var if_branch = false
        var else_branch = false
        if (a and b) and (c or d):
            if_branch = true
        else:
            else_branch = true

        fail = fail or if_branch == else_branch or if_branch != ((a and b) and (c or d))

    prints("(a and b) and (c or d):", "FAIL" if fail else "SUCCESS")

    fail = false

    for i in 2**4:
        var a : bool = i & 0x1 > 0
        var b : bool = i & 0x2 > 0
        var c : bool = i & 0x4 > 0
        var d : bool = i & 0x8 > 0

        var if_branch = false
        var else_branch = false
        if (a and b) or (c and d):
            if_branch = true
        else:
            else_branch = true

        fail = fail or if_branch == else_branch or if_branch != ((a and b) or (c and d))

    prints("(a and b) or (c and d):", "FAIL" if fail else "SUCCESS")

    fail = false

    for i in 2**4:
        var a : bool = i & 0x1 > 0
        var b : bool = i & 0x2 > 0
        var c : bool = i & 0x4 > 0
        var d : bool = i & 0x8 > 0

        var if_branch = false
        var else_branch = false
        if (a and b) or (c or d):
            if_branch = true
        else:
            else_branch = true

        fail = fail or if_branch == else_branch or if_branch != ((a and b) or (c or d))

    prints("(a and b) or (c or d):", "FAIL" if fail else "SUCCESS")

    fail = false

    for i in 2**4:
        var a : bool = i & 0x1 > 0
        var b : bool = i & 0x2 > 0
        var c : bool = i & 0x4 > 0
        var d : bool = i & 0x8 > 0

        var if_branch = false
        var else_branch = false
        if (a or b) and (c and d):
            if_branch = true
        else:
            else_branch = true

        fail = fail or if_branch == else_branch or if_branch != ((a or b) and (c and d))

    prints("(a or b) and (c and d):", "FAIL" if fail else "SUCCESS")

    fail = false

    for i in 2**4:
        var a : bool = i & 0x1 > 0
        var b : bool = i & 0x2 > 0
        var c : bool = i & 0x4 > 0
        var d : bool = i & 0x8 > 0

        var if_branch = false
        var else_branch = false
        if (a or b) and (c or d):
            if_branch = true
        else:
            else_branch = true

        fail = fail or if_branch == else_branch or if_branch != ((a or b) and (c or d))

    prints("(a or b) and (c or d):", "FAIL" if fail else "SUCCESS")

    fail = false

    for i in 2**4:
        var a : bool = i & 0x1 > 0
        var b : bool = i & 0x2 > 0
        var c : bool = i & 0x4 > 0
        var d : bool = i & 0x8 > 0

        var if_branch = false
        var else_branch = false
        if (a or b) or (c and d):
            if_branch = true
        else:
            else_branch = true

        fail = fail or if_branch == else_branch or if_branch != ((a or b) or (c and d))

    prints("(a or b) or (c and d):", "FAIL" if fail else "SUCCESS")

    fail = false

    for i in 2**4:
        var a : bool = i & 0x1 > 0
        var b : bool = i & 0x2 > 0
        var c : bool = i & 0x4 > 0
        var d : bool = i & 0x8 > 0

        var if_branch = false
        var else_branch = false
        if (a or b) or (c or d):
            if_branch = true
        else:
            else_branch = true

        fail = fail or if_branch == else_branch or if_branch != ((a or b) or (c or d))

    prints("(a or b) or (c or d):", "FAIL" if fail else "SUCCESS")
