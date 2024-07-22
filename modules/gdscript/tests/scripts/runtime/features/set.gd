

func test():
    var set_a := {4,5,6}
    var set_b := Set()
    print(set_a)

    print(set_a.has(4))
    print(set_a.has(5))
    print(set_a.has(6))
    print(set_a.has_all([4,5,6]))

    set_b.add(4)
    print(set_b)
    set_b.remove(5)
    set_b.remove(4)
    set_b.add(5)
    print(set_b)

    print(set_a.includes(set_b))
    print(set_b.includes(set_a))
    print(set_b.intersected(set_a))
    print(set_b.intersected(set_a)==set_a.intersected(set_b))
    print(set_b.merged(set_a))
    print(set_b.merged(set_a)==set_a.merged(set_b))
    print(set_b.differentiated(set_a))
    print(set_a.differentiated(set_b))
    print(set_b.symmetric_differentiated(set_a))

    set_b = set_a.differentiated(set_b)
    print(set_b)

    print(set_a.includes(set_b))
    print(set_a.includes(set_a))

    print(set_a.values())
    print(set_b.values())

    for i in [4,192,9,72,44]:
        set_b.add(i)
    print(set_b)
    print(set_b.has_all([4,192,9,72,44]))
