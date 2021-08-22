func test() -> void:
    var numbers := range(10)

    prints('numbers array:', numbers)
    prints('straight copy:', [i for i in numbers])
    prints('only evens:', [i for i in numbers if i % 2 == 0])
    prints('2x2 grid coordinates:', [Vector2(x, y) for x in range(2) for y in range(2)])
    prints('3x3 grid only diagonals:',[Vector2(x, y) for x in range(3) for y in range(3) if x == y])
    prints('only pickout the strings:', [s for s in [1, '1', 1.0, Vector2(), true, "hello world"] if s is String])

    prints('number to string version dictionary:', { i : Vector2(i, i) for i in range(10) })
    prints('using lua syntax:', { id = i for i in range(10) })