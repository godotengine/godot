struct Point:
    var x: float = 0.0
    var y: float = 0.0

struct Enemy:
    var id: int
    var pos: Point
    var health: int = 100

func test():
    var p = Point(10.5, 20.3)
    var e = Enemy(1, p, 150)
    return e
