trait Shearable:
    pass

trait Milkable:
    pass

class FluffyCow:
    uses Shearable, Milkable

class FluffyBull:
    uses Shearable

class Bird:
    pass

func test():
    var my_animals : Array = []
    my_animals.append(FluffyCow.new())
    my_animals.append(FluffyBull.new())
    my_animals.append(Bird.new())
    for animal in my_animals:
        var cast1 = animal as Shearable
        var cast2 = animal as Milkable
        print(cast1 != null)
        print(cast2 != null)

    print("ok")
