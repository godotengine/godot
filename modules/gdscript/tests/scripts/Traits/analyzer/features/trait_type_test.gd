trait Shearable:
    func shear() -> void:
        print("Shearable ok")

trait Milkable:
    func milk() -> void:
        print("Milkable ok")

class FluffyCow:
    uses Shearable, Milkable

class FluffyBull:
    uses Shearable

class Bird:
    pass

func test():
    Utils.check((FluffyCow is Shearable) == true)
    Utils.check((FluffyCow is Milkable) == true)

    var bull = FluffyBull.new()
    Utils.check((bull is Shearable) == true)
    Utils.check((bull is Milkable) == false)

    var my_animals : Array = []
    my_animals.append(FluffyCow.new())
    my_animals.append(FluffyBull.new())
    my_animals.append(Bird.new())
    var count = 1
    for animal in my_animals:
        print("Animal ", count)
        if animal is Shearable:
            animal.shear()
        if animal is Milkable:
            animal.milk()
        count += 1

    print("ok")
