import json

var hisName = "John"
let herAge = 31
var j1 = %*
  [
    { "name": hisName, "age": 30 },
    { "name": "Susan", "age": herAge }
  ]
echo j1

var j2 = %* {"name": "Isaac", "books": ["Robot Dreams"]}
j2["details"] = %* {"age":35, "pi":3.1415}
echo j2


type Animal = ref object of RootObj
  name: string
  age: int
method vocalize(self: Animal): string {.base.} = "..."
method ageHumanYrs(self: Animal): int {.base.} = self.age

type Dog = ref object of Animal
method vocalize(self: Dog): string = "woof"
method ageHumanYrs(self: Dog): int = self.age * 7

type Cat = ref object of Animal
method vocalize(self: Cat): string = "meow"


var animals: seq[Animal] = @[]
animals.add(Dog(name: "Sparky", age: 10))
animals.add(Cat(name: "Mitten", age: 10))

for a in animals:
  echo a.vocalize()
  echo a.ageHumanYrs()
