extends SceneTree

func _init():
    call_deferred("_exercise")

func _exercise():
    var scalar_value = 42
    var huge_array = []
    var huge_dictionary = {}
    var long_text = ""
    var object_value = Node.new()
    for index in range(10000):
        huge_array.append(index)
    for index in range(300):
        huge_dictionary["key_%s" % index] = index
    for _index in range(4096):
        long_text += "x"
    breakpoint
    scalar_value += huge_array[0] + huge_dictionary["key_0"] + long_text.length()
    breakpoint
    object_value.free()
    if scalar_value < 0:
        print(scalar_value)
    quit()
