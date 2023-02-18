# https://github.com/godotengine/godot/issues/64202

class Example extends Resource:
    pass

func test():
    var example = Example.new()
    var shallow_copy = example.duplicate()
    var deep_copy = example.duplicate(true)

    print_resource(inst_to_dict(example))
    print_resource(inst_to_dict(shallow_copy))
    print_resource(inst_to_dict(deep_copy))

func print_resource(resource: Dictionary):
    prints(resource["@subpath"], String(resource["@path"]).get_file())
