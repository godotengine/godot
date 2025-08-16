extends SceneTree

var ThingCounter = load('res://addons/gut/thing_counter.gd')

func search_for_enum(name):
    var classes = ClassDB.get_class_list()
    var found = false
    for c in classes:
        if(ClassDB.class_has_enum(c, name)):
            print(c, ' has ', name)
            found = true

    if(!found):
        print('could not find enum ', name)

func get_all_enums():
    var classes = ClassDB.get_class_list()
    var found = false
    var enums = ThingCounter.new()
    for c in classes:
        var class_enums = ClassDB.class_get_enum_list(c)
        for e in class_enums:
            enums.add(e)
            var enum_constants = ClassDB.class_get_enum_constants(c, e)
            print(e)
            print('    ', enum_constants)
            print()


func get_all_int_constants():
    var classes = ClassDB.get_class_list()
    var found = false
    var all_int_consts = ThingCounter.new()
    for c in classes:
        var int_consts = ClassDB.class_get_integer_constant_list(c)
        # print(c, ':')
        # print('    ', int_consts)
        for ic in int_consts:
            all_int_consts.add(ic)

    print(all_int_consts.to_s())


func get_all_properties():
    var classes = ClassDB.get_class_list()
    var all_properties = ThingCounter.new()
    for c in classes:
        var prop_list = ClassDB.class_get_property_list(c)
        for p in prop_list:
            all_properties.add(p.name)

    print(all_properties.to_s())

func print_all_classes():
    var classes = ClassDB.get_class_list()

    for c in classes:
        print(c)



func _init():
    # get_all_enums()
    # get_all_int_constants()
    # get_all_properties()
    print_all_classes()
    quit()