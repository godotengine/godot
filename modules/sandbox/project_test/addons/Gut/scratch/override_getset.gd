extends SceneTree


class HasAccessors:
    var normal_accessors = 'default' :
        get:
            print('@normal_accessors_getter')
            return normal_accessors
        set(val):
            print('@normal_accessors_setter')
            print('     = ', normal_accessors)
            normal_accessors = val
            print('     = ', normal_accessors)

    var accessor_methods = 'default' :
        get = _get_accessor_methods,
        set = _set_accessor_methods

    func _get_accessor_methods():
        print('[base] _get_accessor_methods')
        return accessor_methods

    func _set_accessor_methods(val):
        print('[base] _set_accessor_methods')
        print('     = ', accessor_methods)
        accessor_methods = val
        print('     = ', accessor_methods)

    func set(property, value):
        print('!!!!!!!!!!!!! in set')
        super.set(property, value)

    func get(property):
        print('!!!!!!!!!!!!! in get')
        return super.get(property)


class OverridesAccessorMethods:
    extends HasAccessors

    func _get_accessor_methods():
        print('[override] _get_accessor_methods')
        return super._get_accessor_methods()

    func _set_accessor_methods(val):
        print('[override] _set_accessor_methods')
        super._set_accessor_methods(val)

# ------------------------------------------------------------------------------

func set_some_values(thing):
    print('-- Setting normal_accessors')
    thing.normal_accessors = 'hello'
    print()
    print('-- Setting accessor_methods')
    thing.accessor_methods = 'world'
    print()
    print('-- Setting with set/get')
    thing.set('accessor_methods', 'hello world')
    print(thing.get('accessor_methods'))


func _init():
    print("starting")

    var h = HasAccessors.new()
    set_some_values(h)
    print("\n\n")
    var o = OverridesAccessorMethods.new()
    set_some_values(o)

    print("done")
    quit();