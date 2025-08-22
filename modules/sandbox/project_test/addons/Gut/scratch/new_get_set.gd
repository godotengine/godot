extends SceneTree

class NewAccessors:
    var foo = 1 :
        get:
            print('get foo')
            return foo
        set(val):
            print('set foo ', foo, ' -> ', val)
            foo = val

    var _bar = 'a'
    var bar = _bar :
        get:
            print('get bar')
            return _bar
        set(val):
            print('set bar ', bar, ' -> ', val)
            _bar = val

    var one_line = 'cool' :
        get: return one_line
        set(val): one_line = val

    var _other_methods = 'hello'
    var other_methods = _other_methods :
        get: return _get_other_methods()
        set(val): _set_other_methods(val)

    var read_only = 'read me' :
        get: return read_only
        set(val): print('READ ONLY, CANNOOT SET')



    func _get_other_methods():
        print('get_other_methods')
        return _other_methods

    func _set_other_methods(val):
        print('set_other_methods ', _other_methods, ' -> ', val)
        _other_methods = val


    func set_foo_internally(val):
        foo = val

    func set_bar_internally(val):
        _bar = val


func _init():
    var na = NewAccessors.new()
    na.foo = 10
    na.set_foo_internally(20)

    na.bar = 'b'
    na.set_bar_internally('c')

    na.one_line = 'man'
    print(na.one_line)

    na.other_methods = 'world'
    print(na.other_methods)

    na.read_only = 'wroten'
    print(na.read_only)
    quit()

