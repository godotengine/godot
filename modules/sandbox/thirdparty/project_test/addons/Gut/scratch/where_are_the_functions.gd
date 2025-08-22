extends SceneTree

var ExtendsWinDialog = load('res://test/resources/doubler_test_objects/double_extends_window_dialog.gd')

class TryThis:
    extends 'res://test/resources/doubler_test_objects/double_extends_window_dialog.gd'

    func emit_changed():
        print('emit_changed')
        # super.emit_changed()
        print('emit_changed 2')

func _init():
    var foo = TryThis.new()
    foo.emit_changed()
    foo.free()

    print(ExtendsWinDialog.get_base_script())
    print(ExtendsWinDialog.get_instance_base_type())

    var base_source = str("extends ", ExtendsWinDialog.get_instance_base_type())
    var script = GDScript.new()
    script.source_code = base_source
    # this is causing an error in 4.0 (does not halt execution, just prints it)
    # ERROR: Attempt to open script '' resulted in error 'File not found'.
    # Everyting seems to work.  I suspect that the path is empty and it
    # is throwing an erro that should not be thrown.  An issue has been
    # created and marked as bug.
    var result = script.reload()
    var inst = script.new()

    print(inst.get_method_list())
    inst.free()
    quit()
