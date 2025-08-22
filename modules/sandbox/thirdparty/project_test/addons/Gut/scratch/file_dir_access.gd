extends SceneTree

func fa_print_open_for_read_result(path):
    var file = FileAccess.open(path, FileAccess.READ)
    var result = FileAccess.get_open_error()
    print('Open(read) ', path)
    print('var file = ', file)
    if(result == OK):
        print(' is OK')
    else:
        print('  Error code ', result)

    return result

func fa_print_open_for_write_result(path):
    var file = FileAccess.open(path, FileAccess.WRITE)
    var result = FileAccess.get_open_error()
    print('Open(write) ', path)
    print('var file = ', file)
    if(result == OK):
        print(' is OK')
    else:
        print('  Error code ', result)

    return result

func da_print_open_result(path):
    var dir = DirAccess.open(path)
    var result = DirAccess.get_open_error()

    print('Open Dir ', path)
    print('var dir = ', dir)
    if(result == OK):
        print(' is OK')
    else:
        print('  Error code ', result)

    return result

func _init():
    fa_print_open_for_read_result('res://addons/gut/gut.gd')
    fa_print_open_for_read_result('res://file_dne.biz')

    fa_print_open_for_write_result('user://foo.txt')
    fa_print_open_for_write_result('bar://foo.txt')

    da_print_open_result('res://addons')
    da_print_open_result('res://dne')

    quit()