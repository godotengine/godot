extends SceneTree

signal the_signal(one)

func signal_callback(one):
    print(one)

func _init():
    var callback = signal_callback.bind("two")
    the_signal.connect(callback)
    the_signal.emit('foo')

    quit()