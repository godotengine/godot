extends Node

signal signal_a(a: int)

func test():
    var other_me = self
    other_me.signal_a.emit(➡)
    pass
