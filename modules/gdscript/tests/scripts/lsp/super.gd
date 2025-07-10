extends Node

class Inner1:
#     ^^^^^^ class1 -> class1
	func _init():
	#    ^^^^^ class1:init
		pass

class Inner2 extends Inner1:
#     |    |         ^^^^^^ -> class1
#     ^^^^^^ class2 -> class2
	func _init():
	#    ^^^^^ class2:init
        super ()
    #   ^^^^^ -> class1:init
		pass
