extends Node

enum {UNIT_NEUTRAL, UNIT_ENEMY, UNIT_ALLY}
#     |          |  |        |  ^^^^^^^^^ enum:unnamed:ally -> enum:unnamed:ally
#     |          |  ^^^^^^^^^^ enum:unnamed:enemy -> enum:unnamed:enemy
#     ^^^^^^^^^^^^ enum:unnamed:neutral -> enum:unnamed:neutral
enum Named {THING_1, THING_2, ANOTHER_THING = -1}
#    |   |  |     |  |     |  ^^^^^^^^^^^^^ enum:named:thing3 -> enum:named:thing3
#    |   |  |     |  ^^^^^^^ enum:named:thing2 -> enum:named:thing2
#    |   |  ^^^^^^^ enum:named:thing1 -> enum:named:thing1
#    ^^^^^ enum:named -> enum:named

func f(arg):
	match arg:
		UNIT_ENEMY: print(UNIT_ENEMY)
		#        |        ^^^^^^^^^^ -> enum:unnamed:enemy
		#<^^^^^^^^ -> enum:unnamed:enemy
		Named.THING_2: print(Named.THING_2)
		#!  | |     |        |   | ^^^^^^^ -> enum:named:thing2
		#   | |     |        ^^^^^ -> enum:named
		#!  | ^^^^^^^ -> enum:named:thing2
		#<^^^ -> enum:named
		_: print(UNIT_ENEMY, Named.ANOTHER_THING)
		#!       |        |  |   | ^^^^^^^^^^^^^ -> enum:named:thing3
		#        |        |  ^^^^^ -> enum:named
		#        ^^^^^^^^^^ -> enum:unnamed:enemy
