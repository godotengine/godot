const SOME_CONSTANT = 5

enum SOME_VALUES {
	VALUE_1,
	VALUE_2,
	VALUE_3
}
func return_passed(p1='a', p2='b'):
	return str(p1, p2)

func call_me(p1, p2 = 2):
	return str('called with ', p1, ', ', p2)

func call_call_me(p1):
	return call_me(p1)

func default_array(p1=[]):
	pass

func default_int(p1=SOME_CONSTANT):
	pass

func default_string(p1='s'):
	pass

func default_last_two_boolean(p1, p2=true, p3=false):
	pass

func default_vector3(p1=Vector3(1, 1, 1)):
	pass

func default_typed_vector3(p1:Vector3=Vector3(1, 1, 1)):
	pass

func default_color(p1=Color(.5, .5, .5)):
	pass

func default_typed_color(p1:Color=Color(.6, .6, .6)):
	pass

func default_enum(p1=SOME_VALUES.VALUE_1):
	pass

func default_typed_enum(p1:SOME_VALUES = SOME_VALUES.VALUE_1):
	pass



func no_defaults(p1, p2, p3):
	pass


