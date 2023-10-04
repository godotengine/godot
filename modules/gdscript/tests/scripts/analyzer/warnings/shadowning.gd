class_name ShadowedClass

var member: int = 0

var print_debug := 'print_debug'
@warning_ignore("shadowed_global_identifier")
var print := 'print'

@warning_ignore("unused_variable")
func test():
	var Array := 'Array'
	var Node := 'Node'
	var is_same := 'is_same'
	var sqrt := 'sqrt'
	var member := 'member'
	var reference := 'reference'
	var ShadowedClass := 'ShadowedClass'

	print('warn')
