extends SceneTree
# ##############################################################################
# I used to this to try and find all the built-in methods that I shouldn't
# double.  it has some useful examples in it so I'll keep it around for a bit.
# ##############################################################################

var Doubler = load('res://addons/gut/doubler.gd')

var DOUBLE_ME_PATH = 'res://test/resources/doubler_test_objects/double_me.gd'
var DoubleMe = load(DOUBLE_ME_PATH)

var DOUBLE_EXTENDS_NODE2D = 'res://test/resources/doubler_test_objects/double_extends_node2d.gd'
var DoubleExtendsNode2d = load(DOUBLE_EXTENDS_NODE2D)

var ARGS = 'args'
var FLAGS = 'flags'
var NAME = 'name'

var _local_black_list = ['draw_char']

func it_worked(path):
	print('it worked:  ', path)

func it_didnt(path):
	print("it didn't work:  ", path)

# gets a list of all the unique methods that aren't of flag==65.
func get_full_blacklist(obj):
	var list = []
	var methods = obj.get_method_list()

	for i in range(methods.size()):
		var flag = methods[i][FLAGS]
		var name = methods[i][NAME]

		if(flag != 65):
			if(!list.has(name)):
				list.append(name)

	return list

# make a new doubler with common props.
func new_doubler():
	var doubler = Doubler.new()
	doubler.set_output_dir('user://doubler_temp_files/')
	return doubler

# Use the output from this method to determine which methods should be added
# to the doubler's blacklist.  Each iteration of the loop a method is removed
# from the blacklist and a double is made.  Check for errors in the output to
# find methods that should not be doubled.
#
# Creates a blacklist and puts it on the doubler.  It then loops through all of
# them removing the first element from the array and trying to double the object.
# It then adds the removed one at the end so that we don't keep getting errors
# for each bad method.
#
# This also uses a local blacklist with methods that cause everytning to blow
# up.  These methods should also be in doubler's blacklist.
func remove_methods_from_blacklist_one_by_one(obj, path):
	var doubler = new_doubler()
	doubler._blacklist = get_full_blacklist(obj)

	for _i in range(doubler._blacklist.size()):

		var removed = doubler._blacklist[0]
		doubler._blacklist.remove_at(0)
		if(!_local_black_list.has(removed)):
			print(removed)
			var inst = doubler.double(path)
			if(inst == null):
				print("didn't work")
		else:
			print('skipped ', removed)

		doubler._blacklist.append(removed)


# given a path it will create a double of it and then create an instance of the
# doubled object checking for nulls along the way.  Thi is what I used to test
# the black lists for various objects.
func double_and_instance_it(path):
	var doubler = new_doubler()

	var doubled = doubler.double(path)
	var inst = null
	if(doubled == null):
		it_didnt(path)
	else:
		inst = doubled.new()
		if(inst != null):
			it_worked(path)
		else:
			it_didnt(path)
	return inst

# main
func _init():
	var n = Node2D.new()
	remove_methods_from_blacklist_one_by_one(DoubleMe.new(), DOUBLE_ME_PATH)
	var inst = double_and_instance_it(DOUBLE_ME_PATH)

	remove_methods_from_blacklist_one_by_one(DoubleExtendsNode2d.new(), DOUBLE_EXTENDS_NODE2D)
	inst = double_and_instance_it(DOUBLE_EXTENDS_NODE2D)
	n.print_orphan_nodes()
	quit()
