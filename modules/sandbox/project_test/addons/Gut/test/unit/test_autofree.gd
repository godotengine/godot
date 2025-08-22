extends GutTest

var AutoFree = load('res://addons/gut/autofree.gd')

var _starting_orphans = 0
func before_all():
	_starting_orphans = gut.get_orphan_counter().orphan_count()

func after_all():
	assert_eq(gut.get_orphan_counter().orphan_count(), _starting_orphans, 'no new orphans during script')

func test_can_make_one():
	assert_not_null(AutoFree.new())

func test_can_add_something():
	var af = AutoFree.new()
	var to_free = Node2D.new()
	af.add_free(to_free)
	assert_eq(af.get_free_count(), 1)
	to_free.free()

func test_calling_free_all_frees_them():
	var af = AutoFree.new()
	var to_free1 = Node2D.new()
	var to_free2 = Node.new()
	af.add_free(to_free1)
	af.add_free(to_free2)
	af.free_all()
	assert_freed(to_free1, 'to free 1')
	assert_freed(to_free2, 'to free 2')
	assert_eq(af.get_free_count(), 0)

func test_does_not_add_basic_types():
	var af = AutoFree.new()
	af.add_free(1)
	assert_eq(af.get_free_count(), 0)

func test_does_not_add_references():
	var af = AutoFree.new()
	var r = RefCounted.new()
	af.add_free(r)
	assert_eq(af.get_free_count(), 0)

func test_add_queue_free():
	var af = AutoFree.new()
	var n1 = Node.new()
	af.add_queue_free(n1)
	assert_eq(af.get_queue_free_count(), 1)
	n1.free()

func test_calling_free_all_queues_free():
	var af = AutoFree.new()
	var to_free1 = Node2D.new()
	var to_free2 = Node.new()
	af.add_queue_free(to_free1)
	af.add_queue_free(to_free2)
	af.free_all()
	assert_not_freed(to_free1, 'free1')
	assert_not_freed(to_free2, 'free2')
	assert_eq(af.get_queue_free_count(), 0)
	await wait_seconds(1)
	assert_freed(to_free1, 'free1')
	assert_freed(to_free2, 'free2')

func test_can_free_things_in_tree():
	var af = AutoFree.new()
	var n = Node.new()
	add_child(n)
	af.add_free(n)
	af.free_all()
	assert_freed(n, 'node')

func test_watch_for_orphans():
	var n = autofree(Node.new())
	pass_test('trying to make orphans')

func test_watch_for_orphans2():
	var n = autoqfree(Node.new())
	pass_test('trying to make orphans')
