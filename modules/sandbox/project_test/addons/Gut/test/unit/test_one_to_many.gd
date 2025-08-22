extends "res://addons/gut/test.gd"

var OneToMany = GutUtils.OneToMany

func test_can_make_one():
	assert_not_null(OneToMany.new())

func test_size():
	var otm  = OneToMany.new()
	assert_eq(otm.size(), 0)

func test_can_add_one():
	var otm = OneToMany.new()
	otm.add('one', 'many1')
	assert_eq(otm.size(), 1)

func test_size_with_name():
	var otm = OneToMany.new()
	otm.add('one', 'many1')
	otm.add('one', 'many2')
	otm.add('one', 'many3')
	otm.add('two', 'two-1')
	assert_eq(otm.size('one'), 3, 'checking one')
	assert_eq(otm.size('two'), 1, 'checking two')

func test_size_when_name_does_not_exist_returns_0():
	var otm = OneToMany.new()
	assert_eq(otm.size('missing'), 0)

func test_ignores_duplicate_many():
	var otm = OneToMany.new()
	otm.add('one', 'many1')
	otm.add('one', 'many1')
	otm.add('one', 'many2')
	assert_eq(otm.size('one'), 2)

func test_clear():
	var otm = OneToMany.new()
	otm.add('one', 'many1')
	otm.add('two', 'many2')
	otm.add('three', 'many3')
	assert_eq(otm.size(), 3, 'check before clear')
	otm.clear()
	assert_eq(otm.size(), 0)

func test_has():
	var otm = OneToMany.new()
	otm.add('one', 'one-1')
	otm.add('two', 'two-1')
	otm.add('two', 'two-2')
	assert_true(otm.has('two', 'two-2'), 'has two/two-2')
	assert_false(otm.has('not', 'in-there'), 'does not exist')

func test_adding_an_existing_value_does_not_clear_out_all_values():
	var otm = OneToMany.new()
	otm.add('two', 'two-1')
	otm.add('two', 'two-2')
	otm.add('two', 'two-1')
	assert_true(otm.has('two', 'two-2'), 'has two/two-2')
	print(otm.to_s())
