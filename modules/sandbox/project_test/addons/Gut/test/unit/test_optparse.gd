extends GutTest


class BaseTest:
	extends GutTest

	var OptParse = load('res://addons/gut/cli/optparse.gd')


class TestOption:
	extends BaseTest

	func test_can_make_one():
		var o = OptParse.Option.new('name', 'default')
		assert_not_null(o)

	func test_init_sets_values():
		var o = OptParse.Option.new('name', 'default', 'description')
		assert_eq(o.option_name, 'name')
		assert_eq(o.default, 'default')
		assert_eq(o.description, 'description')

	func test_has_been_set_false_by_default():
		var o = OptParse.Option.new('name', 'default')
		assert_false(o.has_been_set())

	func test_has_been_set_true_after_setting_value():
		var o = OptParse.Option.new('name', 'default')
		o.value = 'value'
		assert_true(o.has_been_set())

	func test_value_returns_default_when_value_not_set():
		var o = OptParse.Option.new('name', 'default')
		assert_eq(o.value, 'default')

	func test_value_returned_when_value_has_been_set():
		var o = OptParse.Option.new('name', 'default')
		o.value = 'value'
		assert_eq(o.value, 'value')

	func test_to_s_replaces_default_with_default_value():
		var o = OptParse.Option.new('name', 'foobar', 'put default here [default]')
		assert_string_ends_with(o.to_s(), 'put default here foobar')

	func test_to_s_indents_multiple_lines():
		var o = OptParse.Option.new("h", 'default', "line one\nline two\nline three")
		var desc = o.to_s(3)
		var lines = desc.split("\n")
		assert_eq(lines[0], "h   line one")
		assert_eq(lines[1], "    line two")
		assert_eq(lines[2], "    line three")

	func test_to_s_contains_aliases():
		var o = OptParse.Option.new("name", 'default', "description")
		o.aliases.assign(["alias1", "alias2"])
		var desc = o.to_s(4)
		var lines = desc.split("\n")
		assert_eq(lines[0], "name description")
		assert_eq(lines[1], "     aliases: alias1, alias2")

	func test_required_false_by_default():
		var o = OptParse.Option.new('name', 'default')
		assert_false(o.required)

	func test_setting_value_to_an_array_makes_has_been_set_true():
		var o = OptParse.Option.new("name", [])
		o.value = [1, 2, 3]
		assert_true(o.has_been_set())


class TestOptParse:
	extends BaseTest

	func test_can_make_one():
		var opts = OptParse.new()
		assert_not_null(opts)

	func test_assigns_default_value():
		var opts = OptParse.new()
		opts.add('--foo', 'bar', 'this is an argument')
		opts.parse([])

		assert_eq(opts.get_value('--foo'), 'bar')
		if(is_failing()):
			opts.print_options()

	func test_get_help_includes_banner():
		var opts = OptParse.new()
		opts.banner = 'Hello there'
		opts.add('--foo', 'bar', 'this is an argument')
		opts.add('--bar', 'foo', 'what else do you know?')
		var help = opts.get_help()
		assert_string_contains(help, "Hello there\n")

	func test_get_help_includes_all_options():
		var opts = OptParse.new()
		opts.banner = 'Hello there'
		opts.add('--foo', 'bar', 'this is an argument')
		opts.add('--bar', 'foo', 'what else do you know?')
		var help = opts.get_help()
		assert_string_contains(help, "--foo")
		assert_string_contains(help, "--bar")

	func test_get_help_replaces_default_values():
		var opts = OptParse.new()
		opts.banner = 'Hello there'
		opts.add('--foo', 'bar', 'foo = [default]')
		opts.add('--bar', 'foo', 'bar = [default]')
		var help = opts.get_help()
		assert_string_contains(help, "foo = bar")
		assert_string_contains(help, "bar = foo")

	func test_script_option_is_not_in_unused():
		var opts = OptParse.new()
		opts.parse(['-s', 'res://something.gd'])
		assert_eq(opts.unused, [])

	func test_when_script_option_specified_it_is_set():
		var opts = OptParse.new()
		opts.parse(['-s', 'res://something.gd'])
		assert_eq(opts.options.script_option.value, 'res://something.gd')

	func test_when_long_script_option_specified_it_is_set():
		var opts = OptParse.new()
		opts.parse(['--script', 'res://something.gd'])
		assert_eq(opts.options.script_option.value, 'res://something.gd')

	func test_cannot_add_duplicate_options():
		var opts = OptParse.new()
		opts.add('-a', 'a', 'a')
		opts.add('-a', 'a', 'a')
		assert_eq(opts.options.options.size(), 1)

	func test_cannot_add_duplicate_positional_option():
		var opts = OptParse.new()
		opts.add_positional('a', 'a', 'a')
		opts.add_positional('a', 'a', 'a')
		assert_eq(opts.options.positional.size(), 1)

	func test_add_required_sets_required_flag():
		var opts = OptParse.new()
		var result = opts.add_required('-a', 'a', 'a')
		assert_true(result.required)

	func test_add_required_positional_sets_required_flag():
		var opts = OptParse.new()
		var result = opts.add_positional_required('-a', 'a', 'a')
		assert_true(result.required)

	func test_add_required_ignores_duplicates():
		var opts = OptParse.new()
		var first = opts.add('-a', 'a', 'a')
		var result = opts.add_required('-a', 'a', 'a')
		assert_null(result)
		assert_false(first.required)

	func test_add_required_positional_ignores_duplicates():
		var opts = OptParse.new()
		var first = opts.add_positional('-a', 'a', 'a')
		var result = opts.add_positional_required('-a', 'a', 'a')
		assert_null(result)
		assert_false(first.required)

	func test_get_missing_required_options_zero_default():
		var opts = OptParse.new()
		assert_eq(opts.get_missing_required_options().size(), 0)

	func test_non_specified_required_options_included_in_missing():
		var opts = OptParse.new()
		var req1 = opts.add_required('a', 'a', 'a')
		var req2 = opts.add_required('b', 'b', 'b')
		var missing = opts.get_missing_required_options()
		assert_has(missing, req1, 'required 1 in the list')
		assert_has(missing, req2, 'required 2 in the list')

	func test_non_specified_required_positional_options_included_in_missing():
		var opts = OptParse.new()
		var req1 = opts.add_positional_required('a', 'a', 'a')
		var req2 = opts.add_positional_required('b', 'b', 'b')
		var missing = opts.get_missing_required_options()
		assert_has(missing, req1, 'required 1 in the list')
		assert_has(missing, req2, 'required 2 in the list')

	func test_specified_required_options_not_in_missing():
		var opts = OptParse.new()
		var req1 = opts.add_required('-a', 'a', 'a')
		var req2 = opts.add_required('-b', 'b', 'b')
		opts.parse(['-b=something'])
		var missing = opts.get_missing_required_options()
		assert_has(missing, req1, 'required 2 in the list')
		assert_does_not_have(missing, req2, 'required 1 in the list')

	func test_specified_required_positional_options_not_in_missing():
		var opts = OptParse.new()
		var req1 = opts.add_positional_required('a', 'a', 'a')
		var req2 = opts.add_positional_required('b', 'b', 'b')
		opts.parse(['something'])
		var missing = opts.get_missing_required_options()
		assert_does_not_have(missing, req1, 'required 1 in the list')
		assert_has(missing, req2, 'required 2 in the list')

	func test_splits_value_on_equal_sign():
		var opts = OptParse.new()
		var op = opts.add('--foo', 'some string', 'desc')
		opts.parse(['--foo=bar'])
		assert_eq(op.value, 'bar')

	func test_sets_value_when_next_element_when_is_not_an_option():
		var opts = OptParse.new()
		var op = opts.add('--foo', 'some string', 'desc')
		opts.parse(['--foo', 'bar'])
		assert_eq(op.value, 'bar')

	func test_does_not_set_value_when_next_element_when_is_an_option():
		var opts = OptParse.new()
		var op = opts.add('--foo', 'some string', 'desc')
		opts.parse(['--foo', '--bar'])
		assert_eq(op.value, 'some string')

	func test_positional_argument_values_are_parsed_from_a_complicated_set():
		var opts = OptParse.new()
		var pos1 = opts.add_positional('one', 'default', 'one desc')
		var pos2 = opts.add_positional('two', 'default', 'two desc')
		var pos3 = opts.add_positional('three', 'default', 'three desc')
		opts.add('--foo', 'foo', 'foo')
		opts.add('--bar', 'bar', 'bar')

		opts.parse(["--foo=bar", "one_value", "--bar", "asdf", "two_value", "three_value", "--hello", "--world"])

		assert_eq(pos1.value, 'one_value')
		assert_eq(pos2.value, 'two_value')
		assert_eq(pos3.value, 'three_value')

	func test_all_options_are_unused_by_default():
		var opts = OptParse.new()
		opts.parse(['--foo', 'a,b,c,d', '--bar', '--asdf'])
		assert_eq(opts.unused.size(), 4)

	func test_used_options_are_removed_from_unused_options():
		var opts = OptParse.new()
		opts.add('--foo', 'string', 'desc')
		opts.parse(['--foo', 'a,b,c,d', '--bar', '--asdf'])
		assert_eq(opts.unused, ['--bar', '--asdf'])

	func test_flags_are_removed_from_unused_options():
		var opts = OptParse.new()
		opts.add('--foo', false, 'asdf')
		opts.parse(['--foo', 'a,b,c,d', '--bar', '--asdf'])
		assert_eq(opts.unused, ['a,b,c,d', '--bar', '--asdf'])


class TestBooleanValues:
	extends BaseTest

	func test_gets_default_of_false_when_not_specified():
		var op = OptParse.new()
		op.add('--foo', false, 'foo bar')
		op.parse([])
		assert_false(op.get_value('--foo'))

	func test_gets_default_of_true_when_not_specified():
		var op = OptParse.new()
		op.add('--foo', true, 'foo bar')
		op.parse([])
		assert_true(op.get_value('--foo'))

	func test_is_true_when_specified_and_default_false():
		var op = OptParse.new()
		op.add('--foo', false, 'foo bar')
		op.parse(['--foo'])
		assert_true(op.get_value('--foo'))

	func test_is_false_when_specified_and_default_true():
		var op = OptParse.new()
		op.add('--foo', true, 'foo bar')
		op.parse(['--foo'])
		assert_false(op.get_value('--foo'))

	func test_does_not_get_value_of_unnamed_args_after():
		var op = OptParse.new()
		op.add('--foo', false, 'foo bar')
		op.parse(['--foo', 'asdf'])
		assert_true(op.get_value('--foo'))


class TestArrayParameters:
	extends BaseTest

	func test_array_values_parsed_from_commas_when_equal_not_used():
		var op = OptParse.new()
		op.add('--foo', [], 'foo array')
		op.parse(['--foo', 'a,b,c,d'])
		assert_eq(op.get_value('--foo'), ['a', 'b', 'c', 'd'])

	func test_array_values_parsed_from_commas_when_using_equal():
		var op = OptParse.new()
		op.add('--foo', [], 'foo array')
		op.parse(['--foo=a,b,c,d'])
		assert_eq(op.get_value('--foo'), ['a', 'b', 'c', 'd'])

	func test_can_specify_array_values_multiple_times():
		var op = OptParse.new()
		var option = op.add('--foo', [], 'foo array')
		op.parse(['--foo=a,b', '--foo', 'c,d', '--foo', 'e'])
		assert_eq(option.value, ['a', 'b', 'c', 'd', 'e'])

	func test_after_parsing_value_has_been_set_is_true():
		var op = OptParse.new()
		var option = op.add('--foo', [], 'foo array')
		op.parse(['--foo=a,b'])
		assert_eq(option.value, ['a', 'b'])
		assert_true(option.has_been_set())


class TestPositionalArguments:
	extends BaseTest

	func test_can_add_positional_argument():
		var op = OptParse.new()
		op.add_positional('first', '', 'the first one')
		assert_eq(op.options.positional.size(), 1)

	func test_non_named_parameter_1_goes_into_positional():
		var op = OptParse.new()
		op.add_positional('first', '', 'the first one')
		op.parse(['this is a value'])
		assert_eq(op.get_value('first'), 'this is a value')

	func test_two_positional_parameters():
		var op = OptParse.new()
		op.add_positional('first', '', 'the first one')
		op.add_positional('second', 'not_set', 'the second one')
		op.parse(['foo', 'bar'])
		assert_eq(op.get_value('first'), 'foo')
		assert_eq(op.get_value('second'), 'bar')

	func test_second_positional_gets_default_when_not_set():
		var op = OptParse.new()
		op.add_positional('first', '', 'the first one')
		op.add_positional('second', 'not_set', 'the second one')
		op.parse(['foo'])
		assert_eq(op.get_value('first'), 'foo')
		assert_eq(op.get_value('second'), 'not_set')

	func test_when_preceeding_parameter_is_bool_positional_gets_set():
		var op = OptParse.new()
		op.add('--bool', false, 'this is a bool')
		op.add_positional('first', '', 'the first one')
		op.parse(['--bool', 'foo'])
		assert_eq(op.get_value('first'), 'foo')
		assert_true(op.get_value('--bool'))

	func test_can_have_positional_arguments_with_numeric_values():
		var op = OptParse.new()
		op.add_positional('first', 99, 'the first one')
		op.parse([555])
		assert_eq(op.get_value('first'), 555)



class TestValuesDictionary:
	extends BaseTest

	func test_values_dictionary_empty_by_default():
		var op = OptParse.new()
		assert_eq(op.values, {})

	func test_values_contains_options_without_single_dash_after_parse():
		var op = OptParse.new()
		op.add('-foo', false, 'foo')
		op.parse([])
		assert_has(op.values, 'foo')

	func test_values_contains_options_without_two_dashes_after_parse():
		var op = OptParse.new()
		op.add('--foo', false, 'foo')
		op.parse([])
		assert_has(op.values, 'foo')

	func test_values_contains_default_value_when_not_specified():
		var op = OptParse.new()
		op.add('--foo', 'bar', 'foo')
		op.parse([])
		assert_eq(op.values.foo, 'bar')

	func test_values_contains_set_value_when_specified():
		var op = OptParse.new()
		op.add('--foo', 'bar', 'foo')
		op.parse(['--foo', 'i set this'])
		assert_eq(op.values.foo, 'i set this')

	func test_values_contains_positional_arguments():
		var op = OptParse.new()
		op.add_positional("first", 'asdf', 'the first one')
		op.parse(['foo'])
		assert_has(op.values, 'first')

	func test_values_contains_positional_arguments_default_value():
		var op = OptParse.new()
		op.add_positional("first", 'asdf', 'the first one')
		op.parse([])
		assert_eq(op.values.first, 'asdf')

	func test_values_contains_positional_arguments_value():
		var op = OptParse.new()
		op.add_positional("first", 'asdf', 'the first one')
		op.parse(['foo'])
		assert_eq(op.values.first, 'foo')


class TestOptionAliases:
	extends BaseTest

	var op

	func before_each():
		op = OptParse.new()

	func test_options_add_with_alias():
		var opt = OptParse.Option.new("name", "default")
		var opts = OptParse.Options.new()
		opts.add(opt, ["alias1", "alias2"])
		assert_eq(opts.get_by_name("alias1"), opt)
		assert_eq(opts.get_by_name("alias2"), opt)
		assert_has(opt.aliases, "alias1")
		assert_has(opt.aliases, "alias2")

	func test_arguments_by_alias():
		op.add(["--name", "--alias"], "default", "description")
		op.parse(['--alias=value'])
		assert_eq(op.get_value_or_null("--name"), "value")

	func test_alias_doesnt_change_normal_function():
		op.add(["--name", "--alias"], "default", "description")
		op.parse(['--name=value'])
		assert_eq(op.get_value_or_null("--name"), "value")

	func test_argument_accessible_by_alias():
		op.add(["--name", "--alias"], "default", "description")
		op.parse(['--name=value'])
		assert_eq(op.get_value_or_null("--alias"), "value")

	func test_aliases_collide_with_options():
		op.add("--name", "default", "description")
		assert_null(op.add(["--another", "--name"], "default", "description"))

	func test_aliases_collide_with_aliases():
		op.add(["--name", "--alias"], "default", "description")
		assert_null(op.add(["--another", "--alias"], "default", "description"))

	func test_options_collide_with_aliases():
		op.add(["--name", "--alias"], "default", "description")
		assert_null(op.add("--alias", "default", "description"))