extends GutInternalTester

func test_can_make_one():
	assert_not_null(GutUtils.OrphanCounter.new())

func test_can_add_get_counter():
	var oc = partial_double(GutUtils.OrphanCounter).new()
	stub(oc, 'orphan_count').to_return(6)
	oc.add_counter('one')
	stub(oc, 'orphan_count').to_return(10)
	assert_eq(oc.get_orphans_since('one'), 4)

func test_print_singular_orphan():
	var oc = partial_double(GutUtils.OrphanCounter).new()
	var d_logger = double(GutUtils.GutLogger).new()

	stub(oc, 'orphan_count').to_return(1)
	oc.add_counter('one')
	stub(oc, 'orphan_count').to_return(2)
	oc.print_orphans('one', d_logger)
	assert_called(d_logger, 'orphan')
	if(is_passing()):
		var msg = get_call_parameters(d_logger, 'orphan')[0]
		assert_string_contains(msg, 'orphan')

func test_print_plural_orphans():
	var oc = partial_double(GutUtils.OrphanCounter).new()
	var d_logger = double(GutUtils.GutLogger).new()

	stub(oc, 'orphan_count').to_return(1)
	oc.add_counter('one')
	stub(oc, 'orphan_count').to_return(5)
	oc.print_orphans('one', d_logger)
	assert_called(d_logger, 'orphan')
	if(is_passing()):
		var msg = get_call_parameters(d_logger, 'orphan')[0]
		assert_string_contains(msg, 'orphans')

func test_adding_same_name_overwrites_prev_start_val():
	var oc = partial_double(GutUtils.OrphanCounter).new()
	stub(oc, 'orphan_count').to_return(1)
	oc.add_counter('one')
	stub(oc, 'orphan_count').to_return(2)
	oc.add_counter('one')
	stub(oc, 'orphan_count').to_return(10)
	assert_eq(oc.get_orphans_since('one'), 8)

func test_getting_count_for_names_that_dne_returns_neg_1():
	var oc = GutUtils.OrphanCounter.new()
	assert_eq(oc.get_orphans_since('dne'), -1)
