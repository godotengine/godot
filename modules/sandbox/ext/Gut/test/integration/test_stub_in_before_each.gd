extends GutInternalTester


func before_each():
	stub(DoubleMe, 'get_value').to_return(999)


func test_that_it_is_stubbed():
	var inst = double(DoubleMe).new()
	assert_eq(inst.get_value(), 999)


