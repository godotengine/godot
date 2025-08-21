extends GutTest

var CollectedTest = GutUtils.CollectedTest


func test_was_run_defaults_to_false():
    var t = CollectedTest.new()
    assert_false(t.was_run)

func test_add_fail_results_in_is_failing_to_true():
    var t = CollectedTest.new()
    t.add_fail('fail text')
    assert_true(t.is_failing())

func test_add_pending_results_in_is_pending():
    var t = CollectedTest.new()
    t.add_pending('pending text')
    assert_true(t.is_pending())

func test_adding_pending_and_fail_still_results_in_pending_false():
    var t = CollectedTest.new()
    t.add_pending('pending text')
    t.add_fail('fail text')
    assert_false(t.is_pending())

func test_add_pass_results_in_is_passing_true():
    var t= CollectedTest.new()
    t.add_pass('pass text')
    assert_true(t.is_passing())

func test_add_pass_and_fail_results_in_passing_false():
    var t = CollectedTest.new()
    t.add_pass('pass text')
    t.add_fail('fail text')
    assert_false(t.is_passing())

func test_add_pass_and_pending_results_in_passing_false():
    var t = CollectedTest.new()
    t.add_pass('pass text')
    t.add_pending('pending text')
    assert_false(t.is_passing())

func test_get_status_text_is_no_asserts_when_nothing_happened():
    var t = CollectedTest.new()
    t.was_run = true
    assert_eq(t.get_status_text(), 'no asserts')

func test_when_one_pass_added_status_is_pass():
    var t = CollectedTest.new()
    t.was_run = true
    t.add_pass('pass')
    assert_eq(t.get_status_text(), 'pass')

func test_when_one_failed_status_is_fail():
    var t = CollectedTest.new()
    t.was_run = true
    t.add_fail('fail')
    assert_eq(t.get_status_text(), 'fail')

func test_when_one_pending_status_is_pending():
    var t = CollectedTest.new()
    t.was_run = true
    t.add_pending('pending')
    assert_eq(t.get_status_text(), 'pending')

func test_when_should_skip_true_status_is_risky():
    var t = CollectedTest.new()
    t.should_skip = true
    assert_eq(t.get_status_text(), 'skipped')

func test_when_nothing_added_and_test_was_run_then_test_is_risky():
    var t = CollectedTest.new()
    t.was_run = true
    assert_true(t.is_risky())

func test_when_has_pass_test_is_not_risky():
    var t = CollectedTest.new()
    t.was_run = true
    t.add_pass('pass')
    assert_false(t.is_risky())

func test_when_has_pending_test_is_not_risky():
    var t = CollectedTest.new()
    t.was_run = true
    t.add_pending('text')
    assert_false(t.is_risky())

func test_when_has_failure_test_is_not_risky():
    var t = CollectedTest.new()
    t.was_run = true
    t.add_fail('text')
    assert_false(t.is_risky())

func test_when_test_was_not_run_it_is_not_risky():
    var t = CollectedTest.new()
    t.was_run = false
    assert_false(t.is_risky())

# Based on the internal workings of GUT, it does not hit the point where
# the was_run flag is set when skipping tests, so if the flag is set to skip
# then the test is always risky.
func test_when_should_skip_and_not_run_test_is_risky():
    var t = CollectedTest.new()
    t.should_skip = true
    assert_true(t.is_risky())

func test_assert_count_zero_by_default():
    var t = CollectedTest.new()
    assert_eq(t.assert_count, 0)


func test_assert_count_reflects_pass_and_failures():
    var t = CollectedTest.new()
    t.add_pass('pass')
    t.add_pass('pass')
    t.add_fail('fail')
    assert_eq(t.assert_count, 3)


