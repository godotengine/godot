extends GutOutputTest

# Issue 725.
# The following error was created when Tweens were sent off to assert_is.
"""
E 0:00:04:930   utils.gd:434 @ get_native_class_name(): Tween can't be created directly. Use create_tween() method.
  <C++ Error>   Method/function failed.
  <C++ Source>  scene/animation/tween.cpp:537 @ Tween()
  <Stack Trace> utils.gd:434 @ get_native_class_name()
                strutils.gd:120 @ type2str()
                test.gd:121 @ _str()
                test.gd:1681 @ assert_is()
"""
func test_issue_725() -> void:
	assert_is(create_tween(), Tween, 'expected valid tween')
	should_not_error()