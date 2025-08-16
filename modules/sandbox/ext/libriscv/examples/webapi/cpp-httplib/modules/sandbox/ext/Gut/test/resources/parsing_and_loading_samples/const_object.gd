extends 'res://addons/gut/test.gd'
# per issue 290, if a const is defined and it starts with "Test" then GUT will
# treat it like an inner test class.  This should not happen.
const TestConstThing = preload('res://test/resources/parsing_and_loading_samples/ConstObject.tscn')
