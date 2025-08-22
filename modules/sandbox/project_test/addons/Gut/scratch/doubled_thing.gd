# ##############################################################################
# Start Script
# ##############################################################################
extends 'res://test/resources/doubler_test_objects/double_me.gd'




# ------------------------------------------------------------------------------
# THE NEW STYLE
# ------------------------------------------------------------------------------
var __gutdbl = load('res://addons/gut/double_tools.gd').new()

func __gutdbl_init_vals():
	__gutdbl.path = 'res://test/resources/doubler_test_objects/double_me.gd'
	__gutdbl.subpath = ''
	__gutdbl.stubber = __gutdbl.from_id(-9223372003417782807)
	__gutdbl.spy = __gutdbl.from_id(-1)
	__gutdbl.gut = __gutdbl.from_id(29980886386)
	__gutdbl.from_singleton = ''
	__gutdbl.is_partial = false

# ------------------------------------------------------------------------------
# OLD GUT Double properties and methods
# ------------------------------------------------------------------------------

# var __gut_metadata_ = {
#       path = 'res://test/resources/doubler_test_objects/double_me.gd',
#       subpath = '',
#       stubber = __gut_instance_from_id(-9223372003417782807),
#       spy = __gut_instance_from_id(-1),
#       gut = __gut_instance_from_id(29980886386),
#       from_singleton = '',
#       is_partial = false
# }

# func __gut_instance_from_id(inst_id):
#       if(inst_id ==  -1):
#               return null
#       else:
#               return instance_from_id(inst_id)

# func __gut_should_call_super(method_name, called_with):
#       if(__gut_metadata_.stubber != null):
#               return __gut_metadata_.stubber.should_call_super(self, method_name, called_with)
#       else:
#               return false


# func __gut_spy(method_name, called_with):
#       if(__gut_metadata_.spy != null):
#               __gut_metadata_.spy.add_call(self, method_name, called_with)

# func __gut_get_stubbed_return(method_name, called_with):
#       if(__gut_metadata_.stubber != null):
#               return __gut_metadata_.stubber.get_return(self, method_name, called_with)
#       else:
#               return null

# func __gut_default_val(method_name, p_index):
#       if(__gut_metadata_.stubber != null):
#               return __gut_metadata_.stubber.get_default_value(self, method_name, p_index)
#       else:
#               return null

# func __gut_init():
#       if(__gut_metadata_.gut != null):
#               __gut_metadata_.gut.get_autofree().add_free(self)

# ------------------------------------------------------------------------------
# Methods start here
# ------------------------------------------------------------------------------
# {"name":"_ready", "args":[], "default_args":[], "flags":1, "id":0, "return":{"name":"", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":6}}
func _ready():
        __gutdbl.spy_on('_ready', [])
        if(__gutdbl.should_call_super('_ready', [])):
                return await super()
        else:
                return __gutdbl.get_stubbed_return('_ready', [])

func _init():
        __gutdbl_init_vals()
        __gutdbl.init()
        __gutdbl.spy_on('_init', [])

# {"name":"get_value", "args":[], "default_args":[], "flags":1, "id":0, "return":{"name":"", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":262150}}
func get_value():
        __gutdbl.spy_on('get_value', [])
        if(__gutdbl.should_call_super('get_value', [])):
                return await super()
        else:
                return __gutdbl.get_stubbed_return('get_value', [])

# {"name":"set_value", "args":[{"name":"val", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":262150, "default":"__no__default__"}], "default_args":[], "flags":1, "id":0, "return":{"name":"", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":6}}
func set_value(p_val=__gutdbl.default_val("set_value",0)):
        __gutdbl.spy_on('set_value', [p_val])
        if(__gutdbl.should_call_super('set_value', [p_val])):
                return await super(p_val)
        else:
                return __gutdbl.get_stubbed_return('set_value', [p_val])

# {"name":"has_one_param", "args":[{"name":"one", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":262150, "default":"__no__default__"}], "default_args":[], "flags":1, "id":0, "return":{"name":"", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":6}}
func has_one_param(p_one=__gutdbl.default_val("has_one_param",0)):
        __gutdbl.spy_on('has_one_param', [p_one])
        if(__gutdbl.should_call_super('has_one_param', [p_one])):
                return await super(p_one)
        else:
                return __gutdbl.get_stubbed_return('has_one_param', [p_one])

# {"name":"has_two_params_one_default", "args":[{"name":"one", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":262150, "default":"__no__default__"}, {"name":"two", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":262150, "default":<null>}], "default_args":[<null>], "flags":1, "id":0, "return":{"name":"", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":6}}
func has_two_params_one_default(p_one=__gutdbl.default_val("has_two_params_one_default",0), p_two=__gutdbl.default_val("has_two_params_one_default",1)):
        __gutdbl.spy_on('has_two_params_one_default', [p_one, p_two])
        if(__gutdbl.should_call_super('has_two_params_one_default', [p_one, p_two])):
                return await super(p_one, p_two)
        else:
                return __gutdbl.get_stubbed_return('has_two_params_one_default', [p_one, p_two])

# {"name":"get_position", "args":[], "default_args":[], "flags":1, "id":0, "return":{"name":"", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":262150}}
func get_position():
        __gutdbl.spy_on('get_position', [])
        if(__gutdbl.should_call_super('get_position', [])):
                return await super()
        else:
                return __gutdbl.get_stubbed_return('get_position', [])

# {"name":"has_string_and_array_defaults", "args":[{"name":"string_param", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":262150, "default":"__no__default__"}, {"name":"array_param", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":262150, "default":"asdf"}], "default_args":["asdf"], "flags":1, "id":0, "return":{"name":"", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":6}}
func has_string_and_array_defaults(p_string_param=__gutdbl.default_val("has_string_and_array_defaults",0), p_array_param=__gutdbl.default_val("has_string_and_array_defaults",1)):
        __gutdbl.spy_on('has_string_and_array_defaults', [p_string_param, p_array_param])
        if(__gutdbl.should_call_super('has_string_and_array_defaults', [p_string_param, p_array_param])):
                return await super(p_string_param, p_array_param)
        else:
                return __gutdbl.get_stubbed_return('has_string_and_array_defaults', [p_string_param, p_array_param])

# {"name":"this_just_does_an_await", "args":[], "default_args":[], "flags":1, "id":0, "return":{"name":"", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":6}}
func this_just_does_an_await():
        __gutdbl.spy_on('this_just_does_an_await', [])
        if(__gutdbl.should_call_super('this_just_does_an_await', [])):
                return await super()
        else:
                return __gutdbl.get_stubbed_return('this_just_does_an_await', [])

# {"name":"this_is_a_coroutine", "args":[], "default_args":[], "flags":1, "id":0, "return":{"name":"", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":262150}}
func this_is_a_coroutine():
        __gutdbl.spy_on('this_is_a_coroutine', [])
        if(__gutdbl.should_call_super('this_is_a_coroutine', [])):
                return await super()
        else:
                return __gutdbl.get_stubbed_return('this_is_a_coroutine', [])

# {"name":"calls_coroutine", "args":[], "default_args":[], "flags":1, "id":0, "return":{"name":"", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":262150}}
func calls_coroutine():
        __gutdbl.spy_on('calls_coroutine', [])
        if(__gutdbl.should_call_super('calls_coroutine', [])):
                return await super()
        else:
                return __gutdbl.get_stubbed_return('calls_coroutine', [])

# {"name":"does_something_then_calls_coroutine_then_does_something_else", "args":[], "default_args":[], "flags":1, "id":0, "return":{"name":"", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":262150}}
func does_something_then_calls_coroutine_then_does_something_else():
        __gutdbl.spy_on('does_something_then_calls_coroutine_then_does_something_else', [])
        if(__gutdbl.should_call_super('does_something_then_calls_coroutine_then_does_something_else', [])):
                return await super()
        else:
                return __gutdbl.get_stubbed_return('does_something_then_calls_coroutine_then_does_something_else', [])

# {"name":"might_await", "args":[{"name":"should", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":262150, "default":"__no__default__"}, {"name":"some_default", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":262150, "default":3}], "default_args":[3], "flags":1, "id":0, "return":{"name":"", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":262150}}
func might_await(p_should=__gutdbl.default_val("might_await",0), p_some_default=__gutdbl.default_val("might_await",1)):
        __gutdbl.spy_on('might_await', [p_should, p_some_default])
        if(__gutdbl.should_call_super('might_await', [p_should, p_some_default])):
                return await super(p_should, p_some_default)
        else:
                return __gutdbl.get_stubbed_return('might_await', [p_should, p_some_default])

# {"name":"might_await_no_return", "args":[{"name":"some_default", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":262150, "default":3}], "default_args":[3], "flags":1, "id":0, "return":{"name":"", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":6}}
func might_await_no_return(p_some_default=__gutdbl.default_val("might_await_no_return",0)):
        __gutdbl.spy_on('might_await_no_return', [p_some_default])
        if(__gutdbl.should_call_super('might_await_no_return', [p_some_default])):
                return await super(p_some_default)
        else:
                return __gutdbl.get_stubbed_return('might_await_no_return', [p_some_default])

# {"name":"uses_await_response", "args":[], "default_args":[], "flags":1, "id":0, "return":{"name":"", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":6}}
func uses_await_response():
        __gutdbl.spy_on('uses_await_response', [])
        if(__gutdbl.should_call_super('uses_await_response', [])):
                return await super()
        else:
                return __gutdbl.get_stubbed_return('uses_await_response', [])

# {"name":"default_is_value", "args":[{"name":"val", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":262150, "default":"__no__default__"}], "default_args":[], "flags":1, "id":0, "return":{"name":"", "class_name":&"", "type":0, "hint":0, "hint_string":"", "usage":262150}}
func default_is_value(p_val=__gutdbl.default_val("default_is_value",0)):
        __gutdbl.spy_on('default_is_value', [p_val])
        if(__gutdbl.should_call_super('default_is_value', [p_val])):
                return await super(p_val)
        else:
                return __gutdbl.get_stubbed_return('default_is_value', [p_val])

