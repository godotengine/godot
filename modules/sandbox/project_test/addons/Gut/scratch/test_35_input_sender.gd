extends SceneTree
# In 3.5 Input.use_accumulated_input has been enabled by default.  In 3.4 it was
# disabled even though the documentation indicated it was enabled.  In order for
# Input to fire all events as they are sent through an InputSender you must either
# disable use_accumulated_input or call Input.flush_buffered_events() before
# you check that an input was handled correctly.
#
# For all checks other than is_action_just_pressed and is_action_just_released,
# yeilding before checking will also work.  The *just* methods require that you
# either disable accumulated input or flush the buffer so that checks fire
# on the same frame.

var InputSender = load('res://addons/gut/input_sender.gd')

func test_input_vanilla():
    # -- Action checks --
    var action_event = InputEventAction.new()
    action_event.action = 'jump'
    action_event.pressed = true

    Input.parse_input_event(action_event)
    Input.flush_buffered_events()
    print("jump pressed = ", Input.is_action_pressed('jump'))
    print("jump just pressed = ", Input.is_action_just_pressed('jump'))

    # -- Key checks --
    var key_event = InputEventKey.new()
    key_event.pressed = true
    key_event.scancode = KEY_Y

    Input.parse_input_event(key_event)
    Input.flush_buffered_events()
    print('Y pressed = ', Input.is_key_pressed(KEY_Y))


func test_auto_flush():
    var sender = InputSender.new(Input)
    sender._auto_flush_input = false

    var action_event = InputEventAction.new()
    action_event.action = 'jump'
    action_event.pressed = true

    sender.send_event(action_event)
    print("jump pressed = ", Input.is_action_pressed('jump'))
    print("jump just pressed = ", Input.is_action_just_pressed('jump'))


func _init():
    # Input.use_accumulated_input = false
    InputMap.add_action("jump")
    test_auto_flush()

    quit()