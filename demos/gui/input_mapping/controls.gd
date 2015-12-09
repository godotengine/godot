# Note for the reader:
#
# This demo conveniently uses the same names for actions and for the container nodes
# that hold each remapping button. This allow to get back to the button based simply
# on the name of the corresponding action, but it might not be so simple in your project.
#
# A better approach for large-scale input remapping might be to do the connections between
# buttons and wait_for_input through the code, passing as arguments both the name of the
# action and the node, e.g.:
# button.connect("pressed", self, "wait_for_input", [ button, action ])

extends Control

# member variables
var player_actions = [ "move_up", "move_down", "move_left", "move_right", "jump" ]
var action # To register the action the UI is currently handling
var button # Button node corresponding to the above action


func wait_for_input(action_bind):
	action = action_bind
	# See note at the beginning of the script
	button = get_node("bindings").get_node(action).get_node("Button")
	get_node("contextual_help").set_text("Press a key to assign to the '" + action + "' action.")
	set_process_input(true)


func _input(event):
	# Handle the first pressed key
	if (event.type == InputEvent.KEY):
		# Register the event as handled and stop polling
		get_tree().set_input_as_handled()
		set_process_input(false)
		# Reinitialise the contextual help label
		get_node("contextual_help").set_text("Click a key binding to reassign it, or press the Cancel action.")
		if (not event.is_action("ui_cancel")):
			# Display the string corresponding to the pressed key
			button.set_text(OS.get_scancode_string(event.scancode))
			# Start by removing previously key binding(s)
			for old_event in InputMap.get_action_list(action):
				InputMap.action_erase_event(action, old_event)
			# Add the new key binding
			InputMap.action_add_event(action, event)


func _ready():
	# Initialise each button with the default key binding from InputMap
	var input_event
	for action in player_actions:
		# We assume that the key binding that we want is the first one (0), if there are several
		input_event = InputMap.get_action_list(action)[0]
		# See note at the beginning of the script
		get_node("bindings").get_node(action).get_node("Button").set_text(OS.get_scancode_string(input_event.scancode))
