.. _hsm:


State Machines
==============

This guide will show how to set up and use a state machine using :ref:`LimboHSM<class_LimboHSM>`.


Initialization
--------------

To use the :ref:`LimboHSM<class_LimboHSM>` state machine, you first need to initialize it in your code.
This is typically done in the ``_ready`` function of your script. Here's an example of how to do this:

.. code:: gdscript

    @onready var hsm: LimboHSM = $LimboHSM
    @onready var idle_state: LimboState = $LimboHSM/IdleState
    @onready var move_state: LimboState = $LimboHSM/MoveState


    func _ready() -> void:
        _init_state_machine()


    func _init_state_machine() -> void:
        hsm.add_transition(idle_state, move_state, idle_state.EVENT_FINISHED)
        hsm.add_transition(move_state, idle_state, move_state.EVENT_FINISHED)

        hsm.initialize(self)
        hsm.set_active(true)

In this example, we first declare the state machine and the states we want to use.
Then, in the ``_init_state_machine`` function, we add transitions between the states.
Finally, we initialize the state machine and set it to active.


State example
-------------

You can define the behavior of a state in a script and attach it to the state node.
Here's an example of a state that plays an animation on an ``AnimationPlayer`` and waits for it to finish:

.. code:: gdscript

    extends LimboState
    ## PlayAnimation: Play an animation on AnimationPlayer, and wait for it to finish.

    @export var animation_player: AnimationPlayer
    @export var animation_name: StringName

    func _enter() -> void:
        animation_player.play(animation_name)

    func _update(_delta: float) -> void:
        if not animation_player.is_playing() \
                or animation_player.assigned_animation != animation_name:
            dispatch(EVENT_FINISHED)

In this example, the ``_enter`` method is called when the state is entered, and it plays the specified animation.
The ``_update`` method is called every frame, and it checks if the animation is finished.
If it is, it dispatches the ``EVENT_FINISHED`` event, which can result in a transition to the next state.


Events and transitions
----------------------

The :ref:`LimboHSM<class_LimboHSM>` comes with an **event system** that helps
to **decouple transitions** from the state implementations.
A transition is associated with a specific event, a starting state, and a destination state,
and it is performed automatically when such an event is dispatched.

To register a transition and associate it with a specific event, you can use the
:ref:`LimboHSM.add_transition<class_LimboHSM_method_add_transition>` method:

.. code:: gdscript

    hsm.add_transition(idle_state, move_state, &"movement_started")

In this example, we're registering a transition from the ``idle_state`` to the ``move_state``
when the ``movement_started`` event is dispatched.

A transition can be also associated with no particular starting state:

.. code:: gdscript

    hsm.add_transition(hsm.ANYSTATE, move_state, &"movement_started")

**Events are dispatched** with the :ref:`LimboState.dispatch<class_LimboState_method_dispatch>` method.
It's important to note that this method can be called from anywhere in the state machine hierarchy and outside of it.
Events are **propagated from the leaf to the root** state. This means that if an event is consumed by a state,
it won't be propagated to its parent states.

States can also define **event handlers**, which are methods that react to specific events.
These event handlers typically don't result in a state transition;
they are simply a mechanism for states to react to particular events.
You can use the :ref:`LimboState.add_event_handler<class_LimboState_method_add_event_handler>` method
to register event handlers in your states:

.. code:: gdscript

    extends LimboState

    func _setup() -> void:
        add_event_handler("movement_started", _on_movement_started)

    func _on_movement_started(cargo = null) -> bool:
        # Handle the "movement_started" event here.
        # `cargo` can be passed with the event when calling `dispatch()`.
        # It's quite handy when you need to pass some data to the event handler.
        return true

If the event handler returns ``true``, the event will be considered as consumed,
and it won't propagate further or result in a state transition.


State anatomy
-------------

.. code:: gdscript

    extends LimboState

    ## Called once, when state is initialized.
    func _setup() -> void:
        pass

    ## Called when state is entered.
    func _enter() -> void:
        pass

    ## Called when state is exited.
    func _exit() -> void:
        pass

    ## Called each frame when this state is active.
    func _update(delta: float) -> void:
        pass


Using behavior trees with state machines
----------------------------------------

The :ref:`BTState<class_BTState>` is a specialized state node in :ref:`LimboHSM<class_LimboHSM>` that can host a behavior tree.
When a :ref:`BTState<class_BTState>` is active, it executes the hosted behavior tree each frame,
effectively using the behavior tree as its implementation.

This allows you to combine the power of behavior trees with the structure and control of state machines.
Behavior trees are excellent for defining complex, hierarchical behaviors,
while state machines are great for managing the flow and transitions between different behaviors.


Single-file state machine setup
-------------------------------

In certain scenarios, such as prototyping or during game jams,
it's practical to keep the state machine code in a single file.
For such cases, :ref:`LimboHSM<class_LimboHSM>` **supports delegation** and provides **chained methods** for easier setup.
Let's illustrate this with a practical code example:

.. code:: gdscript

    extends CharacterBody2D

    var hsm: LimboHSM

    @onready var animation_player: AnimationPlayer = $AnimationPlayer


    func _ready() -> void:
        _init_state_machine()


    func _init_state_machine() -> void:
        hsm = LimboHSM.new()
        add_child(hsm)

        # Use chained methods and delegation to set up states:
        var idle_state := LimboState.new().named("Idle") \
            .call_on_enter(func(): animation_player.play("idle")) \
            .call_on_update(_idle_update)
        var move_state := LimboState.new().named("Move") \
            .call_on_enter(func(): animation_player.play("walk")) \
            .call_on_update(_move_update)

        hsm.add_child(idle_state)
        hsm.add_child(move_state)

        hsm.add_transition(idle_state, move_state, &"movement_started")
        hsm.add_transition(move_state, idle_state, &"movement_ended")

        hsm.initialize(self)
        hsm.set_active(true)


    func _idle_update(delta: float) -> void:
        var dir: Vector2 = Input.get_vector(
            &"ui_left", &"ui_right", &"ui_up", &"ui_down")
        if dir.is_zero_approx():
            hsm.dispatch(&"movement_started")


    func _move_update(delta: float) -> void:
        var dir: Vector2 = Input.get_vector(
            &"ui_left", &"ui_right", &"ui_up", &"ui_down")
        var desired_velocity: Vector2 = dir * 200.0
        velocity = desired_velocity
        move_and_slide()
        if desired_velocity.is_zero_approx():
            hsm.dispatch(&"movement_ended")
