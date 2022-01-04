// meta-description: Classic movement for gravity games (platformer, ...)

using _BINDINGS_NAMESPACE_;
using System;

public partial class _CLASS_ : _BASE_
{
    public const float Speed = 300.0f;
    public const float JumpForce = -400.0f;

    // Get the gravity from the project settings to be synced with RigidDynamicBody nodes.
    public float gravity = (float)ProjectSettings.GetSetting("physics/2d/default_gravity");

    public override void _PhysicsProcess(float delta)
    {
        Vector2 motionVelocity = MotionVelocity;

        // Add the gravity.
        if (!IsOnFloor())
            motionVelocity.y += gravity * delta;

        // Handle Jump.
        if (Input.IsActionJustPressed("ui_accept") && IsOnFloor())
            motionVelocity.y = JumpForce;

        // Get the input direction and handle the movement/deceleration.
        // As good practice, you should replace UI actions with custom gameplay actions.
        Vector2 direction = Input.GetVector("ui_left", "ui_right", "ui_up", "ui_down");
        if (direction != Vector2.Zero)
        {
            motionVelocity.x = direction.x * Speed;
        }
        else
        {
            motionVelocity.x = Mathf.MoveToward(MotionVelocity.x, 0, Speed);
        }

        MotionVelocity = motionVelocity;
        MoveAndSlide();
    }
}
