// meta-description: Classic movement for gravity games (FPS, TPS, ...)

using _BINDINGS_NAMESPACE_;
using System;

public partial class _CLASS_ : _BASE_
{
    public const float Speed = 5.0f;
    public const float JumpVelocity = 4.5f;

    // Get the gravity from the project settings to be synced with RigidDynamicBody nodes.
    public float gravity = (float)ProjectSettings.GetSetting("physics/3d/default_gravity");

    public override void _PhysicsProcess(float delta)
    {
        Vector3 velocity = Velocity;

        // Add the gravity.
        if (!IsOnFloor())
            velocity.y -= gravity * delta;

        // Handle Jump.
        if (Input.IsActionJustPressed("ui_accept") && IsOnFloor())
             velocity.y = JumpVelocity;

        // Get the input direction and handle the movement/deceleration.
        // As good practice, you should replace UI actions with custom gameplay actions.
        Vector2 inputDir = Input.GetVector("ui_left", "ui_right", "ui_up", "ui_down");
        Vector3 direction = Transform.basis.Xform(new Vector3(inputDir.x, 0, inputDir.y)).Normalized();
        if (direction != Vector3.Zero)
        {
            velocity.x = direction.x * Speed;
            velocity.z = direction.z * Speed;
        }
        else
        {
            velocity.x = Mathf.MoveToward(Velocity.x, 0, Speed);
            velocity.z = Mathf.MoveToward(Velocity.z, 0, Speed);
        }

        Velocity = velocity;
        MoveAndSlide();
    }
}
