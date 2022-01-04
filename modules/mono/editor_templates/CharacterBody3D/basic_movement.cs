// meta-description: Classic movement for gravity games (FPS, TPS, ...)

using _BINDINGS_NAMESPACE_;
using System;

public partial class _CLASS_ : _BASE_
{
    public const float Speed = 5.0f;
    public const float JumpForce = 4.5f;

    // Get the gravity from the project settings to be synced with RigidDynamicBody nodes.
    public float gravity = (float)ProjectSettings.GetSetting("physics/3d/default_gravity");

    public override void _PhysicsProcess(float delta)
    {
        Vector3 motionVelocity = MotionVelocity;

        // Add the gravity.
        if (!IsOnFloor())
            motionVelocity.y -= gravity * delta;

        // Handle Jump.
        if (Input.IsActionJustPressed("ui_accept") && IsOnFloor())
             motionVelocity.y = JumpForce;

        // Get the input direction and handle the movement/deceleration.
        // As good practice, you should replace UI actions with custom gameplay actions.
        Vector2 inputDir = Input.GetVector("ui_left", "ui_right", "ui_up", "ui_down");
        Vector3 direction = Transform.basis.Xform(new Vector3(inputDir.x, 0, inputDir.y)).Normalized();
        if (direction != Vector3.Zero)
        {
            motionVelocity.x = direction.x * Speed;
            motionVelocity.z = direction.z * Speed;
        }
        else
        {
            motionVelocity.x = Mathf.MoveToward(MotionVelocity.x, 0, Speed);
            motionVelocity.z = Mathf.MoveToward(MotionVelocity.z, 0, Speed);
        }

        MotionVelocity = motionVelocity;
        MoveAndSlide();
    }
}
