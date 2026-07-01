// meta-description: Classic movement for topdown games

using _BINDINGS_NAMESPACE_;
using System;

public partial class _CLASS_ : _BASE_
{
    public const float Speed = 300.0f;

    public override void _PhysicsProcess(double delta)
    {
        // As good practice, you should replace UI actions with custom gameplay actions.
        Vector2 direction = Input.GetVector("ui_left", "ui_right", "ui_up", "ui_down");
        if (direction != Vector2.Zero)
        {
            Velocity = direction * Speed;
        }
        else
        {
            Velocity = Velocity.MoveToward(Vector2.Zero, Speed);
        }

        MoveAndSlide();
    }
}
