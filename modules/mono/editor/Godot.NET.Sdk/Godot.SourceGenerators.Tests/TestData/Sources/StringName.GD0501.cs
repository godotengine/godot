using Godot;

public partial class TestStringNameClass : GodotObject
{
    public void TestMethod()
    {
        // this is fine, not in a loop
        _ = Input.IsActionPressed("ui_accept");

        // this however repeatedly creates a new constant StringName object
        while (true)
            _ = Input.IsActionPressed({|GD0501:"ui_accept"|});

        // this is fine, since the name is dynamic
        for (int i = 0; i < 10; ++i)
            _ = Input.IsActionPressed($"ui_accept{i}");

        // this however also repeatedly creates a new constant StringName object
        while (true)
            _ = Input.IsActionPressed({|GD0501:"ui_" + "accept"|});

        // this should not be touched
        TestStringNameClass2.Fn("ui_accept");
    }
}

public partial class TestStringNameClass2 : GodotObject
{
    public void TestMethod()
    {
        // this should not be touched
        _ = Input.IsActionPressed("ui_accept");
    }

    public static void Fn(string s) { }
}

public partial class TestStringNameClass3 : GodotObject
{
    private static readonly StringName uiAcceptStringName = "ui_accept";

    public void TestMethod()
    {
        // this should reuse the existing StringName cache variable
        while (true)
            _ = Input.IsActionPressed({|GD0501:"ui_accept"|});
    }
}