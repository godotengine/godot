using Godot;

public partial class TestStringNameClass : GodotObject
{
    public void TestMethod()
    {
        _ = Input.IsActionPressed({|GD0501:"ui_accept"|});

        while (true)
            _ = Input.IsActionPressed({|GD0501:"ui_accept"|});

        while (true)
            _ = Input.IsActionPressed({|GD0501:"ui_" + "accept"|});

        // this should not be touched
        Fn("ui_accept");
    }

    public static void Fn(string s) { }
}

public partial class TestStringNameClass2 : GodotObject
{
    private static readonly StringName uiAcceptStringName = new("ui_accept");

    public void TestMethod()
    {
        // this should reuse the existing StringName cache variable
        while (true)
            _ = Input.IsActionPressed({|GD0501:"ui_accept"|});
    }
}

public partial class TestStringNameClass3 : GodotObject
{
    public void TestMethod()
    {
        // this should not be touched
        Fn1("ui_accept");

        // this should be touched
        Fn2({|GD0501:"ui_accept"|});

        // this should not be touched, it uses the string overload
        Fn3("ui_accept");
    }

    // regular string parameter
    static void Fn1(string s) { }

    // StringName parameter
    static void Fn2(StringName s) { }

    // overloads for both string and StringName
    static void Fn3(string s) { }
    static void Fn3(StringName s) { }
}
