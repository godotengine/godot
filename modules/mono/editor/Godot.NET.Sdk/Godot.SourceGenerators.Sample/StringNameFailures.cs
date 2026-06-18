using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Godot.SourceGenerators.Sample;
using Godot;

public partial class TestStringNameClass : GodotObject
{
    public void TestMethod()
    {
        _ = Input.IsActionPressed("ui_accept");

        while (true)
            _ = Input.IsActionPressed("ui_accept");

        for (int i = 0; i < 10; ++i)
            _ = Input.IsActionPressed($"ui_accept{i}");

        while (true)
            _ = Input.IsActionPressed("ui_" + "accept");

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
    private static readonly StringName uiAcceptStringName = new("ui_accept");

    public void TestMethod()
    {
        // this should reuse the existing StringName cache variable
        while (true)
            _ = Input.IsActionPressed("ui_accept");
    }
}

public partial class TestStringNameClass4 : GodotObject
{
    public void TestMethod()
    {
        // this should not be touched
        Fn1("ui_accept");

        // this should be touched
        Fn2("ui_accept");

        // this should not be touched, it uses the string overload
        Fn3("ui_accept");
    }

    // regular string parameter
    private static void Fn1(string s) { }

    // StringName parameter
    private static void Fn2(StringName s) { }

    // overloads for both string and StringName
    private static void Fn3(string s) { }
    private static void Fn3(StringName s) { }
}
