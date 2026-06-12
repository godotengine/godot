using Godot;

// This works because it does not inherit from GodotObject
public class CustomParameterlessConstructorClass1
{
    public CustomParameterlessConstructorClass1(int value)
    {
        _ = value;
    }
}

// This works because it inherits from GodotObject and does not declare any constructors
public class CustomParameterlessConstructorClass2 : GodotObject
{

}

// This works because it inherits from GodotObject and declares a parameterless constructor
public class CustomParameterlessConstructorClass3 : GodotObject
{
    public CustomParameterlessConstructorClass3()
    {

    }
}

// This works because it inherits from GodotObject and declares a parameterless constructor
public class CustomParameterlessConstructorClass4 : GodotObject
{
    public CustomParameterlessConstructorClass4() : this(0)
    {

    }

    public CustomParameterlessConstructorClass4(int value)
    {
        _ = value;
    }
}

// This works because it inherits from an object that inherits from GodotObject and does not declare any constructors
public partial class CustomParameterlessConstructorClass5 : Node
{

}

// This works because it inherits from an object that inherits from GodotObject and declares a parameterless constructor
public partial class CustomParameterlessConstructorClass6 : Node
{
    public CustomParameterlessConstructorClass6()
    {

    }
}

// This works because it inherits from an object that inherits from GodotObject and declares a parameterless constructor
public partial class CustomParameterlessConstructorClass7 : Node
{
    public CustomParameterlessConstructorClass7() : this(0)
    {

    }

    public CustomParameterlessConstructorClass7(int value)
    {
        _ = value;
    }
}
