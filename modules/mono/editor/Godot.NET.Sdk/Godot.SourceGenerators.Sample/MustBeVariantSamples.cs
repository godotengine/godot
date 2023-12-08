using System;

namespace Godot.SourceGenerators.Sample;

public class MustBeVariantMethods
{
    public MustBeVariantMethods()
    {
        Method<string>();

        // This call must fail
        // Method<NotVariant>();
    }

    public void Method<[MustBeVariant] T>() where T : class
    {
    }
}

public class MustBeVariantAnnotatedMethods
{
    [GenericTypeAttribute<string>()]
    public void MethodWithAttribute()
    {
    }

    // This method fails
    /*
    [GenericTypeAttribute<NotVariant>()]
    public void MethodWithWrongAttribute()
    {
    }
    */
}

[GenericTypeAttribute<string>()]
public class ClassVariantAnnotated
{
}


// This class fail
/*
[GenericTypeAttribute<NotVariant>()]
public class ClassNonVariantAnnotated
{
}
*/

[AttributeUsage(AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = true)]
public class GenericTypeAttribute<[MustBeVariant] T> : Attribute where T : class
{
}

public class NotVariant
{
}
