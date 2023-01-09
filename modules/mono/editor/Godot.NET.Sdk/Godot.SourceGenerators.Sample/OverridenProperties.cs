using System;
using System.Diagnostics.CodeAnalysis;

#pragma warning disable CS0169
#pragma warning disable CS0414

namespace Godot.SourceGenerators.Sample
{
    public partial class OverridenPropertiesBase : Godot.Object
    {
        // This property is not readonly, has both a getter and a setter.
        public virtual int MyProperty { get; set; } = 1;
    }

    public partial class OverridenPropertiesDerived : OverridenPropertiesBase
    {
        // This property overrides only the getter so it doesn't have a setter,
        // but since it overrides a property with a setter it's not readonly.
        public override int MyProperty => 10;
    }
}
