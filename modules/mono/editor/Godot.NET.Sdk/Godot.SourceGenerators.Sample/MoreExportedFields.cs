using System;
using System.Diagnostics.CodeAnalysis;

#pragma warning disable CS0169
#pragma warning disable CS0414

namespace Godot.SourceGenerators.Sample
{
    [SuppressMessage("ReSharper", "BuiltInTypeReferenceStyle")]
    [SuppressMessage("ReSharper", "RedundantNameQualifier")]
    [SuppressMessage("ReSharper", "ArrangeObjectCreationWhenTypeEvident")]
    [SuppressMessage("ReSharper", "InconsistentNaming")]
    // We split the definition of ExportedFields to verify properties work across multiple files.
    public partial class ExportedFields : GodotObject
    {
        // Note we use Array and not System.Array. This tests the generated namespace qualification.
        [Export] private Int64[] field_empty_Int64Array = Array.Empty<Int64>();
    }
}
