using System;
using System.Linq;
using Microsoft.CodeAnalysis;

namespace Godot.SourceGenerators
{
    internal static class NullableUtils
    {
        internal static bool IsNullableContextEnabledForSymbol(ISymbol symbol, Func<SyntaxTree, SemanticModel> semanticModelProvider)
        {
            // Get the syntax reference for the symbol declaration
            var syntaxReference = symbol.DeclaringSyntaxReferences.FirstOrDefault();
            if (syntaxReference == null)
                return false;

            var syntaxTree = syntaxReference.SyntaxTree;
            var syntaxNode = syntaxReference.GetSyntax();

            // Get the nullable context options at the declaration location
            var semanticModel = semanticModelProvider(syntaxTree);
            var nullableContext = semanticModel.GetNullableContext(syntaxNode.SpanStart);

            // Check if nullable reference types are enabled (either as warnings or errors)
            return (nullableContext & NullableContext.Enabled) != 0;
        }

        internal static bool IsExportedNonNullableGodotType(ISymbol memberSymbol, ITypeSymbol memberType, Func<SyntaxTree, SemanticModel> semanticModelProvider, bool requireNullableContext = true)
        {
            // Check if the member has the [Export] attribute
            bool isExported = memberSymbol.GetAttributes()
                .Any(a => a.AttributeClass?.IsGodotExportAttribute() ?? false);

            if (!isExported)
                return false;

            // Check if it's a reference type and check nullable annotation
            if (!memberType.IsReferenceType)
                return false;

            // Check if member has nullable context enabled (either via #nullable or project settings)
            // This check can be skipped for the suppressor when we just need to identify exported Godot types
            if (requireNullableContext && !IsNullableContextEnabledForSymbol(memberSymbol, semanticModelProvider))
                return false;

            // If the type is nullable annotated (e.g., Node?), skip it
            if (memberType.NullableAnnotation == NullableAnnotation.Annotated)
                return false;

            // Check for string type
            if (memberType.SpecialType == SpecialType.System_String)
                return true;

            // Check for packed arrays (System arrays of Godot-compatible types)
            if (MarshalUtils.IsPackedArrayType(memberType))
                return true;

            // Check if the type is a Godot compatible class type (Node, Resource, or derived)
            // This check must come after array check since arrays are not INamedTypeSymbol
            if (memberType is not INamedTypeSymbol namedType)
                return false;

            // Check if the type inherits from Node or Resource
            bool isNodeOrResource = namedType.InheritsFrom("GodotSharp", GodotClasses.Node) ||
                                    namedType.InheritsFrom("GodotSharp", "Godot.Resource");

            if (isNodeOrResource)
                return true;

            // Check if the type is Godot.Collections.Array or Dictionary (including generic variations)
            string fullTypeName = namedType.ConstructedFrom.ToString();
            bool isGodotCollection = fullTypeName == "Godot.Collections.Array" ||
                                     fullTypeName == "Godot.Collections.Array<T>" ||
                                     fullTypeName == "Godot.Collections.Dictionary" ||
                                     fullTypeName == "Godot.Collections.Dictionary<TKey, TValue>";

            if (isGodotCollection)
                return true;

            // Check for StringName and NodePath
            if (MarshalUtils.IsStringNameType(namedType) || MarshalUtils.IsNodePathType(namedType))
                return true;

            return false;
        }
    }
}
