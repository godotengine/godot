using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace Godot.SourceGenerators;

[Generator]
public class OnReadyGenerator : ISourceGenerator
{
    private const string GetNodeMethod = "GetNode";
    private const string GetNodeOrNullMethod = "GetNodeOrNull";

    public void Initialize(GeneratorInitializationContext context)
    {

    }

    public void Execute(GeneratorExecutionContext context)
    {
        if (context.IsGodotSourceGeneratorDisabled("OnReady"))
            return;

        INamedTypeSymbol[] godotClasses = context
            .Compilation.SyntaxTrees
            .SelectMany(tree =>
                tree.GetRoot().DescendantNodes()
                    .OfType<ClassDeclarationSyntax>()
                    .SelectGodotScriptClasses(context.Compilation)
                    // Report and skip non-partial classes
                    .Where(x =>
                    {
                        if (!x.cds.IsPartial()) return false;

                        if (x.cds.IsNested() && !x.cds.AreAllOuterTypesPartial(out _)) return false;

                        if (!x.symbol.GetMembers().Any(m => m.HasAttribute(GodotClasses.OnReadyAttr))) return false;

                        return true;
                    })
                    .Select(x => x.symbol)
            )
            .Distinct<INamedTypeSymbol>(SymbolEqualityComparer.Default)
            .ToArray();
        foreach (var godotClass in godotClasses)
        {
            VisitGodotScriptClass(context, godotClass);
        }
    }

    private static void VisitGodotScriptClass(
        GeneratorExecutionContext context,
        INamedTypeSymbol typeSymbol
    )
    {
        INamespaceSymbol namespaceSymbol = typeSymbol.ContainingNamespace;
        string classNs = namespaceSymbol is { IsGlobalNamespace: false }
            ? namespaceSymbol.FullQualifiedNameOmitGlobal()
            : string.Empty;

        bool hasNamespace = classNs.Length != 0;

        bool isInnerClass = typeSymbol.ContainingType != null;

        string uniqueHint = typeSymbol.FullQualifiedNameOmitGlobal().SanitizeQualifiedNameForUniqueHint()
                            + "_OnReady.generated";

        var source = new StringBuilder();

        source.Append("using Godot;\n");
        source.Append("\n#nullable enable");
        source.Append("\n\n");

        if (hasNamespace)
        {
            source.Append("namespace ");
            source.Append(classNs);
            source.Append(" {\n\n");
        }

        if (isInnerClass)
        {
            var rootContainingType = typeSymbol.ContainingType;
            AppendPartialContainingTypeDeclarations(rootContainingType);

            void AppendPartialContainingTypeDeclarations(INamedTypeSymbol? containingType)
            {
                if (containingType == null)
                    return;

                AppendPartialContainingTypeDeclarations(containingType.ContainingType);

                source.Append("partial ");
                source.Append(containingType.GetDeclarationKeyword());
                source.Append(" ");
                source.Append(containingType.NameWithTypeParameters());
                source.Append("\n{\n");
            }
        }

        source.Append("partial class ");
        source.Append(typeSymbol.NameWithTypeParameters());
        source.Append("\n{\n");

        var members = typeSymbol.GetMembers();

        var onReadySymbols = members
            .Where(s => s.HasAttribute(GodotClasses.OnReadyAttr));

        foreach (var symbol in onReadySymbols)
        {
            switch (symbol)
            {
                case IMethodSymbol method when Valid(context, method):
                    CreateOnReadyMethod(method, source);
                    break;
                case IPropertySymbol property when Valid(context, property):
                    CreateOnReadyProperty(property, source);
                    break;
            }
        }

        source.Append("}\n"); // partial class

        if (isInnerClass)
        {
            var containingType = typeSymbol.ContainingType;

            while (containingType != null)
            {
                source.Append("}\n"); // outer class

                containingType = containingType.ContainingType;
            }
        }

        if (hasNamespace)
        {
            source.Append("\n}\n");
        }

        source.Append("#nullable restore\n");
        context.AddSource(uniqueHint, SourceText.From(source.ToString(), Encoding.UTF8));
    }

    private static void CreateOnReadyMethod(IMethodSymbol method, StringBuilder source)
    {
        var nodePath = method.GetAttribute(GodotClasses.OnReadyAttr).ConstructorArguments[0].Value;
        var nullable = method.ReturnType.NullableAnnotation == NullableAnnotation.Annotated;
        var returnType = method.ReturnType.FullQualifiedNameIncludeGlobal();
        var returnTypeWithNullable = nullable ? returnType + '?' : returnType;
        var getNodeMethod = nullable ? GetNodeOrNullMethod : GetNodeMethod;
        var modifiers = SyntaxFacts.GetText(method.DeclaredAccessibility);
        var backingField = string.Concat("_", method.Name[0].ToString().ToLower(), method.Name.Substring(1));

        source.Append($"    private {returnType}? {backingField};\n"); // backing field
        source.Append($"    {modifiers} {returnTypeWithNullable} {method.Name}Ready => {backingField} ??= {method.Name}();\n"); // property
        source.Append($"    {modifiers} partial {returnTypeWithNullable} {method.Name}()\n"); // implementing the method
        source.Append($"    {{\n");
        source.Append($"        return {getNodeMethod}<{returnType}>(\"{nodePath}\");\n");
        source.Append($"    }}\n\n");
    }
    private static void CreateOnReadyProperty(IPropertySymbol property, StringBuilder source)
    {
        var nodePath = property.GetAttribute(GodotClasses.OnReadyAttr).ConstructorArguments[0].Value?.ToString() ?? string.Empty;
        var nullable = property.Type.NullableAnnotation == NullableAnnotation.Annotated;
        var type = property.Type.FullQualifiedNameIncludeGlobal();
        var typeWithNullable = nullable ? type + '?' : type;
        var getNodeMethod = nullable ? GetNodeOrNullMethod : GetNodeMethod;
        var modifiers = SyntaxFacts.GetText(property.DeclaredAccessibility);
        var backingField = string.Concat("_", property.Name[0].ToString().ToLower(), property.Name.Substring(1));

        source.Append($"    private {type}? {backingField};\n"); // backing field
        source.Append($"    {modifiers} partial {typeWithNullable} {property.Name} => {backingField} ??= {getNodeMethod}<{type}>(\"{nodePath}\");\n\n"); // property impl
    }

    private static bool Valid(GeneratorExecutionContext context, IPropertySymbol property) =>
        Valid(context, property, property.IsPartialDefinition, property.Type);

    private static bool Valid(GeneratorExecutionContext context, IMethodSymbol property) =>
        Valid(context, property, property.IsPartialDefinition, property.ReturnType);

    private static bool Valid(GeneratorExecutionContext context, ISymbol symbol, bool isPartial, ITypeSymbol type)
    {
        if (symbol.IsStatic)
        {
            context.ReportDiagnostic(Diagnostic.Create(
                Common.OnReadyMemberCannotBeStatic,
                symbol.Locations.FirstLocationWithSourceTreeOrDefault(),
                symbol.ToDisplayString()
            ));
            return false;
        }

        if (!isPartial)
        {
            context.ReportDiagnostic(Diagnostic.Create(
                Common.OnReadyMemberMustBeEmptyPartial,
                symbol.Locations.FirstLocationWithSourceTreeOrDefault(),
                symbol.ToDisplayString()
            ));
            return false;
        }

        if (!type.InheritsFrom("GodotSharp", GodotClasses.Node))
        {
            context.ReportDiagnostic(Diagnostic.Create(
                Common.OnReadyMemberReturnMustDeriveFromNode,
                symbol.Locations.FirstLocationWithSourceTreeOrDefault(),
                symbol.ToDisplayString()
            ));
            return false;
        }

        return true;
    }
}
