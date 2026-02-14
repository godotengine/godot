using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace Godot.SourceGenerators;

[Generator]
public class TrampolineCollectorDispatchGenerator : ISourceGenerator
{
    public void Initialize(GeneratorInitializationContext context)
    {
    }

    public void Execute(GeneratorExecutionContext context)
    {
        if (context.IsGodotSourceGeneratorDisabled("ScriptMethods"))
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
                        if (x.cds.IsPartial())
                        {
                            if (x.cds.IsNested() && !x.cds.AreAllOuterTypesPartial(out _))
                            {
                                return false;
                            }

                            return true;
                        }

                        return false;
                    })
                    .Select(x => x.symbol)
            )
            .Distinct<INamedTypeSymbol>(SymbolEqualityComparer.Default)
            .ToArray();

        if (godotClasses.Length > 0)
        {
            foreach (var godotClass in godotClasses)
            {
                VisitGodotScriptClass(context, godotClass);
            }
        }
    }

    private static void VisitGodotScriptClass(
        GeneratorExecutionContext context,
        INamedTypeSymbol symbol
    )
    {
        INamespaceSymbol namespaceSymbol = symbol.ContainingNamespace;
        string classNs = namespaceSymbol != null && !namespaceSymbol.IsGlobalNamespace
            ? namespaceSymbol.FullQualifiedNameOmitGlobal()
            : string.Empty;
        bool hasNamespace = classNs.Length != 0;

        bool isInnerClass = symbol.ContainingType != null;

        string uniqueHint = symbol.FullQualifiedNameOmitGlobal().SanitizeQualifiedNameForUniqueHint()
                            + "_TrampolineCollectorDispatch.generated";

        var source = new StringBuilder();

        source.Append("using Godot;\n");
        source.Append("using Godot.NativeInterop;\n");
        source.Append("\n");

        if (hasNamespace)
        {
            source.Append("namespace ");
            source.Append(classNs);
            source.Append(";\n\n");
        }

        if (isInnerClass)
        {
            AppendPartialContainingTypeDeclarations(symbol.ContainingType);

            void AppendPartialContainingTypeDeclarations(INamedTypeSymbol? containingType)
            {
                if (containingType == null)
                    return;

                AppendPartialContainingTypeDeclarations(containingType.ContainingType);

                source.Append("partial ");
                source.Append(containingType.GetDeclarationKeyword());
                source.Append(" ");
                source.Append(containingType.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat));
                source.Append("\n{\n");
            }
        }

        source.Append("partial class ");
        source.Append(symbol.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat));
        source.Append("\n{\n");

        source.Append("#pragma warning disable CS0628 // Disable warning about redundant 'new' keyword\n");
        source.Append("#pragma warning disable CS0109")
            .Append(" // Disable warning about new protected member declared in sealed type\n");

        const string CollectorType = "global::Godot.Bridge.ScriptManagerBridge.TrampolineCollectors";
        const string OptionsType = "global::Godot.Bridge.ScriptManagerBridge.TrampolineCollectionOptions";

        source.Append("    /// <summary>\n")
            .Append("    /// Collects the trampolines for all the members visible to Godot.\n")
            .Append(
                "    /// This method is used by Godot to collect the trampolines to call this class's members from the native side.\n")
            .Append("    /// Do not call this method.\n")
            .Append("    /// </summary>\n");

        source.Append(
                "    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]\n")
            .Append("    protected internal new static void GetGodotClassTrampolines(")
            .Append(CollectorType).Append(" collectors, ").Append(OptionsType).Append(" options)\n    {\n");

        source
            .Append("        ").Append(symbol.FullQualifiedNameIncludeGlobal()).Append(".GodotInternal")
            .Append(".GetGodotMethodTrampolines(collectors.MethodTrampolineCollector);\n")
            .Append("        ").Append(symbol.FullQualifiedNameIncludeGlobal()).Append(".GodotInternal")
            .Append(".GetGodotPropertyTrampolines(collectors.PropertyTrampolineCollector);\n")
            .Append("        ").Append(symbol.FullQualifiedNameIncludeGlobal()).Append(".GodotInternal")
            .Append(".GetGodotRaiseSignalTrampolines(collectors.RaiseSignalTrampolineCollector);\n");

        if (!symbol.IsGenericType)
        {
            source
                .Append("        ").Append(symbol.FullQualifiedNameIncludeGlobal()).Append(".GodotInternal")
                .Append(".GetGodotConstructorTrampolines(collectors.ConstructorTrampolineCollector);\n");
        }

        var baseType = symbol.BaseType!;

        if (!SymbolEqualityComparer.Default.Equals(baseType, symbol.GetGodotScriptNativeClass()))
        {
            source
                .Append("        if (options.IncludeAncestors) {\n")
                .Append("            ").Append(baseType.FullQualifiedNameIncludeGlobal())
                .Append(".GetGodotClassTrampolines(collectors, options);")
                .Append("        }\n");
        }

        source
            .Append("    }\n");

        source.Append("#pragma warning restore CS0109\n");
        source.Append("#pragma warning restore CS0628\n");

        source.Append("}\n"); // partial class

        if (isInnerClass)
        {
            var containingType = symbol.ContainingType;

            while (containingType != null)
            {
                source.Append("}\n"); // outer class

                containingType = containingType.ContainingType;
            }
        }

        context.AddSource(uniqueHint, SourceText.From(source.ToString(), Encoding.UTF8));
    }
}
