using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace Godot.SourceGenerators
{
    [Generator]
    public class ScriptInterfacesGenerator : ISourceGenerator
    {
        public void Execute(GeneratorExecutionContext context)
        {
            if (context.IsGodotSourceGeneratorDisabled("ScriptInterfaces"))
                return;

            if (context.IsGodotToolsProject())
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
                var typeCache = new MarshalUtils.TypeCache(context.Compilation);

                foreach (var godotClass in godotClasses)
                {
                    VisitGodotScriptClass(context, typeCache, godotClass);
                }
            }
        }

        private static void VisitGodotScriptClass(
            GeneratorExecutionContext context,
            MarshalUtils.TypeCache typeCache,
            INamedTypeSymbol symbol
        )
        {
            var attributes = new StringBuilder();

            var implementedInterfaces = symbol.AllInterfaces;

            foreach (var implementedInterface in implementedInterfaces)
            {
                if (attributes.Length != 0)
                    attributes.Append("\n");

                attributes.Append(@"[ScriptInterfaceAttribute(""");
                attributes.Append(implementedInterface.FullQualifiedNameOmitGlobal());
                attributes.Append(@""")]");
            }

            INamespaceSymbol namespaceSymbol = symbol.ContainingNamespace;
            string classNs =
                namespaceSymbol != null && !namespaceSymbol.IsGlobalNamespace
                    ? namespaceSymbol.FullQualifiedNameOmitGlobal()
                    : string.Empty;
            bool hasNamespace = classNs.Length != 0;

            string uniqueHint =
                symbol.FullQualifiedNameOmitGlobal().SanitizeQualifiedNameForUniqueHint()
                + "_ScriptInterface.generated";

            var source = new StringBuilder();

            source.Append("using Godot;\n");

            if (hasNamespace)
            {
                source.Append("namespace ");
                source.Append(classNs);
                source.Append(" {\n\n");
            }

            source.Append(attributes);
            source.Append("\npartial class ");
            source.Append(symbol.NameWithTypeParameters());
            source.Append("\n{\n}\n");

            if (hasNamespace)
            {
                source.Append("\n}\n");
            }

            context.AddSource(uniqueHint, SourceText.From(source.ToString(), Encoding.UTF8));
        }

        public void Initialize(GeneratorInitializationContext context) { }
    }
}
