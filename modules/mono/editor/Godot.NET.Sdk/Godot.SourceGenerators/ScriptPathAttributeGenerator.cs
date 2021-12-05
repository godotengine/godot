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
    public class ScriptPathAttributeGenerator : ISourceGenerator
    {
        public void Execute(GeneratorExecutionContext context)
        {
            if (context.TryGetGlobalAnalyzerProperty("GodotScriptPathAttributeGenerator", out string? toggle)
                && toggle == "disabled")
            {
                return;
            }

            // NOTE: IsNullOrEmpty doesn't work well with nullable checks
            // ReSharper disable once ReplaceWithStringIsNullOrEmpty
            if (!context.TryGetGlobalAnalyzerProperty("GodotProjectDir", out string? godotProjectDir)
                || godotProjectDir!.Length == 0)
            {
                throw new InvalidOperationException("Property 'GodotProjectDir' is null or empty.");
            }

            var godotClasses = context.Compilation.SyntaxTrees
                .SelectMany(tree =>
                    tree.GetRoot().DescendantNodes()
                        .OfType<ClassDeclarationSyntax>()
                        // Ignore inner classes
                        .Where(cds => !(cds.Parent is ClassDeclarationSyntax))
                        .SelectGodotScriptClasses(context.Compilation)
                        // Report and skip non-partial classes
                        .Where(x =>
                        {
                            if (x.cds.IsPartial() || x.symbol.HasDisableGeneratorsAttribute())
                                return true;
                            Common.ReportNonPartialGodotScriptClass(context, x.cds, x.symbol);
                            return false;
                        })
                )
                // Ignore classes whose name is not the same as the file name
                .Where(x => Path.GetFileNameWithoutExtension(x.cds.SyntaxTree.FilePath) == x.symbol.Name)
                .GroupBy(x => x.symbol)
                .ToDictionary(g => g.Key, g => g.Select(x => x.cds));

            foreach (var godotClass in godotClasses)
            {
                VisitGodotScriptClass(context, godotProjectDir,
                    symbol: godotClass.Key,
                    classDeclarations: godotClass.Value);
            }

            if (godotClasses.Count <= 0)
                return;

            AddScriptTypesAssemblyAttr(context, godotClasses);
        }

        private static void VisitGodotScriptClass(
            GeneratorExecutionContext context,
            string godotProjectDir,
            INamedTypeSymbol symbol,
            IEnumerable<ClassDeclarationSyntax> classDeclarations
        )
        {
            var attributes = new StringBuilder();

            // Remember syntax trees for which we already added an attribute, to prevent unnecessary duplicates.
            var attributedTrees = new List<SyntaxTree>();

            foreach (var cds in classDeclarations)
            {
                if (attributedTrees.Contains(cds.SyntaxTree))
                    continue;

                attributedTrees.Add(cds.SyntaxTree);

                if (attributes.Length != 0)
                    attributes.Append("\n");

                attributes.Append(@"[ScriptPathAttribute(""res://");
                attributes.Append(RelativeToDir(cds.SyntaxTree.FilePath, godotProjectDir));
                attributes.Append(@""")]");
            }

            string className = symbol.Name;

            INamespaceSymbol namespaceSymbol = symbol.ContainingNamespace;
            string classNs = namespaceSymbol != null && !namespaceSymbol.IsGlobalNamespace ?
                namespaceSymbol.FullQualifiedName() :
                string.Empty;
            bool hasNamespace = classNs.Length != 0;

            var uniqueName = new StringBuilder();
            if (hasNamespace)
                uniqueName.Append($"{classNs}.");
            uniqueName.Append(className);
            if (symbol.IsGenericType)
                uniqueName.Append($"Of{string.Join(string.Empty, symbol.TypeParameters)}");
            uniqueName.Append("_ScriptPath_Generated");

            var source = new StringBuilder();

            // using Godot;
            // namespace {classNs} {
            //     {attributesBuilder}
            //     partial class {className} { }
            // }

            source.Append("using Godot;\n");

            if (hasNamespace)
            {
                source.Append("namespace ");
                source.Append(classNs);
                source.Append(" {\n\n");
            }

            source.Append(attributes);
            source.Append("\n    partial class ");
            source.Append(className);
            if (symbol.IsGenericType)
                source.Append($"<{string.Join(", ", symbol.TypeParameters)}>");
            source.Append("\n{\n}\n");

            if (hasNamespace)
            {
                source.Append("\n}\n");
            }

            context.AddSource(uniqueName.ToString(), SourceText.From(source.ToString(), Encoding.UTF8));
        }

        private static void AddScriptTypesAssemblyAttr(GeneratorExecutionContext context,
            Dictionary<INamedTypeSymbol, IEnumerable<ClassDeclarationSyntax>> godotClasses)
        {
            var sourceBuilder = new StringBuilder();

            sourceBuilder.Append("[assembly:");
            sourceBuilder.Append(GodotClasses.AssemblyHasScriptsAttr);
            sourceBuilder.Append("(new System.Type[] {");

            bool first = true;

            foreach (var godotClass in godotClasses)
            {
                var qualifiedName = godotClass.Key.ToDisplayString(
                    NullableFlowState.NotNull, SymbolDisplayFormat.FullyQualifiedFormat
                        .WithGenericsOptions(SymbolDisplayGenericsOptions.None));
                if (!first)
                    sourceBuilder.Append(", ");
                first = false;
                sourceBuilder.Append("typeof(");
                sourceBuilder.Append(qualifiedName);
                if (godotClass.Key.IsGenericType)
                    sourceBuilder.Append($"<{new string(',', godotClass.Key.TypeParameters.Count() - 1)}>");
                sourceBuilder.Append(")");
            }

            sourceBuilder.Append("})]\n");

            context.AddSource("AssemblyScriptTypes_Generated",
                SourceText.From(sourceBuilder.ToString(), Encoding.UTF8));
        }

        public void Initialize(GeneratorInitializationContext context)
        {
        }

        private static string RelativeToDir(string path, string dir)
        {
            // Make sure the directory ends with a path separator
            dir = Path.Combine(dir, " ").TrimEnd();

            if (Path.DirectorySeparatorChar == '\\')
                dir = dir.Replace("/", "\\") + "\\";

            var fullPath = new Uri(Path.GetFullPath(path), UriKind.Absolute);
            var relRoot = new Uri(Path.GetFullPath(dir), UriKind.Absolute);

            // MakeRelativeUri converts spaces to %20, hence why we need UnescapeDataString
            return Uri.UnescapeDataString(relRoot.MakeRelativeUri(fullPath).ToString());
        }
    }
}
