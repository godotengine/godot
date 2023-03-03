using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace Godot.SourceGenerators
{
    [Generator]
    public class ScriptPathAttributeGenerator : IIncrementalGenerator
    {
        public void Initialize(IncrementalGeneratorInitializationContext context)
        {
            var areGodotSourceGeneratorsDisabled = context.AnalyzerConfigOptionsProvider.Select(static (provider, _) => provider.AreGodotSourceGeneratorsDisabled());

            var isGodotToolsProject = context.AnalyzerConfigOptionsProvider.Select(static (provider, _) => provider.IsGodotToolsProject());

            var godotProjectDir = context.AnalyzerConfigOptionsProvider.Select(static (provider, _) =>
            {
                provider.TryGetGlobalAnalyzerProperty("GodotProjectDir", out string? godotProjectDir);
                return godotProjectDir;
            });

            var godotClasses = context.SyntaxProvider.CreateValuesProviderForGodotClasses(
                // Select class declarations that inherit from something
                // since Godot classes must at least inherit from Godot.Object
                customPredicate: static (s, _) => s is ClassDeclarationSyntax { BaseList.Types.Count: > 0 } cds &&
                    // Ignore inner classes
                    !cds.IsNested());

            var configValues = areGodotSourceGeneratorsDisabled
                .Combine(isGodotToolsProject)
                .Combine(godotProjectDir);

            var values = configValues.Combine(godotClasses.Collect());

            context.RegisterSourceOutput(values, static (spc, source) =>
            {
                var configValues = source.Left;
                var godotClasses = source.Right;

                ((bool areGodotSourceGeneratorsDisabled, bool isGodotToolsProject), string? godotProjectDir) = configValues;

                if (areGodotSourceGeneratorsDisabled)
                    return;

                if (isGodotToolsProject)
                    return;

                // NOTE: NotNullWhen diagnostics don't work on projects targeting .NET Standard 2.0
                // ReSharper disable once ReplaceWithStringIsNullOrEmpty
                if (string.IsNullOrEmpty(godotProjectDir) || godotProjectDir!.Length == 0)
                {
                    throw new InvalidOperationException("Property 'GodotProjectDir' is null or empty.");
                }

                Execute(spc, godotClasses, godotProjectDir);
            });
        }

        private static void Execute(SourceProductionContext context, ImmutableArray<GodotClassData> godotClassDatas, string godotProjectDir)
        {
            Dictionary<INamedTypeSymbol, IEnumerable<ClassDeclarationSyntax>> godotClasses = godotClassDatas.Where(x =>
            {
                // Report and skip non-partial classes
                if (!x.DeclarationSyntax.IsPartial())
                {
                    Common.ReportNonPartialGodotScriptClass(context, x.DeclarationSyntax, x.Symbol);
                    return false;
                }

                // Ignore classes whose name is not the same as the file name
                if (Path.GetFileNameWithoutExtension(x.DeclarationSyntax.SyntaxTree.FilePath) != x.Symbol.Name)
                    return false;

                // Ignore generic classes
                if (x.Symbol.IsGenericType)
                    return false;

                return true;
            }).GroupBy<GodotClassData, INamedTypeSymbol>(x => x.Symbol, SymbolEqualityComparer.Default)
                .ToDictionary<IGrouping<INamedTypeSymbol, GodotClassData>, INamedTypeSymbol, IEnumerable<ClassDeclarationSyntax>>(g => g.Key, g => g.Select(x => x.DeclarationSyntax), SymbolEqualityComparer.Default);

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
            SourceProductionContext context,
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

            INamespaceSymbol namespaceSymbol = symbol.ContainingNamespace;
            string classNs = namespaceSymbol != null && !namespaceSymbol.IsGlobalNamespace ?
                namespaceSymbol.FullQualifiedNameOmitGlobal() :
                string.Empty;
            bool hasNamespace = classNs.Length != 0;

            string uniqueHint = symbol.FullQualifiedNameOmitGlobal().SanitizeQualifiedNameForUniqueHint()
                             + "_ScriptPath.generated";

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
            source.Append("\npartial class ");
            source.Append(symbol.NameWithTypeParameters());
            source.Append("\n{\n}\n");

            if (hasNamespace)
            {
                source.Append("\n}\n");
            }

            context.AddSource(uniqueHint, SourceText.From(source.ToString(), Encoding.UTF8));
        }

        private static void AddScriptTypesAssemblyAttr(SourceProductionContext context,
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
                sourceBuilder.Append(")");
            }

            sourceBuilder.Append("})]\n");

            context.AddSource("AssemblyScriptTypes.generated",
                SourceText.From(sourceBuilder.ToString(), Encoding.UTF8));
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
