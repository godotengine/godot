using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Godot.SourceGenerators
{
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public class ExportMemberAnalyzer : DiagnosticAnalyzer
    {
        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSyntaxNodeAction(AnalyzeNode, SyntaxKind.ClassDeclaration);
        }

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(Common.ExportedMemberMustNotBeDuplicateRule);

        static readonly SymbolDisplayFormat namespaceTypeSymolDisplayFormat =
            new SymbolDisplayFormat(
                typeQualificationStyle: SymbolDisplayTypeQualificationStyle.NameAndContainingTypesAndNamespaces);

        void AnalyzeNode(SyntaxNodeAnalysisContext context)
        {
            var godotClasses = context
                .Compilation.SyntaxTrees
                .SelectMany(tree =>
                    tree.GetRoot().DescendantNodes()
                        .OfType<ClassDeclarationSyntax>()
                        .SelectGodotScriptClasses(context.Compilation)
                        // Report and skip non-partial classes
                        .Where(x => x.cds.IsPartial() && (!x.cds.IsNested() || x.cds.AreAllOuterTypesPartial(out _)))
                        .Select(x => x.symbol)
                )
                .Distinct<INamedTypeSymbol>(SymbolEqualityComparer.Default)
                .ToArray();

            foreach (var godotClass in godotClasses)
            {
                var members = new List<ISymbol>();
                var potentialMemberContainer = godotClass;
                while (potentialMemberContainer != null)
                {
                    var exportedFieldsAndProperties = potentialMemberContainer.GetMembers()
                        .Where(s => !s.IsStatic
                                    && ((s is IPropertySymbol ps && !ps.IsIndexer &&
                                         ps.ExplicitInterfaceImplementations.Length == 0)
                                        || s is IFieldSymbol fs && !fs.IsImplicitlyDeclared)
                                    && s.GetAttributes().Any(attr =>
                                        attr.AttributeClass?.ToDisplayString(namespaceTypeSymolDisplayFormat) ==
                                        "Godot.ExportAttribute")
                        );
                    members.AddRange(exportedFieldsAndProperties);
                    potentialMemberContainer = potentialMemberContainer.BaseType;
                }
                FindAndReportDuplicates(members, context);
            }
        }

        private static void FindAndReportDuplicates(IEnumerable<ISymbol> symbols,
            SyntaxNodeAnalysisContext context)
        {
            var groupedDuplicates = symbols.GroupBy(
                sym => sym switch
                {
                    IPropertySymbol p => sym.Name + p.Type,
                    IFieldSymbol f => sym.Name + f.Type,
                    _ => string.Empty
                }).Where(g => g.Key != string.Empty && g.Count() > 1);

            foreach (var duplicateGroup in groupedDuplicates)
            {
#pragma warning disable RS1024
                var distinctPerContainer = duplicateGroup.Distinct(SymbolContainerEqualityComparer.Instance).ToArray();
#pragma warning restore RS1024
                foreach (var duplicateSymbol in distinctPerContainer)
                {
                    context.ReportDiagnostic(Diagnostic.Create(
                        Common.ExportedMemberMustNotBeDuplicateRule,
                        duplicateSymbol.Locations.FirstOrDefault() ?? context.Node.GetLocation(),
                        duplicateSymbol.Kind,
                        duplicateSymbol.Name,
                        duplicateSymbol.ContainingType.ToDisplayString(),
                        (distinctPerContainer.FirstOrDefault(s =>
                            !SymbolEqualityComparer.Default.Equals(s, duplicateSymbol)) ?? duplicateSymbol)
                        .ContainingType.ToDisplayString()
                    ));
                }
            }
        }

        private class SymbolContainerEqualityComparer : IEqualityComparer<ISymbol>
        {
            public static readonly SymbolContainerEqualityComparer Instance = new SymbolContainerEqualityComparer();

            public bool Equals(ISymbol x, ISymbol y) =>
                SymbolEqualityComparer.Default.Equals(x.ContainingSymbol, y.ContainingSymbol);

            public int GetHashCode(ISymbol obj) => SymbolEqualityComparer.Default.GetHashCode(obj.ContainingSymbol);
        }
    }
}
