using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace Godot.SourceGenerators
{
    [Generator]
    public class ScriptPropertyDefValGenerator : ISourceGenerator
    {
        public void Initialize(GeneratorInitializationContext context)
        {
        }

        public void Execute(GeneratorExecutionContext context)
        {
            if (context.IsGodotSourceGeneratorDisabled("ScriptPropertyDefVal"))
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
                                if (x.cds.IsNested() && !x.cds.AreAllOuterTypesPartial(out var typeMissingPartial))
                                {
                                    Common.ReportNonPartialGodotScriptOuterClass(context, typeMissingPartial!);
                                    return false;
                                }

                                return true;
                            }

                            Common.ReportNonPartialGodotScriptClass(context, x.cds, x.symbol);
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
            INamespaceSymbol namespaceSymbol = symbol.ContainingNamespace;
            string classNs = namespaceSymbol != null && !namespaceSymbol.IsGlobalNamespace ?
                namespaceSymbol.FullQualifiedNameOmitGlobal() :
                string.Empty;
            bool hasNamespace = classNs.Length != 0;

            bool isInnerClass = symbol.ContainingType != null;

            string uniqueHint = symbol.FullQualifiedNameOmitGlobal().SanitizeQualifiedNameForUniqueHint()
                                + "_ScriptPropertyDefVal.generated";

            var source = new StringBuilder();

            if (hasNamespace)
            {
                source.Append("namespace ");
                source.Append(classNs);
                source.Append(" {\n\n");
            }

            if (isInnerClass)
            {
                var containingType = symbol.ContainingType;
                AppendPartialContainingTypeDeclarations(containingType);

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
            source.Append(symbol.NameWithTypeParameters());
            source.Append("\n{\n");

            var exportedMembers = new List<ExportedPropertyMetadata>();

            var members = symbol.GetMembers();

            var exportedProperties = members
                .Where(s => !s.IsStatic && s.Kind == SymbolKind.Property)
                .Cast<IPropertySymbol>()
                .Where(s => s.GetAttributes()
                    .Any(a => a.AttributeClass?.IsGodotExportAttribute() ?? false))
                .ToArray();

            var exportedFields = members
                .Where(s => !s.IsStatic && s.Kind == SymbolKind.Field && !s.IsImplicitlyDeclared)
                .Cast<IFieldSymbol>()
                .Where(s => s.GetAttributes()
                    .Any(a => a.AttributeClass?.IsGodotExportAttribute() ?? false))
                .ToArray();

            foreach (var property in exportedProperties)
            {
                if (property.IsStatic)
                {
                    Common.ReportExportedMemberIsStatic(context, property);
                    continue;
                }

                if (property.IsIndexer)
                {
                    Common.ReportExportedMemberIsIndexer(context, property);
                    continue;
                }

                // TODO: We should still restore read-only properties after reloading assembly. Two possible ways: reflection or turn RestoreGodotObjectData into a constructor overload.
                // Ignore properties without a getter, without a setter or with an init-only setter. Godot properties must be both readable and writable.
                if (property.IsWriteOnly)
                {
                    Common.ReportExportedMemberIsWriteOnly(context, property);
                    continue;
                }

                if (property.IsReadOnly || property.SetMethod!.IsInitOnly)
                {
                    Common.ReportExportedMemberIsReadOnly(context, property);
                    continue;
                }

                if (property.ExplicitInterfaceImplementations.Length > 0)
                {
                    Common.ReportExportedMemberIsExplicitInterfaceImplementation(context, property);
                    continue;
                }

                var propertyType = property.Type;
                var marshalType = MarshalUtils.ConvertManagedTypeToMarshalType(propertyType, typeCache);

                if (marshalType == null)
                {
                    Common.ReportExportedMemberTypeNotSupported(context, property);
                    continue;
                }

                if (marshalType == MarshalType.GodotObjectOrDerived)
                {
                    if (!symbol.InheritsFrom("GodotSharp", "Godot.Node") &&
                        propertyType.InheritsFrom("GodotSharp", "Godot.Node"))
                    {
                        Common.ReportOnlyNodesShouldExportNodes(context, property);
                    }
                }

                var propertyDeclarationSyntax = property.DeclaringSyntaxReferences
                    .Select(r => r.GetSyntax() as PropertyDeclarationSyntax).FirstOrDefault();

                // Fully qualify the value to avoid issues with namespaces.
                string? value = null;
                if (propertyDeclarationSyntax != null)
                {
                    if (propertyDeclarationSyntax.Initializer != null)
                    {
                        var sm = context.Compilation.GetSemanticModel(propertyDeclarationSyntax.Initializer.SyntaxTree);
                        value = propertyDeclarationSyntax.Initializer.Value.FullQualifiedSyntax(sm);
                    }
                    else
                    {
                        var propertyGet = propertyDeclarationSyntax.AccessorList?.Accessors
                            .Where(a => a.Keyword.IsKind(SyntaxKind.GetKeyword)).FirstOrDefault();
                        if (propertyGet != null)
                        {
                            if (propertyGet.ExpressionBody != null)
                            {
                                if (propertyGet.ExpressionBody.Expression is IdentifierNameSyntax identifierNameSyntax)
                                {
                                    var sm = context.Compilation.GetSemanticModel(identifierNameSyntax.SyntaxTree);
                                    var fieldSymbol = sm.GetSymbolInfo(identifierNameSyntax).Symbol as IFieldSymbol;
                                    EqualsValueClauseSyntax? initializer = fieldSymbol?.DeclaringSyntaxReferences
                                        .Select(r => r.GetSyntax())
                                        .OfType<VariableDeclaratorSyntax>()
                                        .Select(s => s.Initializer)
                                        .FirstOrDefault(i => i != null);

                                    if (initializer != null)
                                    {
                                        sm = context.Compilation.GetSemanticModel(initializer.SyntaxTree);
                                        value = initializer.Value.FullQualifiedSyntax(sm);
                                    }
                                }
                            }
                            else
                            {
                                var returns = propertyGet.DescendantNodes().OfType<ReturnStatementSyntax>();
                                if (returns.Count() == 1)
                                {
                                    // Generate only single return
                                    var returnStatementSyntax = returns.Single();
                                    if (returnStatementSyntax.Expression is IdentifierNameSyntax identifierNameSyntax)
                                    {
                                        var sm = context.Compilation.GetSemanticModel(identifierNameSyntax.SyntaxTree);
                                        var fieldSymbol = sm.GetSymbolInfo(identifierNameSyntax).Symbol as IFieldSymbol;
                                        EqualsValueClauseSyntax? initializer = fieldSymbol?.DeclaringSyntaxReferences
                                            .Select(r => r.GetSyntax())
                                            .OfType<VariableDeclaratorSyntax>()
                                            .Select(s => s.Initializer)
                                            .FirstOrDefault(i => i != null);

                                        if (initializer != null)
                                        {
                                            sm = context.Compilation.GetSemanticModel(initializer.SyntaxTree);
                                            value = initializer.Value.FullQualifiedSyntax(sm);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                exportedMembers.Add(new ExportedPropertyMetadata(
                    property.Name, marshalType.Value, propertyType, value));
            }

            foreach (var field in exportedFields)
            {
                if (field.IsStatic)
                {
                    Common.ReportExportedMemberIsStatic(context, field);
                    continue;
                }

                // TODO: We should still restore read-only fields after reloading assembly. Two possible ways: reflection or turn RestoreGodotObjectData into a constructor overload.
                // Ignore properties without a getter or without a setter. Godot properties must be both readable and writable.
                if (field.IsReadOnly)
                {
                    Common.ReportExportedMemberIsReadOnly(context, field);
                    continue;
                }

                var fieldType = field.Type;
                var marshalType = MarshalUtils.ConvertManagedTypeToMarshalType(fieldType, typeCache);

                if (marshalType == null)
                {
                    Common.ReportExportedMemberTypeNotSupported(context, field);
                    continue;
                }

                if (marshalType == MarshalType.GodotObjectOrDerived)
                {
                    if (!symbol.InheritsFrom("GodotSharp", "Godot.Node") &&
                        fieldType.InheritsFrom("GodotSharp", "Godot.Node"))
                    {
                        Common.ReportOnlyNodesShouldExportNodes(context, field);
                    }
                }

                EqualsValueClauseSyntax? initializer = field.DeclaringSyntaxReferences
                    .Select(r => r.GetSyntax())
                    .OfType<VariableDeclaratorSyntax>()
                    .Select(s => s.Initializer)
                    .FirstOrDefault(i => i != null);

                // This needs to be fully qualified to avoid issues with namespaces.
                string? value = null;
                if (initializer != null)
                {
                    var sm = context.Compilation.GetSemanticModel(initializer.SyntaxTree);
                    value = initializer.Value.FullQualifiedSyntax(sm);
                }

                exportedMembers.Add(new ExportedPropertyMetadata(
                    field.Name, marshalType.Value, fieldType, value));
            }

            // Generate GetGodotExportedProperties

            if (exportedMembers.Count > 0)
            {
                source.Append("#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword\n");

                const string dictionaryType =
                    "global::System.Collections.Generic.Dictionary<global::Godot.StringName, global::Godot.Variant>";

                source.Append("#if TOOLS\n");

                source.Append("    /// <summary>\n")
                    .Append("    /// Get the default values for all properties declared in this class.\n")
                    .Append("    /// This method is used by Godot to determine the value that will be\n")
                    .Append("    /// used by the inspector when resetting properties.\n")
                    .Append("    /// Do not call this method.\n")
                    .Append("    /// </summary>\n");

                source.Append("    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]\n");

                source.Append("    internal new static ");
                source.Append(dictionaryType);
                source.Append(" GetGodotPropertyDefaultValues()\n    {\n");

                source.Append("        var values = new ");
                source.Append(dictionaryType);
                source.Append("(");
                source.Append(exportedMembers.Count);
                source.Append(");\n");

                foreach (var exportedMember in exportedMembers)
                {
                    string defaultValueLocalName = string.Concat("__", exportedMember.Name, "_default_value");

                    source.Append("        ");
                    source.Append(exportedMember.TypeSymbol.FullQualifiedNameIncludeGlobal());
                    source.Append(" ");
                    source.Append(defaultValueLocalName);
                    source.Append(" = ");
                    source.Append(exportedMember.Value ?? "default");
                    source.Append(";\n");
                    source.Append("        values.Add(PropertyName.");
                    source.Append(exportedMember.Name);
                    source.Append(", ");
                    source.AppendManagedToVariantExpr(defaultValueLocalName,
                        exportedMember.TypeSymbol, exportedMember.Type);
                    source.Append(");\n");
                }

                source.Append("        return values;\n");
                source.Append("    }\n");

                source.Append("#endif // TOOLS\n");

                source.Append("#pragma warning restore CS0109\n");
            }

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

            if (hasNamespace)
            {
                source.Append("\n}\n");
            }

            context.AddSource(uniqueHint, SourceText.From(source.ToString(), Encoding.UTF8));
        }

        private struct ExportedPropertyMetadata
        {
            public ExportedPropertyMetadata(string name, MarshalType type, ITypeSymbol typeSymbol, string? value)
            {
                Name = name;
                Type = type;
                TypeSymbol = typeSymbol;
                Value = value;
            }

            public string Name { get; }
            public MarshalType Type { get; }
            public ITypeSymbol TypeSymbol { get; }
            public string? Value { get; }
        }
    }
}
