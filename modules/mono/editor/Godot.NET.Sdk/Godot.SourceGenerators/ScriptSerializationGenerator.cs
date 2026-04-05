using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace Godot.SourceGenerators
{
    [Generator]
    public class ScriptSerializationGenerator : ISourceGenerator
    {
        public void Initialize(GeneratorInitializationContext context)
        {
        }

        public void Execute(GeneratorExecutionContext context)
        {
            if (context.IsGodotSourceGeneratorDisabled("ScriptSerialization"))
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
            INamespaceSymbol namespaceSymbol = symbol.ContainingNamespace;
            string classNs = namespaceSymbol != null && !namespaceSymbol.IsGlobalNamespace ?
                namespaceSymbol.FullQualifiedNameOmitGlobal() :
                string.Empty;
            bool hasNamespace = classNs.Length != 0;

            bool isInnerClass = symbol.ContainingType != null;

            string uniqueHint = symbol.FullQualifiedNameOmitGlobal().SanitizeQualifiedNameForUniqueHint()
                                + "_ScriptSerialization.generated";

            var source = new StringBuilder();

            source.Append("using Godot;\n");
            source.Append("using Godot.NativeInterop;\n");
            source.Append("\n");

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
                    source.Append(containingType.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat));
                    source.Append("\n{\n");
                }
            }

            source.Append("partial class ");
            source.Append(symbol.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat));
            source.Append("\n{\n");

            var members = symbol.GetMembers();

            var propertySymbols = members
                .Where(s => !s.IsStatic && s.Kind == SymbolKind.Property)
                .Cast<IPropertySymbol>()
                .Where(s => !s.IsIndexer && s.ExplicitInterfaceImplementations.Length == 0);

            var fieldSymbols = members
                .Where(s => !s.IsStatic && s.Kind == SymbolKind.Field && !s.IsImplicitlyDeclared)
                .Cast<IFieldSymbol>();

            // TODO: We should still restore read-only properties after reloading assembly. Two possible ways: reflection or turn RestoreGodotObjectData into a constructor overload.
            // Ignore properties without a getter, without a setter or with an init-only setter. Godot properties must be both readable and writable.
            var godotClassProperties = propertySymbols.Where(property => !(property.IsReadOnly || property.IsWriteOnly || property.SetMethod!.IsInitOnly))
                .WhereIsGodotCompatibleType(typeCache)
                .ToArray();
            var godotClassFields = fieldSymbols.Where(property => !property.IsReadOnly)
                .WhereIsGodotCompatibleType(typeCache)
                .ToArray();

            var signalDelegateSymbols = members
                .Where(s => s.Kind == SymbolKind.NamedType)
                .Cast<INamedTypeSymbol>()
                .Where(namedTypeSymbol => namedTypeSymbol.TypeKind == TypeKind.Delegate)
                .Where(s => s.GetAttributes()
                    .Any(a => a.AttributeClass?.IsGodotSignalAttribute() ?? false));

            List<GodotSignalDelegateData> godotSignalDelegates = new();

            foreach (var signalDelegateSymbol in signalDelegateSymbols)
            {
                if (!signalDelegateSymbol.Name.EndsWith(ScriptSignalsGenerator.SignalDelegateSuffix))
                    continue;

                string signalName = signalDelegateSymbol.Name;
                signalName = signalName.Substring(0,
                    signalName.Length - ScriptSignalsGenerator.SignalDelegateSuffix.Length);

                var invokeMethodData = signalDelegateSymbol
                    .DelegateInvokeMethod?.HasGodotCompatibleSignature(typeCache);

                if (invokeMethodData == null)
                    continue;

                godotSignalDelegates.Add(new(signalName, signalDelegateSymbol, invokeMethodData.Value));
            }

            source.Append("    /// <inheritdoc/>\n");
            source.Append("    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]\n");
            source.Append(
                "    protected override void SaveGodotObjectData(global::Godot.Bridge.GodotSerializationInfo info)\n    {\n");
            source.Append("        base.SaveGodotObjectData(info);\n");

            // Save properties

            foreach (var property in godotClassProperties)
            {
                string propertyName = property.PropertySymbol.Name;

                source.Append("        info.AddProperty(PropertyName.@")
                    .Append(propertyName)
                    .Append(", ")
                    .AppendManagedToVariantExpr(string.Concat("this.@", propertyName),
                        property.PropertySymbol.Type, property.Type)
                    .Append(");\n");
            }

            // Save fields

            foreach (var field in godotClassFields)
            {
                string fieldName = field.FieldSymbol.Name;

                source.Append("        info.AddProperty(PropertyName.@")
                    .Append(fieldName)
                    .Append(", ")
                    .AppendManagedToVariantExpr(string.Concat("this.@", fieldName),
                        field.FieldSymbol.Type, field.Type)
                    .Append(");\n");
            }

            // Save signal events

            foreach (var signalDelegate in godotSignalDelegates)
            {
                string signalName = signalDelegate.Name;

                source.Append("        info.AddSignalEventDelegate(SignalName.@")
                    .Append(signalName)
                    .Append(", this.backing_")
                    .Append(signalName)
                    .Append(");\n");
            }

            source.Append("    }\n");

            source.Append("    /// <inheritdoc/>\n");
            source.Append("    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]\n");
            source.Append(
                "    protected override void RestoreGodotObjectData(global::Godot.Bridge.GodotSerializationInfo info)\n    {\n");
            source.Append("        base.RestoreGodotObjectData(info);\n");

            // Restore properties

            foreach (var property in godotClassProperties)
            {
                string propertyName = property.PropertySymbol.Name;

                source.Append("        if (info.TryGetProperty(PropertyName.@")
                    .Append(propertyName)
                    .Append(", out var _value_")
                    .Append(propertyName)
                    .Append("))\n")
                    .Append("            this.@")
                    .Append(propertyName)
                    .Append(" = ")
                    .AppendVariantToManagedExpr(string.Concat("_value_", propertyName),
                        property.PropertySymbol.Type, property.Type)
                    .Append(";\n");
            }

            // Restore fields

            foreach (var field in godotClassFields)
            {
                string fieldName = field.FieldSymbol.Name;

                source.Append("        if (info.TryGetProperty(PropertyName.@")
                    .Append(fieldName)
                    .Append(", out var _value_")
                    .Append(fieldName)
                    .Append("))\n")
                    .Append("            this.@")
                    .Append(fieldName)
                    .Append(" = ")
                    .AppendVariantToManagedExpr(string.Concat("_value_", fieldName),
                        field.FieldSymbol.Type, field.Type)
                    .Append(";\n");
            }

            // Restore signal events

            foreach (var signalDelegate in godotSignalDelegates)
            {
                string signalName = signalDelegate.Name;
                string signalDelegateQualifiedName = signalDelegate.DelegateSymbol.FullQualifiedNameIncludeGlobal();

                source.Append("        if (info.TryGetSignalEventDelegate<")
                    .Append(signalDelegateQualifiedName)
                    .Append(">(SignalName.@")
                    .Append(signalName)
                    .Append(", out var _value_")
                    .Append(signalName)
                    .Append("))\n")
                    .Append("            this.backing_")
                    .Append(signalName)
                    .Append(" = _value_")
                    .Append(signalName)
                    .Append(";\n");
            }

            source.Append("    }\n");

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
    }
}
