using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace Godot.SourceGenerators
{
    [Generator]
    public class ScriptBoilerplateGenerator : ISourceGenerator
    {
        public void Execute(GeneratorExecutionContext context)
        {
            if (context.AreGodotSourceGeneratorsDisabled())
                return;

            // False positive for RS1024. We're already using `SymbolEqualityComparer.Default`...
#pragma warning disable RS1024
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
                                return true;
                            Common.ReportNonPartialGodotScriptClass(context, x.cds, x.symbol);
                            return false;
                        })
                        .Select(x => x.symbol)
                )
                .Distinct<INamedTypeSymbol>(SymbolEqualityComparer.Default)
                .ToArray();
#pragma warning restore RS1024

            if (godotClasses.Length > 0)
            {
                var typeCache = new MarshalUtils.TypeCache(context);

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
            string className = symbol.Name;

            INamespaceSymbol namespaceSymbol = symbol.ContainingNamespace;
            string classNs = namespaceSymbol != null && !namespaceSymbol.IsGlobalNamespace ?
                namespaceSymbol.FullQualifiedName() :
                string.Empty;
            bool hasNamespace = classNs.Length != 0;

            string uniqueName = hasNamespace ?
                classNs + "." + className + "_ScriptBoilerplate_Generated" :
                className + "_ScriptBoilerplate_Generated";

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

            source.Append("partial class ");
            source.Append(className);
            source.Append("\n{\n");

            var members = symbol.GetMembers();

            // TODO: Static static marshaling (no reflection, no runtime type checks)

            var methodSymbols = members
                .Where(s => s.Kind == SymbolKind.Method)
                .Cast<IMethodSymbol>()
                .Where(m => m.MethodKind == MethodKind.Ordinary && !m.IsImplicitlyDeclared);

            var propertySymbols = members
                .Where(s => s.Kind == SymbolKind.Property)
                .Cast<IPropertySymbol>();

            var fieldSymbols = members
                .Where(s => s.Kind == SymbolKind.Field)
                .Cast<IFieldSymbol>()
                .Where(p => !p.IsImplicitlyDeclared);

            var methods = WhereHasCompatibleGodotType(methodSymbols, typeCache).ToArray();
            var properties = WhereIsCompatibleGodotType(propertySymbols, typeCache).ToArray();
            var fields = WhereIsCompatibleGodotType(fieldSymbols, typeCache).ToArray();

            source.Append("    private class GodotInternal {\n");

            // Generate cached StringNames for methods and properties, for fast lookup

            foreach (var method in methods)
            {
                string methodName = method.Method.Name;
                source.Append("        public static readonly StringName MethodName_");
                source.Append(methodName);
                source.Append(" = \"");
                source.Append(methodName);
                source.Append("\";\n");
            }

            foreach (var property in properties)
            {
                string propertyName = property.Property.Name;
                source.Append("        public static readonly StringName PropName_");
                source.Append(propertyName);
                source.Append(" = \"");
                source.Append(propertyName);
                source.Append("\";\n");
            }

            foreach (var field in fields)
            {
                string fieldName = field.Field.Name;
                source.Append("        public static readonly StringName PropName_");
                source.Append(fieldName);
                source.Append(" = \"");
                source.Append(fieldName);
                source.Append("\";\n");
            }

            source.Append("    }\n");

            if (methods.Length > 0)
            {
                source.Append("    protected override bool InvokeGodotClassMethod(in godot_string_name method, ");
                source.Append("NativeVariantPtrArgs args, int argCount, out godot_variant ret)\n    {\n");

                foreach (var method in methods)
                {
                    GenerateMethodInvoker(method, source);
                }

                source.Append("        return base.InvokeGodotClassMethod(method, args, argCount, out ret);\n");

                source.Append("    }\n");
            }

            if (properties.Length > 0 || fields.Length > 0)
            {
                // Setters

                source.Append("    protected override bool SetGodotClassPropertyValue(in godot_string_name name, ");
                source.Append("in godot_variant value)\n    {\n");

                foreach (var property in properties)
                {
                    GeneratePropertySetter(property.Property.Name,
                        property.Property.Type.FullQualifiedName(), source);
                }

                foreach (var field in fields)
                {
                    GeneratePropertySetter(field.Field.Name,
                        field.Field.Type.FullQualifiedName(), source);
                }

                source.Append("        return base.SetGodotClassPropertyValue(name, value);\n");

                source.Append("    }\n");

                // Getters

                source.Append("    protected override bool GetGodotClassPropertyValue(in godot_string_name name, ");
                source.Append("out godot_variant value)\n    {\n");

                foreach (var property in properties)
                {
                    GeneratePropertyGetter(property.Property.Name, source);
                }

                foreach (var field in fields)
                {
                    GeneratePropertyGetter(field.Field.Name, source);
                }

                source.Append("        return base.GetGodotClassPropertyValue(name, out value);\n");

                source.Append("    }\n");
            }

            source.Append("}\n");

            if (hasNamespace)
            {
                source.Append("\n}\n");
            }

            context.AddSource(uniqueName, SourceText.From(source.ToString(), Encoding.UTF8));
        }

        private static void GenerateMethodInvoker(
            GodotMethodInfo method,
            StringBuilder source
        )
        {
            string methodName = method.Method.Name;

            source.Append("        if (method == GodotInternal.MethodName_");
            source.Append(methodName);
            source.Append(" && argCount == ");
            source.Append(method.ParamTypes.Length);
            source.Append(") {\n");

            if (method.RetType != null)
                source.Append("            object retBoxed = ");
            else
                source.Append("            ");

            source.Append(methodName);
            source.Append("(");

            for (int i = 0; i < method.ParamTypes.Length; i++)
            {
                if (i != 0)
                    source.Append(", ");

                // TODO: static marshaling (no reflection, no runtime type checks)

                string paramTypeQualifiedName = method.ParamTypeSymbols[i].FullQualifiedName();

                source.Append("(");
                source.Append(paramTypeQualifiedName);
                source.Append(")Marshaling.ConvertVariantToManagedObjectOfType(args[");
                source.Append(i);
                source.Append("], typeof(");
                source.Append(paramTypeQualifiedName);
                source.Append("))");
            }

            source.Append(");\n");

            if (method.RetType != null)
            {
                // TODO: static marshaling (no reflection, no runtime type checks)
                source.Append("            ret = Marshaling.ConvertManagedObjectToVariant(retBoxed);\n");
                source.Append("            return true;\n");
            }
            else
            {
                source.Append("            ret = default;\n");
                source.Append("            return true;\n");
            }

            source.Append("        }\n");
        }

        private static void GeneratePropertySetter(
            string propertyMemberName,
            string propertyTypeQualifiedName,
            StringBuilder source
        )
        {
            source.Append("        if (name == GodotInternal.PropName_");
            source.Append(propertyMemberName);
            source.Append(") {\n");

            source.Append("            ");
            source.Append(propertyMemberName);
            source.Append(" = ");

            // TODO: static marshaling (no reflection, no runtime type checks)

            source.Append("(");
            source.Append(propertyTypeQualifiedName);
            source.Append(")Marshaling.ConvertVariantToManagedObjectOfType(value, typeof(");
            source.Append(propertyTypeQualifiedName);
            source.Append("));\n");

            source.Append("            return true;\n");

            source.Append("        }\n");
        }

        private static void GeneratePropertyGetter(
            string propertyMemberName,
            StringBuilder source
        )
        {
            source.Append("        if (name == GodotInternal.PropName_");
            source.Append(propertyMemberName);
            source.Append(") {\n");

            // TODO: static marshaling (no reflection, no runtime type checks)

            source.Append("            value = Marshaling.ConvertManagedObjectToVariant(");
            source.Append(propertyMemberName);
            source.Append(");\n");
            source.Append("            return true;\n");

            source.Append("        }\n");
        }

        public void Initialize(GeneratorInitializationContext context)
        {
        }

        private struct GodotMethodInfo
        {
            public GodotMethodInfo(IMethodSymbol method, ImmutableArray<MarshalType> paramTypes,
                ImmutableArray<ITypeSymbol> paramTypeSymbols, MarshalType? retType)
            {
                Method = method;
                ParamTypes = paramTypes;
                ParamTypeSymbols = paramTypeSymbols;
                RetType = retType;
            }

            public IMethodSymbol Method { get; }
            public ImmutableArray<MarshalType> ParamTypes { get; }
            public ImmutableArray<ITypeSymbol> ParamTypeSymbols { get; }
            public MarshalType? RetType { get; }
        }

        private struct GodotPropertyInfo
        {
            public GodotPropertyInfo(IPropertySymbol property, MarshalType type)
            {
                Property = property;
                Type = type;
            }

            public IPropertySymbol Property { get; }
            public MarshalType Type { get; }
        }

        private struct GodotFieldInfo
        {
            public GodotFieldInfo(IFieldSymbol field, MarshalType type)
            {
                Field = field;
                Type = type;
            }

            public IFieldSymbol Field { get; }
            public MarshalType Type { get; }
        }

        private static IEnumerable<GodotMethodInfo> WhereHasCompatibleGodotType(
            IEnumerable<IMethodSymbol> methods,
            MarshalUtils.TypeCache typeCache
        )
        {
            foreach (var method in methods)
            {
                var retType = method.ReturnsVoid ?
                    null :
                    MarshalUtils.ConvertManagedTypeToVariantType(method.ReturnType, typeCache);

                if (retType == null && !method.ReturnsVoid)
                    continue;

                var parameters = method.Parameters;

                var paramTypes = parameters.Select(p =>
                        MarshalUtils.ConvertManagedTypeToVariantType(p.Type, typeCache))
                    .Where(t => t != null).Cast<MarshalType>().ToImmutableArray();

                if (parameters.Length > paramTypes.Length)
                    continue; // Some param types weren't compatible

                yield return new GodotMethodInfo(method, paramTypes, parameters
                    .Select(p => p.Type).ToImmutableArray(), retType);
            }
        }

        private static IEnumerable<GodotPropertyInfo> WhereIsCompatibleGodotType(
            IEnumerable<IPropertySymbol> properties,
            MarshalUtils.TypeCache typeCache
        )
        {
            foreach (var property in properties)
            {
                var marshalType = MarshalUtils.ConvertManagedTypeToVariantType(property.Type, typeCache);

                if (marshalType == null)
                    continue;

                yield return new GodotPropertyInfo(property, marshalType.Value);
            }
        }

        private static IEnumerable<GodotFieldInfo> WhereIsCompatibleGodotType(
            IEnumerable<IFieldSymbol> fields,
            MarshalUtils.TypeCache typeCache
        )
        {
            foreach (var field in fields)
            {
                var marshalType = MarshalUtils.ConvertManagedTypeToVariantType(field.Type, typeCache);

                if (marshalType == null)
                    continue;

                yield return new GodotFieldInfo(field, marshalType.Value);
            }
        }
    }
}
