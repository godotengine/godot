using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace Godot.SourceGenerators
{
    [Generator]
    public class ScriptMemberInvokerGenerator : ISourceGenerator
    {
        public void Execute(GeneratorExecutionContext context)
        {
            if (context.AreGodotSourceGeneratorsDisabled())
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
            INamespaceSymbol namespaceSymbol = symbol.ContainingNamespace;
            string classNs = namespaceSymbol != null && !namespaceSymbol.IsGlobalNamespace ?
                namespaceSymbol.FullQualifiedName() :
                string.Empty;
            bool hasNamespace = classNs.Length != 0;

            bool isInnerClass = symbol.ContainingType != null;

            string uniqueHint = symbol.FullQualifiedName().SanitizeQualifiedNameForUniqueHint()
                                + "_ScriptMemberInvoker_Generated";

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

                while (containingType != null)
                {
                    source.Append("partial ");
                    source.Append(containingType.GetDeclarationKeyword());
                    source.Append(" ");
                    source.Append(containingType.NameWithTypeParameters());
                    source.Append("\n{\n");

                    containingType = containingType.ContainingType;
                }
            }

            source.Append("partial class ");
            source.Append(symbol.NameWithTypeParameters());
            source.Append("\n{\n");

            var members = symbol.GetMembers();

            var methodSymbols = members
                .Where(s => !s.IsStatic && s.Kind == SymbolKind.Method && !s.IsImplicitlyDeclared)
                .Cast<IMethodSymbol>()
                .Where(m => m.MethodKind == MethodKind.Ordinary);

            var propertySymbols = members
                .Where(s => !s.IsStatic && s.Kind == SymbolKind.Property)
                .Cast<IPropertySymbol>();

            var fieldSymbols = members
                .Where(s => !s.IsStatic && s.Kind == SymbolKind.Field && !s.IsImplicitlyDeclared)
                .Cast<IFieldSymbol>();

            var godotClassMethods = methodSymbols.WhereHasGodotCompatibleSignature(typeCache).ToArray();
            var godotClassProperties = propertySymbols.WhereIsGodotCompatibleType(typeCache).ToArray();
            var godotClassFields = fieldSymbols.WhereIsGodotCompatibleType(typeCache).ToArray();

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

            source.Append("    private partial class GodotInternal {\n");

            // Generate cached StringNames for methods and properties, for fast lookup

            // TODO: Move the generation of these cached StringNames to its own generator

            foreach (var method in godotClassMethods)
            {
                string methodName = method.Method.Name;
                source.Append("        public static readonly StringName MethodName_");
                source.Append(methodName);
                source.Append(" = \"");
                source.Append(methodName);
                source.Append("\";\n");
            }

            source.Append("    }\n"); // class GodotInternal

            // Generate InvokeGodotClassMethod

            if (godotClassMethods.Length > 0)
            {
                source.Append("    protected override bool InvokeGodotClassMethod(in godot_string_name method, ");
                source.Append("NativeVariantPtrArgs args, int argCount, out godot_variant ret)\n    {\n");

                foreach (var method in godotClassMethods)
                {
                    GenerateMethodInvoker(method, source);
                }

                source.Append("        return base.InvokeGodotClassMethod(method, args, argCount, out ret);\n");

                source.Append("    }\n");
            }

            // Generate HasGodotClassMethod

            if (godotClassMethods.Length > 0)
            {
                source.Append("    protected override bool HasGodotClassMethod(in godot_string_name method)\n    {\n");

                bool isFirstEntry = true;
                foreach (var method in godotClassMethods)
                {
                    GenerateHasMethodEntry(method, source, isFirstEntry);
                    isFirstEntry = false;
                }

                source.Append("        return base.HasGodotClassMethod(method);\n");

                source.Append("    }\n");
            }

            // Generate RaiseGodotClassSignalCallbacks

            if (godotSignalDelegates.Count > 0)
            {
                source.Append(
                    "    protected override void RaiseGodotClassSignalCallbacks(in godot_string_name signal, ");
                source.Append("NativeVariantPtrArgs args, int argCount)\n    {\n");

                foreach (var signal in godotSignalDelegates)
                {
                    GenerateSignalEventInvoker(signal, source);
                }

                source.Append("        base.RaiseGodotClassSignalCallbacks(signal, args, argCount);\n");

                source.Append("    }\n");
            }

            // Generate Set/GetGodotClassPropertyValue

            if (godotClassProperties.Length > 0 || godotClassFields.Length > 0)
            {
                bool isFirstEntry;

                // Setters

                bool allPropertiesAreReadOnly = godotClassFields.All(fi => fi.FieldSymbol.IsReadOnly) &&
                                                godotClassProperties.All(pi => pi.PropertySymbol.IsReadOnly);

                if (!allPropertiesAreReadOnly)
                {
                    source.Append("    protected override bool SetGodotClassPropertyValue(in godot_string_name name, ");
                    source.Append("in godot_variant value)\n    {\n");

                    isFirstEntry = true;
                    foreach (var property in godotClassProperties)
                    {
                        if (property.PropertySymbol.IsReadOnly)
                            continue;

                        GeneratePropertySetter(property.PropertySymbol.Name,
                            property.PropertySymbol.Type, property.Type, source, isFirstEntry);
                        isFirstEntry = false;
                    }

                    foreach (var field in godotClassFields)
                    {
                        if (field.FieldSymbol.IsReadOnly)
                            continue;

                        GeneratePropertySetter(field.FieldSymbol.Name,
                            field.FieldSymbol.Type, field.Type, source, isFirstEntry);
                        isFirstEntry = false;
                    }

                    source.Append("        return base.SetGodotClassPropertyValue(name, value);\n");

                    source.Append("    }\n");
                }

                // Getters

                source.Append("    protected override bool GetGodotClassPropertyValue(in godot_string_name name, ");
                source.Append("out godot_variant value)\n    {\n");

                isFirstEntry = true;
                foreach (var property in godotClassProperties)
                {
                    GeneratePropertyGetter(property.PropertySymbol.Name,
                        property.Type, source, isFirstEntry);
                    isFirstEntry = false;
                }

                foreach (var field in godotClassFields)
                {
                    GeneratePropertyGetter(field.FieldSymbol.Name,
                        field.Type, source, isFirstEntry);
                    isFirstEntry = false;
                }

                source.Append("        return base.GetGodotClassPropertyValue(name, out value);\n");

                source.Append("    }\n");
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

        private static void GenerateMethodInvoker(
            GodotMethodData method,
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
                source.Append("            var callRet = ");
            else
                source.Append("            ");

            source.Append(methodName);
            source.Append("(");

            for (int i = 0; i < method.ParamTypes.Length; i++)
            {
                if (i != 0)
                    source.Append(", ");

                source.AppendVariantToManagedExpr(string.Concat("args[", i.ToString(), "]"),
                    method.ParamTypeSymbols[i], method.ParamTypes[i]);
            }

            source.Append(");\n");

            if (method.RetType != null)
            {
                source.Append("            ret = ");

                source.AppendManagedToVariantExpr("callRet", method.RetType.Value);
                source.Append(";\n");

                source.Append("            return true;\n");
            }
            else
            {
                source.Append("            ret = default;\n");
                source.Append("            return true;\n");
            }

            source.Append("        }\n");
        }

        private static void GenerateSignalEventInvoker(
            GodotSignalDelegateData signal,
            StringBuilder source
        )
        {
            string signalName = signal.Name;
            var invokeMethodData = signal.InvokeMethodData;

            source.Append("        if (signal == GodotInternal.SignalName_");
            source.Append(signalName);
            source.Append(" && argCount == ");
            source.Append(invokeMethodData.ParamTypes.Length);
            source.Append(") {\n");
            source.Append("            backing_");
            source.Append(signalName);
            source.Append("?.Invoke(");

            for (int i = 0; i < invokeMethodData.ParamTypes.Length; i++)
            {
                if (i != 0)
                    source.Append(", ");

                source.AppendVariantToManagedExpr(string.Concat("args[", i.ToString(), "]"),
                    invokeMethodData.ParamTypeSymbols[i], invokeMethodData.ParamTypes[i]);
            }

            source.Append(");\n");

            source.Append("            return;\n");

            source.Append("        }\n");
        }

        private static void GeneratePropertySetter(
            string propertyMemberName,
            ITypeSymbol propertyTypeSymbol,
            MarshalType propertyMarshalType,
            StringBuilder source,
            bool isFirstEntry
        )
        {
            source.Append("        ");

            if (!isFirstEntry)
                source.Append("else ");

            source.Append("if (name == GodotInternal.PropName_")
                .Append(propertyMemberName)
                .Append(") {\n")
                .Append("            ")
                .Append(propertyMemberName)
                .Append(" = ")
                .AppendVariantToManagedExpr("value", propertyTypeSymbol, propertyMarshalType)
                .Append(";\n")
                .Append("            return true;\n")
                .Append("        }\n");
        }

        private static void GeneratePropertyGetter(
            string propertyMemberName,
            MarshalType propertyMarshalType,
            StringBuilder source,
            bool isFirstEntry
        )
        {
            source.Append("        ");

            if (!isFirstEntry)
                source.Append("else ");

            source.Append("if (name == GodotInternal.PropName_")
                .Append(propertyMemberName)
                .Append(") {\n")
                .Append("            value = ")
                .AppendManagedToVariantExpr(propertyMemberName, propertyMarshalType)
                .Append(";\n")
                .Append("            return true;\n")
                .Append("        }\n");
        }

        private static void GenerateHasMethodEntry(
            GodotMethodData method,
            StringBuilder source,
            bool isFirstEntry
        )
        {
            string methodName = method.Method.Name;

            source.Append("        ");
            if (!isFirstEntry)
                source.Append("else ");
            source.Append("if (method == GodotInternal.MethodName_");
            source.Append(methodName);
            source.Append(") {\n           return true;\n        }\n");
        }

        public void Initialize(GeneratorInitializationContext context)
        {
        }
    }
}
