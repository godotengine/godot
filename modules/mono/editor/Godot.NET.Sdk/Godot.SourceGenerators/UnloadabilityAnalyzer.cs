using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Operations;

namespace Godot.SourceGenerators
{
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class UnloadabilityAnalyzer : DiagnosticAnalyzer
    {
        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } =
            ImmutableArray.Create(
                Common.GDU0001_SubscriptionToExternalStaticEventRule,
                Common.GDU0002_GCHandleAllocRule,
                Common.GDU0003_ThreadPoolRegisterWaitForSingleObjectRule,
                Common.GDU0005_NewtonsoftJsonSerializationRule,
                Common.GDU0006_TypeDescriptorModificationRule,
                Common.GDU0007_ThreadCreationRule,
                Common.GDU0008_TimerCreationRule,
                Common.GDU0009_EncodingRegisterProviderRule,
                Common.GDU0010_TaskRunRule,
                Common.GDU0011_ThreadPoolQueueUserWorkItemRule);

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();

            context.RegisterOperationAction(AnalyzeEventAssignment, OperationKind.EventAssignment);
            context.RegisterOperationAction(AnalyzeInvocation, OperationKind.Invocation);
            context.RegisterOperationAction(AnalyzeObjectCreation, OperationKind.ObjectCreation);
        }

        private static bool IsInToolType(ISymbol symbol)
        {
            var type = symbol as INamedTypeSymbol ?? symbol.ContainingType;
            while (type != null)
            {
                if (type.GetAttributes().Any(a =>
                        a.AttributeClass?.ToDisplayString() == GodotClasses.ToolAttr))
                    return true;
                type = type.ContainingType;
            }
            return false;
        }

        private static bool IsRootAlcAssembly(IAssemblySymbol assembly)
        {
            var name = assembly?.Name;
            if (name == null) return false;

            return name == "mscorlib"
                   || name == "netstandard"
                   || name == "System"
                   || name.StartsWith("System.")
                   || name.StartsWith("Microsoft.")
                   || name == "GodotSharp"
                   || name == "GodotSharpEditor"
                   || name.StartsWith("Godot.");
        }

        private static bool IsWeakGCHandleAlloc(IInvocationOperation operation)
        {
            var method = operation.TargetMethod;
            if (method.Parameters.Length != 2)
                return false;

            var typeArg = operation.Arguments[1].Value;
            if (!typeArg.ConstantValue.HasValue)
                return false;

            // GCHandleType.Weak = 0, GCHandleType.WeakTrackResurrection = 1
            return typeArg.ConstantValue.Value is int handleType && (handleType == 0 || handleType == 1);
        }

        private static void AnalyzeEventAssignment(OperationAnalysisContext context)
        {
            if (!IsInToolType(context.ContainingSymbol))
                return;

            var operation = (IEventAssignmentOperation)context.Operation;

            if (!operation.Adds)
                return;

            if (!(operation.EventReference is IEventReferenceOperation eventRef))
                return;

            var eventSymbol = eventRef.Event;

            if (!eventSymbol.IsStatic)
                return;

            if (SymbolEqualityComparer.Default.Equals(
                    eventSymbol.ContainingAssembly, context.Compilation.Assembly))
                return;

            if (!IsRootAlcAssembly(eventSymbol.ContainingAssembly))
                return;

            context.ReportDiagnostic(Diagnostic.Create(
                Common.GDU0001_SubscriptionToExternalStaticEventRule,
                operation.Syntax.GetLocation(),
                eventSymbol.ContainingType?.ToDisplayString(),
                eventSymbol.Name));
        }

        private static void AnalyzeInvocation(OperationAnalysisContext context)
        {
            if (!IsInToolType(context.ContainingSymbol))
                return;

            var operation = (IInvocationOperation)context.Operation;
            var method = operation.TargetMethod;
            var containingType = method.ContainingType?.ToDisplayString();

            if (containingType == null)
                return;

            DiagnosticDescriptor? descriptor = null;

            if (containingType == "System.Runtime.InteropServices.GCHandle" && method.Name == "Alloc"
                && !IsWeakGCHandleAlloc(operation))
                descriptor = Common.GDU0002_GCHandleAllocRule;
            else if (containingType == "System.Threading.ThreadPool" && method.Name == "RegisterWaitForSingleObject")
                descriptor = Common.GDU0003_ThreadPoolRegisterWaitForSingleObjectRule;
            else if ((containingType == "Newtonsoft.Json.JsonConvert"
                      && (method.Name == "SerializeObject" || method.Name == "DeserializeObject"))
                     || (containingType == "Newtonsoft.Json.JsonSerializer"
                         && (method.Name == "Serialize" || method.Name == "Deserialize")))
                descriptor = Common.GDU0005_NewtonsoftJsonSerializationRule;
            else if (containingType == "System.ComponentModel.TypeDescriptor"
                     && (method.Name == "AddAttributes" || method.Name == "AddProvider"
                         || method.Name == "AddProviderTransparent" || method.Name == "Refresh"))
                descriptor = Common.GDU0006_TypeDescriptorModificationRule;
            else if (containingType == "System.Text.Encoding" && method.Name == "RegisterProvider")
                descriptor = Common.GDU0009_EncodingRegisterProviderRule;
            else if (containingType == "System.Threading.Tasks.Task" && method.Name == "Run")
                descriptor = Common.GDU0010_TaskRunRule;
            else if (containingType == "System.Threading.ThreadPool" && method.Name == "QueueUserWorkItem")
                descriptor = Common.GDU0011_ThreadPoolQueueUserWorkItemRule;

            if (descriptor != null)
            {
                var diagnostic = descriptor.Id == "GDU0006"
                    ? Diagnostic.Create(descriptor, operation.Syntax.GetLocation(), method.Name)
                    : Diagnostic.Create(descriptor, operation.Syntax.GetLocation());
                context.ReportDiagnostic(diagnostic);
            }
        }

        private static void AnalyzeObjectCreation(OperationAnalysisContext context)
        {
            if (!IsInToolType(context.ContainingSymbol))
                return;

            var operation = (IObjectCreationOperation)context.Operation;
            var createdType = operation.Type?.ToDisplayString();

            if (createdType == "System.Threading.Thread")
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    Common.GDU0007_ThreadCreationRule,
                    operation.Syntax.GetLocation()));
            }
            else if (createdType == "System.Threading.Timer" || createdType == "System.Timers.Timer")
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    Common.GDU0008_TimerCreationRule,
                    operation.Syntax.GetLocation()));
            }
        }
    }
}
