using System.Collections.Frozen;
using System.Collections.Generic;

namespace Godot.Bridge
{
    public sealed class ScriptMethodRegistry<T> :
        ScriptRegistry<T, ScriptMethod<GodotObject>, ScriptCache<ScriptMethod<GodotObject>>,
            ScriptMethodRegistry<T>>
        where T : GodotObject
    {
        private static List<string> _mostUsedMethods = [
            GodotObject.MethodName.Notification,
            Node.MethodName._Process,
            Node.MethodName._PhysicsProcess,
            CanvasItem.MethodName._Draw,
            Node.MethodName._Input,
            Node.MethodName._UnhandledInput,
            Control.MethodName._GuiInput,
            Node.MethodName._Ready,
            Node.MethodName._EnterTree,
            Node.MethodName._ExitTree,
        ];

        protected override ScriptCache<ScriptMethod<GodotObject>> InitializeCache((MethodKey, ScriptMethod<GodotObject>)[] methods)
        {
            return new ScriptCache<ScriptMethod<GodotObject>>(methods, _mostUsedMethods);
        }
    }
}
