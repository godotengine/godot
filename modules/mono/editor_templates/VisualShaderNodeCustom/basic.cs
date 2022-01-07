// meta-description: Visual shader's node plugin template

using _BINDINGS_NAMESPACE_;
using System;

public partial class VisualShaderNode_CLASS_ : _BASE_
{
    public override string _GetName()
    {
        return "_CLASS_";
    }

    public override string _GetCategory()
    {
        return "";
    }

    public override string _GetDescription()
    {
        return "";
    }

    public override int _GetReturnIconType()
    {
        return 0;
    }

    public override int _GetInputPortCount()
    {
        return 0;
    }

    public override string _GetInputPortName(int port)
    {
        return "";
    }

    public override int _GetInputPortType(int port)
    {
        return 0;
    }

    public override int _GetOutputPortCount()
    {
        return 1;
    }

    public override string _GetOutputPortName(int port)
    {
        return "result";
    }

    public override int _GetOutputPortType(int port)
    {
        return 0;
    }

    public override string _GetGlobalCode(Shader.Mode mode)
    {
        return "";
    }

    public override string _GetCode(Godot.Collections.Array inputVars, Godot.Collections.Array outputVars, Shader.Mode mode, VisualShader.Type type)
    {
        return "";
    }
}
