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

    public override long _GetReturnIconType()
    {
        return 0;
    }

    public override long _GetInputPortCount()
    {
        return 0;
    }

    public override string _GetInputPortName(long port)
    {
        return "";
    }

    public override long _GetInputPortType(long port)
    {
        return 0;
    }

    public override long _GetOutputPortCount()
    {
        return 1;
    }

    public override string _GetOutputPortName(long port)
    {
        return "result";
    }

    public override long _GetOutputPortType(long port)
    {
        return 0;
    }

    public override string _GetCode(Godot.Collections.Array<string> inputVars, Godot.Collections.Array<string> outputVars, Shader.Mode mode, VisualShader.Type type)
    {
        return "";
    }
}
