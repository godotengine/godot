using System;

namespace Godot
{
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public class ExportAttribute : Attribute
    {
        private PropertyHint hint;
        private string hintString;

        public ExportAttribute(PropertyHint hint = PropertyHint.None, string hintString = "")
        {
            this.hint = hint;
            this.hintString = hintString;
        }

        public ExportAttribute(string className)
        {
            if (ClassDB.ClassExists(className) || ScriptServer.IsGlobalClass(className))
            {
                this.hint = PropertyHint.ResourceType;
                this.hintString = className;
            }
            else
            {
                this.hint = PropertyHint.None;
                this.hintString = "";
            }
        }
    }
}
