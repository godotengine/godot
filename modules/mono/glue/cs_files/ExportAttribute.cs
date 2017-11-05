using System;

namespace Godot
{
	[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
	public class ExportAttribute : Attribute
	{
		private int hint;
		private string hint_string;

		public ExportAttribute(int hint = GD.PROPERTY_HINT_NONE, string hint_string = "")
		{
			this.hint = hint;
			this.hint_string = hint_string;
		}
	}
}
