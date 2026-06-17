using System;

#nullable enable
namespace Godot
{
	public readonly struct Signal
	{
		private readonly long _targetId;
		private readonly string _name;

		public long TargetId => _targetId;
		public string Name => _name;

		public Signal(long targetId, string name)
		{
			_targetId = targetId;
			_name = name;
		}
	}
}
