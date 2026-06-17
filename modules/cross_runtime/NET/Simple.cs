/*
using Godot;
using Timer = Godot.Timer;

public partial class Main : Control
{
    public Main(ulong id) : base(id) { }
    private Button _targetButton;
    private Label _scoreLabel;
    private Timer _gameTimer;
    private int _score = 0;
    private readonly Random _random = new Random();

    public override void _Ready()
    {
        GD.Print("This is ready");
        _targetButton = this.GetNode<Button>("TargetButton");
        _scoreLabel = this.GetNode<Label>("ScoreLabel");
        _gameTimer = this.GetNode<Timer>("GameTimer");

        _targetButton.Pressed += OnTargetButtonPressed;
        _gameTimer.Timeout += OnGameTimerTimeout;

        _scoreLabel.Text = "Score: 0";

        // Wait one frame so the button has a real size before moving it
        CallDeferred(nameof(MoveButtonRandomly));
    }

    private void OnTargetButtonPressed()
    {
        _score++;
        _scoreLabel.Text = $"Score: {_score}";
        MoveButtonRandomly();
    }

    private void OnGameTimerTimeout()
    {
        MoveButtonRandomly();
    }

    

    private void MoveButtonRandomly()
    {
        Vector2 viewportSize = GetViewportRect().Size;

        Vector2 buttonSize = _targetButton.Size;
        if (buttonSize == Vector2.Zero)
        {
            buttonSize = _targetButton.GetCombinedMinimumSize();
        }

        float maxX = Math.Max(0, viewportSize.X - buttonSize.X);
        float maxY = Math.Max(0, viewportSize.Y - buttonSize.Y);

        float randomX = (float)_random.NextDouble() * maxX;
        float randomY = (float)_random.NextDouble() * maxY;

        _targetButton.Position = new Vector2(randomX, randomY);
    }
}
 */