Param (
	[string]$file = "",
	[string]$text = "",
	[string]$buttons = "OK:0",
	[string]$default = "",
	[switch]$nearmouse = $false,
	[switch]$center = $false,
	[string]$geometry = "",
	[int32]$timeout = 0,
	[string]$title = "Message"
)
Add-Type -assembly System.Windows.Forms

$global:Result = 0

$main_form = New-Object System.Windows.Forms.Form
$main_form.Text = $title

$geometry_data = $geometry.Split("+")
if ($geometry_data.Length -ge 1) {
	$size_data = $geometry_data[0].Split("x")
	if ($size_data.Length -eq 2) {
		$main_form.Width = $size_data[0]
		$main_form.Height = $size_data[1]
	}
}
if ($geometry_data.Length -eq 3) {
	$main_form.StartPosition = [System.Windows.Forms.FormStartPosition]::Manual
	$main_form.Location = New-Object System.Drawing.Point($geometry_data[1], $geometry_data[2])
}
if ($nearmouse) {
	$main_form.StartPosition = [System.Windows.Forms.FormStartPosition]::Manual
	$main_form.Location = System.Windows.Forms.Cursor.Position
}
if ($center) {
	$main_form.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
}

$main_form.SuspendLayout()

$button_panel = New-Object System.Windows.Forms.FlowLayoutPanel
$button_panel.SuspendLayout()
$button_panel.FlowDirection = [System.Windows.Forms.FlowDirection]::RightToLeft
$button_panel.Dock = [System.Windows.Forms.DockStyle]::Bottom
$button_panel.Autosize = $true

if ($file -ne "") {
	$text = [IO.File]::ReadAllText($file).replace("`n", "`r`n")
}

if ($text -ne "") {
	$text_box = New-Object System.Windows.Forms.TextBox
	$text_box.Multiline = $true
	$text_box.ReadOnly = $true
	$text_box.Autosize = $true
	$text_box.Text = $text
	$text_box.Select(0,0)
	$text_box.Dock = [System.Windows.Forms.DockStyle]::Fill
	$main_form.Controls.Add($text_box)
}

$buttons_array = $buttons.Split(",")
foreach ($button in $buttons_array) {
	$button_data = $button.Split(":")
	$button_ctl = New-Object System.Windows.Forms.Button
	if ($button_data.Length -eq 2) {
		$button_ctl.Tag = $button_data[1]
	} else {
		$button_ctl.Tag = 100 + $buttons_array.IndexOf($button)
	}
	if ($default -eq $button_data[0]) {
		$main_form.AcceptButton = $button_ctl
	}
	$button_ctl.Autosize = $true
	$button_ctl.Text = $button_data[0]
	$button_ctl.Add_Click(
		{
			Param($sender)
			$global:Result = $sender.Tag
			$main_form.Close()
		}
	)
	$button_panel.Controls.Add($button_ctl)
}
$main_form.Controls.Add($button_panel)

$button_panel.ResumeLayout($false)
$main_form.ResumeLayout($false)

if ($timeout -gt 0) {
	$timer = New-Object System.Windows.Forms.Timer
	$timer.Add_Tick(
		{
			$global:Result = 0
			$main_form.Close()
		}
	)
	$timer.Interval = $timeout
	$timer.Start()
}
$dlg_res = $main_form.ShowDialog()

[Environment]::Exit($global:Result)
