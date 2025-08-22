extends Node2D

var count = 1


func _on_ShortcutButton_changed():
	print('Shortcut changed to ', $ShortcutButton.to_s())
	$Button.shortcut = $ShortcutButton.get_shortcut()


func _on_Button_pressed():
	print('hello world ', count)
	count += 1


func _on_show_dialog_pressed():
	$BottomPanelShortcuts.popup_centered()
