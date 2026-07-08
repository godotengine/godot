const π = PI
@warning_ignore("confusable_identifier")
var ㄥ = π

func test():
	var փորձարկում = "test"
	prints("փորձարկում", փորձարկում)
	var امتحان = "test"
	prints("امتحان", امتحان)
	var পরীক্ষা = "test"
	prints("পরীক্ষা", পরীক্ষা)
	var тест = "test"
	prints("тест", тест)
	var जाँच = "test"
	prints("जाँच", जाँच)
	var 기준 = "test"
	prints("기준", 기준)
	var 测试 = "test"
	prints("测试", 测试)
	var テスト = "test"
	prints("テスト", テスト)
	var 試験 = "test"
	prints("試験", 試験)
	var പരീക്ഷ = "test"
	prints("പരീക്ഷ", പരീക്ഷ)
	var ทดสอบ = "test"
	prints("ทดสอบ", ทดสอบ)
	var δοκιμή = "test"
	prints("δοκιμή", δοκιμή)

	const d = 1.1
	_process(d)
	@warning_ignore("unsafe_call_argument")
	print(is_equal_approx(ㄥ, PI + (d * PI)))

func _process(Δ: float) -> void:
	ㄥ += Δ * π
