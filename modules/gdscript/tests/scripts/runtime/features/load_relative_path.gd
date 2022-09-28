func test():
    var file1 = load("./load_relative_path_external.notest.gd")
    var inst1 = file1.new()
    inst1.hello()

    var file2 = load("./some_dir/../load_relative_path_external.notest.gd")
    var inst2 = file2.new()
    inst2.hello()

    var file3 = load("../features/load_relative_path_external.notest.gd")
    var inst3 = file3.new()
    inst3.hello()

    var withdot = load("./.load_relative_path_dot.notest.gd")
    var instwithdot = withdot.new()
    instwithdot.hello()

    var withdot2 = load("./..load_relative_path_dot.notest.gd")
    var instwithdot2 = withdot2.new()
    instwithdot2.hello()
