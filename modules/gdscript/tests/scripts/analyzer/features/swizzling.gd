extends Node

func test():
    var v2 = Vector2(1, 2)

    Utils.check(v2.xy == Vector2(1, 2))
    Utils.check(v2.xx == Vector2(1, 1))
    Utils.check(v2.yy == Vector2(2, 2))
    Utils.check(v2.yx == Vector2(2, 1))

    var v3 = Vector3(1, 2, 3)

    Utils.check(v3.xyz == Vector3(1, 2, 3))
    Utils.check(v3.xxx == Vector3(1, 1, 1))
    Utils.check(v3.yyy == Vector3(2, 2, 2))
    Utils.check(v3.zzz == Vector3(3, 3, 3))

    Utils.check(v3.xyy == Vector3(1, 2, 2))
    Utils.check(v3.yxy == Vector3(2, 1, 2))
    Utils.check(v3.xxy == Vector3(1, 1, 2))
    Utils.check(v3.xyx == Vector3(1, 2, 1))

    Utils.check(v3.zyy == Vector3(3, 2, 2))
    Utils.check(v3.yzy == Vector3(2, 3, 2))
    Utils.check(v3.zzy == Vector3(3, 3, 2))
    Utils.check(v3.zyz == Vector3(3, 2, 3))

    Utils.check(v3.zxx == Vector3(3, 1, 1))
    Utils.check(v3.xzx == Vector3(1, 3, 1))
    Utils.check(v3.zzx == Vector3(3, 3, 1))
    Utils.check(v3.zxz == Vector3(3, 1, 3))

    Utils.check(v3.zyx == Vector3(3, 2, 1))
    Utils.check(v3.xzy == Vector3(1, 3, 2))
    Utils.check(v3.yzx == Vector3(2, 3, 1))
    Utils.check(v3.yxz == Vector3(2, 1, 3))

    print('ok')
