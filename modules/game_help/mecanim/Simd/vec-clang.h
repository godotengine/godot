#pragma once

namespace math
{
namespace meta
{
    template<typename T> struct v4
    {
        typedef T  packed;

        template_decl(struct SWIZ, int SWZ)
        {
        };
        template_spec(struct SWIZ, SWZ_ANY)
        {
            static MATH_FORCEINLINE D f(const T &p) { return p; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_X, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xxxx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_X, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yxxx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_X, COMP_X, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zxxx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_X, COMP_X, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wxxx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_X, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xyxx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_X, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yyxx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Y, COMP_X, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zyxx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Y, COMP_X, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wyxx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Z, COMP_X, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xzxx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Z, COMP_X, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yzxx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Z, COMP_X, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zzxx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Z, COMP_X, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wzxx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_W, COMP_X, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xwxx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_W, COMP_X, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.ywxx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_W, COMP_X, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zwxx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_W, COMP_X, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wwxx; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_Y, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xxyx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_Y, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yxyx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_X, COMP_Y, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zxyx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_X, COMP_Y, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wxyx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_Y, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xyyx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_Y, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yyyx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Y, COMP_Y, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zyyx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Y, COMP_Y, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wyyx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Z, COMP_Y, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xzyx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Z, COMP_Y, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yzyx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Z, COMP_Y, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zzyx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Z, COMP_Y, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wzyx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_W, COMP_Y, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xwyx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_W, COMP_Y, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.ywyx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_W, COMP_Y, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zwyx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_W, COMP_Y, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wwyx; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_Z, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xxzx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_Z, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yxzx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_X, COMP_Z, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zxzx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_X, COMP_Z, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wxzx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_Z, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xyzx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_Z, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yyzx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Y, COMP_Z, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zyzx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Y, COMP_Z, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wyzx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Z, COMP_Z, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xzzx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Z, COMP_Z, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yzzx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Z, COMP_Z, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zzzx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Z, COMP_Z, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wzzx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_W, COMP_Z, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xwzx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_W, COMP_Z, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.ywzx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_W, COMP_Z, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zwzx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_W, COMP_Z, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wwzx; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_W, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xxwx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_W, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yxwx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_X, COMP_W, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zxwx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_X, COMP_W, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wxwx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_W, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xywx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_W, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yywx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Y, COMP_W, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zywx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Y, COMP_W, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wywx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Z, COMP_W, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xzwx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Z, COMP_W, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yzwx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Z, COMP_W, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zzwx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Z, COMP_W, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wzwx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_W, COMP_W, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xwwx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_W, COMP_W, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.ywwx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_W, COMP_W, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zwwx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_W, COMP_W, COMP_X))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wwwx; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_X, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xxxy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_X, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yxxy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_X, COMP_X, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zxxy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_X, COMP_X, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wxxy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_X, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xyxy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_X, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yyxy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Y, COMP_X, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zyxy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Y, COMP_X, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wyxy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Z, COMP_X, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xzxy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Z, COMP_X, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yzxy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Z, COMP_X, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zzxy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Z, COMP_X, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wzxy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_W, COMP_X, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xwxy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_W, COMP_X, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.ywxy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_W, COMP_X, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zwxy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_W, COMP_X, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wwxy; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_Y, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xxyy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_Y, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yxyy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_X, COMP_Y, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zxyy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_X, COMP_Y, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wxyy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_Y, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xyyy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_Y, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yyyy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Y, COMP_Y, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zyyy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Y, COMP_Y, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wyyy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Z, COMP_Y, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xzyy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Z, COMP_Y, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yzyy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Z, COMP_Y, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zzyy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Z, COMP_Y, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wzyy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_W, COMP_Y, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xwyy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_W, COMP_Y, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.ywyy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_W, COMP_Y, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zwyy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_W, COMP_Y, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wwyy; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_Z, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xxzy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_Z, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yxzy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_X, COMP_Z, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zxzy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_X, COMP_Z, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wxzy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_Z, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xyzy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_Z, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yyzy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Y, COMP_Z, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zyzy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Y, COMP_Z, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wyzy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Z, COMP_Z, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xzzy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Z, COMP_Z, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yzzy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Z, COMP_Z, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zzzy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Z, COMP_Z, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wzzy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_W, COMP_Z, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xwzy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_W, COMP_Z, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.ywzy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_W, COMP_Z, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zwzy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_W, COMP_Z, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wwzy; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_W, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xxwy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_W, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yxwy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_X, COMP_W, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zxwy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_X, COMP_W, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wxwy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_W, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xywy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_W, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yywy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Y, COMP_W, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zywy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Y, COMP_W, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wywy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Z, COMP_W, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xzwy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Z, COMP_W, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yzwy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Z, COMP_W, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zzwy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Z, COMP_W, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wzwy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_W, COMP_W, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xwwy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_W, COMP_W, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.ywwy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_W, COMP_W, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zwwy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_W, COMP_W, COMP_Y))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wwwy; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_X, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xxxz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_X, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yxxz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_X, COMP_X, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zxxz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_X, COMP_X, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wxxz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_X, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xyxz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_X, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yyxz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Y, COMP_X, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zyxz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Y, COMP_X, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wyxz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Z, COMP_X, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xzxz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Z, COMP_X, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yzxz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Z, COMP_X, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zzxz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Z, COMP_X, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wzxz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_W, COMP_X, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xwxz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_W, COMP_X, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.ywxz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_W, COMP_X, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zwxz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_W, COMP_X, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wwxz; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_Y, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xxyz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_Y, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yxyz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_X, COMP_Y, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zxyz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_X, COMP_Y, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wxyz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_Y, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xyyz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_Y, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yyyz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Y, COMP_Y, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zyyz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Y, COMP_Y, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wyyz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Z, COMP_Y, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xzyz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Z, COMP_Y, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yzyz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Z, COMP_Y, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zzyz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Z, COMP_Y, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wzyz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_W, COMP_Y, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xwyz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_W, COMP_Y, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.ywyz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_W, COMP_Y, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zwyz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_W, COMP_Y, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wwyz; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_Z, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xxzz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_Z, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yxzz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_X, COMP_Z, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zxzz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_X, COMP_Z, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wxzz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_Z, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xyzz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_Z, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yyzz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Y, COMP_Z, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zyzz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Y, COMP_Z, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wyzz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Z, COMP_Z, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xzzz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Z, COMP_Z, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yzzz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Z, COMP_Z, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zzzz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Z, COMP_Z, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wzzz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_W, COMP_Z, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xwzz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_W, COMP_Z, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.ywzz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_W, COMP_Z, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zwzz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_W, COMP_Z, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wwzz; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_W, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xxwz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_W, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yxwz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_X, COMP_W, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zxwz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_X, COMP_W, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wxwz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_W, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xywz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_W, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yywz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Y, COMP_W, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zywz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Y, COMP_W, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wywz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Z, COMP_W, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xzwz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Z, COMP_W, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yzwz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Z, COMP_W, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zzwz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Z, COMP_W, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wzwz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_W, COMP_W, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xwwz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_W, COMP_W, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.ywwz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_W, COMP_W, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zwwz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_W, COMP_W, COMP_Z))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wwwz; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_X, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xxxw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_X, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yxxw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_X, COMP_X, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zxxw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_X, COMP_X, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wxxw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_X, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xyxw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_X, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yyxw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Y, COMP_X, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zyxw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Y, COMP_X, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wyxw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Z, COMP_X, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xzxw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Z, COMP_X, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yzxw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Z, COMP_X, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zzxw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Z, COMP_X, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wzxw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_W, COMP_X, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xwxw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_W, COMP_X, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.ywxw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_W, COMP_X, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zwxw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_W, COMP_X, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wwxw; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_Y, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xxyw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_Y, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yxyw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_X, COMP_Y, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zxyw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_X, COMP_Y, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wxyw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_Y, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xyyw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_Y, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yyyw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Y, COMP_Y, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zyyw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Y, COMP_Y, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wyyw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Z, COMP_Y, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xzyw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Z, COMP_Y, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yzyw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Z, COMP_Y, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zzyw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Z, COMP_Y, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wzyw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_W, COMP_Y, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xwyw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_W, COMP_Y, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.ywyw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_W, COMP_Y, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zwyw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_W, COMP_Y, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wwyw; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_Z, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xxzw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_Z, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yxzw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_X, COMP_Z, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zxzw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_X, COMP_Z, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wxzw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_Z, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xyzw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_Z, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yyzw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Y, COMP_Z, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zyzw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Y, COMP_Z, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wyzw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Z, COMP_Z, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xzzw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Z, COMP_Z, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yzzw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Z, COMP_Z, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zzzw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Z, COMP_Z, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wzzw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_W, COMP_Z, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xwzw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_W, COMP_Z, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.ywzw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_W, COMP_Z, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zwzw; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_W, COMP_Z, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wwzw; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_W, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xxww; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_W, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yxww; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_X, COMP_W, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zxww; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_X, COMP_W, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wxww; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_W, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xyww; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_W, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yyww; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Y, COMP_W, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zyww; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Y, COMP_W, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wyww; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Z, COMP_W, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xzww; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Z, COMP_W, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yzww; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Z, COMP_W, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zzww; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_Z, COMP_W, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wzww; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_W, COMP_W, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xwww; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_W, COMP_W, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.ywww; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_W, COMP_W, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zwww; }
        };
        template_spec(struct SWIZ, SWZ(COMP_W, COMP_W, COMP_W, COMP_W))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.wwww; }
        };
    };

    template<typename T> struct v3
    {
        typedef T  packed;

        template_decl(struct SWIZ, int SWZ)
        {
        };
        template_spec(struct SWIZ, SWZ_ANY)
        {
            static MATH_FORCEINLINE T f(const T &p) { return p; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_X, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xxx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_X, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yxx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_X, COMP_X, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zxx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_X, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xyx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_X, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yyx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Y, COMP_X, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zyx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Z, COMP_X, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xzx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Z, COMP_X, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yzx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Z, COMP_X, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zzx; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_Y, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xxy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_Y, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yxy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_X, COMP_Y, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zxy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_Y, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xyy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_Y, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yyy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Y, COMP_Y, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zyy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Z, COMP_Y, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xzy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Z, COMP_Y, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yzy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Z, COMP_Y, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zzy; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_Z, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xxz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_Z, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yxz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_X, COMP_Z, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zxz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_Z, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xyz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_Z, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yyz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Y, COMP_Z, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zyz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Z, COMP_Z, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xzz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Z, COMP_Z, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yzz; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Z, COMP_Z, COMP_Z, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.zzz; }
        };
    };

    template<typename T> struct v2
    {
        typedef T  packed;

        template_decl(struct SWIZ, int SWZ)
        {
        };
        template_spec(struct SWIZ, SWZ_ANY)
        {
            static MATH_FORCEINLINE T f(const T &p) { return p; }
        };

        template_spec(struct SWIZ, SWZ(COMP_X, COMP_X, COMP_N, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_X, COMP_N, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yx; }
        };
        template_spec(struct SWIZ, SWZ(COMP_X, COMP_Y, COMP_N, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.xy; }
        };
        template_spec(struct SWIZ, SWZ(COMP_Y, COMP_Y, COMP_N, COMP_N))
        {
            static MATH_FORCEINLINE T f(const T &p) { return p.yy; }
        };
    };
}

#define _COMP_x COMP_X
#define _COMP_y COMP_Y
#define _COMP_z COMP_Z
#define _COMP_w COMP_W

    template<typename T, int c0, int c1, int c2, int c3> MATH_FORCEINLINE T _ext_swizzle4(const T &v)
    {
        return meta::v4<T>::f<SWZ(c0, c1, c2, c3)>(v);
    }

    template<typename T, int c0, int c1, int c2> MATH_FORCEINLINE T _ext_swizzle3(const T &v)
    {
        return meta::v3<T>::f<SWZ(c0, c1, c2, COMP_N)>(v);
    }

    template<typename T, int c0, int c1> MATH_FORCEINLINE T _ext_swizzle3(const T &v)
    {
        return meta::v2<T>::f<SWZ(c0, c1, COMP_N, COMP_N)>(v);
    }

#define swizzle_float4(v, c0, c1, c2, c3)   _ext_swizzle4<float4, _COMP_##c0, _COMP_##c1, _COMP_##c2, _COMP_##c3>(v)
#define swizzle_float3(v, c0, c1, c2)       _ext_swizzle3<float3, _COMP_##c0, _COMP_##c1, _COMP_##c2>(v)
#define swizzle_float2(v, c0, c1)           _ext_swizzle2<float2, _COMP_##c0, _COMP_##c1>(v)
}
