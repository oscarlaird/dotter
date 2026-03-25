// Lean compiler output
// Module: vec2_standalone_probe.Vec2Standalone
// Imports: public import Init
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
LEAN_EXPORT lean_object* l_sumVec2Boxed___boxed(lean_object*);
LEAN_EXPORT lean_object* l_sumVec2___boxed(lean_object*);
LEAN_EXPORT double sum_vec2_xy(double, double);
LEAN_EXPORT double sum_vec2_boxed(lean_object*);
double lean_float_add(double, double);
LEAN_EXPORT lean_object* l_sumVec2XY___boxed(lean_object*, lean_object*);
LEAN_EXPORT double l_sumVec2(lean_object*);
LEAN_EXPORT double l_sumVec2(lean_object* x_1) {
_start:
{
double x_2; double x_3; double x_4; 
x_2 = lean_ctor_get_float(x_1, 0);
x_3 = lean_ctor_get_float(x_1, 8);
x_4 = lean_float_add(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* l_sumVec2___boxed(lean_object* x_1) {
_start:
{
double x_2; lean_object* x_3; 
x_2 = l_sumVec2(x_1);
lean_dec_ref(x_1);
x_3 = lean_box_float(x_2);
return x_3;
}
}
LEAN_EXPORT double sum_vec2_boxed(lean_object* x_1) {
_start:
{
double x_2; 
x_2 = l_sumVec2(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* l_sumVec2Boxed___boxed(lean_object* x_1) {
_start:
{
double x_2; lean_object* x_3; 
x_2 = sum_vec2_boxed(x_1);
x_3 = lean_box_float(x_2);
return x_3;
}
}
LEAN_EXPORT double sum_vec2_xy(double x_1, double x_2) {
_start:
{
lean_object* x_3; double x_4; 
x_3 = lean_alloc_ctor(0, 0, 16);
lean_ctor_set_float(x_3, 0, x_1);
lean_ctor_set_float(x_3, 8, x_2);
x_4 = l_sumVec2(x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* l_sumVec2XY___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
double x_3; double x_4; double x_5; lean_object* x_6; 
x_3 = lean_unbox_float(x_1);
lean_dec_ref(x_1);
x_4 = lean_unbox_float(x_2);
lean_dec_ref(x_2);
x_5 = sum_vec2_xy(x_3, x_4);
x_6 = lean_box_float(x_5);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_vec2__standalone__probe_Vec2Standalone(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
