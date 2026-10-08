#ifndef PACKET_NEON_IPP
#define PACKET_NEON_IPP

// Definitions for packet_neon.hpp. Do not include this file directly.

#include <arm_neon.h>
#include <complex>

namespace mlinalg {

// ------------------------------------------------------------------------------------------------
// float x4
// ------------------------------------------------------------------------------------------------

inline Packet<float>::Packet(native_type value) noexcept : m_value(value) {}

inline auto Packet<float>::native() const noexcept -> native_type {
    return m_value;
}

inline auto Packet<float>::load(const float* ptr) noexcept -> Packet {
    return Packet(vld1q_f32(ptr));
}

inline auto Packet<float>::loadu(const float* ptr) noexcept -> Packet {
    return Packet(vld1q_f32(ptr)); // NEON loads have no alignment requirement.
}

inline auto Packet<float>::broadcast(float value) noexcept -> Packet {
    return Packet(vdupq_n_f32(value));
}

inline void Packet<float>::store(float* ptr) const noexcept {
    vst1q_f32(ptr, m_value);
}

inline void Packet<float>::storeu(float* ptr) const noexcept {
    vst1q_f32(ptr, m_value);
}

inline auto Packet<float>::reduce_add() const noexcept -> float {
    return vaddvq_f32(m_value);
}

inline auto add(Packet<float> lhs, Packet<float> rhs) noexcept -> Packet<float> {
    return Packet<float>(vaddq_f32(lhs.native(), rhs.native()));
}

inline auto sub(Packet<float> lhs, Packet<float> rhs) noexcept -> Packet<float> {
    return Packet<float>(vsubq_f32(lhs.native(), rhs.native()));
}

inline auto mul(Packet<float> lhs, Packet<float> rhs) noexcept -> Packet<float> {
    return Packet<float>(vmulq_f32(lhs.native(), rhs.native()));
}

inline auto div(Packet<float> lhs, Packet<float> rhs) noexcept -> Packet<float> {
    return Packet<float>(vdivq_f32(lhs.native(), rhs.native()));
}

inline auto fma(Packet<float> a, Packet<float> b, Packet<float> c) noexcept -> Packet<float> {
    return Packet<float>(vfmaq_f32(c.native(), a.native(), b.native()));
}

// ------------------------------------------------------------------------------------------------
// double x2
// ------------------------------------------------------------------------------------------------

inline Packet<double>::Packet(native_type value) noexcept : m_value(value) {}

inline auto Packet<double>::native() const noexcept -> native_type {
    return m_value;
}

inline auto Packet<double>::load(const double* ptr) noexcept -> Packet {
    return Packet(vld1q_f64(ptr));
}

inline auto Packet<double>::loadu(const double* ptr) noexcept -> Packet {
    return Packet(vld1q_f64(ptr));
}

inline auto Packet<double>::broadcast(double value) noexcept -> Packet {
    return Packet(vdupq_n_f64(value));
}

inline void Packet<double>::store(double* ptr) const noexcept {
    vst1q_f64(ptr, m_value);
}

inline void Packet<double>::storeu(double* ptr) const noexcept {
    vst1q_f64(ptr, m_value);
}

inline auto Packet<double>::reduce_add() const noexcept -> double {
    return vaddvq_f64(m_value);
}

inline auto add(Packet<double> lhs, Packet<double> rhs) noexcept -> Packet<double> {
    return Packet<double>(vaddq_f64(lhs.native(), rhs.native()));
}

inline auto sub(Packet<double> lhs, Packet<double> rhs) noexcept -> Packet<double> {
    return Packet<double>(vsubq_f64(lhs.native(), rhs.native()));
}

inline auto mul(Packet<double> lhs, Packet<double> rhs) noexcept -> Packet<double> {
    return Packet<double>(vmulq_f64(lhs.native(), rhs.native()));
}

inline auto div(Packet<double> lhs, Packet<double> rhs) noexcept -> Packet<double> {
    return Packet<double>(vdivq_f64(lhs.native(), rhs.native()));
}

inline auto fma(Packet<double> a, Packet<double> b, Packet<double> c) noexcept -> Packet<double> {
    return Packet<double>(vfmaq_f64(c.native(), a.native(), b.native()));
}

// ------------------------------------------------------------------------------------------------
// int x4
// ------------------------------------------------------------------------------------------------

inline Packet<int>::Packet(native_type value) noexcept : m_value(value) {}

inline auto Packet<int>::native() const noexcept -> native_type {
    return m_value;
}

inline auto Packet<int>::load(const int* ptr) noexcept -> Packet {
    return Packet(vld1q_s32(ptr));
}

inline auto Packet<int>::loadu(const int* ptr) noexcept -> Packet {
    return Packet(vld1q_s32(ptr));
}

inline auto Packet<int>::broadcast(int value) noexcept -> Packet {
    return Packet(vdupq_n_s32(value));
}

inline void Packet<int>::store(int* ptr) const noexcept {
    vst1q_s32(ptr, m_value);
}

inline void Packet<int>::storeu(int* ptr) const noexcept {
    vst1q_s32(ptr, m_value);
}

inline auto Packet<int>::reduce_add() const noexcept -> int {
    return vaddvq_s32(m_value);
}

inline auto add(Packet<int> lhs, Packet<int> rhs) noexcept -> Packet<int> {
    return Packet<int>(vaddq_s32(lhs.native(), rhs.native()));
}

inline auto sub(Packet<int> lhs, Packet<int> rhs) noexcept -> Packet<int> {
    return Packet<int>(vsubq_s32(lhs.native(), rhs.native()));
}

inline auto mul(Packet<int> lhs, Packet<int> rhs) noexcept -> Packet<int> {
    return Packet<int>(vmulq_s32(lhs.native(), rhs.native()));
}

inline auto div(Packet<int> lhs, Packet<int> rhs) noexcept -> Packet<int> {
    // NEON has no integer division. Every int32 is exact in double and the rounding error of the
    // double quotient is smaller than its distance to the next integer, so truncating it is exact.
    const int32x4_t   a    = lhs.native();
    const int32x4_t   b    = rhs.native();
    const float64x2_t a_lo = vcvtq_f64_s64(vmovl_s32(vget_low_s32(a)));
    const float64x2_t a_hi = vcvtq_f64_s64(vmovl_high_s32(a));
    const float64x2_t b_lo = vcvtq_f64_s64(vmovl_s32(vget_low_s32(b)));
    const float64x2_t b_hi = vcvtq_f64_s64(vmovl_high_s32(b));
    const int64x2_t   q_lo = vcvtq_s64_f64(vdivq_f64(a_lo, b_lo)); // rounds toward zero
    const int64x2_t   q_hi = vcvtq_s64_f64(vdivq_f64(a_hi, b_hi));
    return Packet<int>(vcombine_s32(vmovn_s64(q_lo), vmovn_s64(q_hi)));
}

inline auto fma(Packet<int> a, Packet<int> b, Packet<int> c) noexcept -> Packet<int> {
    return Packet<int>(vmlaq_s32(c.native(), a.native(), b.native()));
}

// ------------------------------------------------------------------------------------------------
// complex<float> x2
// ------------------------------------------------------------------------------------------------

inline detail::PacketCF::Packet(native_type value) noexcept : m_value(value) {}

inline auto detail::PacketCF::native() const noexcept -> native_type {
    return m_value;
}

inline auto detail::PacketCF::load(const scalar_type* ptr) noexcept -> Packet {
    return Packet(vld1q_f32(detail::as_scalars(ptr)));
}

inline auto detail::PacketCF::loadu(const scalar_type* ptr) noexcept -> Packet {
    return Packet(vld1q_f32(detail::as_scalars(ptr)));
}

inline auto detail::PacketCF::broadcast(scalar_type value) noexcept -> Packet {
    const auto re = value.real();
    const auto im = value.imag();
    return Packet(float32x4_t{re, im, re, im});
}

inline void detail::PacketCF::store(scalar_type* ptr) const noexcept {
    vst1q_f32(detail::as_scalars(ptr), m_value);
}

inline void detail::PacketCF::storeu(scalar_type* ptr) const noexcept {
    vst1q_f32(detail::as_scalars(ptr), m_value);
}

inline auto detail::PacketCF::reduce_add() const noexcept -> scalar_type {
    const float32x2_t sum = vadd_f32(vget_low_f32(m_value), vget_high_f32(m_value));
    return {vget_lane_f32(sum, 0), vget_lane_f32(sum, 1)};
}

inline auto add(detail::PacketCF lhs, detail::PacketCF rhs) noexcept -> detail::PacketCF {
    return detail::PacketCF(vaddq_f32(lhs.native(), rhs.native()));
}

inline auto sub(detail::PacketCF lhs, detail::PacketCF rhs) noexcept -> detail::PacketCF {
    return detail::PacketCF(vsubq_f32(lhs.native(), rhs.native()));
}

inline auto mul(detail::PacketCF lhs, detail::PacketCF rhs) noexcept -> detail::PacketCF {
    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    const float32x4_t a       = lhs.native();
    const float32x4_t b       = rhs.native();
    const float32x4_t a_re    = vtrn1q_f32(a, a); // ar ar
    const float32x4_t a_im    = vtrn2q_f32(a, a); // ai ai
    const float32x4_t b_swap  = vrev64q_f32(b);   // bi br
    const float32x4_t sign    = {-1.0F, 1.0F, -1.0F, 1.0F};
    const float32x4_t partial = vmulq_f32(a_re, b); // ar*br  ar*bi
    return detail::PacketCF(vfmaq_f32(partial, a_im, vmulq_f32(b_swap, sign)));
}

inline auto div(detail::PacketCF lhs, detail::PacketCF rhs) noexcept -> detail::PacketCF {
    // a / b = a * conj(b) / |b|^2  (no overflow-avoiding scaling, unlike std::complex
    // division)
    const float32x4_t a       = lhs.native();
    const float32x4_t b       = rhs.native();
    const float32x4_t b_re    = vtrn1q_f32(b, b); // br br
    const float32x4_t b_im    = vtrn2q_f32(b, b); // bi bi
    const float32x4_t a_swap  = vrev64q_f32(a);   // ai ar
    const float32x4_t sign    = {1.0F, -1.0F, 1.0F, -1.0F};
    const float32x4_t partial = vmulq_f32(b_re, a); // br*ar  br*ai
    const float32x4_t num     = vfmaq_f32(partial, b_im, vmulq_f32(a_swap, sign));
    const float32x4_t den     = vfmaq_f32(vmulq_f32(b_re, b_re), b_im, b_im);
    return detail::PacketCF(vdivq_f32(num, den));
}

inline auto fma(detail::PacketCF a, detail::PacketCF b,
                detail::PacketCF c) noexcept -> detail::PacketCF {
    return add(mul(a, b), c);
}

// ------------------------------------------------------------------------------------------------
// complex<double> x1
// ------------------------------------------------------------------------------------------------

inline detail::PacketCD::Packet(native_type value) noexcept : m_value(value) {}

inline auto detail::PacketCD::native() const noexcept -> native_type {
    return m_value;
}

inline auto detail::PacketCD::load(const scalar_type* ptr) noexcept -> Packet {
    return Packet(vld1q_f64(detail::as_scalars(ptr)));
}

inline auto detail::PacketCD::loadu(const scalar_type* ptr) noexcept -> Packet {
    return Packet(vld1q_f64(detail::as_scalars(ptr)));
}

inline auto detail::PacketCD::broadcast(scalar_type value) noexcept -> Packet {
    const auto re = value.real();
    const auto im = value.imag();
    return Packet(float64x2_t{re, im});
}

inline void detail::PacketCD::store(scalar_type* ptr) const noexcept {
    vst1q_f64(detail::as_scalars(ptr), m_value);
}

inline void detail::PacketCD::storeu(scalar_type* ptr) const noexcept {
    vst1q_f64(detail::as_scalars(ptr), m_value);
}

inline auto detail::PacketCD::reduce_add() const noexcept -> scalar_type {
    return {vgetq_lane_f64(m_value, 0), vgetq_lane_f64(m_value, 1)};
}

inline auto add(detail::PacketCD lhs, detail::PacketCD rhs) noexcept -> detail::PacketCD {
    return detail::PacketCD(vaddq_f64(lhs.native(), rhs.native()));
}

inline auto sub(detail::PacketCD lhs, detail::PacketCD rhs) noexcept -> detail::PacketCD {
    return detail::PacketCD(vsubq_f64(lhs.native(), rhs.native()));
}

inline auto mul(detail::PacketCD lhs, detail::PacketCD rhs) noexcept -> detail::PacketCD {
    const float64x2_t a       = lhs.native();
    const float64x2_t b       = rhs.native();
    const float64x2_t a_re    = vdupq_laneq_f64(a, 0); // ar ar
    const float64x2_t a_im    = vdupq_laneq_f64(a, 1); // ai ai
    const float64x2_t b_swap  = vextq_f64(b, b, 1);    // bi br
    const float64x2_t sign    = {-1.0, 1.0};
    const float64x2_t partial = vmulq_f64(a_re, b);
    return detail::PacketCD(vfmaq_f64(partial, a_im, vmulq_f64(b_swap, sign)));
}

inline auto div(detail::PacketCD lhs, detail::PacketCD rhs) noexcept -> detail::PacketCD {
    const float64x2_t a       = lhs.native();
    const float64x2_t b       = rhs.native();
    const float64x2_t b_re    = vdupq_laneq_f64(b, 0);
    const float64x2_t b_im    = vdupq_laneq_f64(b, 1);
    const float64x2_t a_swap  = vextq_f64(a, a, 1); // ai ar
    const float64x2_t sign    = {1.0, -1.0};
    const float64x2_t partial = vmulq_f64(b_re, a);
    const float64x2_t num     = vfmaq_f64(partial, b_im, vmulq_f64(a_swap, sign));
    const float64x2_t den     = vfmaq_f64(vmulq_f64(b_re, b_re), b_im, b_im);
    return detail::PacketCD(vdivq_f64(num, den));
}

inline auto fma(detail::PacketCD a, detail::PacketCD b,
                detail::PacketCD c) noexcept -> detail::PacketCD {
    return add(mul(a, b), c);
}

} // namespace mlinalg

#endif // PACKET_NEON_IPP
