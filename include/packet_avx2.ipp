#ifndef PACKET_AVX2_IPP
#define PACKET_AVX2_IPP

// Definitions for packet_avx2.hpp. Do not include this file directly.

#include <complex>
#include <immintrin.h>

namespace mlinalg {

// ------------------------------------------------------------------------------------------------
// float x8
// ------------------------------------------------------------------------------------------------

inline Packet<float>::Packet(native_type value) noexcept : m_value(value) {}

inline auto Packet<float>::native() const noexcept -> native_type {
    return m_value;
}

inline auto Packet<float>::load(const float* ptr) noexcept -> Packet {
    return Packet(_mm256_load_ps(ptr));
}

inline auto Packet<float>::loadu(const float* ptr) noexcept -> Packet {
    return Packet(_mm256_loadu_ps(ptr));
}

inline auto Packet<float>::broadcast(float value) noexcept -> Packet {
    return Packet(_mm256_set1_ps(value));
}

inline void Packet<float>::store(float* ptr) const noexcept {
    _mm256_store_ps(ptr, m_value);
}

inline void Packet<float>::storeu(float* ptr) const noexcept {
    _mm256_storeu_ps(ptr, m_value);
}

inline auto Packet<float>::reduce_add() const noexcept -> float {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(m_value), _mm256_extractf128_ps(m_value, 1));
    sum        = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum        = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

inline auto add(Packet<float> lhs, Packet<float> rhs) noexcept -> Packet<float> {
    return Packet<float>(_mm256_add_ps(lhs.native(), rhs.native()));
}

inline auto sub(Packet<float> lhs, Packet<float> rhs) noexcept -> Packet<float> {
    return Packet<float>(_mm256_sub_ps(lhs.native(), rhs.native()));
}

inline auto mul(Packet<float> lhs, Packet<float> rhs) noexcept -> Packet<float> {
    return Packet<float>(_mm256_mul_ps(lhs.native(), rhs.native()));
}

inline auto div(Packet<float> lhs, Packet<float> rhs) noexcept -> Packet<float> {
    return Packet<float>(_mm256_div_ps(lhs.native(), rhs.native()));
}

inline auto fma(Packet<float> a, Packet<float> b, Packet<float> c) noexcept -> Packet<float> {
    return Packet<float>(_mm256_fmadd_ps(a.native(), b.native(), c.native()));
}

// ------------------------------------------------------------------------------------------------
// double x4
// ------------------------------------------------------------------------------------------------

inline Packet<double>::Packet(native_type value) noexcept : m_value(value) {}

inline auto Packet<double>::native() const noexcept -> native_type {
    return m_value;
}

inline auto Packet<double>::load(const double* ptr) noexcept -> Packet {
    return Packet(_mm256_load_pd(ptr));
}

inline auto Packet<double>::loadu(const double* ptr) noexcept -> Packet {
    return Packet(_mm256_loadu_pd(ptr));
}

inline auto Packet<double>::broadcast(double value) noexcept -> Packet {
    return Packet(_mm256_set1_pd(value));
}

inline void Packet<double>::store(double* ptr) const noexcept {
    _mm256_store_pd(ptr, m_value);
}

inline void Packet<double>::storeu(double* ptr) const noexcept {
    _mm256_storeu_pd(ptr, m_value);
}

inline auto Packet<double>::reduce_add() const noexcept -> double {
    __m128d sum = _mm_add_pd(_mm256_castpd256_pd128(m_value), _mm256_extractf128_pd(m_value, 1));
    sum         = _mm_add_sd(sum, _mm_unpackhi_pd(sum, sum));
    return _mm_cvtsd_f64(sum);
}

inline auto add(Packet<double> lhs, Packet<double> rhs) noexcept -> Packet<double> {
    return Packet<double>(_mm256_add_pd(lhs.native(), rhs.native()));
}

inline auto sub(Packet<double> lhs, Packet<double> rhs) noexcept -> Packet<double> {
    return Packet<double>(_mm256_sub_pd(lhs.native(), rhs.native()));
}

inline auto mul(Packet<double> lhs, Packet<double> rhs) noexcept -> Packet<double> {
    return Packet<double>(_mm256_mul_pd(lhs.native(), rhs.native()));
}

inline auto div(Packet<double> lhs, Packet<double> rhs) noexcept -> Packet<double> {
    return Packet<double>(_mm256_div_pd(lhs.native(), rhs.native()));
}

inline auto fma(Packet<double> a, Packet<double> b, Packet<double> c) noexcept -> Packet<double> {
    return Packet<double>(_mm256_fmadd_pd(a.native(), b.native(), c.native()));
}

// ------------------------------------------------------------------------------------------------
// int x8
// ------------------------------------------------------------------------------------------------

namespace detail {

// The AVX2 integer load/store intrinsics take __m256i pointers; this is the intended usage.
[[nodiscard]] inline auto as_m256i(const int* ptr) noexcept -> const __m256i* {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<const __m256i*>(ptr);
}

[[nodiscard]] inline auto as_m256i(int* ptr) noexcept -> __m256i* {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<__m256i*>(ptr);
}

} // namespace detail

inline Packet<int>::Packet(native_type value) noexcept : m_value(value) {}

inline auto Packet<int>::native() const noexcept -> native_type {
    return m_value;
}

inline auto Packet<int>::load(const int* ptr) noexcept -> Packet {
    return Packet(_mm256_load_si256(detail::as_m256i(ptr)));
}

inline auto Packet<int>::loadu(const int* ptr) noexcept -> Packet {
    return Packet(_mm256_loadu_si256(detail::as_m256i(ptr)));
}

inline auto Packet<int>::broadcast(int value) noexcept -> Packet {
    return Packet(_mm256_set1_epi32(value));
}

inline void Packet<int>::store(int* ptr) const noexcept {
    _mm256_store_si256(detail::as_m256i(ptr), m_value);
}

inline void Packet<int>::storeu(int* ptr) const noexcept {
    _mm256_storeu_si256(detail::as_m256i(ptr), m_value);
}

inline auto Packet<int>::reduce_add() const noexcept -> int {
    __m128i sum =
        _mm_add_epi32(_mm256_castsi256_si128(m_value), _mm256_extracti128_si256(m_value, 1));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E)); // swap 64-bit halves
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1)); // swap adjacent lanes
    return _mm_cvtsi128_si32(sum);
}

inline auto add(Packet<int> lhs, Packet<int> rhs) noexcept -> Packet<int> {
    return Packet<int>(_mm256_add_epi32(lhs.native(), rhs.native()));
}

inline auto sub(Packet<int> lhs, Packet<int> rhs) noexcept -> Packet<int> {
    return Packet<int>(_mm256_sub_epi32(lhs.native(), rhs.native()));
}

inline auto mul(Packet<int> lhs, Packet<int> rhs) noexcept -> Packet<int> {
    return Packet<int>(_mm256_mullo_epi32(lhs.native(), rhs.native()));
}

inline auto div(Packet<int> lhs, Packet<int> rhs) noexcept -> Packet<int> {
    // AVX2 has no integer division. Every int32 is exact in double and the rounding error of the
    // double quotient is smaller than its distance to the next integer, so truncating it is exact.
    const __m256i a    = lhs.native();
    const __m256i b    = rhs.native();
    const __m256d a_lo = _mm256_cvtepi32_pd(_mm256_castsi256_si128(a));
    const __m256d a_hi = _mm256_cvtepi32_pd(_mm256_extracti128_si256(a, 1));
    const __m256d b_lo = _mm256_cvtepi32_pd(_mm256_castsi256_si128(b));
    const __m256d b_hi = _mm256_cvtepi32_pd(_mm256_extracti128_si256(b, 1));
    const __m128i q_lo = _mm256_cvttpd_epi32(_mm256_div_pd(a_lo, b_lo)); // truncates
    const __m128i q_hi = _mm256_cvttpd_epi32(_mm256_div_pd(a_hi, b_hi));
    return Packet<int>(_mm256_inserti128_si256(_mm256_castsi128_si256(q_lo), q_hi, 1));
}

inline auto fma(Packet<int> a, Packet<int> b, Packet<int> c) noexcept -> Packet<int> {
    return Packet<int>(_mm256_add_epi32(_mm256_mullo_epi32(a.native(), b.native()), c.native()));
}

// ------------------------------------------------------------------------------------------------
// complex<float> x4
// ------------------------------------------------------------------------------------------------

inline detail::PacketCF::Packet(native_type value) noexcept : m_value(value) {}

inline auto detail::PacketCF::native() const noexcept -> native_type {
    return m_value;
}

inline auto detail::PacketCF::load(const scalar_type* ptr) noexcept -> Packet {
    return Packet(_mm256_load_ps(detail::as_scalars(ptr)));
}

inline auto detail::PacketCF::loadu(const scalar_type* ptr) noexcept -> Packet {
    return Packet(_mm256_loadu_ps(detail::as_scalars(ptr)));
}

inline auto detail::PacketCF::broadcast(scalar_type value) noexcept -> Packet {
    const auto re = value.real();
    const auto im = value.imag();
    return Packet(_mm256_setr_ps(re, im, re, im, re, im, re, im));
}

inline void detail::PacketCF::store(scalar_type* ptr) const noexcept {
    _mm256_store_ps(detail::as_scalars(ptr), m_value);
}

inline void detail::PacketCF::storeu(scalar_type* ptr) const noexcept {
    _mm256_storeu_ps(detail::as_scalars(ptr), m_value);
}

inline auto detail::PacketCF::reduce_add() const noexcept -> scalar_type {
    // re0 im0 re1 im1 (+) re2 im2 re3 im3
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(m_value), _mm256_extractf128_ps(m_value, 1));
    sum        = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    return {_mm_cvtss_f32(sum), _mm_cvtss_f32(_mm_movehdup_ps(sum))};
}

inline auto add(detail::PacketCF lhs, detail::PacketCF rhs) noexcept -> detail::PacketCF {
    return detail::PacketCF(_mm256_add_ps(lhs.native(), rhs.native()));
}

inline auto sub(detail::PacketCF lhs, detail::PacketCF rhs) noexcept -> detail::PacketCF {
    return detail::PacketCF(_mm256_sub_ps(lhs.native(), rhs.native()));
}

inline auto mul(detail::PacketCF lhs, detail::PacketCF rhs) noexcept -> detail::PacketCF {
    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    const __m256 a      = lhs.native();
    const __m256 b      = rhs.native();
    const __m256 a_re   = _mm256_moveldup_ps(a);       // ar ar
    const __m256 a_im   = _mm256_movehdup_ps(a);       // ai ai
    const __m256 b_swap = _mm256_permute_ps(b, 0xB1);  // bi br
    const __m256 cross  = _mm256_mul_ps(a_im, b_swap); // ai*bi  ai*br
    // even lanes: ar*br - ai*bi, odd lanes: ar*bi + ai*br
    return detail::PacketCF(_mm256_fmaddsub_ps(a_re, b, cross));
}

inline auto div(detail::PacketCF lhs, detail::PacketCF rhs) noexcept -> detail::PacketCF {
    // a / b = a * conj(b) / |b|^2  (no overflow-avoiding scaling, unlike std::complex
    // division)
    const __m256 a      = lhs.native();
    const __m256 b      = rhs.native();
    const __m256 b_re   = _mm256_moveldup_ps(b);       // br br
    const __m256 b_im   = _mm256_movehdup_ps(b);       // bi bi
    const __m256 a_swap = _mm256_permute_ps(a, 0xB1);  // ai ar
    const __m256 cross  = _mm256_mul_ps(b_im, a_swap); // bi*ai  bi*ar
    // even lanes: br*ar + bi*ai, odd lanes: br*ai - bi*ar
    const __m256 num = _mm256_fmsubadd_ps(b_re, a, cross);
    const __m256 den = _mm256_fmadd_ps(b_re, b_re, _mm256_mul_ps(b_im, b_im));
    return detail::PacketCF(_mm256_div_ps(num, den));
}

inline auto fma(detail::PacketCF a, detail::PacketCF b,
                detail::PacketCF c) noexcept -> detail::PacketCF {
    return add(mul(a, b), c);
}

// ------------------------------------------------------------------------------------------------
// complex<double> x2
// ------------------------------------------------------------------------------------------------

inline detail::PacketCD::Packet(native_type value) noexcept : m_value(value) {}

inline auto detail::PacketCD::native() const noexcept -> native_type {
    return m_value;
}

inline auto detail::PacketCD::load(const scalar_type* ptr) noexcept -> Packet {
    return Packet(_mm256_load_pd(detail::as_scalars(ptr)));
}

inline auto detail::PacketCD::loadu(const scalar_type* ptr) noexcept -> Packet {
    return Packet(_mm256_loadu_pd(detail::as_scalars(ptr)));
}

inline auto detail::PacketCD::broadcast(scalar_type value) noexcept -> Packet {
    const auto re = value.real();
    const auto im = value.imag();
    return Packet(_mm256_setr_pd(re, im, re, im));
}

inline void detail::PacketCD::store(scalar_type* ptr) const noexcept {
    _mm256_store_pd(detail::as_scalars(ptr), m_value);
}

inline void detail::PacketCD::storeu(scalar_type* ptr) const noexcept {
    _mm256_storeu_pd(detail::as_scalars(ptr), m_value);
}

inline auto detail::PacketCD::reduce_add() const noexcept -> scalar_type {
    const __m128d sum =
        _mm_add_pd(_mm256_castpd256_pd128(m_value), _mm256_extractf128_pd(m_value, 1));
    return {_mm_cvtsd_f64(sum), _mm_cvtsd_f64(_mm_unpackhi_pd(sum, sum))};
}

inline auto add(detail::PacketCD lhs, detail::PacketCD rhs) noexcept -> detail::PacketCD {
    return detail::PacketCD(_mm256_add_pd(lhs.native(), rhs.native()));
}

inline auto sub(detail::PacketCD lhs, detail::PacketCD rhs) noexcept -> detail::PacketCD {
    return detail::PacketCD(_mm256_sub_pd(lhs.native(), rhs.native()));
}

inline auto mul(detail::PacketCD lhs, detail::PacketCD rhs) noexcept -> detail::PacketCD {
    const __m256d a      = lhs.native();
    const __m256d b      = rhs.native();
    const __m256d a_re   = _mm256_movedup_pd(a);      // ar ar
    const __m256d a_im   = _mm256_permute_pd(a, 0xF); // ai ai
    const __m256d b_swap = _mm256_permute_pd(b, 0x5); // bi br
    const __m256d cross  = _mm256_mul_pd(a_im, b_swap);
    return detail::PacketCD(_mm256_fmaddsub_pd(a_re, b, cross));
}

inline auto div(detail::PacketCD lhs, detail::PacketCD rhs) noexcept -> detail::PacketCD {
    const __m256d a      = lhs.native();
    const __m256d b      = rhs.native();
    const __m256d b_re   = _mm256_movedup_pd(b);      // br br
    const __m256d b_im   = _mm256_permute_pd(b, 0xF); // bi bi
    const __m256d a_swap = _mm256_permute_pd(a, 0x5); // ai ar
    const __m256d cross  = _mm256_mul_pd(b_im, a_swap);
    const __m256d num    = _mm256_fmsubadd_pd(b_re, a, cross);
    const __m256d den    = _mm256_fmadd_pd(b_re, b_re, _mm256_mul_pd(b_im, b_im));
    return detail::PacketCD(_mm256_div_pd(num, den));
}

inline auto fma(detail::PacketCD a, detail::PacketCD b,
                detail::PacketCD c) noexcept -> detail::PacketCD {
    return add(mul(a, b), c);
}

} // namespace mlinalg

#endif // PACKET_AVX2_IPP
