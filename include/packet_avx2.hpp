#ifndef PACKET_AVX2_HPP
#define PACKET_AVX2_HPP

// AVX2 + FMA (x86_64) Packet specializations: 256-bit registers.
// Do not include this file directly; include packet.hpp.

#include <complex>
#include <immintrin.h>

static_assert(sizeof(int) == 4, "Packet<int> assumes a 32-bit int");

namespace mlinalg {

// ------------------------------------------------------------------------------------------------
// float x8
// ------------------------------------------------------------------------------------------------

template <> class Packet<float> {
  public:
    using scalar_type = float;
    using native_type = __m256;

    static constexpr int size = 8;

    Packet() = default;
    explicit Packet(native_type value) noexcept;

    [[nodiscard]] auto native() const noexcept -> native_type;

    [[nodiscard]] static auto load(const float* ptr) noexcept -> Packet;
    [[nodiscard]] static auto loadu(const float* ptr) noexcept -> Packet;
    [[nodiscard]] static auto broadcast(float value) noexcept -> Packet;

    void store(float* ptr) const noexcept;
    void storeu(float* ptr) const noexcept;

    [[nodiscard]] auto reduce_add() const noexcept -> float;

  private:
    native_type m_value;
};

[[nodiscard]] inline auto add(Packet<float> lhs, Packet<float> rhs) noexcept -> Packet<float>;
[[nodiscard]] inline auto sub(Packet<float> lhs, Packet<float> rhs) noexcept -> Packet<float>;
[[nodiscard]] inline auto mul(Packet<float> lhs, Packet<float> rhs) noexcept -> Packet<float>;
[[nodiscard]] inline auto div(Packet<float> lhs, Packet<float> rhs) noexcept -> Packet<float>;
[[nodiscard]] inline auto fma(Packet<float> a, Packet<float> b,
                              Packet<float> c) noexcept -> Packet<float>;

// ------------------------------------------------------------------------------------------------
// double x4
// ------------------------------------------------------------------------------------------------

template <> class Packet<double> {
  public:
    using scalar_type = double;
    using native_type = __m256d;

    static constexpr int size = 4;

    Packet() = default;
    explicit Packet(native_type value) noexcept;

    [[nodiscard]] auto native() const noexcept -> native_type;

    [[nodiscard]] static auto load(const double* ptr) noexcept -> Packet;
    [[nodiscard]] static auto loadu(const double* ptr) noexcept -> Packet;
    [[nodiscard]] static auto broadcast(double value) noexcept -> Packet;

    void store(double* ptr) const noexcept;
    void storeu(double* ptr) const noexcept;

    [[nodiscard]] auto reduce_add() const noexcept -> double;

  private:
    native_type m_value;
};

[[nodiscard]] inline auto add(Packet<double> lhs, Packet<double> rhs) noexcept -> Packet<double>;
[[nodiscard]] inline auto sub(Packet<double> lhs, Packet<double> rhs) noexcept -> Packet<double>;
[[nodiscard]] inline auto mul(Packet<double> lhs, Packet<double> rhs) noexcept -> Packet<double>;
[[nodiscard]] inline auto div(Packet<double> lhs, Packet<double> rhs) noexcept -> Packet<double>;
[[nodiscard]] inline auto fma(Packet<double> a, Packet<double> b,
                              Packet<double> c) noexcept -> Packet<double>;

// ------------------------------------------------------------------------------------------------
// int x8
// ------------------------------------------------------------------------------------------------

template <> class Packet<int> {
  public:
    using scalar_type = int;
    using native_type = __m256i;

    static constexpr int size = 8;

    Packet() = default;
    explicit Packet(native_type value) noexcept;

    [[nodiscard]] auto native() const noexcept -> native_type;

    [[nodiscard]] static auto load(const int* ptr) noexcept -> Packet;
    [[nodiscard]] static auto loadu(const int* ptr) noexcept -> Packet;
    [[nodiscard]] static auto broadcast(int value) noexcept -> Packet;

    void store(int* ptr) const noexcept;
    void storeu(int* ptr) const noexcept;

    [[nodiscard]] auto reduce_add() const noexcept -> int;

  private:
    native_type m_value;
};

[[nodiscard]] inline auto add(Packet<int> lhs, Packet<int> rhs) noexcept -> Packet<int>;
[[nodiscard]] inline auto sub(Packet<int> lhs, Packet<int> rhs) noexcept -> Packet<int>;
[[nodiscard]] inline auto mul(Packet<int> lhs, Packet<int> rhs) noexcept -> Packet<int>;
/// Truncates toward zero like built-in int division; division by zero is undefined.
[[nodiscard]] inline auto div(Packet<int> lhs, Packet<int> rhs) noexcept -> Packet<int>;
[[nodiscard]] inline auto fma(Packet<int> a, Packet<int> b, Packet<int> c) noexcept -> Packet<int>;

// ------------------------------------------------------------------------------------------------
// complex<float> x4  (interleaved: re0 im0 re1 im1 ...)
// ------------------------------------------------------------------------------------------------

template <> class Packet<std::complex<float>> {
  public:
    using scalar_type = std::complex<float>;
    using native_type = __m256;

    static constexpr int size = 4;

    Packet() = default;
    explicit Packet(native_type value) noexcept;

    [[nodiscard]] auto native() const noexcept -> native_type;

    [[nodiscard]] static auto load(const scalar_type* ptr) noexcept -> Packet;
    [[nodiscard]] static auto loadu(const scalar_type* ptr) noexcept -> Packet;
    [[nodiscard]] static auto broadcast(scalar_type value) noexcept -> Packet;

    void store(scalar_type* ptr) const noexcept;
    void storeu(scalar_type* ptr) const noexcept;

    [[nodiscard]] auto reduce_add() const noexcept -> scalar_type;

  private:
    native_type m_value;
};

namespace detail {
using PacketCF = Packet<std::complex<float>>;
} // namespace detail

[[nodiscard]] inline auto add(detail::PacketCF lhs,
                              detail::PacketCF rhs) noexcept -> detail::PacketCF;
[[nodiscard]] inline auto sub(detail::PacketCF lhs,
                              detail::PacketCF rhs) noexcept -> detail::PacketCF;
[[nodiscard]] inline auto mul(detail::PacketCF lhs,
                              detail::PacketCF rhs) noexcept -> detail::PacketCF;
[[nodiscard]] inline auto div(detail::PacketCF lhs,
                              detail::PacketCF rhs) noexcept -> detail::PacketCF;
[[nodiscard]] inline auto fma(detail::PacketCF a, detail::PacketCF b,
                              detail::PacketCF c) noexcept -> detail::PacketCF;

// ------------------------------------------------------------------------------------------------
// complex<double> x2  (interleaved: re0 im0 re1 im1)
// ------------------------------------------------------------------------------------------------

template <> class Packet<std::complex<double>> {
  public:
    using scalar_type = std::complex<double>;
    using native_type = __m256d;

    static constexpr int size = 2;

    Packet() = default;
    explicit Packet(native_type value) noexcept;

    [[nodiscard]] auto native() const noexcept -> native_type;

    [[nodiscard]] static auto load(const scalar_type* ptr) noexcept -> Packet;
    [[nodiscard]] static auto loadu(const scalar_type* ptr) noexcept -> Packet;
    [[nodiscard]] static auto broadcast(scalar_type value) noexcept -> Packet;

    void store(scalar_type* ptr) const noexcept;
    void storeu(scalar_type* ptr) const noexcept;

    [[nodiscard]] auto reduce_add() const noexcept -> scalar_type;

  private:
    native_type m_value;
};

namespace detail {
using PacketCD = Packet<std::complex<double>>;
} // namespace detail

[[nodiscard]] inline auto add(detail::PacketCD lhs,
                              detail::PacketCD rhs) noexcept -> detail::PacketCD;
[[nodiscard]] inline auto sub(detail::PacketCD lhs,
                              detail::PacketCD rhs) noexcept -> detail::PacketCD;
[[nodiscard]] inline auto mul(detail::PacketCD lhs,
                              detail::PacketCD rhs) noexcept -> detail::PacketCD;
[[nodiscard]] inline auto div(detail::PacketCD lhs,
                              detail::PacketCD rhs) noexcept -> detail::PacketCD;
[[nodiscard]] inline auto fma(detail::PacketCD a, detail::PacketCD b,
                              detail::PacketCD c) noexcept -> detail::PacketCD;

} // namespace mlinalg

#include "packet_avx2.ipp" // IWYU pragma: keep

#endif // PACKET_AVX2_HPP
