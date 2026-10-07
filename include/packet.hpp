#ifndef PACKET_HPP
#define PACKET_HPP

// Module 2: SIMD Abstraction Layer.
//
// Packet<T> is a fixed-width vector of `Packet<T>::size` scalars of type T. It is the ONLY place in
// the library where platform intrinsics may appear; everything above this layer (in particular the
// math evaluator) must be written against the Packet<T> API:
//
//   Packet<T>::load(ptr)    aligned load      (ptr must be aligned to the Packet's register width;
//   Packet<T>::loadu(ptr)   unaligned load     storage from matrix_storage.hpp always is)
//   p.store(ptr)            aligned store
//   p.storeu(ptr)           unaligned store
//   Packet<T>::broadcast(v) all lanes = v
//   p.reduce_add()          horizontal sum of all lanes
//   add / sub / mul / div   lane-wise arithmetic (free functions, found via ADL)
//   fma(a, b, c)            lane-wise a * b + c
//
// Backend selection (at compile time, from the target flags):
//   * AVX2 + FMA (x86_64): packet_avx2.hpp
//       float x8, double x4, int x8, complex<float> x4, complex<double> x2
//   * NEON (AArch64):      packet_neon.hpp
//       float x4, double x2, int x4, complex<float> x2, complex<double> x1
//   * otherwise:           the one-lane portable primary template below.
// Scalar types without a specialization (e.g. long double) also use the primary template.

#include <complex>
#include <concepts>

// Backend detection must be visible to the preprocessor, hence macros.
// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#if defined(__AVX2__) && defined(__FMA__)
#define MLINALG_SIMD_AVX2 1
#elif defined(__aarch64__) && defined(__ARM_NEON)
#define MLINALG_SIMD_NEON 1
#endif
// NOLINTEND(cppcoreguidelines-macro-usage)

namespace mlinalg {

namespace detail {

template <typename T> inline constexpr bool is_complex_v = false;

template <std::floating_point T> inline constexpr bool is_complex_v<std::complex<T>> = true;

// std::complex<T> is guaranteed to be layout-compatible with T[2] ([complex.numbers.general]),
// so N complex values can be viewed as 2N interleaved scalars.
template <std::floating_point T>
[[nodiscard]] inline auto as_scalars(const std::complex<T>* ptr) noexcept -> const T* {
    return reinterpret_cast<const T*>(ptr); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

template <std::floating_point T>
[[nodiscard]] inline auto as_scalars(std::complex<T>* ptr) noexcept -> T* {
    return reinterpret_cast<T*>(ptr); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

} // namespace detail

/// Scalar types a Packet can hold: real floating-point types, std::complex thereof, and int.
template <typename T>
concept PacketScalar = std::floating_point<T> || std::same_as<T, int> || detail::is_complex_v<T>;

/// Portable one-lane packet. Used when no SIMD specialization exists for T on the current target.
template <PacketScalar T> class Packet {
  public:
    using scalar_type = T;
    using native_type = T;

    static constexpr int size = 1;

    Packet() = default;
    explicit Packet(native_type value) noexcept;

    [[nodiscard]] auto native() const noexcept -> native_type;

    [[nodiscard]] static auto load(const T* ptr) noexcept -> Packet;
    [[nodiscard]] static auto loadu(const T* ptr) noexcept -> Packet;
    [[nodiscard]] static auto broadcast(T value) noexcept -> Packet;

    void store(T* ptr) const noexcept;
    void storeu(T* ptr) const noexcept;

    [[nodiscard]] auto reduce_add() const noexcept -> T;

  private:
    native_type m_value;
};

template <PacketScalar T>
[[nodiscard]] auto add(Packet<T> lhs, Packet<T> rhs) noexcept -> Packet<T>;

template <PacketScalar T>
[[nodiscard]] auto sub(Packet<T> lhs, Packet<T> rhs) noexcept -> Packet<T>;

template <PacketScalar T>
[[nodiscard]] auto mul(Packet<T> lhs, Packet<T> rhs) noexcept -> Packet<T>;

template <PacketScalar T>
[[nodiscard]] auto div(Packet<T> lhs, Packet<T> rhs) noexcept -> Packet<T>;

/// a * b + c. Not guaranteed to be fused in the portable fallback.
template <PacketScalar T>
[[nodiscard]] auto fma(Packet<T> a, Packet<T> b, Packet<T> c) noexcept -> Packet<T>;

} // namespace mlinalg

#if defined(MLINALG_SIMD_AVX2)
#include "packet_avx2.hpp" // IWYU pragma: export
#elif defined(MLINALG_SIMD_NEON)
#include "packet_neon.hpp" // IWYU pragma: export
#endif

#include "packet.ipp" // IWYU pragma: keep

#endif // PACKET_HPP
