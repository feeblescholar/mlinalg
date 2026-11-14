#ifndef GEMM_MICROKERNEL_IPP
#define GEMM_MICROKERNEL_IPP

// Definitions for microkernel.hpp. Do not include this file directly.

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace mlinalg::detail {

/// Calls f(std::integral_constant<std::size_t, I>{}) for I = 0 .. N-1, fully unrolled, so that
/// every index is a constant and arrays indexed by it can live in registers.
///
/// Forced inline: if the inliner leaves one level out of line, everything the body touches (the
/// microkernel's accumulators) goes through memory on every call. Seen with clang for int on NEON.
template <std::size_t N, typename F> [[gnu::always_inline]] constexpr void static_for(const F& f) {
    [&]<std::size_t... I> [[gnu::always_inline]] (std::index_sequence<I...>) {
        (f(std::integral_constant<std::size_t, I>{}), ...);
    }(std::make_index_sequence<N>{});
}

template <GemmScalar T, int Packets, int NR>
    requires(Packets > 0 && NR > 0)
void gemm_microkernel(Index kc, const T* ap, const T* bp, T* c, Index ldc) noexcept {
    using P = Packet<T>;

    constexpr auto mp    = static_cast<std::size_t>(Packets);
    constexpr auto nr    = static_cast<std::size_t>(NR);
    constexpr auto width = static_cast<std::size_t>(P::size);
    constexpr auto mr    = mp * width;

    assert(kc >= 0 && ldc >= static_cast<Index>(mr));

    // acc[j * mp + i] holds rows [i * width, (i + 1) * width) of column j of the tile. Packet's
    // default constructor is not user-provided, so {} zero-initializes every lane.
    std::array<P, mp * nr> acc{};

    for (Index k = 0; k < kc; ++k) {
        std::array<P, mp> a{};
        static_for<mp>([&](auto i) { a[i] = P::load(ap + (i * width)); });
        static_for<nr>([&](auto j) {
            const P b = P::broadcast(bp[j]);
            static_for<mp>([&](auto i) { acc[(j * mp) + i] = fma(a[i], b, acc[(j * mp) + i]); });
        });
        ap += mr;
        bp += nr;
    }

    static_for<nr>([&](auto j) {
        T* column = c + (static_cast<Index>(j) * ldc);
        static_for<mp>([&](auto i) {
            T* dst = column + (i * width);
            add(P::loadu(dst), acc[(j * mp) + i]).storeu(dst);
        });
    });
}

template <GemmScalar T, int Packets, int NR>
    requires(Packets > 0 && NR > 0)
void gemm_microkernel_edge(TileSize size, Index kc, const T* ap, const T* bp, T* c,
                           Index ldc) noexcept {
    constexpr Index mr = Index{Packets} * Packet<T>::size;
    assert(size.rows > 0 && size.rows <= mr && size.cols > 0 && size.cols <= NR &&
           ldc >= size.rows);

    std::array<T, static_cast<std::size_t>(mr * NR)> tile{};
    gemm_microkernel<T, Packets, NR>(kc, ap, bp, tile.data(), mr);

    for (Index j = 0; j < size.cols; ++j) {
        for (Index i = 0; i < size.rows; ++i) {
            c[(j * ldc) + i] += tile[(j * mr) + i];
        }
    }
}

} // namespace mlinalg::detail

#endif // GEMM_MICROKERNEL_IPP
