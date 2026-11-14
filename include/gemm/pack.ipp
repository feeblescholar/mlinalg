#ifndef GEMM_PACK_IPP
#define GEMM_PACK_IPP

// Definitions for pack.hpp. Do not include this file directly.

#include <algorithm>
#include <cassert>

namespace mlinalg::detail {

template <GemmScalar T>
constexpr auto StridedView<T>::operator()(Index row, Index col) const noexcept -> const T& {
    return data[(row * row_stride) + (col * col_stride)];
}

template <GemmScalar T>
constexpr auto StridedView<T>::offset(Index row, Index col) const noexcept -> StridedView {
    return {&(*this)(row, col), row_stride, col_stride};
}

template <GemmScalar T, int MR> constexpr auto packed_a_size(Index mc, Index kc) noexcept -> Index {
    return (mc + MR - 1) / MR * MR * kc;
}

template <GemmScalar T, int NR> constexpr auto packed_b_size(Index kc, Index nc) noexcept -> Index {
    return (nc + NR - 1) / NR * NR * kc;
}

template <GemmScalar T, int MR> void pack_a(StridedView<T> a, Index mc, Index kc, T* ap) noexcept {
    assert(mc >= 0 && kc >= 0);
    for (Index panel = 0; panel < mc; panel += MR) {
        const Index rows = std::min<Index>(MR, mc - panel);
        for (Index k = 0; k < kc; ++k) {
            for (Index i = 0; i < rows; ++i) {
                ap[i] = a(panel + i, k);
            }
            for (Index i = rows; i < MR; ++i) {
                ap[i] = T{};
            }
            ap += MR;
        }
    }
}

template <GemmScalar T, int NR> void pack_b(StridedView<T> b, Index kc, Index nc, T* bp) noexcept {
    assert(kc >= 0 && nc >= 0);
    for (Index panel = 0; panel < nc; panel += NR) {
        const Index cols = std::min<Index>(NR, nc - panel);
        for (Index k = 0; k < kc; ++k) {
            for (Index j = 0; j < cols; ++j) {
                bp[j] = b(k, panel + j);
            }
            for (Index j = cols; j < NR; ++j) {
                bp[j] = T{};
            }
            bp += NR;
        }
    }
}

} // namespace mlinalg::detail

#endif // GEMM_PACK_IPP
