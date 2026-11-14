#ifndef GEMM_PACK_HPP
#define GEMM_PACK_HPP

// Module 5: GEMM packing - copies blocks of the operands into the layouts the microkernel reads
// (see microkernel.hpp), so that it streams through contiguous memory whatever the source layout.
//
//   * pack_a: an mc x kc block of A becomes ceil(mc / mr) panels of kc columns of mr coefficients.
//   * pack_b: a kc x nc block of B becomes ceil(nc / nr) panels of kc rows of nr coefficients.
//
// The last panel is zero-padded to a full mr (nr) when mc (nc) is not a multiple of it, so the
// microkernel always works on full slivers; gemm_microkernel_edge discards the padded results.
//
// Sources are read through a StridedView, so a transposed operand is packed by swapping its strides
// and needs no separate code path.

#include "../matrix_storage.hpp"
#include "microkernel.hpp"

namespace mlinalg::detail {

/// Read-only view of a matrix whose coefficient (i, j) is at data[i * row_stride + j * col_stride].
/// A column-major matrix with leading dimension ld is {data, 1, ld}; {data, ld, 1} is its
/// transpose.
template <GemmScalar T> struct StridedView {
    const T* data;
    Index    row_stride;
    Index    col_stride;

    [[nodiscard]] constexpr auto operator()(Index row, Index col) const noexcept -> const T&;

    /// The view of the same matrix whose (0, 0) is this view's (row, col).
    [[nodiscard]] constexpr auto offset(Index row, Index col) const noexcept -> StridedView;
};

/// Number of coefficients pack_a writes for an mc x kc block.
template <GemmScalar T, int MR = gemm_kernel_shape<T>::mr>
[[nodiscard]] constexpr auto packed_a_size(Index mc, Index kc) noexcept -> Index;

/// Number of coefficients pack_b writes for a kc x nc block.
template <GemmScalar T, int NR = gemm_kernel_shape<T>::nr>
[[nodiscard]] constexpr auto packed_b_size(Index kc, Index nc) noexcept -> Index;

/// Packs a[0:mc, 0:kc] into ap, which must hold packed_a_size(mc, kc) coefficients and be aligned
/// to the Packet register width (aligned_allocator storage is).
template <GemmScalar T, int MR = gemm_kernel_shape<T>::mr>
void pack_a(StridedView<T> a, Index mc, Index kc, T* ap) noexcept;

/// Packs b[0:kc, 0:nc] into bp, which must hold packed_b_size(kc, nc) coefficients.
template <GemmScalar T, int NR = gemm_kernel_shape<T>::nr>
void pack_b(StridedView<T> b, Index kc, Index nc, T* bp) noexcept;

} // namespace mlinalg::detail

#include "pack.ipp" // IWYU pragma: keep

#endif // GEMM_PACK_HPP
