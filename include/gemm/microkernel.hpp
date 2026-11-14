#ifndef GEMM_MICROKERNEL_HPP
#define GEMM_MICROKERNEL_HPP

// Module 5: GEMM microkernel - the innermost loop of the blocked matrix product.
//
// The microkernel updates one mr x nr tile of C with the product of a packed sliver of A and a
// packed sliver of B:
//
//   C[0:mr, 0:nr] += Ap[0:mr, 0:kc] * Bp[0:kc, 0:nr]
//
// The tile is held in mr / Packet<T>::size * nr packet accumulators for the whole kc loop, and C is
// only read and written once at the end. Each step of the kc loop loads one column of Ap as
// packets, broadcasts each of the nr coefficients of one row of Bp and accumulates with fma.
//
// Packed layouts (produced by the packing routines):
//   * Ap: kc consecutive columns of mr coefficients, i.e. Ap[i, k] at ap[k * mr + i]. Must be
//     aligned to the Packet register width; mr is a whole number of packets, so every column is.
//   * Bp: kc consecutive rows of nr coefficients, i.e. Bp[k, j] at bp[k * nr + j]. No alignment
//     requirement, since its coefficients are only broadcast.
//   * C: column-major with leading dimension ldc >= mr; no alignment requirement.

#include "../matrix_storage.hpp"
#include "../packet.hpp"

namespace mlinalg::detail {

/// Scalar types the GEMM kernels handle. Complex products use the coefficient-wise path.
template <typename T>
concept GemmScalar = PacketScalar<T> && !is_complex_v<T>;

/// Register tile of the microkernel for T on the current target: mr = packets * Packet<T>::size
/// rows by nr columns. The accumulators, one column of Ap and a broadcast coefficient of Bp must
/// all fit in the vector register file.
template <GemmScalar T> struct gemm_kernel_shape {
  private:
    static constexpr bool simd = Packet<T>::size > 1;

  public:
#if defined(MLINALG_SIMD_AVX2)
    // 16 ymm registers: 2 x 6 = 12 accumulators, 2 for Ap, 1 for the broadcast.
    static constexpr int packets = simd ? 2 : 4;
    static constexpr int nr      = simd ? 6 : 4;
#elif defined(MLINALG_SIMD_NEON)
    // 32 v registers: 4 x 6 = 24 accumulators, 4 for Ap, 1 for the broadcast.
    static constexpr int packets = 4;
    static constexpr int nr      = simd ? 6 : 4;
#else
    static constexpr int packets = 4;
    static constexpr int nr      = 4;
#endif
    static constexpr int mr = packets * Packet<T>::size;
};

/// C[0:MR, 0:NR] += Ap * Bp over kc steps, with MR = Packets * Packet<T>::size. See the layouts
/// above. kc may be 0, in which case C is left unchanged.
template <GemmScalar T, int Packets = gemm_kernel_shape<T>::packets,
          int NR = gemm_kernel_shape<T>::nr>
    requires(Packets > 0 && NR > 0)
void gemm_microkernel(Index kc, const T* ap, const T* bp, T* c, Index ldc) noexcept;

/// Extent of a partial tile; a struct so that rows and cols can't be swapped at the call site.
struct TileSize {
    Index rows;
    Index cols;
};

/// C[0:rows, 0:cols] += (Ap * Bp)[0:rows, 0:cols] for a tile at the bottom or right edge of C,
/// where 0 < rows <= MR and 0 < cols <= NR. Ap and Bp are full, zero-padded slivers; the kernel
/// runs on a local tile and only its valid part is added to C, so C needs no room beyond
/// rows x cols (ldc >= rows).
template <GemmScalar T, int Packets = gemm_kernel_shape<T>::packets,
          int NR = gemm_kernel_shape<T>::nr>
    requires(Packets > 0 && NR > 0)
void gemm_microkernel_edge(TileSize size, Index kc, const T* ap, const T* bp, T* c,
                           Index ldc) noexcept;

} // namespace mlinalg::detail

#include "microkernel.ipp" // IWYU pragma: keep

#endif // GEMM_MICROKERNEL_HPP
