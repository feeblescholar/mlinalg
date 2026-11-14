#include "../include/gemm/microkernel.hpp"
#include "../include/gemm/pack.hpp"
#include "../include/matrix_storage.hpp"
#include "test.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

using mlinalg::Index;
using mlinalg::detail::gemm_kernel_shape;
using mlinalg::detail::StridedView;

namespace {

template <typename T> using aligned_vector = std::vector<T, mlinalg::aligned_allocator<T>>;

/// Small integers in [-4, 4], so every product below is exact even in float.
template <typename T> auto make_value(Index i, int salt) -> T {
    return static_cast<T>((((i * 7) + (salt * 13)) % 9) - 4);
}

template <typename T> constexpr T sentinel = static_cast<T>(-99);

/// A rows x cols matrix in a column-major buffer with leading dimension rows + 2, so that packing
/// has to honour the stride. With `transposed`, the buffer holds the transpose and the view swaps
/// the strides, the way a TransposeOp operand would be packed.
template <typename T> class Source {
  public:
    Source(Index rows, Index cols, bool transposed, int salt)
        : m_rows(rows), m_cols(cols), m_ld((transposed ? cols : rows) + 2),
          m_data(static_cast<std::size_t>(m_ld * (transposed ? rows : cols)), sentinel<T>),
          m_transposed(transposed) {
        for (Index j = 0; j < cols; ++j) {
            for (Index i = 0; i < rows; ++i) {
                at(i, j) = make_value<T>((j * rows) + i, salt);
            }
        }
    }

    [[nodiscard]] auto view() const -> StridedView<T> {
        return m_transposed ? StridedView<T>{m_data.data(), m_ld, 1}
                            : StridedView<T>{m_data.data(), 1, m_ld};
    }

    [[nodiscard]] auto operator()(Index i, Index j) const -> T { return view()(i, j); }

  private:
    auto at(Index i, Index j) -> T& {
        return m_transposed ? m_data[static_cast<std::size_t>((i * m_ld) + j)]
                            : m_data[static_cast<std::size_t>((j * m_ld) + i)];
    }

    Index          m_rows;
    Index          m_cols;
    Index          m_ld;
    std::vector<T> m_data;
    bool           m_transposed;
};

template <typename T> class GemmPackTest : public ::testing::Test {};

using GemmScalars = ::testing::Types<float, double, int>;
TYPED_TEST_SUITE(GemmPackTest, GemmScalars);

/// Sizes around a tile dimension r: below, at and above one tile, and a ragged multiple.
auto edge_sizes(Index r) -> std::vector<Index> {
    return {1, r - 1, r, r + 1, (2 * r) + 3};
}

} // namespace

TYPED_TEST(GemmPackTest, PackAWritesZeroPaddedColumnPanels) {
    using T            = TypeParam;
    constexpr Index mr = gemm_kernel_shape<T>::mr;
    for (const bool transposed : {false, true}) {
        for (const Index mc : edge_sizes(mr)) {
            for (const Index kc : {Index{1}, Index{5}}) {
                const Source<T>   a(mc, kc, transposed, 1);
                const Index       size = mlinalg::detail::packed_a_size<T>(mc, kc);
                aligned_vector<T> ap(static_cast<std::size_t>(size + 1), sentinel<T>);
                mlinalg::detail::pack_a<T>(a.view(), mc, kc, ap.data());

                for (Index n = 0; n < size; ++n) {
                    const Index panel = n / (mr * kc);
                    const Index k     = (n % (mr * kc)) / mr;
                    const Index i     = (panel * mr) + (n % mr);
                    const T     want  = i < mc ? a(i, k) : T{};
                    ASSERT_EQ(ap[n], want) << "mc " << mc << ", kc " << kc << ", offset " << n
                                           << ", transposed " << transposed;
                }
                EXPECT_EQ(ap[size], sentinel<T>) << "pack_a wrote past packed_a_size";
            }
        }
    }
}

TYPED_TEST(GemmPackTest, PackBWritesZeroPaddedRowPanels) {
    using T            = TypeParam;
    constexpr Index nr = gemm_kernel_shape<T>::nr;
    for (const bool transposed : {false, true}) {
        for (const Index nc : edge_sizes(nr)) {
            for (const Index kc : {Index{1}, Index{5}}) {
                const Source<T> b(kc, nc, transposed, 2);
                const Index     size = mlinalg::detail::packed_b_size<T>(kc, nc);
                std::vector<T>  bp(static_cast<std::size_t>(size + 1), sentinel<T>);
                mlinalg::detail::pack_b<T>(b.view(), kc, nc, bp.data());

                for (Index n = 0; n < size; ++n) {
                    const Index panel = n / (nr * kc);
                    const Index k     = (n % (nr * kc)) / nr;
                    const Index j     = (panel * nr) + (n % nr);
                    const T     want  = j < nc ? b(k, j) : T{};
                    ASSERT_EQ(bp[static_cast<std::size_t>(n)], want)
                        << "nc " << nc << ", kc " << kc << ", offset " << n << ", transposed "
                        << transposed;
                }
                EXPECT_EQ(bp[static_cast<std::size_t>(size)], sentinel<T>)
                    << "pack_b wrote past packed_b_size";
            }
        }
    }
}

// Packs whole operands and sweeps every tile with the full or the edge kernel: a single-block
// product, which checks that packing and both kernels agree on the layouts at every edge.
TYPED_TEST(GemmPackTest, SingleBlockProductMatchesReference) {
    using T            = TypeParam;
    using shape        = gemm_kernel_shape<T>;
    constexpr Index mr = shape::mr;
    constexpr Index nr = shape::nr;

    for (const bool transposed : {false, true}) {
        for (const Index m : edge_sizes(mr)) {
            for (const Index n : edge_sizes(nr)) {
                for (const Index k : {Index{1}, Index{2}, Index{17}}) {
                    const Source<T> a(m, k, transposed, 1);
                    const Source<T> b(k, n, !transposed, 2);

                    aligned_vector<T> ap(
                        static_cast<std::size_t>(mlinalg::detail::packed_a_size<T>(m, k)));
                    aligned_vector<T> bp(
                        static_cast<std::size_t>(mlinalg::detail::packed_b_size<T>(k, n)));
                    mlinalg::detail::pack_a<T>(a.view(), m, k, ap.data());
                    mlinalg::detail::pack_b<T>(b.view(), k, n, bp.data());

                    // C has a leading dimension larger than m, filled with values to accumulate
                    // into and sentinels in the rows the product must not touch.
                    const Index    ldc = m + 1;
                    std::vector<T> c(static_cast<std::size_t>(ldc * n), sentinel<T>);
                    for (Index j = 0; j < n; ++j) {
                        for (Index i = 0; i < m; ++i) {
                            c[static_cast<std::size_t>((j * ldc) + i)] =
                                make_value<T>((j * m) + i, 3);
                        }
                    }
                    std::vector<T> expected(c);
                    for (Index j = 0; j < n; ++j) {
                        for (Index i = 0; i < m; ++i) {
                            T sum = expected[static_cast<std::size_t>((j * ldc) + i)];
                            for (Index p = 0; p < k; ++p) {
                                sum += a(i, p) * b(p, j);
                            }
                            expected[static_cast<std::size_t>((j * ldc) + i)] = sum;
                        }
                    }

                    for (Index jr = 0; jr < n; jr += nr) {
                        for (Index ir = 0; ir < m; ir += mr) {
                            const T*    ap_sliver = ap.data() + (ir * k);
                            const T*    bp_sliver = bp.data() + (jr * k);
                            T*          tile      = c.data() + (jr * ldc) + ir;
                            const Index rows      = std::min(mr, m - ir);
                            const Index cols      = std::min(nr, n - jr);
                            if (rows == mr && cols == nr) {
                                mlinalg::detail::gemm_microkernel<T>(k, ap_sliver, bp_sliver, tile,
                                                                     ldc);
                            } else {
                                mlinalg::detail::gemm_microkernel_edge<T>(
                                    {.rows = rows, .cols = cols}, k, ap_sliver, bp_sliver, tile,
                                    ldc);
                            }
                        }
                    }

                    for (std::size_t idx = 0; idx < c.size(); ++idx) {
                        ASSERT_EQ(c[idx], expected[idx])
                            << "m " << m << ", n " << n << ", k " << k << ", offset " << idx
                            << ", transposed " << transposed;
                    }
                }
            }
        }
    }
}
