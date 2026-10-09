#include "../include/matrix.hpp"
#include "../include/partial_piv_lu.hpp"
#include "test.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

using mlinalg::Dynamic;
using mlinalg::Index;
using mlinalg::Matrix;
using mlinalg::PartialPivLU;

namespace {

using M22  = Matrix<double, 2, 2>;
using M33  = Matrix<double, 3, 3>;
using M44  = Matrix<double, 4, 4>;
using M23  = Matrix<double, 2, 3>;
using MX3  = Matrix<double, Dynamic, 3>;
using MXX  = Matrix<double, Dynamic, Dynamic>;
using MF33 = Matrix<float, 3, 3>;
using C    = std::complex<double>;
using MCX  = Matrix<C, Dynamic, Dynamic>;

template <typename M>
concept HasLu = requires(const M& m) { m.lu(); };

template <typename M>
concept HasDet = requires(const M& m) { m.det(); };

/// Reference determinant by cofactor expansion along the first row. O(n!), small n only.
template <typename T> auto laplace_det(const std::vector<std::vector<T>>& a) -> T {
    const std::size_t n = a.size();
    if (n == 0) {
        return T{1};
    }
    T    det{};
    auto sign = T{1};
    for (std::size_t col = 0; col < n; ++col) {
        std::vector<std::vector<T>> minor;
        for (std::size_t i = 1; i < n; ++i) {
            std::vector<T> row;
            for (std::size_t j = 0; j < n; ++j) {
                if (j != col) {
                    row.push_back(a[i][j]);
                }
            }
            minor.push_back(std::move(row));
        }
        det += sign * a[0][col] * laplace_det(minor);
        sign = -sign;
    }
    return det;
}

template <typename M> auto to_rows(const M& m) {
    std::vector<std::vector<typename M::value_type>> rows(static_cast<std::size_t>(m.rows()));
    for (Index i = 0; i < m.rows(); ++i) {
        for (Index j = 0; j < m.cols(); ++j) {
            rows[static_cast<std::size_t>(i)].push_back(m(i, j));
        }
    }
    return rows;
}

/// Deterministic, well-mixed test matrix with entries in [-1, 1].
auto pseudo_random(Index n, unsigned seed) -> MXX {
    MXX      m(n, n);
    unsigned state = seed;
    for (double& c : m) {
        state = (state * 1103515245U) + 12345U;
        c     = (static_cast<double>((state >> 8U) % 2001U) / 1000.0) - 1.0;
    }
    return m;
}

/// Checks P * A == L * U and the structural properties of the factors.
template <typename M> void expect_valid_lu(const M& a, double tol = 1e-12) {
    const auto  lu = a.lu();
    const auto& f  = lu.matrix_lu();
    const Index n  = a.rows();
    ASSERT_EQ(lu.rows(), n);
    ASSERT_EQ(lu.cols(), n);

    std::vector<bool> seen(static_cast<std::size_t>(n), false);
    for (Index i = 0; i < n; ++i) {
        const Index p = lu.permutation()[i];
        ASSERT_TRUE(p >= 0 && p < n);
        EXPECT_FALSE(seen[static_cast<std::size_t>(p)]) << "permutation repeats row " << p;
        seen[static_cast<std::size_t>(p)] = true;
    }

    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < n; ++j) {
            // (L * U)(i, j) with L unit lower and U upper triangular.
            typename M::value_type sum{};
            for (Index k = 0; k <= std::min(i, j); ++k) {
                sum += (k == i ? typename M::value_type{1} : f(i, k)) * f(k, j);
            }
            EXPECT_NEAR(std::abs(sum - a(lu.permutation()[i], j)), 0.0, tol)
                << "at (" << i << ", " << j << ")";
            if (j < i) {
                // Partial pivoting bounds the multipliers: |l| <= 1 for reals. Complex pivots are
                // chosen by |re| + |im|, which only guarantees |l| <= sqrt(2).
                const double bound =
                    mlinalg::detail::is_complex_v<typename M::value_type> ? std::sqrt(2.0) : 1.0;
                EXPECT_LE(std::abs(f(i, j)), bound * (1.0 + 1e-15));
            }
        }
    }
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Compile-time properties
// ------------------------------------------------------------------------------------------------

static_assert(HasLu<M33> && HasLu<MXX> && HasLu<MX3> && HasLu<MF33> && HasLu<MCX>);
static_assert(HasDet<M22> && HasDet<MXX> && HasDet<MF33> && HasDet<MCX>);
static_assert(std::is_same_v<decltype(std::declval<const M44&>().lu()), PartialPivLU<M44>>);
static_assert(std::is_same_v<decltype(std::declval<const MCX&>().det()), C>);

// Non-square at compile time.
static_assert(!HasLu<M23> && !HasDet<M23>);

// Integer division is not exact, so LU-based functionality is not offered.
static_assert(!HasLu<Matrix<int, 3, 3>> && !HasDet<Matrix<int, 3, 3>>);

// The permutation of a fixed-size decomposition is fixed-size too.
static_assert(PartialPivLU<M44>::permutation_type::is_fixed);
static_assert(!PartialPivLU<MXX>::permutation_type::is_fixed);

// Fixed-size determinants are constant expressions, both closed-form and through LU.
static_assert(M22{{1, 2}, {3, 4}}.det() == -2.0);
static_assert(M33{{6, 1, 1}, {4, -2, 5}, {2, 8, 7}}.det() == -306.0);
static_assert(M44{{0, 1, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}}.det() == 1.0);
static_assert(M44{{0, 0, 0, 2}, {0, 0, 3, 0}, {0, 5, 0, 0}, {7, 0, 0, 0}}.det() == 210.0);

// ------------------------------------------------------------------------------------------------
// PartialPivLU
// ------------------------------------------------------------------------------------------------

TEST(PartialPivLU, ReconstructsPermutedMatrix) {
    for (Index n = 1; n <= 8; ++n) {
        SCOPED_TRACE(n);
        expect_valid_lu(pseudo_random(n, static_cast<unsigned>(n)));
    }
}

TEST(PartialPivLU, PivotsOnLargestMagnitude) {
    const MXX  a{{1, 2, 3}, {-4, 5, 6}, {2, 8, 9}};
    const auto lu = a.lu();
    // Column 0: |-4| is largest, so row 1 comes first.
    EXPECT_EQ(lu.permutation()[0], 1);
    EXPECT_DOUBLE_EQ(lu.matrix_lu()(0, 0), -4.0);
    expect_valid_lu(a);
}

TEST(PartialPivLU, ZeroLeadingEntryNeedsPivot) {
    const M44 a{{0, 2, 1, 4}, {1, 1, 1, 1}, {3, 0, 2, 1}, {1, 5, 0, 2}};
    expect_valid_lu(a);
    EXPECT_NEAR(a.lu().determinant(), laplace_det(to_rows(a)), 1e-12);
}

TEST(PartialPivLU, PermutationSign) {
    const M44 identity{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    EXPECT_EQ(identity.lu().permutation_sign(), 1);

    const M44 one_swap{{0, 1, 0, 0}, {1, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    EXPECT_EQ(one_swap.lu().permutation_sign(), -1);
    EXPECT_DOUBLE_EQ(one_swap.lu().determinant(), -1.0);
}

TEST(PartialPivLU, SingularMatrixIsNotAnError) {
    // Zero column: the pivot is exactly zero and elimination skips it.
    const MXX a{{1, 0, 2, 3}, {4, 0, 5, 6}, {7, 0, 8, 9}, {1, 0, 1, 1}};
    expect_valid_lu(a);
    EXPECT_EQ(a.lu().determinant(), 0.0);

    // Linearly dependent rows: zero up to rounding.
    const MXX b{{1, 2, 3, 4}, {2, 4, 6, 8}, {0, 1, 0, 1}, {5, 1, 2, 2}};
    EXPECT_NEAR(b.lu().determinant(), 0.0, 1e-12);
}

TEST(PartialPivLU, DoesNotModifyTheMatrix) {
    const MXX                   a    = pseudo_random(5, 7);
    const MXX                   copy = a;
    [[maybe_unused]] const auto lu   = a.lu();
    for (Index i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a.data()[i], copy.data()[i]);
    }
}

TEST(PartialPivLU, ComplexScalars) {
    const MCX a{{C(1, 1), C(2, 0), C(0, 1), C(1, 0)},
                {C(0, 0), C(1, -1), C(3, 0), C(0, 2)},
                {C(2, 1), C(0, 0), C(1, 1), C(1, 0)},
                {C(1, 0), C(1, 1), C(0, 0), C(2, -1)}};
    expect_valid_lu(a);
    const C expected = laplace_det(to_rows(a));
    EXPECT_NEAR(std::abs(a.det() - expected), 0.0, 1e-12);
}

TEST(PartialPivLU, PartiallyFixedSquare) {
    MX3 a{{2, 1, 0}, {1, 3, 1}, {0, 1, 4}};
    expect_valid_lu(a);
    EXPECT_NEAR(a.lu().determinant(), 18.0, 1e-12);
}

// ------------------------------------------------------------------------------------------------
// Matrix::det
// ------------------------------------------------------------------------------------------------

TEST(MatrixDet, ClosedFormSizes) {
    EXPECT_EQ(MXX(0, 0).det(), 1.0);
    EXPECT_EQ(MXX{{-3.5}}.det(), -3.5);
    EXPECT_EQ((MXX{{1, 2}, {3, 4}}.det()), -2.0);
    EXPECT_EQ((MXX{{6, 1, 1}, {4, -2, 5}, {2, 8, 7}}.det()), -306.0);
}

TEST(MatrixDet, ClosedFormMatchesLu) {
    for (Index n = 1; n <= 3; ++n) {
        SCOPED_TRACE(n);
        const MXX a = pseudo_random(n, 100 + static_cast<unsigned>(n));
        EXPECT_NEAR(a.det(), a.lu().determinant(), 1e-14);
    }
}

TEST(MatrixDet, LargerSizesMatchCofactorExpansion) {
    for (Index n = 4; n <= 7; ++n) {
        SCOPED_TRACE(n);
        const MXX    a        = pseudo_random(n, 200 + static_cast<unsigned>(n));
        const double expected = laplace_det(to_rows(a));
        EXPECT_NEAR(a.det(), expected, 1e-12 * std::max(1.0, std::abs(expected)));
    }
}

TEST(MatrixDet, TriangularAndDiagonal) {
    const MXX upper{
        {2, 7, 1, 8, 2}, {0, 3, 1, 4, 1}, {0, 0, 5, 9, 2}, {0, 0, 0, -1, 6}, {0, 0, 0, 0, 4}};
    EXPECT_DOUBLE_EQ(upper.det(), -120.0);

    MXX diag(6, 6);
    for (Index i = 0; i < 6; ++i) {
        diag(i, i) = static_cast<double>(i + 1);
    }
    EXPECT_DOUBLE_EQ(diag.det(), 720.0);
}

TEST(MatrixDet, FloatScalars) {
    const MF33 a{{2, 1, 0}, {1, 3, 1}, {0, 1, 4}};
    EXPECT_FLOAT_EQ(a.det(), 18.0F);

    const Matrix<float, 4, 4> b{{4, 3, 2, 1}, {0, 1, 2, 3}, {1, 0, 1, 0}, {2, 2, 0, 1}};
    EXPECT_NEAR(b.det(), 8.0F, 1e-5F);
}
