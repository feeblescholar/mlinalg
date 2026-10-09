#include "../include/bareiss.hpp"
#include "../include/matrix.hpp"
#include "test.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

using mlinalg::Dynamic;
using mlinalg::Index;
using mlinalg::Matrix;

namespace {

using MI44 = Matrix<int, 4, 4>;
using MIX  = Matrix<int, Dynamic, Dynamic>;
using MLX  = Matrix<std::int64_t, Dynamic, Dynamic>;

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

/// Deterministic integer test matrix with entries in [-bound, bound].
template <typename M> auto pseudo_random(Index n, unsigned seed, int bound) -> M {
    M        m(n, n);
    unsigned state = seed;
    for (auto& c : m) {
        state = (state * 1103515245U) + 12345U;
        c     = static_cast<typename M::value_type>(
            static_cast<int>((state >> 8U) % static_cast<unsigned>((2 * bound) + 1)) - bound);
    }
    return m;
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Compile-time properties
// ------------------------------------------------------------------------------------------------

static_assert(HasDet<MI44> && HasDet<MIX> && HasDet<MLX> && HasDet<Matrix<short, 3, 3>>);

// Unsigned determinants can be negative; bool is not an arithmetic scalar.
static_assert(!HasDet<Matrix<unsigned, 4, 4>> && !HasDet<Matrix<bool, 2, 2>>);

// Non-square at compile time.
static_assert(!HasDet<Matrix<int, 2, 3>>);

// Exact and usable in constant expressions.
static_assert(MI44{{2, -1, 0, 3}, {1, 4, 2, -2}, {0, 3, -1, 1}, {5, 0, 2, 2}}.det() == 48);
static_assert(Matrix<int, 3, 3>{{6, 1, 1}, {4, -2, 5}, {2, 8, 7}}.det() == -306);

// ------------------------------------------------------------------------------------------------
// Runtime behaviour
// ------------------------------------------------------------------------------------------------

TEST(Bareiss, MatchesCofactorExpansion) {
    // Entries in [-3, 3] keep every intermediate minor product well inside int for n <= 6.
    for (Index n = 4; n <= 6; ++n) {
        SCOPED_TRACE(n);
        const auto a = pseudo_random<MIX>(n, 300 + static_cast<unsigned>(n), 3);
        EXPECT_EQ(a.det(), laplace_det(to_rows(a)));
    }
}

TEST(Bareiss, ZeroPivotDuringEliminationSwapsRows) {
    // After the first step a(1, 1) becomes 0 and row 2 is swapped in.
    const MI44 a{{1, 1, 1, 1}, {1, 1, 2, 3}, {1, 2, 1, 1}, {1, 1, 1, 2}};
    EXPECT_EQ(a.det(), -1);
}

TEST(Bareiss, LeadingZeroSwapsRows) {
    const MI44 a{{0, 2, 1, 4}, {1, 1, 1, 1}, {3, 0, 2, 1}, {1, 5, 0, 2}};
    EXPECT_EQ(a.det(), -17);
}

TEST(Bareiss, SingularMatrices) {
    // Zero column: no row can be swapped in.
    const MI44 zero_col{{1, 0, 2, 3}, {4, 0, 5, 6}, {7, 0, 8, 9}, {1, 0, 1, 1}};
    EXPECT_EQ(zero_col.det(), 0);

    // Dependent rows: exactly zero, no rounding.
    const MIX dependent{{1, 2, 3, 4}, {2, 4, 6, 8}, {0, 1, 0, 1}, {5, 1, 2, 2}};
    EXPECT_EQ(dependent.det(), 0);
}

TEST(Bareiss, ClosedFormSizes) {
    EXPECT_EQ(MIX(0, 0).det(), 1);
    EXPECT_EQ(MIX{{-7}}.det(), -7);
    EXPECT_EQ((MIX{{1, 2}, {3, 4}}.det()), -2);
    EXPECT_EQ((MIX{{6, 1, 1}, {4, -2, 5}, {2, 8, 7}}.det()), -306);
}

TEST(Bareiss, DiagonalAndPermutation) {
    MIX diag(6, 6);
    for (Index i = 0; i < 6; ++i) {
        diag(i, i) = static_cast<int>(i + 1);
    }
    EXPECT_EQ(diag.det(), 720);

    // Reversal of 5 rows: 2 swaps, even.
    MIX reversal(5, 5);
    for (Index i = 0; i < 5; ++i) {
        reversal(i, 4 - i) = 1;
    }
    EXPECT_EQ(reversal.det(), 1);
}

TEST(Bareiss, WideIntegerType) {
    // Intermediate minor products reach ~1e14 here, far beyond int but fine for int64.
    const MLX a{{97, -42, 13, 88, -71},
                {-5, 64, -99, 23, 40},
                {31, 7, 58, -16, 92},
                {-83, 29, -4, 77, 11},
                {60, -38, 45, -27, 3}};
    EXPECT_EQ(a.det(), 1212060782);

    const auto b = pseudo_random<MLX>(8, 42, 5);
    EXPECT_EQ(b.det(), laplace_det(to_rows(b)));
}

TEST(Bareiss, DoesNotModifyTheMatrix) {
    const MI44                 a{{0, 2, 1, 4}, {1, 1, 1, 1}, {3, 0, 2, 1}, {1, 5, 0, 2}};
    const MI44                 copy = a;
    [[maybe_unused]] const int det  = a.det();
    for (Index i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a.data()[i], copy.data()[i]);
    }
}
