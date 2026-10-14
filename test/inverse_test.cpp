#include "../include/matrix.hpp"
#include "../include/partial_piv_lu.hpp"
#include "test.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <type_traits>
#include <utility>

using mlinalg::Dynamic;
using mlinalg::Index;
using mlinalg::Matrix;

namespace {

using M22  = Matrix<double, 2, 2>;
using M33  = Matrix<double, 3, 3>;
using M44  = Matrix<double, 4, 4>;
using M23  = Matrix<double, 2, 3>;
using MX3  = Matrix<double, Dynamic, 3>;
using MXX  = Matrix<double, Dynamic, Dynamic>;
using MF44 = Matrix<float, 4, 4>;
using C    = std::complex<double>;
using MCX  = Matrix<C, Dynamic, Dynamic>;

template <typename M>
concept HasInv = requires(const M& m) { m.inv(); };

/// Deterministic, well-conditioned test matrix: entries in [-1, 1] plus n on the diagonal.
auto pseudo_random(Index n, unsigned seed) -> MXX {
    MXX      m(n, n);
    unsigned state = seed;
    for (Index j = 0; j < n; ++j) {
        for (Index i = 0; i < n; ++i) {
            state   = (state * 1103515245U) + 12345U;
            m(i, j) = (static_cast<double>((state >> 8U) % 2001U) / 1000.0) - 1.0;
        }
        m(j, j) += static_cast<double>(n);
    }
    return m;
}

/// max |m(i, j) - (i == j)|
template <typename M> auto distance_to_identity(const M& m) -> double {
    double worst = 0.0;
    for (Index i = 0; i < m.rows(); ++i) {
        for (Index j = 0; j < m.cols(); ++j) {
            const auto expected = i == j ? typename M::value_type{1} : typename M::value_type{0};
            worst = std::max(worst, static_cast<double>(std::abs(m(i, j) - expected)));
        }
    }
    return worst;
}

template <typename M> void expect_near(const M& a, const M& b, double tol) {
    ASSERT_EQ(a.rows(), b.rows());
    ASSERT_EQ(a.cols(), b.cols());
    for (Index i = 0; i < a.rows(); ++i) {
        for (Index j = 0; j < a.cols(); ++j) {
            EXPECT_NEAR(std::abs(a(i, j) - b(i, j)), 0.0, tol) << "at (" << i << ", " << j << ")";
        }
    }
}

template <typename M> void expect_coeffs(const M& m, std::initializer_list<double> expected) {
    ASSERT_EQ(static_cast<std::size_t>(m.rows() * m.cols()), expected.size());
    const auto* it = expected.begin();
    for (Index i = 0; i < m.rows(); ++i) {
        for (Index j = 0; j < m.cols(); ++j) {
            EXPECT_DOUBLE_EQ(m(i, j), *it++) << "at (" << i << ", " << j << ")";
        }
    }
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Compile-time properties
// ------------------------------------------------------------------------------------------------

static_assert(HasInv<M22> && HasInv<M44> && HasInv<MXX> && HasInv<MX3> && HasInv<MF44> &&
              HasInv<MCX>);
static_assert(std::is_same_v<decltype(std::declval<const M44&>().inv()), M44>);
static_assert(std::is_same_v<decltype(std::declval<const MX3&>().inv()), MX3>);

// Non-square at compile time; integer inverses are not integers in general.
static_assert(!HasInv<M23> && !HasInv<Matrix<int, 3, 3>>);

// Fixed-size inverses are constant expressions, both closed-form and through LU.
static_assert([] {
    const M22 a = M22{{1, 2}, {3, 5}}.inv(); // det -1
    return a(0, 0) == -5.0 && a(0, 1) == 2.0 && a(1, 0) == 3.0 && a(1, 1) == -1.0;
}());

static_assert([] {
    const M33 a = M33{{2, 3, 1}, {1, 2, 1}, {1, 1, 1}}.inv(); // det 1
    return a(0, 0) == 1.0 && a(0, 1) == -2.0 && a(0, 2) == 1.0 && a(1, 0) == 0.0 &&
           a(1, 1) == 1.0 && a(1, 2) == -1.0 && a(2, 0) == -1.0 && a(2, 1) == 1.0 && a(2, 2) == 1.0;
}());

static_assert([] {
    // Scaled permutation: the inverse is the transpose with reciprocal entries.
    const M44 a = M44{{0, 0, 2, 0}, {4, 0, 0, 0}, {0, 0, 0, 8}, {0, 1, 0, 0}}.inv();
    return a(0, 1) == 0.25 && a(1, 3) == 1.0 && a(2, 0) == 0.5 && a(3, 2) == 0.125 &&
           a(0, 0) == 0.0;
}());

// ------------------------------------------------------------------------------------------------
// PartialPivLU::inverse and is_invertible
// ------------------------------------------------------------------------------------------------

TEST(PartialPivLUInverse, ExactForUnitTriangular) {
    const M44 a{{1, 2, 0, 0}, {0, 1, 2, 0}, {0, 0, 1, 2}, {0, 0, 0, 1}};
    expect_coeffs(a.lu().inverse(), {1, -2, 4, -8, 0, 1, -2, 4, 0, 0, 1, -2, 0, 0, 0, 1});
}

TEST(PartialPivLUInverse, ProductIsIdentity) {
    for (Index n = 1; n <= 10; ++n) {
        SCOPED_TRACE(n);
        const MXX a   = pseudo_random(n, 500 + static_cast<unsigned>(n));
        const MXX ai  = a.lu().inverse();
        const MXX lhs = ai * a;
        const MXX rhs = a * ai;
        EXPECT_LT(distance_to_identity(lhs), 1e-13);
        EXPECT_LT(distance_to_identity(rhs), 1e-13);
    }
}

TEST(PartialPivLUInverse, IsInvertible) {
    EXPECT_TRUE(pseudo_random(5, 1).lu().is_invertible());

    const M44 singular{{1, 1, 0, 0}, {0, 1, 1, 0}, {0, 0, 1, 1}, {1, 0, 0, 1}};
    EXPECT_FALSE(singular.lu().is_invertible());

    const MXX zero_col{{1, 0, 2}, {4, 0, 5}, {7, 0, 8}};
    EXPECT_FALSE(zero_col.lu().is_invertible());

    EXPECT_TRUE(MXX(0, 0).lu().is_invertible()); // empty product of pivots
}

// ------------------------------------------------------------------------------------------------
// Matrix::inv
// ------------------------------------------------------------------------------------------------

TEST(MatrixInv, ClosedFormSizes) {
    EXPECT_EQ(MXX(0, 0).inv().size(), 0);
    expect_coeffs(MXX{{4}}.inv(), {0.25});
    expect_coeffs((MXX{{1, 2}, {3, 5}}.inv()), {-5, 2, 3, -1});
    expect_coeffs((MXX{{2, 3, 1}, {1, 2, 1}, {1, 1, 1}}.inv()), {1, -2, 1, 0, 1, -1, -1, 1, 1});
}

TEST(MatrixInv, ClosedFormMatchesLu) {
    for (Index n = 1; n <= 3; ++n) {
        SCOPED_TRACE(n);
        const MXX a = pseudo_random(n, 600 + static_cast<unsigned>(n));
        expect_near(a.inv(), a.lu().inverse(), 1e-14);
    }
}

TEST(MatrixInv, ClosedFormHandlesZeroLeadingEntry) {
    // Zero in the top-left corner: fine for the adjugate formulas.
    const MXX a{{0, 1, 2}, {1, 0, 3}, {4, -3, 8}};
    const MXX ai      = a.inv();
    const MXX product = a * ai;
    EXPECT_LT(distance_to_identity(product), 1e-14);
}

TEST(MatrixInv, InverseOfInverse) {
    const MXX a = pseudo_random(6, 7);
    expect_near(a.inv().inv(), a, 1e-13);
}

TEST(MatrixInv, FixedAndPartiallyFixed) {
    M44       a;
    const MXX source = pseudo_random(4, 11);
    for (Index i = 0; i < 4; ++i) {
        for (Index j = 0; j < 4; ++j) {
            a(i, j) = source(i, j);
        }
    }
    const M44 ai      = a.inv();
    const M44 product = a * ai;
    EXPECT_LT(distance_to_identity(product), 1e-14);

    const MX3 p{{2, 1, 0}, {1, 3, 1}, {0, 1, 4}};
    const MX3 pi = p.inv();
    const MX3 q  = p * pi;
    EXPECT_LT(distance_to_identity(q), 1e-15);
}

TEST(MatrixInv, ComplexScalars) {
    const MCX a{{C(4, 1), C(2, 0), C(0, 1), C(1, 0)},
                {C(0, 0), C(5, -1), C(3, 0), C(0, 2)},
                {C(2, 1), C(0, 0), C(6, 1), C(1, 0)},
                {C(1, 0), C(1, 1), C(0, 0), C(4, -1)}};
    const MCX ai      = a.inv();
    const MCX product = a * ai;
    EXPECT_LT(distance_to_identity(product), 1e-14);

    const MCX small{{C(1, 1), C(2, 0)}, {C(0, -1), C(3, 2)}};
    const MCX small_inv     = small.inv();
    const MCX small_product = small_inv * small;
    EXPECT_LT(distance_to_identity(small_product), 1e-15);
}

TEST(MatrixInv, FloatScalars) {
    const MF44 a{{4, 3, 2, 1}, {0, 1, 2, 3}, {1, 0, 1, 0}, {2, 2, 0, 1}}; // det 8
    const MF44 ai      = a.inv();
    const MF44 product = a * ai;
    EXPECT_LT(distance_to_identity(product), 1e-5);
}

TEST(MatrixInv, DoesNotModifyTheMatrix) {
    const MXX                  a    = pseudo_random(5, 3);
    const MXX                  copy = a;
    [[maybe_unused]] const MXX ai   = a.inv();
    for (Index i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a.data()[i], copy.data()[i]);
    }
}
