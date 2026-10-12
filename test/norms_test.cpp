#include "../include/matrix.hpp"
#include "test.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <utility>

using mlinalg::conj_transpose;
using mlinalg::Dynamic;
using mlinalg::Index;
using mlinalg::Matrix;
using mlinalg::RowVector;
using mlinalg::transpose;
using mlinalg::Vector;

namespace {

using M22 = Matrix<double, 2, 2>;
using M23 = Matrix<double, 2, 3>;
using MXX = Matrix<double, Dynamic, Dynamic>;
using MFX = Matrix<float, Dynamic, Dynamic>;
using C   = std::complex<double>;
using MCX = Matrix<C, Dynamic, Dynamic>;
using MIX = Matrix<int, Dynamic, Dynamic>;
using VX  = Vector<double>;
using RX  = RowVector<double>;

template <typename M>
concept HasSumNorms = requires(const M& m) {
    m.norm1();
    m.norminf();
};

template <typename M>
concept HasEuclideanNorms = requires(const M& m) {
    m.norm2();
    m.normF();
};

/// Deterministic test matrix with entries in [-1, 1].
auto pseudo_random(Index rows, Index cols, unsigned seed) -> MXX {
    MXX      m(rows, cols);
    unsigned state = seed;
    for (double& c : m) {
        state = (state * 1103515245U) + 12345U;
        c     = (static_cast<double>((state >> 8U) % 2001U) / 1000.0) - 1.0;
    }
    return m;
}

auto pseudo_random_complex(Index rows, Index cols, unsigned seed) -> MCX {
    const MXX re = pseudo_random(rows, cols, seed);
    const MXX im = pseudo_random(rows, cols, seed + 1000);
    MCX       m(rows, cols);
    for (Index i = 0; i < rows; ++i) {
        for (Index j = 0; j < cols; ++j) {
            m(i, j) = C(re(i, j), im(i, j));
        }
    }
    return m;
}

/// Reference spectral norm: power iteration on A^H A, an algorithm unrelated to the library's.
template <typename M> auto power_iteration_norm2(const M& a) -> double {
    using T                                = typename M::value_type;
    const Matrix<T, Dynamic, Dynamic> gram = conj_transpose(a) * a;
    Matrix<T, Dynamic, 1>             x(gram.rows());
    for (Index i = 0; i < x.size(); ++i) {
        x[i] = T(1.0 + (0.1 * static_cast<double>(i)));
    }
    double lambda = 0.0;
    for (int iteration = 0; iteration < 20000; ++iteration) {
        const Matrix<T, Dynamic, 1> y = gram * x;
        lambda                        = y.normF() / x.normF();
        x                             = (T(1.0) / T(y.normF())) * y;
    }
    return std::sqrt(lambda);
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Compile-time properties
// ------------------------------------------------------------------------------------------------

static_assert(HasSumNorms<M23> && HasSumNorms<MXX> && HasSumNorms<MCX> && HasSumNorms<MIX>);
static_assert(HasEuclideanNorms<M23> && HasEuclideanNorms<MFX> && HasEuclideanNorms<MCX>);

// Integer matrices have exact sum norms but no (irrational) Euclidean ones; unsigned is excluded.
static_assert(!HasEuclideanNorms<MIX>);
static_assert(!HasSumNorms<Matrix<unsigned, 2, 2>>);

// Norms are real, also for complex matrices.
static_assert(std::is_same_v<decltype(std::declval<const MCX&>().norm1()), double>);
static_assert(std::is_same_v<decltype(std::declval<const MCX&>().norm2()), double>);
static_assert(std::is_same_v<decltype(std::declval<const MFX&>().normF()), float>);
static_assert(std::is_same_v<decltype(std::declval<const MIX&>().norminf()), int>);

// The sum norms of real fixed-size matrices are constant expressions.
static_assert(M22{{1, -2}, {3, 4}}.norm1() == 6.0);
static_assert(M22{{1, -2}, {3, 4}}.norminf() == 7.0);
static_assert(Matrix<int, 2, 3>{{1, -2, 3}, {-4, 5, -6}}.norm1() == 9);

// ------------------------------------------------------------------------------------------------
// norm1, norminf, normF
// ------------------------------------------------------------------------------------------------

TEST(Norms, KnownSmallMatrix) {
    const MXX a{{1, -2}, {3, 4}};
    EXPECT_DOUBLE_EQ(a.norm1(), 6.0);   // columns: 4, 6
    EXPECT_DOUBLE_EQ(a.norminf(), 7.0); // rows: 3, 7
    EXPECT_DOUBLE_EQ(a.normF(), std::sqrt(30.0));
    // Singular values squared: 15 +- sqrt(125).
    EXPECT_NEAR(a.norm2(), std::sqrt(15.0 + std::sqrt(125.0)), 1e-14);
}

TEST(Norms, NonSquare) {
    const M23 a{{1, -2, 3}, {-4, 5, -6}};
    EXPECT_DOUBLE_EQ(a.norm1(), 9.0);    // columns: 5, 7, 9
    EXPECT_DOUBLE_EQ(a.norminf(), 15.0); // rows: 6, 15
    EXPECT_DOUBLE_EQ(a.normF(), std::sqrt(91.0));
}

TEST(Norms, EmptyAndZero) {
    const MXX empty;
    EXPECT_EQ(empty.norm1(), 0.0);
    EXPECT_EQ(empty.norm2(), 0.0);
    EXPECT_EQ(empty.norminf(), 0.0);
    EXPECT_EQ(empty.normF(), 0.0);

    const MXX zero(3, 4);
    EXPECT_EQ(zero.norm1(), 0.0);
    EXPECT_EQ(zero.norm2(), 0.0);
    EXPECT_EQ(zero.norminf(), 0.0);
    EXPECT_EQ(zero.normF(), 0.0);
}

TEST(Norms, ColumnAndRowVectors) {
    const VX col{3, -4};
    EXPECT_DOUBLE_EQ(col.norm1(), 7.0);   // one column: the sum
    EXPECT_DOUBLE_EQ(col.norminf(), 4.0); // rows of one entry: the max
    EXPECT_DOUBLE_EQ(col.norm2(), 5.0);
    EXPECT_DOUBLE_EQ(col.normF(), 5.0);

    // As a 1 x n matrix the two sum norms swap roles.
    const RX row{3, -4};
    EXPECT_DOUBLE_EQ(row.norm1(), 4.0);
    EXPECT_DOUBLE_EQ(row.norminf(), 7.0);
    EXPECT_DOUBLE_EQ(row.norm2(), 5.0);
    EXPECT_DOUBLE_EQ(row.normF(), 5.0);
}

TEST(Norms, IntegerSumNormsAreExact) {
    const MIX a{{1, -2, 3}, {-4, 5, -6}};
    EXPECT_EQ(a.norm1(), 9);
    EXPECT_EQ(a.norminf(), 15);

    const Matrix<std::int64_t, 2, 2> b{{-5000000000, 1}, {2, 3}};
    EXPECT_EQ(b.norm1(), 5000000002);
}

TEST(Norms, ComplexMagnitudes) {
    const MCX a{{C(3, 4), C(0, -1)}, {C(0, 0), C(5, 12)}};
    EXPECT_DOUBLE_EQ(a.norm1(), 14.0);   // columns: 5, 1 + 13
    EXPECT_DOUBLE_EQ(a.norminf(), 13.0); // rows: 5 + 1, 13
    EXPECT_DOUBLE_EQ(a.normF(), std::sqrt(25.0 + 1.0 + 169.0));
}

TEST(Norms, FloatScalars) {
    const MFX a{{1, -2}, {3, 4}};
    EXPECT_FLOAT_EQ(a.norm1(), 6.0F);
    EXPECT_FLOAT_EQ(a.norminf(), 7.0F);
    EXPECT_FLOAT_EQ(a.normF(), std::sqrt(30.0F));
    EXPECT_NEAR(a.norm2(), std::sqrt(15.0F + std::sqrt(125.0F)), 1e-6F);
}

// ------------------------------------------------------------------------------------------------
// norm2
// ------------------------------------------------------------------------------------------------

TEST(Norm2, DiagonalAndScaledPermutation) {
    MXX diag(4, 4);
    diag(0, 0) = 2.0;
    diag(1, 1) = -7.0;
    diag(2, 2) = 0.5;
    diag(3, 3) = 3.0;
    EXPECT_DOUBLE_EQ(diag.norm2(), 7.0);

    const MXX permuted{{0, 0, 2}, {-9, 0, 0}, {0, 4, 0}};
    EXPECT_DOUBLE_EQ(permuted.norm2(), 9.0);
}

TEST(Norm2, OrthogonalMatrixHasUnitNorm) {
    const double c = std::cos(0.7);
    const double s = std::sin(0.7);
    const MXX    rotation{{c, -s}, {s, c}};
    EXPECT_NEAR(rotation.norm2(), 1.0, 1e-15);

    // Repeated singular values (all equal): no rotation can be selected by magnitude.
    const MXX scaled_identity{{3, 0, 0}, {0, 3, 0}, {0, 0, 3}};
    EXPECT_DOUBLE_EQ(scaled_identity.norm2(), 3.0);
}

TEST(Norm2, RankOneOuterProduct) {
    const VX  u{1, 2, 2};    // length 3
    const VX  v{3, 0, 4, 0}; // length 5
    const MXX outer = u * transpose(v);
    EXPECT_NEAR(outer.norm2(), 15.0, 1e-13);
    EXPECT_NEAR(outer.normF(), 15.0, 1e-13); // rank one: both norms agree
}

TEST(Norm2, MatchesPowerIteration) {
    for (Index n = 2; n <= 8; ++n) {
        SCOPED_TRACE(n);
        const MXX    a        = pseudo_random(n, n, 700 + static_cast<unsigned>(n));
        const double expected = power_iteration_norm2(a);
        EXPECT_NEAR(a.norm2(), expected, 1e-9 * expected);
    }
}

TEST(Norm2, TallAndWideAgree) {
    const MXX tall = pseudo_random(7, 3, 21);
    const MXX wide = transpose(tall);
    ASSERT_EQ(wide.rows(), 3);
    EXPECT_NEAR(tall.norm2(), wide.norm2(), 1e-14);
    EXPECT_NEAR(tall.norm2(), power_iteration_norm2(tall), 1e-9);

    const M23 fixed{{1, -2, 3}, {-4, 5, -6}}; // wide, fixed size: goes through a 3 x 2 copy
    EXPECT_NEAR(fixed.norm2(), power_iteration_norm2(fixed), 1e-9);
}

TEST(Norm2, ComplexScalars) {
    // Hermitian with eigenvalues 0 and 2.
    const MCX hermitian{{C(1, 0), C(0, 1)}, {C(0, -1), C(1, 0)}};
    EXPECT_NEAR(hermitian.norm2(), 2.0, 1e-15);

    // Scaled permutation with complex phases: singular values 3 and 2.
    const MCX phases{{C(0, 0), C(0, 2)}, {C(-3, 0), C(0, 0)}};
    EXPECT_NEAR(phases.norm2(), 3.0, 1e-15);

    for (Index n = 2; n <= 6; ++n) {
        SCOPED_TRACE(n);
        const MCX    a        = pseudo_random_complex(n + 1, n, 900 + static_cast<unsigned>(n));
        const double expected = power_iteration_norm2(a);
        EXPECT_NEAR(a.norm2(), expected, 1e-9 * expected);

        const MCX adjoint = conj_transpose(a);
        EXPECT_NEAR(adjoint.norm2(), a.norm2(), 1e-13);
    }
}

TEST(Norm2, NormInequalities) {
    for (unsigned seed = 1; seed <= 5; ++seed) {
        SCOPED_TRACE(seed);
        const MXX    a   = pseudo_random(5, 4, seed);
        const double n2  = a.norm2();
        const double eps = 1e-12;
        EXPECT_LE(n2, a.normF() + eps);
        EXPECT_LE(a.normF(), (std::sqrt(4.0) * n2) + eps); // rank <= 4
        EXPECT_LE(n2 * n2, (a.norm1() * a.norminf()) + eps);

        // Absolute homogeneity.
        const MXX scaled = -2.5 * a;
        EXPECT_NEAR(scaled.norm2(), 2.5 * n2, 1e-13);
    }
}

TEST(Norm2, DoesNotModifyTheMatrix) {
    const MXX                     a    = pseudo_random(4, 6, 5);
    const MXX                     copy = a;
    [[maybe_unused]] const double n2   = a.norm2();
    for (Index i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a.data()[i], copy.data()[i]);
    }
}
