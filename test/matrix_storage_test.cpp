#include "../include/matrix_storage.hpp"
#include "test.hpp"

#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

using mlinalg::aligned_allocator;
using mlinalg::Dynamic;
using mlinalg::DynamicStorage;
using mlinalg::FixedStorage;
using mlinalg::Index;
using mlinalg::MatrixStorage;
using mlinalg::SboCapacity;

namespace {

auto is_aligned(const void* ptr, std::size_t alignment = 32) -> bool {
    return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

template <typename Storage> void fill_iota(Storage& storage) {
    for (Index i = 0; i < storage.size(); ++i) {
        storage[i] = static_cast<typename Storage::value_type>(i);
    }
}

using DynD = DynamicStorage<double, Dynamic, Dynamic>;

} // namespace

// ------------------------------------------------------------------------------------------------
// Compile-time properties
// ------------------------------------------------------------------------------------------------

static_assert(std::is_same_v<MatrixStorage<float, 3, 3>, FixedStorage<float, 3, 3>>);
static_assert(std::is_same_v<MatrixStorage<float, Dynamic, 3>, DynamicStorage<float, Dynamic, 3>>);
static_assert(std::is_same_v<MatrixStorage<float, 1, Dynamic>, DynamicStorage<float, 1, Dynamic>>);
static_assert(std::is_same_v<MatrixStorage<float, Dynamic, Dynamic>,
                             DynamicStorage<float, Dynamic, Dynamic>>);

static_assert(alignof(FixedStorage<float, 2, 2>) >= 32);
static_assert(alignof(FixedStorage<double, 5, 5>) >= 32);
static_assert(alignof(DynamicStorage<double, Dynamic, Dynamic>) >= 32);

static_assert(std::is_nothrow_move_constructible_v<DynD>);
static_assert(std::is_nothrow_move_assignable_v<DynD>);

// FixedStorage must be usable in constant expressions (needed by Module 3).
static_assert([] {
    FixedStorage<int, 2, 3> storage;
    for (Index i = 0; i < storage.size(); ++i) {
        storage[i] = static_cast<int>(i);
    }
    int sum = 0;
    for (const int value : storage) {
        sum += value;
    }
    return sum;
}() == 15);

static_assert(FixedStorage<int, 4, 7>::rows() == 4 && FixedStorage<int, 4, 7>::cols() == 7 &&
              FixedStorage<int, 4, 7>::size() == 28);

// ------------------------------------------------------------------------------------------------
// aligned_allocator
// ------------------------------------------------------------------------------------------------

TEST(AlignedAllocator, ReturnsAlignedMemory) {
    aligned_allocator<double> alloc;
    for (std::size_t n : {1U, 3U, 7U, 64U, 1001U}) {
        double* ptr = alloc.allocate(n);
        EXPECT_TRUE(is_aligned(ptr)) << "n = " << n;
        alloc.deallocate(ptr, n);
    }
}

TEST(AlignedAllocator, HonoursCustomAlignment) {
    aligned_allocator<float, 64> alloc;
    float*                       ptr = alloc.allocate(10);
    EXPECT_TRUE(is_aligned(ptr, 64));
    alloc.deallocate(ptr, 10);
}

TEST(AlignedAllocator, ThrowsOnOverflow) {
    aligned_allocator<double> alloc;
    EXPECT_THROW((void)alloc.allocate(alloc.max_size() + 1), std::bad_array_new_length);
}

TEST(AlignedAllocator, InstancesCompareEqualAndRebind) {
    aligned_allocator<double, 32> a;
    aligned_allocator<float, 32>  b;
    EXPECT_TRUE(a == b);

    using Rebound = std::allocator_traits<aligned_allocator<double, 32>>::rebind_alloc<int>;
    static_assert(std::is_same_v<Rebound, aligned_allocator<int, 32>>);
}

// ------------------------------------------------------------------------------------------------
// FixedStorage
// ------------------------------------------------------------------------------------------------

TEST(FixedStorage, DimensionsAndValueInitialization) {
    FixedStorage<double, 3, 4> storage;
    EXPECT_EQ(storage.rows(), 3);
    EXPECT_EQ(storage.cols(), 4);
    EXPECT_EQ(storage.size(), 12);
    for (const double value : storage) {
        EXPECT_EQ(value, 0.0);
    }
}

TEST(FixedStorage, DataIsAligned) {
    FixedStorage<float, 4, 4> on_stack;
    EXPECT_TRUE(is_aligned(on_stack.data()));

    auto on_heap = std::make_unique<FixedStorage<float, 3, 3>>();
    EXPECT_TRUE(is_aligned(on_heap->data()));
}

TEST(FixedStorage, ElementAccessAndIterators) {
    FixedStorage<int, 2, 2> storage;
    fill_iota(storage);
    EXPECT_EQ(storage[3], 3);
    EXPECT_EQ(storage.end() - storage.begin(), 4);
    EXPECT_EQ(storage.data(), storage.begin());
}

TEST(FixedStorage, CopyAndSwap) {
    FixedStorage<int, 2, 2> a;
    fill_iota(a);
    FixedStorage<int, 2, 2> b = a;
    EXPECT_EQ(b[2], 2);
    EXPECT_NE(a.data(), b.data());

    FixedStorage<int, 2, 2> c;
    swap(b, c);
    EXPECT_EQ(b[2], 0);
    EXPECT_EQ(c[2], 2);
}

TEST(FixedStorage, SizedConstructorAndResizeAcceptMatchingDims) {
    FixedStorage<double, 2, 3> storage(2, 3);
    storage.resize(2, 3);
    EXPECT_EQ(storage.size(), 6);
}

// ------------------------------------------------------------------------------------------------
// DynamicStorage: construction & SBO
// ------------------------------------------------------------------------------------------------

TEST(DynamicStorage, DefaultIsEmpty) {
    DynD storage;
    EXPECT_EQ(storage.rows(), 0);
    EXPECT_EQ(storage.cols(), 0);
    EXPECT_EQ(storage.size(), 0);
    EXPECT_FALSE(storage.is_heap_allocated());
}

TEST(DynamicStorage, DefaultKeepsFixedDimension) {
    DynamicStorage<float, Dynamic, 1> column;
    EXPECT_EQ(column.rows(), 0);
    EXPECT_EQ(column.cols(), 1);

    DynamicStorage<float, 1, Dynamic> row;
    EXPECT_EQ(row.rows(), 1);
    EXPECT_EQ(row.cols(), 0);
}

TEST(DynamicStorage, SmallSizesUseInlineBuffer) {
    DynD storage(5, 6);
    EXPECT_EQ(storage.size(), 30);
    EXPECT_FALSE(storage.is_heap_allocated());
    EXPECT_TRUE(is_aligned(storage.data()));
}

TEST(DynamicStorage, SboBoundary) {
    DynD at_capacity(1, SboCapacity);
    EXPECT_FALSE(at_capacity.is_heap_allocated());

    DynD over_capacity(1, SboCapacity + 1);
    EXPECT_TRUE(over_capacity.is_heap_allocated());
    EXPECT_TRUE(is_aligned(over_capacity.data()));
}

TEST(DynamicStorage, ElementsAreValueInitialized) {
    DynD small(3, 3);
    DynD large(10, 10);
    for (const double value : small) {
        EXPECT_EQ(value, 0.0);
    }
    for (const double value : large) {
        EXPECT_EQ(value, 0.0);
    }
}

TEST(DynamicStorage, ThrowsOnSizeOverflow) {
    const Index big = std::numeric_limits<Index>::max() / 2;
    EXPECT_THROW(DynD(big, 3), std::length_error);
}

// ------------------------------------------------------------------------------------------------
// DynamicStorage: copy / move / swap
// ------------------------------------------------------------------------------------------------

TEST(DynamicStorage, CopyConstructHeap) {
    DynD a(10, 10);
    fill_iota(a);
    DynD b = a;
    EXPECT_TRUE(b.is_heap_allocated());
    EXPECT_NE(a.data(), b.data());
    EXPECT_EQ(b.rows(), 10);
    EXPECT_EQ(b[99], 99.0);
}

TEST(DynamicStorage, CopyConstructInline) {
    DynD a(2, 3);
    fill_iota(a);
    DynD b = a;
    EXPECT_FALSE(b.is_heap_allocated());
    EXPECT_EQ(b[5], 5.0);
}

TEST(DynamicStorage, MoveConstructStealsHeapBuffer) {
    DynD          a(10, 10);
    const double* original = a.data();
    DynD          b        = std::move(a);
    EXPECT_EQ(b.data(), original);
    EXPECT_EQ(a.size(), 0); // NOLINT(bugprone-use-after-move): moved-from state is specified
    EXPECT_FALSE(a.is_heap_allocated());
}

TEST(DynamicStorage, MoveConstructInline) {
    DynD a(2, 2);
    fill_iota(a);
    DynD b = std::move(a);
    EXPECT_EQ(b[3], 3.0);
    EXPECT_EQ(a.size(), 0); // NOLINT(bugprone-use-after-move)
}

TEST(DynamicStorage, MovedFromKeepsFixedDimension) {
    DynamicStorage<float, Dynamic, 1> a(100, 1);
    auto                              b = std::move(a);
    EXPECT_EQ(a.cols(), 1); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(a.size(), 0);
}

TEST(DynamicStorage, CopyAssignAcrossStorageKinds) {
    DynD small(2, 2);
    fill_iota(small);
    DynD large(10, 10);
    fill_iota(large);

    DynD target(3, 3);
    target = large; // inline -> heap
    EXPECT_TRUE(target.is_heap_allocated());
    EXPECT_EQ(target[42], 42.0);

    target = small; // heap -> inline
    EXPECT_FALSE(target.is_heap_allocated());
    EXPECT_EQ(target.rows(), 2);
    EXPECT_EQ(target[3], 3.0);
}

TEST(DynamicStorage, CopyAssignSameSizeReusesBuffer) {
    DynD a(10, 10);
    fill_iota(a);
    DynD          b(20, 5);
    const double* buffer = b.data();
    b                    = a;
    EXPECT_EQ(b.data(), buffer);
    EXPECT_EQ(b.rows(), 10);
    EXPECT_EQ(b.cols(), 10);
    EXPECT_EQ(b[99], 99.0);
}

TEST(DynamicStorage, SelfAssignment) {
    DynD a(10, 10);
    fill_iota(a);
    DynD& alias = a;
    a           = alias;
    a           = std::move(alias);
    EXPECT_EQ(a.size(), 100);
    EXPECT_EQ(a[50], 50.0);
}

TEST(DynamicStorage, MoveAssignReleasesOldHeap) {
    DynD a(10, 10);
    DynD b(2, 2);
    fill_iota(b);
    a = std::move(b);
    EXPECT_FALSE(a.is_heap_allocated());
    EXPECT_EQ(a[3], 3.0);
}

TEST(DynamicStorage, SwapMixedStorage) {
    DynD small(2, 2);
    fill_iota(small);
    DynD large(10, 10);
    fill_iota(large);

    swap(small, large);
    EXPECT_TRUE(small.is_heap_allocated());
    EXPECT_EQ(small.size(), 100);
    EXPECT_EQ(small[99], 99.0);
    EXPECT_FALSE(large.is_heap_allocated());
    EXPECT_EQ(large.size(), 4);
    EXPECT_EQ(large[3], 3.0);
}

// ------------------------------------------------------------------------------------------------
// DynamicStorage: resize
// ------------------------------------------------------------------------------------------------

TEST(DynamicStorage, ResizeSameSizeKeepsElements) {
    DynD a(2, 3);
    fill_iota(a);
    a.resize(3, 2);
    EXPECT_EQ(a.rows(), 3);
    EXPECT_EQ(a.cols(), 2);
    EXPECT_EQ(a[5], 5.0);
}

TEST(DynamicStorage, ResizeMovesBetweenInlineAndHeap) {
    DynD a(2, 2);
    a.resize(40, 40);
    EXPECT_TRUE(a.is_heap_allocated());
    EXPECT_TRUE(is_aligned(a.data()));
    EXPECT_EQ(a.size(), 1600);

    a.resize(3, 3);
    EXPECT_FALSE(a.is_heap_allocated());
    EXPECT_EQ(a.size(), 9);

    a.resize(0, 0);
    EXPECT_EQ(a.size(), 0);
}

TEST(DynamicStorage, ResizeVector) {
    DynamicStorage<double, Dynamic, 1> vec;
    vec.resize(100, 1);
    EXPECT_EQ(vec.size(), 100);
    EXPECT_TRUE(vec.is_heap_allocated());
}

// ------------------------------------------------------------------------------------------------
// DynamicStorage: element types
// ------------------------------------------------------------------------------------------------

TEST(DynamicStorage, ComplexElements) {
    DynamicStorage<std::complex<double>, Dynamic, Dynamic> a(3, 3);
    a[0]   = {1.0, 2.0};
    auto b = a;
    EXPECT_EQ(b[0], std::complex<double>(1.0, 2.0));
    EXPECT_EQ(b[8], std::complex<double>());
}

TEST(DynamicStorage, NonTrivialElementsAreManagedCorrectly) {
    // std::string owns heap memory, so leaks or double frees show up under AddressSanitizer.
    DynamicStorage<std::string, Dynamic, Dynamic> a(6, 6);
    a[35]  = std::string(100, 'x');
    auto b = a;
    a      = std::move(b);
    a.resize(2, 2);
    a[0] = "small";
    DynamicStorage<std::string, Dynamic, Dynamic> c(8, 8);
    swap(a, c);
    EXPECT_EQ(c[0], "small");
    EXPECT_EQ(a.size(), 64);
}
