# mlinalg specification

_Note: This is the initial specification. Subject to change during the implementation._

**Target Standard:** C++20/23
**Target Architectures:** x86_64 (AVX2 + FMA), ARM64 (NEON)
**Design Philosophy:** Zero-overhead abstractions, cache-friendliness, expression templates for lazy evaluation, and strict separation of data storage from math logic.

## Module 1: Memory & Storage Policies
**Goal:** Guarantee cache-aligned memory and implement Small Buffer Optimization (SBO) to avoid heap allocation for small matrices.

*   **Alignment:** All data must be 32-byte aligned to support AVX2 memory operations natively (`alignas(32)`).
*   **Allocator:** Implement an `aligned_allocator<T, Alignment>` to be used for all heap allocations.
*   **Storage Policy Template:** `template <typename T, int Rows, int Cols>`
    *   If `Rows` and `Cols` are known at compile time (> 0), use an internal `std::array`-like `alignas(32)` static buffer.
    *   **Dimension Marker:** Use `-1` to denote `Dynamic` dimensions.
    *   **SBO Fallback:** If dimension is `Dynamic`, evaluate total requested size at runtime. If $Rows \times Cols \le 32$ elements (up to ~5x5 matrices + padding), allocate on a pre-allocated stack buffer. Only invoke `aligned_allocator` if size > 32 elements.

## Module 2: SIMD Abstraction Layer
**Goal:** Isolate platform-specific intrinsics from the rest of the codebase. Agents must NEVER write intrinsics directly in the matrix math logic.

*   **Structure:** Create a `Packet<T>` template specialized for `int`, `float`, `double`, and `std::complex<T>`.
*   **Hardware Mapping:**

| Architecture | Intrinsic Set | Vector Size | Elements (Float / Double) |
| :--- | :--- | :--- | :--- |
| **x86_64** | AVX2 + FMA | 256-bit | 8 / 4 |
| **ARM64** | NEON | 128-bit | 4 / 2 |

*   **Required `Packet<T>` Operations:**
    *   `load(const T* ptr)`, `store(T* ptr)`, `loadu`, `storeu` (unaligned fallbacks).
    *   `add`, `sub`, `mul`, `div`.
    *   `fma(a, b, c)` (computes $a \times b + c$).
    *   `broadcast(T val)`.
    *   `reduce_add()` (horizontal sum of packet elements).

## Module 3: Expression Templates (AST Frontend)
**Goal:** Eliminate temporary allocations during chained operations (e.g., `A = B + C * D`).

*   **Design:** Implement lightweight proxy objects representing nodes in an Abstract Syntax Tree (AST).
*   **Nodes required:** `MatrixWrapper`, `AddOp`, `SubOp`, `MulOp`, `ScalarMulOp`.
*   **Compile-Time Traits:** Each AST node must expose `Rows`, `Cols`, and `ScalarType` via traits. If combining `Dynamic` (-1) and static sizes, the result trait evaluates to `Dynamic`.
*   **Constexpr Support:** All proxy object construction and static dimension deduction must be `constexpr` to allow compile-time precomputations for static matrices.

## Module 4: The Matrix & Vector API
**Goal:** The user-facing frontend. A Vector is strictly a `Matrix<T, -1, 1>` or `Matrix<T, 1, -1>`.

*   **API Requirements:**
    *   Constructors: Default, sized (for dynamic), initializer list, and evaluation constructor (accepting an Expression Template).
    *   Overloaded Operators: `+`, `-`, `*` (matrix-matrix and matrix-scalar). These must return AST Proxy objects, **not** evaluated matrices.
    *   Assignment Operator (`operator=`): This is where the AST is evaluated and written to storage.
*   **Functionality Set:**
    *   `det()`: Determinant.
    *   `inv()`: Inverse.
    *   `norm1()`, `norm2()`, `norminf()`, `normF()` (Frobenius).
    *   `rank()` and `dim()`.

## Module 5: Math Evaluator & Threading
**Goal:** The backend that compiles the AST into highly unrolled, SIMD-vectorized, and multi-threaded loops.

*   **SIMD Loop Unrolling:** The evaluator must process data in blocks of `Packet<T>`. Use a main loop for aligned/vectorized chunks, and a scalar fallback loop for the remaining elements (tail handling).
*   **Lockless Multithreading Model:**
    *   Implement a `SpinWorkerPool` strictly initialized once on first use (or statically).
    *   Threads sleep on a `std::atomic<bool>` flag (using `std::atomic::wait` in C++20).
    *   The evaluator divides large matrix loops (e.g., dot products for matrix multiplication) into row-blocks.
    *   Pointers to the AST evaluation lambda and memory blocks are written to atomic variables. Workers wake up, compute their chunk, and atomically decrement a completion counter. No mutexes.
*   **Thresholding:** Threading is only dispatched if the estimated flop count (e.g., $M \times N \times K$ for multiplication) exceeds a defined heuristic threshold (e.g., 64x64 matrices). Otherwise, evaluation executes synchronously.
