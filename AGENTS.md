# Agent Guidelines (AGENTS.md)

**CRITICAL: You are an AI agent working on a high-performance C++20/23 linear algebra library. Read these rules before taking any action.**

## 1. The Source of Truth
*   **NEVER change `spec.md`.** It is the ultimate source of truth for the architecture. If you hit a roadblock, implement exactly what is in the spec, or ask the human for clarification.
*   Do not add new features, modules, or external dependencies outside of what is defined in `spec.md`.

## 2. File Structure & Boundaries
*   **DO NOT scan or modify** the following directories: `target/`, `build/`, `.gihub/`, `.git/`, `lib/`. (This prevents context bloat and breaking the repository).
*   **Source Code:** Put `.cpp` implementation files in the `src/` directory. Put public `.hpp` or `.h` header files in the `include/` directory.
*   **Tests:** All test files must go into the `test/` directory.
*   **Documentation:** All project documentation goes into the `docs/` directory.

## 3. Workflow & Git
*   **NEVER commit or push using git.** You are only allowed to stage changes (`git add`). The human developer will review and commit.
*   Make incremental, focused changes. Do not rewrite massive files all at once unless instructed.

## 4. Testing Guardrails
*   **ALWAYS run the tests** after modifying code to verify your changes.
*   **NEVER delete or comment out a failing test** just to make the build pass. If a test fails, your code is wrong. Fix the implementation.
*   If you genuinely believe a test contradicts `spec.md`, stop and ask the human.
*   The testing framework used is GoogleTest.

## 5. C++ & Architecture Specifics (Crucial for AI)
*   **Template Metaprogramming (TMP) Errors:** C++ template errors can be massive. When a build fails, do not panic and change the whole architecture. Look at the *first* instantiation error in the compiler output to find the root cause.
*   **Header-Heavy Design:** Because we are using Expression Templates and TMP, expect most of the logic to live in header files (`.hpp`). Ensure strict use of header guards.
*   **SIMD Intrinsics Containment:** NEVER write raw platform intrinsics (AVX/NEON) outside of the `Packet<T>` abstraction layer (Module 2). The math evaluator must only use the `Packet<T>` API.
*   **C++ Standard:** Stick strictly to C++20/23 features. Do not use legacy C++98/11 paradigms where modern equivalents (Concepts, constexpr, type traits) exist.
