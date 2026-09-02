// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "amx-config.hpp"
#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <numa.h>
#include <numeric> // for std::accumulate
#include <random>
#include <algorithm>

void mKernel_m16(uint16_t *C, float *temp, uint16_t* A, uint16_t* B,
                    long M, long K, long stride_K, long stride_M, long k_tag, long k_end,
                    uint8_t* vpermt2w_control)
{


    asm volatile(

    "BEGIN_KERNEKL_16x32_%=:                        \n"

    "   mov       %[A], %%rax                       \n"
    "   mov       %[B], %%rbx                       \n"
    "   mov       %[C], %%rcx                       \n"
    "   mov       %[M], %%r8                        \n"
    "   mov       %[K], %%r9                        \n"
    "   mov       %[stride_K], %%r11                \n"
    "   mov       %[vpermt2w_control], %%rdx        \n"
    "   mov       %[stride_M], %%r14                \n"
    "   mov       %[k_tag], %%r15                   \n"
    "   mov       %[k_end], %%r12                   \n"
    "   vpmovzxbw   (%%rdx), %%zmm31                \n"

    "   mov       %[temp], %%rdx                    \n"

    "   mov       %%r8, %%r13                       \n"
    "   shl       $6, %%r13                         \n"  // stride_A
//mask
    "   mov       $1, %%rdi                         \n"
    "   shlx      %%r8, %%rdi, %%rdi                \n"
    "   sub       $1, %%rdi                         \n"
    "   kmovq     %%rdi, %%k2                       \n"  // mask1

    "   mov      %%rsp, %%r10                       \n"
    "   and      $63, %%r10                         \n"
    "   addq     $1024, %%r10                       \n"
    "   subq     %%r10, %%rsp                       \n"
    "   kmovq    %%r10, %%k1                        \n"

    "   mov      %%r14, %%r10                       \n"
    "   shl      $4, %%r10                          \n"  //16*stride_M


//-----------------------------------------------------
    "   leaq        (%%rbx, %%r11, 8), %%rdi        \n"
    "   leaq        (%%rdi, %%r11, 8), %%rdi        \n"
//-----------------------------------------------------------
    "   cmp       $0, %%r15                         \n"  //判断K维度是否被切分
    "   jg        BEGIN_LOAD_%=                     \n"

    "   tilezero    %%tmm6                          \n"
    "   tilezero    %%tmm7                          \n"
    "   jmp         BEGIN_K_%=                      \n"

    "BEGIN_LOAD_%=:                                 \n"
    "   tileloadd   (%%rdx, %%r14), %%tmm6          \n"
    "   addq        %%r10, %%rdx                    \n"
    "   tileloadd   (%%rdx, %%r14), %%tmm7          \n"
    "   subq        %%r10, %%rdx                    \n"

    "BEGIN_K_%=:                                    \n"

    "   subq        $32, %%r9                       \n"
    "   tileloadd   (%%rax, %%r8, 4), %%tmm1        \n"
    "   addq        %%r13, %%rax                    \n"
    "   tileloadd   (%%rbx, %%r11, 1), %%tmm2       \n"
    "   addq        $64, %%rbx                      \n"
    "   tdpbf16ps   %%tmm1, %%tmm2, %%tmm6          \n"
    "   tileloadd   (%%rdi, %%r11, 1), %%tmm3       \n"
    "   addq        $64, %%rdi                      \n"
    "   tdpbf16ps   %%tmm1, %%tmm3, %%tmm7          \n"
    "   cmpq        $0, %%r9                        \n"
    "   jg          BEGIN_K_%=                      \n"

    "END_%=:                                        \n"

    "   cmp         $0, %%r12                       \n"
    "   je          8f                              \n"

    //temp
    "   tilestored   %%tmm6, (%%rdx, %%r14)         \n"
    "   addq         %%r10, %%rdx                   \n"
    "   tilestored   %%tmm7, (%%rdx, %%r14)         \n"

    "  jmp          2f                              \n"


    "8:                                             \n"
    "   mov         $2, %%rax                       \n"
    "   tilestored   %%tmm6, (%%rsp, %%r8, 4)       \n"

    "1:                                             \n"
    "   vmovups      (%%rsp), %%zmm0%{%%k2}         \n"
    "   vmovups      (%%rsp, %%r8, 4), %%zmm1%{%%k2}\n"
    "   vmovups      (%%rsp, %%r8, 8), %%zmm2%{%%k2}\n"
    "   leaq         (%%rsp, %%r8, 8), %%r9         \n"
    "   vmovups      (%%r9, %%r8, 4), %%zmm3%{%%k2}\n"

    "   vmovups      (%%r9, %%r8, 8), %%zmm4%{%%k2}\n"
    "   leaq         (%%r9, %%r8, 8), %%r9         \n"
    "   vmovups      (%%r9, %%r8, 4), %%zmm5%{%%k2}\n"
    "   vmovups      (%%r9, %%r8, 8), %%zmm6%{%%k2}\n"
    "   leaq         (%%r9, %%r8, 8), %%r9         \n"
    "   vmovups      (%%r9, %%r8, 4), %%zmm7%{%%k2}\n"

    "   vmovups      (%%r9, %%r8, 8), %%zmm8%{%%k2}\n"
    "   leaq         (%%r9, %%r8, 8), %%r9         \n"
    "   vmovups      (%%r9, %%r8, 4), %%zmm9%{%%k2}\n"
    "   vmovups      (%%r9, %%r8, 8), %%zmm10%{%%k2}\n"
    "   leaq         (%%r9, %%r8, 8), %%r9         \n"
    "   vmovups      (%%r9, %%r8, 4), %%zmm11%{%%k2}\n"

    "   vmovups      (%%r9, %%r8, 8), %%zmm12%{%%k2}\n"
    "   leaq         (%%r9, %%r8, 8), %%r9         \n"
    "   vmovups      (%%r9, %%r8, 4), %%zmm13%{%%k2}\n"
    "   vmovups      (%%r9, %%r8, 8), %%zmm14%{%%k2}\n"
    "   leaq         (%%r9, %%r8, 8), %%r9         \n"
    "   vmovups      (%%r9, %%r8, 4), %%zmm15%{%%k2}\n"

    "   vcvtne2ps2bf16     %%zmm0, %%zmm1, %%zmm0   \n"
    "   vpermt2w    %%zmm0, %%zmm31, %%zmm0         \n"
    "   vcvtne2ps2bf16     %%zmm2, %%zmm3, %%zmm1   \n"
    "   vpermt2w    %%zmm1, %%zmm31, %%zmm1         \n"
    "   vcvtne2ps2bf16     %%zmm4, %%zmm5, %%zmm2   \n"
    "   vpermt2w    %%zmm2, %%zmm31, %%zmm2         \n"
    "   vcvtne2ps2bf16     %%zmm6, %%zmm7, %%zmm3   \n"
    "   vpermt2w    %%zmm3, %%zmm31, %%zmm3         \n"


    "   vcvtne2ps2bf16     %%zmm8, %%zmm9, %%zmm4   \n"
    "   vpermt2w    %%zmm4, %%zmm31, %%zmm4         \n"
    "   vcvtne2ps2bf16     %%zmm10, %%zmm11, %%zmm5 \n"
    "   vpermt2w    %%zmm5, %%zmm31, %%zmm5         \n"
    "   vcvtne2ps2bf16     %%zmm12, %%zmm13, %%zmm6 \n"
    "   vpermt2w    %%zmm6, %%zmm31, %%zmm6         \n"
    "   vcvtne2ps2bf16     %%zmm14, %%zmm15, %%zmm7 \n"
    "   vpermt2w    %%zmm7, %%zmm31, %%zmm7         \n"

    "   vmovups    %%zmm0, (%%rcx)%{%%k2}           \n"
    "   vmovups    %%zmm1, (%%rcx, %%r8, 4)%{%%k2}  \n"
    "   vmovups    %%zmm2, (%%rcx, %%r8, 8)%{%%k2}  \n"
    "   leaq       (%%rcx, %%r8, 8), %%rcx          \n"
    "   vmovups    %%zmm3, (%%rcx, %%r8, 4)%{%%k2}  \n"
    "   vmovups    %%zmm4, (%%rcx, %%r8, 8)%{%%k2}  \n"
    "   leaq       (%%rcx, %%r8, 8), %%rcx          \n"
    "   vmovups    %%zmm5, (%%rcx, %%r8, 4)%{%%k2}  \n"
    "   vmovups    %%zmm6, (%%rcx, %%r8, 8)%{%%k2}  \n"
    "   leaq       (%%rcx, %%r8, 8), %%rcx          \n"
    "   vmovups    %%zmm7, (%%rcx, %%r8, 4)%{%%k2}  \n"
    "   leaq       (%%rcx, %%r8, 8), %%rcx          \n"

    "   subq        $1, %%rax                       \n"
    "   cmp         $1, %%rax                       \n"
    "   jl          2f                              \n"

    "   tilestored   %%tmm7, (%%rsp, %%r8, 4)       \n"
    "   jmp         1b                              \n"

    "2:                                             \n"

    "   kmovq       %%k1, %%R12                     \n"
    "   addq        %%r12, %%rsp                    \n"

    :
    :
      [A] "m" (A),
      [B] "m" (B),
      [C] "m" (C),
      [temp]"m"(temp),
      [M] "m" (M),
      [K] "m" (K),
      [k_tag]"m"(k_tag),
      [k_end]"m"(k_end),
      [stride_K]"m"(stride_K),
      [stride_M]"m"(stride_M),
      [vpermt2w_control]"m"(vpermt2w_control)
    :"rax", "rbx", "rcx", "rdx", "rdi", "rsi","r8", "r9", "r10", "r11", "r12",
     "r13", "r14", "r15","k1","k2","k3","zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
     "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
     "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
     "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
     "zmm30", "zmm31", "memory"
    );

}


void mKernel_m32(uint16_t *C, float *temp, uint16_t* A, uint16_t* B,
                long K, long stride_N,  long stride_K,
                long stride_M, long k_tag, long k_end,
                uint8_t* vpermt2w_control)
{

    asm volatile(


    "BEGIN_KERNEKL_32x32_%=:                        \n"

    "   mov       %[A], %%rax                       \n"
    "   mov       %[B], %%rbx                       \n"
    "   mov       %[C], %%rcx                       \n"
    "   mov       %[K], %%r9                        \n"
    "   mov       %[temp], %%rdx                    \n"
    "   mov       %[stride_K], %%r11                \n"
    "   mov       %[stride_N], %%r10                \n"
    "   kmovq     %%r10, %%k3                       \n"
    "   mov       %[k_tag], %%r15                   \n"
    "   mov       %[k_end], %%r12                   \n"

    "   mov       %[vpermt2w_control], %%r14        \n"
    "   vpmovzxbw   (%%r14), %%zmm31                \n"
    "   mov       %[stride_M], %%r14                \n"

    "   mov      %%rsp, %%r10                       \n"
    "   and      $63, %%r10                         \n"
    "   addq     $1024, %%r10                       \n"
    "   subq     %%r10, %%rsp                       \n"
    "   kmovq    %%r10, %%k1                        \n"

    "   mov      %%r14, %%r13                       \n"
    "   shl      $4, %%r13                          \n"


    "   mov         $64, %%r8                       \n"
    "   leaq        (%%rax, %%r11, 8), %%rsi        \n"
    "   leaq        (%%rsi, %%r11, 8), %%rsi        \n"

//-----------------------------------------------------
    "   leaq        (%%rbx, %%r11, 8), %%rdi        \n"
    "   leaq        (%%rdi, %%r11, 8), %%rdi        \n"

    "   cmp       $0, %%r15                         \n"
    "   jg        BEGIN_LOAD_%=                     \n"

    "   tilezero    %%tmm4                          \n"
    "   tilezero    %%tmm5                          \n"
    "   tilezero    %%tmm6                          \n"
    "   tilezero    %%tmm7                          \n"
    "   jmp         BEGIN_Reduce_K_%=               \n"


    "BEGIN_LOAD_%=:                                 \n"
    "   tileloadd   (%%rdx, %%r14), %%tmm4          \n"
    "   tileloadd   64(%%rdx, %%r14), %%tmm6        \n"
    "   addq        %%r13, %%rdx                    \n"
    "   tileloadd   (%%rdx, %%r14), %%tmm5          \n"
    "   tileloadd   64(%%rdx, %%r14), %%tmm7        \n"
    "   subq        %%r13, %%rdx                    \n"

//-----------------------------------------------------------
    "BEGIN_Reduce_K_%=:                             \n"

    "   subq        $32, %%r9                       \n"
    "   tileloadd   (%%rax, %%r8), %%tmm0           \n"
    "   addq        $1024, %%rax                    \n"
    "   tileloadd   (%%rbx, %%r11), %%tmm2          \n"
    "   addq        $64, %%rbx                      \n"
    "   tdpbf16ps   %%tmm0, %%tmm2, %%tmm4          \n"
    "   tileloadd   (%%rdi, %%r11), %%tmm3          \n"
    "   addq        $64, %%rdi                      \n"
    "   tdpbf16ps   %%tmm0, %%tmm3, %%tmm5          \n"

    "   tileloadd   (%%rsi, %%r8), %%tmm1           \n"
    "   addq        $1024, %%rsi                    \n"
    "   tdpbf16ps   %%tmm1, %%tmm2, %%tmm6          \n"
    "   tdpbf16ps   %%tmm1, %%tmm3, %%tmm7          \n"

    "   cmpq        $0, %%r9                        \n"
    "   jg          BEGIN_Reduce_K_%=               \n"


//-----------------------------------------------------------

    "   cmp         $0, %%r12                       \n"
    "   je          8f                              \n"

    "   tilestored   %%tmm4, (%%rdx, %%r14)         \n"
    "   tilestored   %%tmm6, 64(%%rdx, %%r14)       \n"
    "   addq        %%r13, %%rdx                    \n"
    "   tilestored   %%tmm5, (%%rdx, %%r14)         \n"
    "   tilestored   %%tmm7, 64(%%rdx, %%r14)       \n"
    "  jmp          5f                              \n"

    "8:                                             \n"


//存结果
    "   mov         $4, %%rax                       \n"
    "   tilestored   %%tmm4, (%%rsp, %%r8)          \n"

    "0:                                             \n"
    "   vmovups      (%%rsp), %%zmm0                \n"
    "   vmovups      64(%%rsp), %%zmm1              \n"
    "   vmovups      128(%%rsp), %%zmm2             \n"
    "   vmovups      192(%%rsp), %%zmm3             \n"
    "   vmovups      256(%%rsp), %%zmm4             \n"
    "   vmovups      320(%%rsp), %%zmm5             \n"
    "   vmovups      384(%%rsp), %%zmm6             \n"
    "   vmovups      448(%%rsp), %%zmm7             \n"
    "   vmovups      512(%%rsp), %%zmm8             \n"
    "   vmovups      576(%%rsp), %%zmm9             \n"
    "   vmovups      640(%%rsp), %%zmm10            \n"
    "   vmovups      704(%%rsp), %%zmm11            \n"
    "   vmovups      768(%%rsp), %%zmm12            \n"
    "   vmovups      832(%%rsp), %%zmm13            \n"
    "   vmovups      896(%%rsp), %%zmm14            \n"
    "   vmovups      960(%%rsp), %%zmm15            \n"

    "   vcvtne2ps2bf16     %%zmm0, %%zmm1, %%zmm0   \n"
    "   vpermt2w    %%zmm0, %%zmm31, %%zmm0         \n"
    "   vcvtne2ps2bf16     %%zmm2, %%zmm3, %%zmm1   \n"
    "   vpermt2w    %%zmm1, %%zmm31, %%zmm1         \n"
    "   vcvtne2ps2bf16     %%zmm4, %%zmm5, %%zmm2   \n"
    "   vpermt2w    %%zmm2, %%zmm31, %%zmm2         \n"
    "   vcvtne2ps2bf16     %%zmm6, %%zmm7, %%zmm3   \n"
    "   vpermt2w    %%zmm3, %%zmm31, %%zmm3         \n"


    "   vcvtne2ps2bf16     %%zmm8, %%zmm9, %%zmm4   \n"
    "   vpermt2w    %%zmm4, %%zmm31, %%zmm4         \n"
    "   vcvtne2ps2bf16     %%zmm10, %%zmm11, %%zmm5 \n"
    "   vpermt2w    %%zmm5, %%zmm31, %%zmm5         \n"
    "   vcvtne2ps2bf16     %%zmm12, %%zmm13, %%zmm6 \n"
    "   vpermt2w    %%zmm6, %%zmm31, %%zmm6         \n"
    "   vcvtne2ps2bf16     %%zmm14, %%zmm15, %%zmm7 \n"
    "   vpermt2w    %%zmm7, %%zmm31, %%zmm7         \n"

    "   vmovdqu16    %%zmm0, (%%rcx)                \n"
    "   vmovdqu16    %%zmm1, 64(%%rcx)              \n"
    "   vmovdqu16    %%zmm2, 128(%%rcx)             \n"
    "   vmovdqu16    %%zmm3, 192(%%rcx)             \n"
    "   vmovdqu16    %%zmm4, 256(%%rcx)             \n"
    "   vmovdqu16    %%zmm5, 320(%%rcx)             \n"
    "   vmovdqu16    %%zmm6, 384(%%rcx)             \n"
    "   vmovdqu16    %%zmm7, 448(%%rcx)             \n"

    "   subq        $1, %%rax                       \n"
    "   cmp         $0, %%rax                       \n"
    "   je          5f                              \n"

    "2:                                             \n"
    "   addq        $512, %%rcx                     \n"
    "   cmp         $3, %%rax                       \n"
    "   jl          3f                              \n"
    "   tilestored   %%tmm5, (%%rsp, %%r8)          \n"
    "   jmp         0b                              \n"

    "3:                                             \n"

    "   cmp         $1, %%rax                       \n"
    "   je          4f                              \n"

    "   kmovq       %%k3, %%rsi                     \n"
    "   shl         $4, %%rsi                       \n"  // 16 * N * 2
    "   leaq        -1024(%%rcx, %%rsi), %%rcx      \n"
    "   tilestored   %%tmm6, (%%rsp, %%r8)          \n"
    "   jmp         0b                              \n"

    "4:                                             \n"
    "   tilestored   %%tmm7, (%%rsp, %%r8)          \n"
    "   jmp         0b                              \n"

    "5:                                             \n"

    "   kmovq    %%k1, %%r12                        \n"
    "   addq     %%r12, %%rsp                       \n"


    :
    :
      [A] "m" (A),
      [B] "m" (B),
      [C] "m" (C),
      [temp]"m"(temp),
      [K] "m" (K),
      [k_tag]"m"(k_tag),
      [k_end]"m"(k_end),
      [stride_N]"m"(stride_N),
      [stride_K]"m"(stride_K),
      [stride_M]"m"(stride_M),
      [vpermt2w_control]"m"(vpermt2w_control)
    :"rax", "rbx", "rcx", "rdx", "rdi", "rsi","r8", "r9", "r10", "r11", "r12",
     "r13", "r14", "r15","k1","k2","k3","k4","zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
     "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
     "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
     "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
     "zmm30", "zmm31", "memory"
    );

}


void mKernel_m16_down(uint16_t *C, float *temp, uint16_t* A, uint16_t* B,
                    long M, long K, long stride_N, long stride_K,
                    long stride_M, long k_tag, long k_end,
                    uint8_t* vpermt2w_control)
{


    asm volatile(

    "mKernel_m16_down%=:                            \n"

    "   mov       %[A], %%rax                       \n"
    "   mov       %[B], %%rbx                       \n"
    "   mov       %[C], %%rcx                       \n"
    "   mov       %[M], %%r8                        \n"
    "   mov       %[K], %%r9                        \n"
    "   mov       %[stride_K], %%r11                \n"
    "   mov       %[vpermt2w_control], %%rdx        \n"
    "   mov       %[stride_M], %%r14                \n"
    "   mov       %[k_tag], %%r15                   \n"
    "   mov       %[k_end], %%r12                   \n"

    "   mov       %[stride_N], %%r10                \n"
    "   kmovq     %%r10, %%k3                       \n"
    "   vpmovzxbw   (%%rdx), %%zmm31                \n"
    "   mov       %[temp], %%rdx                    \n"

    "   mov       %%r8, %%r13                       \n"
    "   shl       $6, %%r13                         \n"  // stride_A

//mask
    "   mov       $1, %%rdi                         \n"
    "   shlx      %%r8, %%rdi, %%rdi                \n"
    "   sub       $1, %%rdi                         \n"
    "   kmovq     %%rdi, %%k2                       \n"  // mask1

    "   mov      %%rsp, %%r10                       \n"
    "   and      $63, %%r10                         \n"
    "   addq     $1024, %%r10                       \n"
    "   subq     %%r10, %%rsp                       \n"
    "   kmovq    %%r10, %%k1                        \n"

    "   mov      %%r14, %%r10                       \n"
    "   shl      $4, %%r10                          \n"


//-----------------------------------------------------
    "   leaq        (%%rbx, %%r11, 8), %%rdi        \n"
    "   leaq        (%%rdi, %%r11, 8), %%rdi        \n"
//-----------------------------------------------------

    "   cmp       $0, %%r15                         \n"  //判断K维度是否被切分
    "   jg        BEGIN_LOAD_%=                     \n"

    "   tilezero    %%tmm6                          \n"
    "   tilezero    %%tmm7                          \n"
    "   jmp         BEGIN_K_%=                      \n"

    "BEGIN_LOAD_%=:                                 \n"
    "   tileloadd   (%%rdx, %%r14), %%tmm6          \n"
    "   addq        %%r10, %%rdx                    \n"
    "   tileloadd   (%%rdx, %%r14), %%tmm7          \n"
    "   subq        %%r10, %%rdx                    \n"

    "BEGIN_K_%=:                                    \n"

    "   subq        $32, %%r9                       \n"
    "   tileloadd   (%%rax, %%r8, 4), %%tmm1        \n"
    "   addq        %%r13, %%rax                    \n"
    "   tileloadd   (%%rbx, %%r11, 1), %%tmm2       \n"
    "   addq        $64, %%rbx                      \n"
    "   tdpbf16ps   %%tmm1, %%tmm2, %%tmm6          \n"
    "   tileloadd   (%%rdi, %%r11, 1), %%tmm3       \n"
    "   addq        $64, %%rdi                      \n"
    "   tdpbf16ps   %%tmm1, %%tmm3, %%tmm7          \n"
    "   cmpq        $0, %%r9                        \n"
    "   jg          BEGIN_K_%=                      \n"


    "END_%=:                                        \n"


    "   cmp         $0, %%r12                       \n"
    "   je          BEGINN_Store_%=                 \n"

    //temp
    "   tilestored   %%tmm6, (%%rdx, %%r14)         \n"
    "   addq         %%r10, %%rdx                   \n"
    "   tilestored   %%tmm7, (%%rdx, %%r14)         \n"
    "   jmp         5f                              \n"


    "BEGINN_Store_%=:                               \n"

    "   mov         $2, %%rax                       \n"
    "   tilestored   %%tmm6, (%%rsp, %%r8, 4)       \n"
    "   kmovq       %%k3, %%r10                     \n"//stride_N

    "0:                                             \n"
    "   vmovups      (%%rsp), %%zmm0%{%%k2}         \n"
    "   vmovups      (%%rsp, %%r8, 4), %%zmm1%{%%k2}\n"
    "   vmovups      (%%rsp, %%r8, 8), %%zmm2%{%%k2}\n"
    "   leaq         (%%rsp, %%r8, 8), %%r9         \n"
    "   vmovups      (%%r9, %%r8, 4), %%zmm3%{%%k2}\n"

    "   vmovups      (%%r9, %%r8, 8), %%zmm4%{%%k2}\n"
    "   leaq         (%%r9, %%r8, 8), %%r9         \n"
    "   vmovups      (%%r9, %%r8, 4), %%zmm5%{%%k2}\n"
    "   vmovups      (%%r9, %%r8, 8), %%zmm6%{%%k2}\n"
    "   leaq         (%%r9, %%r8, 8), %%r9         \n"
    "   vmovups      (%%r9, %%r8, 4), %%zmm7%{%%k2}\n"

    "   vmovups      (%%r9, %%r8, 8), %%zmm8%{%%k2}\n"
    "   leaq         (%%r9, %%r8, 8), %%r9         \n"
    "   vmovups      (%%r9, %%r8, 4), %%zmm9%{%%k2}\n"
    "   vmovups      (%%r9, %%r8, 8), %%zmm10%{%%k2}\n"
    "   leaq         (%%r9, %%r8, 8), %%r9         \n"
    "   vmovups      (%%r9, %%r8, 4), %%zmm11%{%%k2}\n"

    "   vmovups      (%%r9, %%r8, 8), %%zmm12%{%%k2}\n"
    "   leaq         (%%r9, %%r8, 8), %%r9         \n"
    "   vmovups      (%%r9, %%r8, 4), %%zmm13%{%%k2}\n"
    "   vmovups      (%%r9, %%r8, 8), %%zmm14%{%%k2}\n"
    "   leaq         (%%r9, %%r8, 8), %%r9         \n"
    "   vmovups      (%%r9, %%r8, 4), %%zmm15%{%%k2}\n"

    "   vcvtne2ps2bf16     %%zmm0, %%zmm1, %%zmm0   \n"
    "   vpermt2w    %%zmm0, %%zmm31, %%zmm0         \n"
    "   vcvtne2ps2bf16     %%zmm2, %%zmm3, %%zmm1   \n"
    "   vpermt2w    %%zmm1, %%zmm31, %%zmm1         \n"
    "   vcvtne2ps2bf16     %%zmm4, %%zmm5, %%zmm2   \n"
    "   vpermt2w    %%zmm2, %%zmm31, %%zmm2         \n"
    "   vcvtne2ps2bf16     %%zmm6, %%zmm7, %%zmm3   \n"
    "   vpermt2w    %%zmm3, %%zmm31, %%zmm3         \n"


    "   vcvtne2ps2bf16     %%zmm8, %%zmm9, %%zmm4   \n"
    "   vpermt2w    %%zmm4, %%zmm31, %%zmm4         \n"
    "   vcvtne2ps2bf16     %%zmm10, %%zmm11, %%zmm5 \n"
    "   vpermt2w    %%zmm5, %%zmm31, %%zmm5         \n"
    "   vcvtne2ps2bf16     %%zmm12, %%zmm13, %%zmm6 \n"
    "   vpermt2w    %%zmm6, %%zmm31, %%zmm6         \n"
    "   vcvtne2ps2bf16     %%zmm14, %%zmm15, %%zmm7 \n"
    "   vpermt2w    %%zmm7, %%zmm31, %%zmm7         \n"


    "   vunpcklps %%zmm1, %%zmm0, %%zmm24                   \n"
    "   vunpcklps %%zmm3, %%zmm2, %%zmm25                   \n"
    "   vunpcklpd %%zmm25, %%zmm24, %%zmm26                 \n"

    "   vunpcklps %%zmm5, %%zmm4, %%zmm29                   \n"
    "   vunpcklps %%zmm7, %%zmm6, %%zmm30                   \n"
    "   vunpcklpd %%zmm30, %%zmm29, %%zmm27                 \n"

    "   vinsertf32x4    $1, %%xmm27, %%ymm26, %%ymm8        \n"
    "   vmovdqu16    %%ymm8, (%%rcx)                        \n"
    "   cmp     $5, %%r8                                    \n"
    "   jl      1f                                          \n"

    "   vextractf32x4   $1, %%zmm26, %%xmm9                 \n"
    "   vextractf32x4   $1, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm9, %%ymm9         \n"

    "   vmovdqu16    %%ymm9, (%%rcx, %%r10, 4)              \n"
    "   cmp     $9, %%r8                                    \n"
    "   jl      1f                                          \n"

    "   vextractf32x4   $2, %%zmm26, %%xmm10                \n"
    "   vextractf32x4   $2, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm10, %%ymm10       \n"

    "   vmovdqu16    %%ymm10, (%%rcx, %%r10, 8)             \n"
    "   cmp     $13, %%r8                                    \n"
    "   jl      1f                                          \n"

    "   vextractf32x4   $3, %%zmm26, %%xmm11                \n"
    "   vextractf32x4   $3, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm11, %%ymm11       \n"

    "   leaq        (%%rcx, %%r10, 8), %%r14                \n"
    "   vmovdqu16    %%ymm11, (%%r14, %%r10, 4)             \n"

//--------------------------------------------------------------------

    "1:                                                     \n"
    "   cmp     $2, %%r8                                    \n"
    "   jl      4f                                          \n"

    "   vunpckhpd %%zmm25, %%zmm24, %%zmm26                 \n"
    "   vunpckhpd %%zmm30, %%zmm29, %%zmm27                 \n"

    "   vinsertf32x4    $1, %%xmm27, %%ymm26, %%ymm12       \n"
    "   vmovdqu16    %%ymm12, (%%rcx, %%r10)                \n"
    "   cmp     $6, %%r8                                    \n"
    "   jl      2f                                          \n"
    "   vextractf32x4   $1, %%zmm26, %%xmm13                \n"
    "   vextractf32x4   $1, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm13, %%ymm13       \n"

    "   leaq        (%%rcx, %%r10), %%r14                   \n"
    "   vmovdqu16    %%ymm13, (%%r14, %%r10, 4)             \n"
    "   cmp     $10, %%r8                                   \n"
    "   jl      2f                                          \n"

    "   vextractf32x4   $2, %%zmm26, %%xmm14                \n"
    "   vextractf32x4   $2, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm14, %%ymm14       \n"

    "   vmovdqu16    %%ymm14, (%%r14, %%r10,8)              \n"

    "   cmp     $14, %%r8                                   \n"
    "   jl      2f                                          \n"

    "   vextractf32x4   $3, %%zmm26, %%xmm15                \n"
    "   vextractf32x4   $3, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm15, %%ymm15       \n"

    "   leaq        (%%r14, %%r10, 8), %%r14                \n"
    "   vmovdqu16    %%ymm15, (%%r14, %%r10, 4)             \n"

//--------------------------------------------------------------------

    "2:                                                     \n"
    "   cmp      $3, %%r8                                   \n"
    "   jl       4f                                         \n"

    "   vunpckhps %%zmm1, %%zmm0, %%zmm24                   \n"
    "   vunpckhps %%zmm3, %%zmm2, %%zmm25                   \n"
    "   vunpcklpd %%zmm25, %%zmm24, %%zmm26                 \n"

    "   vunpckhps %%zmm5, %%zmm4, %%zmm29                   \n"
    "   vunpckhps %%zmm7, %%zmm6, %%zmm30                   \n"
    "   vunpcklpd %%zmm30, %%zmm29, %%zmm27                 \n"

    "   vinsertf32x4    $1, %%xmm27, %%ymm26, %%ymm16       \n"
    "   vmovdqu16    %%ymm16, (%%rcx, %%r10, 2)             \n"
    "   cmp     $7, %%r8                                    \n"
    "   jl      3f                                          \n"
    "   vextractf32x4   $1, %%zmm26, %%xmm17                \n"
    "   vextractf32x4   $1, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm17, %%ymm17       \n"

    "   leaq        (%%rcx, %%r10, 2), %%r14                \n"
    "   vmovdqu16    %%ymm17, (%%r14, %%r10, 4)             \n"
    "   cmp     $11, %%r8                                   \n"
    "   jl      3f                                          \n"

    "   vextractf32x4   $2, %%zmm26, %%xmm18                \n"
    "   vextractf32x4   $2, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm18, %%ymm18       \n"

    "   vmovdqu16    %%ymm18, (%%r14, %%r10, 8)             \n"
    "   cmp     $15, %%r8                                   \n"
    "   jl      3f                                          \n"

    "   vextractf32x4   $3, %%zmm26, %%xmm19                \n"
    "   vextractf32x4   $3, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm19, %%ymm19       \n"

    "   leaq        (%%r14, %%r10, 8), %%r14                \n"
    "   vmovdqu16    %%ymm19, (%%r14, %%r10, 4)             \n"


//--------------------------------------------------------------------

    "3:                                                    \n"
    "   cmp      $4, %%r8                                  \n"
    "   jl       4f                                        \n"

    "   vunpckhpd %%zmm25, %%zmm24, %%zmm26                \n"
    "   vunpckhpd %%zmm30, %%zmm29, %%zmm27                \n"

    "   vinsertf32x4    $1, %%xmm27, %%ymm26, %%ymm20       \n"
    "   leaq            (%%rcx, %%r10, 2), %%r14            \n"
    "   leaq            (%%r14, %%r10), %%r14               \n"
    "   vmovdqu16    %%ymm20, (%%r14)                       \n"
    "   cmp          $8, %%r8                               \n"
    "   jl           4f                                     \n"

    "   vextractf32x4   $1, %%zmm26, %%xmm21                \n"
    "   vextractf32x4   $1, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm21, %%ymm21       \n"

    "   vmovdqu16    %%ymm21, (%%r14, %%r10, 4)             \n"
    "   cmp     $12, %%r8                                   \n"
    "   jl      4f                                          \n"

    "   vextractf32x4   $2, %%zmm26, %%xmm22                \n"
    "   vextractf32x4   $2, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm22, %%ymm22       \n"

    "   vmovdqu16    %%ymm22, (%%r14, %%r10, 8)             \n"
    "   cmp     $16, %%r8                                   \n"
    "   jl      4f                                          \n"

    "   vextractf32x4   $3, %%zmm26, %%xmm23                \n"
    "   vextractf32x4   $3, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm23, %%ymm23       \n"

    "   leaq        (%%r14, %%r10, 8), %%r14                \n"
    "   vmovdqu16    %%ymm23, (%%r14, %%r10, 4)             \n"

    "4:                                             \n"
    "   subq        $1, %%rax                       \n"
    "   cmp         $1, %%rax                       \n"
    "   jl          5f                              \n"

    "   addq        $32, %%rcx                      \n"
    "   tilestored   %%tmm7, (%%rsp, %%r8, 4)       \n"
    "   jmp         0b                              \n"

    "5:                                             \n"

    "   kmovq    %%k1, %%r12                        \n"
    "   addq        %%r12, %%rsp                    \n"

    :
    :
      [A] "m" (A),
      [B] "m" (B),
      [C] "m" (C),
      [temp]"m"(temp),
      [M] "m" (M),
      [K] "m" (K),
      [k_tag]"m"(k_tag),
      [k_end]"m"(k_end),
      [stride_N]"m"(stride_N),
      [stride_K]"m"(stride_K),
      [stride_M]"m"(stride_M),
      [vpermt2w_control]"m"(vpermt2w_control)
    :"rax", "rbx", "rcx", "rdx", "rdi", "rsi","r8", "r9", "r10", "r11", "r12",
     "r13", "r14", "r15","k2","k3","zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
     "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
     "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
     "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
     "zmm30", "zmm31", "memory"
    );

}


void mKernel_m32_down(uint16_t *C, float* temp, uint16_t* A, uint16_t* B,
                long K, long stride_N, long stride_K, long stride_M,
                long k_tag, long k_end,
                uint8_t* vpermt2w_control)
{

    asm volatile(


    "mKernel_m32_down%=:                            \n"

    "   mov       %[A], %%rax                       \n"
    "   mov       %[B], %%rbx                       \n"
    "   mov       %[C], %%rcx                       \n"
    "   mov       %[temp], %%rdx                    \n"
    "   mov       %[K], %%r9                        \n"
    "   mov       %[stride_N], %%r10                \n"
    "   mov       %[stride_K], %%r11                \n"
    "   kmovq     %%r10, %%k3                       \n"

    "   mov       %[k_tag], %%r15                   \n"
    "   mov       %[k_end], %%r12                   \n"
    "   mov       %[vpermt2w_control], %%r14        \n"
    "   vpmovzxbw   (%%r14), %%zmm31                \n"
    "   mov       %[stride_M], %%r14                \n"


    "   mov      %%rsp, %%r10                       \n"
    "   and      $63, %%r10                         \n"
    "   addq     $1024, %%r10                       \n"
    "   subq     %%r10, %%rsp                       \n"
    "   kmovq    %%r10, %%k1                        \n"

    "   mov      %%r14, %%r13                       \n"
    "   shl      $4, %%r13                          \n"

    "   mov         $64, %%r8                       \n"
    "   leaq        (%%rax, %%r11, 8), %%rsi        \n"
    "   leaq        (%%rsi, %%r11, 8), %%rsi        \n"
//-----------------------------------------------------
    "   leaq        (%%rbx, %%r11, 8), %%rdi        \n"
    "   leaq        (%%rdi, %%r11, 8), %%rdi        \n"


    "   cmp       $0, %%r15                         \n"
    "   jg        BEGIN_LOAD_%=                     \n"

    "   tilezero    %%tmm4                          \n"
    "   tilezero    %%tmm5                          \n"
    "   tilezero    %%tmm6                          \n"
    "   tilezero    %%tmm7                          \n"
    "   jmp         BEGIN_Reduce_K_%=               \n"

    "BEGIN_LOAD_%=:                                 \n"
    "   tileloadd   (%%rdx, %%r14), %%tmm4          \n"
    "   tileloadd   64(%%rdx, %%r14), %%tmm6        \n"
    "   addq        %%r13, %%rdx                    \n"
    "   tileloadd   (%%rdx, %%r14), %%tmm5          \n"
    "   tileloadd   64(%%rdx, %%r14), %%tmm7        \n"
    "   subq        %%r13, %%rdx                    \n"

//-----------------------------------------------------------
    "BEGIN_Reduce_K_%=:                             \n"

    "   subq        $32, %%r9                       \n"
    "   tileloadd   (%%rax, %%r8), %%tmm0           \n"
    "   addq        $1024, %%rax                    \n"
    "   tileloadd   (%%rbx, %%r11), %%tmm2          \n"
    "   addq        $64, %%rbx                      \n"
    "   tdpbf16ps   %%tmm0, %%tmm2, %%tmm4          \n"
    "   tileloadd   (%%rdi, %%r11), %%tmm3          \n"
    "   addq        $64, %%rdi                      \n"
    "   tdpbf16ps   %%tmm0, %%tmm3, %%tmm5          \n"

    "   tileloadd   (%%rsi, %%r8), %%tmm1           \n"
    "   addq        $1024, %%rsi                   \n"
    "   tdpbf16ps   %%tmm1, %%tmm2, %%tmm6          \n"
    "   tdpbf16ps   %%tmm1, %%tmm3, %%tmm7          \n"

    "   cmpq        $0, %%r9                        \n"
    "   jg          BEGIN_Reduce_K_%=               \n"

//-----------------------------------------------------------
    "   cmp         $0, %%r12                       \n"
    "   je          8f                              \n"

    "   tilestored   %%tmm4, (%%rdx, %%r14)         \n"
    "   tilestored   %%tmm6, 64(%%rdx, %%r14)       \n"
    "   addq        %%r13, %%rdx                    \n"
    "   tilestored   %%tmm5, (%%rdx, %%r14)         \n"
    "   tilestored   %%tmm7, 64(%%rdx, %%r14)       \n"
    "   subq        %%r13, %%rdx                    \n"

    "  jmp          5f                              \n"

    "8:                                             \n"


//存结果
    "   mov         $4, %%rax                       \n"
    "   kmovq       %%k3, %%r10                     \n"
    "   tilestored   %%tmm4, (%%rsp, %%r8)          \n"


    "0:                                             \n"
    "   vmovups      (%%rsp), %%zmm0                \n"
    "   vmovups      64(%%rsp), %%zmm1              \n"
    "   vmovups      128(%%rsp), %%zmm2             \n"
    "   vmovups      192(%%rsp), %%zmm3             \n"

    "   vmovups      256(%%rsp), %%zmm4             \n"
    "   vmovups      320(%%rsp), %%zmm5             \n"
    "   vmovups      384(%%rsp), %%zmm6             \n"
    "   vmovups      448(%%rsp), %%zmm7             \n"

    "   vmovups      512(%%rsp), %%zmm8             \n"
    "   vmovups      576(%%rsp), %%zmm9             \n"
    "   vmovups      640(%%rsp), %%zmm10            \n"
    "   vmovups      704(%%rsp), %%zmm11            \n"

    "   vmovups      768(%%rsp), %%zmm12            \n"
    "   vmovups      832(%%rsp), %%zmm13            \n"
    "   vmovups      896(%%rsp), %%zmm14            \n"
    "   vmovups      960(%%rsp), %%zmm15            \n"

    "   vcvtne2ps2bf16     %%zmm0, %%zmm1, %%zmm0   \n"
    "   vpermt2w    %%zmm0, %%zmm31, %%zmm0         \n"
    "   vcvtne2ps2bf16     %%zmm2, %%zmm3, %%zmm1   \n"
    "   vpermt2w    %%zmm1, %%zmm31, %%zmm1         \n"
    "   vcvtne2ps2bf16     %%zmm4, %%zmm5, %%zmm2   \n"
    "   vpermt2w    %%zmm2, %%zmm31, %%zmm2         \n"
    "   vcvtne2ps2bf16     %%zmm6, %%zmm7, %%zmm3   \n"
    "   vpermt2w    %%zmm3, %%zmm31, %%zmm3         \n"


    "   vcvtne2ps2bf16     %%zmm8, %%zmm9, %%zmm4   \n"
    "   vpermt2w    %%zmm4, %%zmm31, %%zmm4         \n"
    "   vcvtne2ps2bf16     %%zmm10, %%zmm11, %%zmm5 \n"
    "   vpermt2w    %%zmm5, %%zmm31, %%zmm5         \n"
    "   vcvtne2ps2bf16     %%zmm12, %%zmm13, %%zmm6 \n"
    "   vpermt2w    %%zmm6, %%zmm31, %%zmm6         \n"
    "   vcvtne2ps2bf16     %%zmm14, %%zmm15, %%zmm7 \n"
    "   vpermt2w    %%zmm7, %%zmm31, %%zmm7         \n"


    "   vunpcklps %%zmm1, %%zmm0, %%zmm24           \n"
    "   vunpcklps %%zmm3, %%zmm2, %%zmm25           \n"
    "   vunpcklpd %%zmm25, %%zmm24, %%zmm26         \n"

    "   vunpcklps %%zmm5, %%zmm4, %%zmm29                   \n"
    "   vunpcklps %%zmm7, %%zmm6, %%zmm30                   \n"
    "   vunpcklpd %%zmm30, %%zmm29, %%zmm27                 \n"

    "   vinsertf32x4    $1, %%xmm27, %%ymm26, %%ymm8        \n"
    "   vmovdqu16    %%ymm8, (%%rcx)                        \n"
    "   vextractf32x4   $1, %%zmm26, %%xmm9                 \n"
    "   vextractf32x4   $1, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm9, %%ymm9         \n"
    "   vmovdqu16    %%ymm9, (%%rcx, %%r10, 4)              \n"
    "   vextractf32x4   $2, %%zmm26, %%xmm10                \n"
    "   vextractf32x4   $2, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm10, %%ymm10       \n"
    "   vmovdqu16    %%ymm10, (%%rcx, %%r10, 8)             \n"
    "   vextractf32x4   $3, %%zmm26, %%xmm11                \n"
    "   vextractf32x4   $3, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm11, %%ymm11       \n"
    "   leaq        (%%rcx, %%r10, 8), %%r14                \n"
    "   vmovdqu16    %%ymm11, (%%r14, %%r10, 4)             \n"

//--------------------------------------------------------------------


    "   vunpckhpd %%zmm25, %%zmm24, %%zmm26                 \n"
    "   vunpckhpd %%zmm30, %%zmm29, %%zmm27                 \n"
    "   vinsertf32x4    $1, %%xmm27, %%ymm26, %%ymm12       \n"
    "   vmovdqu16    %%ymm12, (%%rcx, %%r10)                \n"
    "   vextractf32x4   $1, %%zmm26, %%xmm13                \n"
    "   vextractf32x4   $1, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm13, %%ymm13       \n"
    "   leaq        (%%rcx, %%r10), %%r14                   \n"
    "   vmovdqu16    %%ymm13, (%%r14, %%r10, 4)             \n"
    "   vextractf32x4   $2, %%zmm26, %%xmm14                \n"
    "   vextractf32x4   $2, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm14, %%ymm14       \n"
    "   vmovdqu16    %%ymm14, (%%r14, %%r10,8)              \n"
    "   vextractf32x4   $3, %%zmm26, %%xmm15                \n"
    "   vextractf32x4   $3, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm15, %%ymm15       \n"
    "   leaq        (%%r14, %%r10, 8), %%r14                \n"
    "   vmovdqu16    %%ymm15, (%%r14, %%r10, 4)             \n"

//--------------------------------------------------------------------

    "   vunpckhps %%zmm1, %%zmm0, %%zmm24                   \n"
    "   vunpckhps %%zmm3, %%zmm2, %%zmm25                   \n"
    "   vunpcklpd %%zmm25, %%zmm24, %%zmm26                 \n"
    "   vunpckhps %%zmm5, %%zmm4, %%zmm29                   \n"
    "   vunpckhps %%zmm7, %%zmm6, %%zmm30                   \n"
    "   vunpcklpd %%zmm30, %%zmm29, %%zmm27                 \n"
    "   vinsertf32x4    $1, %%xmm27, %%ymm26, %%ymm16       \n"
    "   vmovdqu16    %%ymm16, (%%rcx, %%r10, 2)             \n"
    "   vextractf32x4   $1, %%zmm26, %%xmm17                \n"
    "   vextractf32x4   $1, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm17, %%ymm17       \n"
    "   leaq        (%%rcx, %%r10, 2), %%r14                \n"
    "   vmovdqu16    %%ymm17, (%%r14, %%r10, 4)             \n"
    "   vextractf32x4   $2, %%zmm26, %%xmm18                \n"
    "   vextractf32x4   $2, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm18, %%ymm18       \n"
    "   vmovdqu16    %%ymm18, (%%r14, %%r10, 8)             \n"
    "   vextractf32x4   $3, %%zmm26, %%xmm19                \n"
    "   vextractf32x4   $3, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm19, %%ymm19       \n"
    "   leaq        (%%r14, %%r10, 8), %%r14                \n"
    "   vmovdqu16    %%ymm19, (%%r14, %%r10, 4)             \n"


//--------------------------------------------------------------------

    "   vunpckhpd %%zmm25, %%zmm24, %%zmm26                \n"
    "   vunpckhpd %%zmm30, %%zmm29, %%zmm27                \n"
    "   vinsertf32x4    $1, %%xmm27, %%ymm26, %%ymm20       \n"
    "   leaq            (%%rcx, %%r10, 2), %%r14            \n"
    "   leaq            (%%r14, %%r10), %%r14               \n"
    "   vmovdqu16    %%ymm20, (%%r14)                       \n"
    "   vextractf32x4   $1, %%zmm26, %%xmm21                \n"
    "   vextractf32x4   $1, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm21, %%ymm21       \n"
    "   vmovdqu16    %%ymm21, (%%r14, %%r10, 4)             \n"
    "   vextractf32x4   $2, %%zmm26, %%xmm22                \n"
    "   vextractf32x4   $2, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm22, %%ymm22       \n"
    "   vmovdqu16    %%ymm22, (%%r14, %%r10, 8)             \n"
    "   vextractf32x4   $3, %%zmm26, %%xmm23                \n"
    "   vextractf32x4   $3, %%zmm27, %%xmm28                \n"
    "   vinsertf32x4    $1, %%xmm28, %%ymm23, %%ymm23       \n"
    "   leaq        (%%r14, %%r10, 8), %%r14                \n"
    "   vmovdqu16    %%ymm23, (%%r14, %%r10, 4)             \n"

    "   subq        $1, %%rax                       \n"
    "   cmp         $0, %%rax                       \n"
    "   je          5f                              \n"

    "2:                                             \n"
    "   addq        $32, %%rcx                     \n"
    "   cmp         $3, %%rax                       \n"
    "   jl          3f                              \n"
    "   tilestored   %%tmm5, (%%rsp, %%r8)          \n"
    "   jmp         0b                              \n"

    "3:                                             \n"

    "   cmp         $1, %%rax                       \n"
    "   je          4f                              \n"

    "   kmovq       %%k3, %%rsi                     \n"
    "   shl         $4, %%rsi                       \n"
    "   leaq        -64(%%rcx, %%rsi), %%rcx        \n"
    "   tilestored   %%tmm6, (%%rsp, %%r8)          \n"
    "   jmp         0b                              \n"

    "4:                                             \n"
    "   tilestored   %%tmm7, (%%rsp, %%r8)          \n"
    "   jmp         0b                              \n"

    "5:                                             \n"

    "   kmovq    %%k1, %%r12                        \n"
    "   addq     %%r12, %%rsp                       \n"


    :
    :
      [A] "m" (A),
      [B] "m" (B),
      [C] "m" (C),
      [temp] "m" (temp),
      [K] "m" (K),
      [k_tag]"m"(k_tag),
      [k_end]"m"(k_end),
      [stride_N]"m"(stride_N),
      [stride_M]"m"(stride_M),
      [stride_K]"m"(stride_K),
      [vpermt2w_control]"m"(vpermt2w_control)
    :"rax", "rbx", "rcx", "rdx", "rdi", "rsi","r8", "r9", "r10", "r11", "r12",
     "r13", "r14", "r15","k1","k2","k3","zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
     "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
     "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
     "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
     "zmm30", "zmm31", "memory"
    );

}



void bf16gemm_gate_up_kernel(uint16_t *gate_C, uint16_t *up_C,
                            float*temp,   //用来存放中间的计算结果
                            uint16_t *input, uint16_t *gate, uint16_t *up,
                            long M, long N, long K, long stride_N, long n_offset)
{

    int i, j, ii, kk, kc, mc, nr, t;

    tile_config tile_data[2]={0,0};
    init_tile_config(tile_data,16);
    init_tile_config(tile_data+1, M%16);

/*
    for(int ij; ij < M*K; ij++)
    {
        input[ij] = FP32_toBF16(1.0);
    }

    for(int ij; ij < N*K; ij++)
    {
        up[ij] = FP32_toBF16(1.0);
    }
*/

    long GEMM_M = M>224?224:M;
    long GEMM_K = (573440 / GEMM_M / 32) * 32;

    for(t = 0; t < 2; t++)
    {

        uint16_t *C = gate_C;
        uint16_t *B = gate;

        if(t==1)
        {
            C = up_C;
            B = up;
        }

        for(kk = 0; kk < K; kk = kk+kc)
        {
            kc = GEMM_K;
            if(K-kk<GEMM_K)
            {
                kc  = K- kk;
            }

            for(ii =0; ii < M; ii = ii+mc)
            {
                mc = GEMM_M;
                if(M-ii<GEMM_M)
                {
                    mc  = M- ii;
                }

                int Int_M = mc - mc % 32;
                uint8_t tail_M = mc - (mc-1)/16*16;

                for(j = 0; j < N; j = j + nr)
                {
                    nr =32;
                    if(N - j < 32)
                    {
                        nr = N - j;
                    }

                    uint16_t* B0 = B + j*K + kk;  // B
                    _tile_loadconfig(tile_data);

                    if(Int_M >= 32)
                    {

                        for(i = ii; i < ii+Int_M; i = i + 32)
                        {
                            uint16_t* C0 = C + (n_offset+j)*16 + i*stride_N;
                            uint16_t* input0 = input + i*K + kk*16;
                            float*temp0 = temp + M*j + i;

                            mKernel_m32(C0, temp0, input0, B0,
                                       kc, stride_N<<1, K<<1,
                                       M<<2, kk, K-kk-kc, mydata::vpermt2w_control);
                        }
                    }
                    // EGDE_M

                    if(Int_M < mc)
                    {
                        uint32_t edge_M;
                        for(i = ii+Int_M; i < ii+mc; i = i + edge_M)
                        {
                            edge_M = 16;
                            if(ii+mc - i < 16)
                            {
                                edge_M= ii+mc - i;
                                _tile_loadconfig(tile_data+1);
                            }

                            mKernel_m16(C + (n_offset+j)*edge_M + i*stride_N,
                                temp + M*j + i,
                                input + i*K + kk*edge_M,
                                B0, edge_M, kc, K<<1, M<<2, kk, K-kk-kc, mydata::vpermt2w_control);
                        }
                    }


                }
            }
        }
    }

    //printf("+++++++++++++++%f\n", BF16_toFP32(up_C[123]) );

/*
    //Check
    void *ptr;
    posix_memalign(&ptr, 64, M*N*sizeof( float ));
    float *temp1 = ( float * ) ptr;


    for(int i = 0; i < M; i++)
    {
        int block = i / 16;
        int intra_i = i % 16;
        int rows_in_block = ((block + 1) * 16 <= M) ? 16 : (M - block * 16);
        int base = block * 16 * K;

        for(int j = 0; j < N; j++)
        {
            temp1[j*M + i] = 0;
           for(int k = 0; k < K; k++)
           {
                int pair_k = k / 2;
                int intra_k = k % 2;

                temp1[j*M + i] += BF16_toFP32(input[base + pair_k * (rows_in_block * 2) + intra_i * 2 + intra_k])
                            * BF16_toFP32(up[j*K + k]);
           }
        }
    }


    int flag = 0, mr;
    for(i = 0; i < M; i++)
    {
        mr = 16;
        if(i >= M-M%16)
        {
            mr = M%16;
        }
        for(j = 0; j < N; j++)
        {
            if( fabs( temp1[i+j*M]- BF16_toFP32(
                up_C[ i/16*16*stride_N + (j+n_offset)/2 *2 *mr + (i%16)*2 + (j+n_offset)%2 ] ) )>1)
            {
                flag = 1;
                printf("error: M = %d, i = %d, j=%d, %f %f\n",
                    M, i, j, temp1[i+j*M], BF16_toFP32(
                        up_C[ i/16*16*stride_N + (j+n_offset)/2 *2 *mr + (i%16)*2 + (j+n_offset)%2 ] ) );
                break;
            }
        }
        if(flag==1)
        {
            break;
        }

    }

    free(temp1);
*/
}


void bf16gemm_down_kernel(uint16_t *down_C, float*temp, uint16_t *input, uint16_t *down,
                            long M, long N, long K, long stride_N)
{

    int i, ii, kk, j, nr, mc, kc;

    tile_config tile_data[2]={0,0};
    init_tile_config(tile_data,16);
    init_tile_config(tile_data+1, M%16);


    long GEMM_M = M>224?224:M;
    long GEMM_K = (573440 / GEMM_M / 32) * 32;

    for(kk = 0; kk < K; kk = kk + kc)
    {
        kc = GEMM_K;
        if(K-kk<GEMM_K)
        {
            kc  = K- kk;
        }

        for(ii = 0; ii < M; ii = ii + mc)
        {
            mc = GEMM_M;

            if(M-ii<GEMM_M)
            {
                mc = M - ii;
            }

            int Int_M = mc - mc % 32;
            uint8_t tail_M = mc - (mc-1)/16*16;

            for(j = 0; j < N; j = j + nr)
            {
                nr =32;
                if(N - j < 32)
                {
                    nr = N - j;
                }

                uint16_t* B0 = down + j*K + kk;  // B

                _tile_loadconfig(tile_data);

                if(Int_M >= 32)
                {

                    for(i = ii; i < ii+Int_M; i = i + 32)
                    {
                        uint16_t* C0 = down_C + i*stride_N + j;

                        uint16_t* input0 = input + i*K + 16*kk;

                        mKernel_m32_down(C0, temp + M*j + i, input0, B0,
                                   kc, stride_N<<1, K<<1, M<<2, kk, K-kk-kc,
                                   mydata::vpermt2w_control);
                    }
                }

                // EGDE_M
                if(Int_M < mc)
                {
                    uint32_t edge_M;
                    for(i = ii+Int_M; i < ii+mc; i = i + edge_M)
                    {
                        edge_M = 16;
                        if(ii+mc - i < 16)
                        {
                            edge_M= ii+mc - i;
                            _tile_loadconfig(tile_data+1);
                        }

                        mKernel_m16_down(down_C + j + i*stride_N,
                            temp + M*j + i,
                            input + i*K + edge_M*kk, B0,
                            edge_M, kc, stride_N<<1, K<<1, M<<2, kk, K-kk-kc,
                            mydata::vpermt2w_control);
                    }
                }

            }

        }
    }

/*
    //Check
    void *ptr;
    posix_memalign(&ptr, 64, M*N*sizeof( float ));
    float *temp1 = ( float * ) ptr;


    for(int i = 0; i < M; i++)
    {
        int block = i / 16;
        int intra_i = i % 16;
        int rows_in_block = ((block + 1) * 16 <= M) ? 16 : (M - block * 16);
        int base = block * 16 * K;

        for(int j = 0; j < N; j++)
        {
            temp1[i*N + j] = 0;
           for(int k = 0; k < K; k++)
           {
                int pair_k = k / 2;
                int intra_k = k % 2;

                temp1[i*N + j] += BF16_toFP32(input[base + pair_k * (rows_in_block * 2) + intra_i * 2 + intra_k])
                            * BF16_toFP32(down[j*K + k]);
           }
        }
    }

    int flag = 0, mr;
    for(i = 0; i < M; i++)
    {
        mr = 16;
        if(i >= M-M%16)
        {
            mr = M%16;
        }
        for(j = 0; j < N; j++)
        {
            if( fabs( temp1[i*N + j]- BF16_toFP32( down_C[ i*stride_N + j ] ) )>1100)
            {
                flag = 1;
                printf("error: M = %d, i = %d, j=%d, %f %f\n",
                    M, i, j, temp1[i*N + j], BF16_toFP32( down_C[ i*stride_N + j ] ) );
                break;
            }
        }
        if(flag==1)
        {
            break;
        }

    }

    free(temp1);
*/

}
