// SPDX-License-Identifier: Apache-2.0
#pragma once

void Gather_expert_tokens(uint16_t *dst, uint16_t *src, uint32_t K,
                    uint32_t M, uint32_t* token_ids)
{

        asm volatile(

        "   mov       %[src], %%rax                             \n"
        "   mov       %[dst], %%rdx                             \n"
        "   movl      %[K], %%r9d                               \n"
        "   movl      %[M], %%r12d                              \n"

        "   mov       %[token_ids], %%rbx                       \n"
        "   movl      (%%rbx), %%esi                            \n"

        "   movl     $1, %%r8d                                  \n"
        "   shlx     %%r12, %%r8, %%r8                          \n"
        "   subq     $1, %%r8                                   \n"
        "   kmovq    %%r8, %%k2                                 \n"

        "   mov     %%r12, %%r14                                \n"
        "   shl     $2, %%r14                                   \n"

        "   cmp     $9, %%r12                                   \n"
        "   jl      Gather_M8                                   \n"

        "BEGIN_Gather_K_M16:                                    \n"

        "   vmovdqu16  (%%rax, %%rsi), %%zmm0                   \n"
        "   movl       4(%%rbx), %%r10d                         \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm1                   \n"
        "   movl       8(%%rbx), %%r10d                         \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm2                   \n"

        "   movl       12(%%rbx), %%r10d                        \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm3                   \n"
        "   movl       16(%%rbx), %%r10d                        \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm4                   \n"

        "   movl       20(%%rbx), %%r10d                        \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm5                   \n"
        "   movl       24(%%rbx), %%r10d                        \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm6                   \n"

        "   movl       28(%%rbx), %%r10d                        \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm7                   \n"
        "   movl       32(%%rbx), %%r10d                        \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm8                   \n"

        "   cmp         $10, %%r12                              \n"
        "   jl          1f                                      \n"
        "   movl       36(%%rbx), %%r10d                        \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm9                   \n"
        "   cmp         $11, %%r12                              \n"
        "   jl          1f                                      \n"
        "   movl       40(%%rbx), %%r10d                        \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm10                  \n"

        "   cmp         $12, %%r12                              \n"
        "   jl          1f                                      \n"
        "   movl       44(%%rbx), %%r10d                        \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm11                  \n"
        "   cmp         $13, %%r12                              \n"
        "   jl          1f                                      \n"
        "   movl       48(%%rbx), %%r10d                        \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm12                  \n"

        "   cmp         $14, %%r12                              \n"
        "   jl          1f                                      \n"
        "   movl       52(%%rbx), %%r10d                        \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm13                  \n"

        "   cmp         $15, %%r12                              \n"
        "   jl          1f                                      \n"
        "   movl       56(%%rbx), %%r10d                        \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm14                  \n"

        "   cmp         $16, %%r12                              \n"
        "   jl          1f                                      \n"
        "   movl       60(%%rbx), %%r10d                        \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm15                  \n"

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "1:                                                     \n"

        "   mov        $4, %%rcx                                \n"
// Transpose
//********************************************************************
        "   vunpcklps %%zmm1, %%zmm0, %%zmm24                   \n"
        "   vunpcklps %%zmm3, %%zmm2, %%zmm25                   \n"
        "   vunpcklpd %%zmm25, %%zmm24, %%zmm19                 \n"

        "   vunpcklps %%zmm5, %%zmm4, %%zmm26                   \n"
        "   vunpcklps %%zmm7, %%zmm6, %%zmm27                   \n"
        "   vunpcklpd %%zmm27, %%zmm26, %%zmm20                 \n"

        "   vunpcklps %%zmm9, %%zmm8, %%zmm28                   \n"
        "   vunpcklps %%zmm11, %%zmm10, %%zmm29                 \n"
        "   vunpcklpd %%zmm29, %%zmm28, %%zmm21                 \n"

        "   vunpcklps %%zmm13, %%zmm12, %%zmm30                 \n"
        "   vunpcklps %%zmm15, %%zmm14, %%zmm31                 \n"
        "   vunpcklpd %%zmm31, %%zmm30, %%zmm22                 \n"

        "   jmp       Transpose_END                             \n"

        "Transpose3:                                            \n"
//********************************************************************
        "   vunpckhps %%zmm1, %%zmm0, %%zmm24                   \n"
        "   vunpckhps %%zmm3, %%zmm2, %%zmm25                   \n"
        "   vunpcklpd %%zmm25, %%zmm24, %%zmm19                 \n"

        "   vunpckhps %%zmm5, %%zmm4, %%zmm26                   \n"
        "   vunpckhps %%zmm7, %%zmm6, %%zmm27                   \n"
        "   vunpcklpd %%zmm27, %%zmm26, %%zmm20                 \n"

        "   vunpckhps %%zmm9, %%zmm8, %%zmm28                   \n"
        "   vunpckhps %%zmm11, %%zmm10, %%zmm29                 \n"
        "   vunpcklpd %%zmm29, %%zmm28, %%zmm21                 \n"

        "   vunpckhps %%zmm13, %%zmm12, %%zmm30                 \n"
        "   vunpckhps %%zmm15, %%zmm14, %%zmm31                 \n"
        "   vunpcklpd %%zmm31, %%zmm30, %%zmm22                 \n"


        "   jmp       Transpose_END                             \n"

        "Transpose4:                                            \n"
//********************************************************************
        "   vunpckhpd %%zmm25, %%zmm24, %%zmm19                 \n"
        "   vunpckhpd %%zmm27, %%zmm26, %%zmm20                 \n"
        "   vunpckhpd %%zmm29, %%zmm28, %%zmm21                 \n"
        "   vunpckhpd %%zmm31, %%zmm30, %%zmm22                 \n"

        "Transpose_END:                                         \n"
//********************************************************************
        "   vinsertf32x4    $1, %%xmm20, %%ymm19, %%ymm17       \n"
        "   vinsertf32x4    $1, %%xmm22, %%ymm21, %%ymm18       \n"
        "   vinsertf64x4    $1, %%ymm18, %%zmm17, %%zmm17       \n"  //

        "   vmovups       %%zmm17, (%%rdx)%{%%k2}               \n"

        "   vextractf32x4   $1, %%zmm19, %%xmm17                \n"
        "   vinsertf32x4    $0, %%xmm17, %%ymm20, %%ymm17       \n"
        "   vextractf32x4   $1, %%zmm21, %%xmm18                \n"
        "   vinsertf32x4    $0, %%xmm18, %%ymm22, %%ymm18       \n"
        "   vinsertf64x4    $1, %%ymm18, %%zmm17, %%zmm17       \n" //

        "   vmovups       %%zmm17, (%%rdx, %%r14, 4)%{%%k2}     \n"

        "   vextractf32x4   $2, %%zmm19, %%xmm17                \n"
        "   vextractf32x4   $2, %%zmm20, %%xmm18                \n"
        "   vinsertf32x4    $1, %%xmm18, %%ymm17, %%ymm17       \n"
        "   vextractf32x4   $2, %%zmm21, %%xmm18                \n"
        "   vextractf32x4   $2, %%zmm22, %%xmm23                \n"
        "   vinsertf32x4    $1, %%xmm23, %%ymm18, %%ymm18       \n"
        "   vinsertf64x4    $1, %%ymm18, %%zmm17, %%zmm17       \n" //

        "   vmovups       %%zmm17, (%%rdx, %%r14, 8)%{%%k2}     \n"

        "   vextractf32x4   $3, %%zmm19, %%xmm17                \n"
        "   vextractf32x4   $3, %%zmm20, %%xmm18                \n"
        "   vinsertf32x4    $1, %%xmm18, %%ymm17, %%ymm17       \n"
        "   vextractf32x4   $3, %%zmm21, %%xmm18                \n"
        "   vextractf32x4   $3, %%zmm22, %%xmm23                \n"
        "   vinsertf32x4    $1, %%xmm23, %%ymm18, %%ymm18       \n"
        "   vinsertf64x4    $1, %%ymm18, %%zmm17, %%zmm17       \n" //

        "   leaq            (%%rdx, %%r14, 8), %%rbp            \n"
        "   vmovups         %%zmm17, (%%rbp, %%r14, 4)%{%%k2}   \n"
        "   addq            %%r14, %%rdx                        \n"
//********************************************************************
        "   sub      $1, %%rcx                                  \n"
        "   cmp      $2, %%rcx                                  \n"
        "   je       Transpose3                                 \n"
        "   cmp      $0, %%rcx                                  \n"
        "   jg       Transpose4                                 \n"

        "   leaq     (%%rdx, %%r14, 8), %%rdx                   \n"
        "   leaq     (%%rdx, %%r14, 4), %%rdx                   \n"

        "   addq     $64, %%rax                                 \n"
        "   subq     $32, %%r9                                  \n"
        "   cmp      $0, %%r9                                   \n"
        "   jg       BEGIN_Gather_K_M16                         \n"

        "   jmp     4f                                          \n"

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        "Gather_M8:                                             \n"
        "   cmp         $2, %%r12                               \n"
        "   jl          BEGIN_Gather_K_M1                       \n"

        "BEGIN_Gather_K_M8:                                     \n"

        "   vmovdqu16  (%%rax, %%rsi), %%zmm0                   \n"
        "   movl       4(%%rbx), %%r10d                         \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm1                   \n"
        "   cmp         $3, %%r12                               \n"
        "   jl          2f                                      \n"
        "   movl       8(%%rbx), %%r10d                         \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm2                   \n"

        "   cmp         $4, %%r12                               \n"
        "   jl          2f                                      \n"
        "   movl       12(%%rbx), %%r10d                         \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm3                   \n"
        "   cmp         $5, %%r12                               \n"
        "   jl          2f                                      \n"
        "   movl       16(%%rbx), %%r10d                         \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm4                   \n"

        "   cmp         $6, %%r12                               \n"
        "   jl          2f                                      \n"
        "   movl       20(%%rbx), %%r10d                         \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm5                   \n"
        "   cmp         $7, %%r12                               \n"
        "   jl          2f                                      \n"
        "   movl       24(%%rbx), %%r10d                         \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm6                   \n"

        "   cmp         $8, %%r12                               \n"
        "   jl          2f                                      \n"
        "   movl       28(%%rbx), %%r10d                         \n"
        "   vmovdqu16  (%%rax, %%r10), %%zmm7                   \n"

        "2:                                                     \n"

        "   mov        $4, %%rcx                                \n"
// Transpose
//********************************************************************
        "   vunpcklps %%zmm1, %%zmm0, %%zmm24                   \n"
        "   vunpcklps %%zmm3, %%zmm2, %%zmm25                   \n"
        "   vunpcklpd %%zmm25, %%zmm24, %%zmm19                 \n"

        "   vunpcklps %%zmm5, %%zmm4, %%zmm26                   \n"
        "   vunpcklps %%zmm7, %%zmm6, %%zmm27                   \n"
        "   vunpcklpd %%zmm27, %%zmm26, %%zmm20                 \n"

        "   jmp       Transpose_END_M8                          \n"

        "Transpose3_M8:                                         \n"
//********************************************************************
        "   vunpckhps %%zmm1, %%zmm0, %%zmm24                   \n"
        "   vunpckhps %%zmm3, %%zmm2, %%zmm25                   \n"
        "   vunpcklpd %%zmm25, %%zmm24, %%zmm19                 \n"

        "   vunpckhps %%zmm5, %%zmm4, %%zmm26                   \n"
        "   vunpckhps %%zmm7, %%zmm6, %%zmm27                   \n"
        "   vunpcklpd %%zmm27, %%zmm26, %%zmm20                 \n"

        "   jmp       Transpose_END_M8                          \n"

        "Transpose4_M8:                                         \n"
//********************************************************************

        "   vunpckhpd %%zmm25, %%zmm24, %%zmm19                 \n"
        "   vunpckhpd %%zmm27, %%zmm26, %%zmm20                 \n"

        "Transpose_END_M8:                                      \n"
//********************************************************************
        "   vinsertf32x4    $1, %%xmm20, %%ymm19, %%ymm17       \n"
        "   vmovups         %%ymm17, (%%rdx)%{%%k2}             \n"

        "   vextractf32x4   $1, %%ymm19, %%xmm17                \n"
        "   vinsertf32x4    $0, %%xmm17, %%ymm20, %%ymm17       \n"
        "   vmovups         %%ymm17, (%%rdx, %%r14, 4)%{%%k2}   \n"

        "   vextractf32x4   $2, %%zmm19, %%xmm17                \n"
        "   vextractf32x4   $2, %%zmm20, %%xmm18                \n"
        "   vinsertf32x4    $1, %%xmm18, %%ymm17, %%ymm17       \n"
        "   vmovups         %%ymm17, (%%rdx, %%r14, 8)%{%%k2}   \n"

        "   vextractf32x4   $3, %%zmm19, %%xmm17                \n"
        "   vextractf32x4   $3, %%zmm20, %%xmm18                \n"
        "   vinsertf32x4    $1, %%xmm18, %%ymm17, %%ymm17       \n"
        "   leaq            (%%rdx, %%r14, 8), %%rbp            \n"
        "   vmovups         %%ymm17, (%%rbp, %%r14, 4)%{%%k2}   \n"
        "   addq            %%r14, %%rdx                        \n"
//********************************************************************

        "   sub      $1, %%rcx                                  \n"
        "   cmp      $2, %%rcx                                  \n"
        "   je       Transpose3_M8                              \n"
        "   cmp      $0, %%rcx                                  \n"
        "   jg       Transpose4_M8                              \n"

        "   leaq     (%%rdx, %%r14, 8), %%rdx                   \n"
        "   leaq     (%%rdx, %%r14, 4), %%rdx                   \n"

        "   addq     $64, %%rax                                 \n"
        "   subq     $32, %%r9                                  \n"
        "   cmp      $0, %%r9                                   \n"
        "   jg       BEGIN_Gather_K_M8                          \n"

        "   jmp     4f                                          \n"

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        "BEGIN_Gather_K_M1:                                     \n"

        "   vmovdqu16  (%%rax, %%rsi), %%zmm0                   \n"
        "   vmovdqu16  %%zmm0, (%%rdx)                          \n"

        "   addq     $64, %%rax                                 \n"
        "   addq     $64, %%rdx                                 \n"
        "   subq     $32, %%r9                                  \n"
        "   cmp      $0, %%r9                                   \n"
        "   jg       BEGIN_Gather_K_M1                          \n"

        "4:                                                     \n"

        :
        :
        [src] "m" (src),
        [dst] "m" (dst),
        [K]"m" (K),
        [M]"m"(M),
        [token_ids]"m"(token_ids)
        :"rax", "rbx", "rcx", "rdx", "rbp","rsi","r8", "r9", "r10", "r11", "r12",
        "r13", "r14","k2","zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
        "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
        "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
        "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
        "zmm30", "zmm31", "memory"
    );

}
