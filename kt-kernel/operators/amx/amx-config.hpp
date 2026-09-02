// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <immintrin.h>


struct ExpertBlockTask {
    int expert_id;
    int block_start;   // token 起始位置
    int block_tokens;  // 这个块实际 token 数
};

typedef struct tile_config {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} tile_config;


static void init_tile_config (tile_config *tileinfo, uint8_t M)
{
    int i;
    tileinfo->palette_id = 1;
    tileinfo->start_row = 0;

    for (i = 0; i < 14; i++)
    {
        tileinfo->reserved_0[i] = 0;
    }


    tileinfo->colsb[0] = 64;
    tileinfo->rows[0] =  16;

    tileinfo->colsb[1] = M<<2;
    tileinfo->rows[1] =  16;

    tileinfo->colsb[2] = 64;
    tileinfo->rows[2] =  16;

    tileinfo->colsb[3] = 64;
    tileinfo->rows[3] =  16;

    tileinfo->colsb[4] = 64;
    tileinfo->rows[4] =  16;

    tileinfo->colsb[5] = 64;
    tileinfo->rows[5] =  16;

    tileinfo->colsb[6] = M<<2;
    tileinfo->rows[6] =  16;

    tileinfo->colsb[7] = M<<2;
    tileinfo->rows[7] =  16;

    //_tile_loadconfig(tileinfo);
}


namespace mydata {
    uint8_t vpermt2w_control[] = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23,
                            8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};//交叉数据

    float Va[]={0x1.715476p+0f, -0x1.62E430p-1f,
              0x1.0F9F9Cp-7f, 0x1.573A1Ap-5f, 0x1.555A80p-3f, 0x1.FFFDC6p-2f,
              0x1.FFFFF6p-1f, 1.0f};   // exp计算

}

// 将 FP32 转换为 BF16
uint16_t FP32_toBF16(float fp32_value)
{
    // 将浮点数的二进制表示强制转换为无符号 32 位整数
    uint32_t fp32_bits;
    memcpy(&fp32_bits, &fp32_value, sizeof(fp32_bits));

    // 截取高 16 位作为 BF16 格式，低 16 位丢弃（精度截断）
    uint16_t bf16_bits = (uint16_t)(fp32_bits >> 16);

    return bf16_bits;
}

// 将 BF16 转换回 FP32
float BF16_toFP32(uint16_t bf16_value) {
    // 创建一个 32 位的无符号整数，将 BF16 的高 16 位放入 FP32 的高位
    uint32_t fp32_bits = (uint32_t)bf16_value << 16;

    // 将二进制表示强制转换回浮点数
    float fp32_value;
    memcpy(&fp32_value, &fp32_bits, sizeof(fp32_value));

    return fp32_value;
}



void Check_Transpose(uint16_t*A, uint32_t* token_ids, uint16_t*Ac, int M, int K)
{

    int i, j, flag=0, mr;
    for(i = 0; i < M; i++)
    {
        mr = 16;
        if(i >= M-M%16)
        {
            mr = M%16;
        }
        for(j = 0; j < K; j++)
        {
            if( BF16_toFP32(A[ token_ids[i]/(K*2)*K + j ] ) !=
                    BF16_toFP32( Ac[ i/16*16*K + j/2 *2 *mr + (i%16)*2 + j%2 ] ))
            {
                printf("转换error: i = %d, j = %d: %f %f\n", i,j,
                        BF16_toFP32(A[ token_ids[i]/(K*2)*K + j ] ),
                        BF16_toFP32( Ac[ i/16*16*K + j/2 *2 *mr + (i%16)*2 + j%2 ] ));
                flag = 1;
                break;
            }
        }
        if(flag==1)
        {
            break;
        }
    }
}
