__kernel void
reducedot(         __global real_t* __restrict     src1,
                   __global real_t* __restrict     src2,
          volatile __global real_t* __restrict      dst,
                            real_t              initVal,
                               int                    n,
          volatile __local  real_t*            scratch){

    ulong const block_size = get_local_size(0);
    ulong const idx_in_block = get_local_id(0);
    ulong idx_global = get_group_id(0) * (get_local_size(0) * 2) + get_local_id(0);
    ulong const grid_size = block_size * 2 * get_num_groups(0);
    real_t mine = (idx_global == 0) ? initVal : 0;

    // We reduce multiple elements per thread.
    // The number is determined by the number of active thread blocks (via gridDim).
    // More blocks will result in a larger grid_size and therefore fewer elements per thread.
    while (idx_global < n) {
        real_t a1 = src1[idx_global];
        real_t b1 = src2[idx_global];
        mine += a1*b1;
        // Ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays.
        if (idx_global + block_size < n) {
            a1 = src1[idx_global + block_size];
            b1 = src2[idx_global + block_size];
            mine += a1*b1;
        }
        idx_global += grid_size;
    }

    scratch[idx_in_block] = mine;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in the shared memory.
    if (block_size > 512) {
        if (idx_in_block < 512)
            scratch[idx_in_block] += scratch[idx_in_block + 512];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (block_size > 256) {
        if (idx_in_block < 256)
            scratch[idx_in_block] += scratch[idx_in_block + 256];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (block_size > 128) {
        if (idx_in_block < 128)
            scratch[idx_in_block] += scratch[idx_in_block + 128];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (block_size > 64) {
        if (idx_in_block < 64)
            scratch[idx_in_block] += scratch[idx_in_block + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (idx_in_block < 32) {
        volatile __local real_t* smem = scratch;
        smem[idx_in_block] += smem[idx_in_block + 32];
        smem[idx_in_block] += smem[idx_in_block + 16];
        smem[idx_in_block] += smem[idx_in_block + 8];
        smem[idx_in_block] += smem[idx_in_block + 4];
        smem[idx_in_block] += smem[idx_in_block + 2];
        smem[idx_in_block] += smem[idx_in_block + 1];
    }

    // Add atomically to global buffer
    if (idx_in_block == 0) {
        atomicAdd_r(dst, scratch[0]);
//            dst[get_group_id(0)] = scratch[0];
    }

}
