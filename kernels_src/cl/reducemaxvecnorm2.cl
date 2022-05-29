__kernel void
reducemaxvecnorm2(__global real_t* __restrict       x,
                  __global real_t* __restrict       y,
                  __global real_t* __restrict       z,
                  __global real_t* __restrict     dst,
                           real_t             initVal,
                              int                   n,
                  __local  real_t*            scratch) {

    // Calculate indices
    int    local_idx = get_local_id(0);   // Work-item index within workgroup
    int       grp_sz = get_local_size(0); // Total number of work-items in each workgroup
    real_t       res = 0.0;

    for (int idx_base = 0; idx_base < n; idx_base += grp_sz) {
        int global_idx = idx_base + local_idx;
        scratch[local_idx] = 0.0;
        if (global_idx < n) {
            real_t x0 = x[global_idx];
            scratch[local_idx] = x0*x0;
            real_t y0 = y[global_idx];
            scratch[local_idx] += y0*y0;
            real_t z0 = z[global_idx];
            scratch[local_idx] += z0*z0;
        }

        // Add barrier to sync all threads
        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int s = (grp_sz >> 1); s > 1; s >>= 1 ) {
            if (local_idx < s) {
                real_t other = scratch[local_idx + s];
                real_t  mine = scratch[local_idx];
                scratch[local_idx] = fmax(mine, other);
            }

            // Synchronize work-group
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Store reduction result for each iteration and move to next
        if (local_idx == 0) {
            res = fmax(scratch[0], scratch[1]);
        }

    }

    // Store reduction result for each iteration and move to next
    if (local_idx == 0) {
        dst[0] = fmax(res, initVal);
    }

}
