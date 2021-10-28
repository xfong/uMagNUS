// normalize vector {vx, vy, vz} to unit length, unless length or vol are zero.
__kernel void
normalize2(__global float* __restrict vx, __global float* __restrict vy, __global float* __restrict vz, __global float* __restrict vol, int N) {

    int gid = get_global_id(0);
    int gsize = get_global_size(0);
    for (int i = gid; i < N; i += gsize) {

        float v = (vol == NULL) ? 1.0f : vol[i];
        float3 V = {v*vx[i], v*vy[i], v*vz[i]};
        V = normalize(V);
        if (v == 0.0f) {
            vx[i] = 0.0;
            vy[i] = 0.0;
            vz[i] = 0.0;
        } else {
            vx[i] = V.x;
            vy[i] = V.y;
            vz[i] = V.z;
        }
    }
}
