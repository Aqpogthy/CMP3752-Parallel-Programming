kernel void intensityHistogram(global const uchar* A, global int* H) {

	int id = get_global_id(0);
	int bin_index = A[id];
	//assumes bins are set to 0
	atomic_inc(&H[bin_index]);//increments histogram bin

}

kernel void intensityHistogramLocal(global const uchar* A, local int* LH, global int* H, const int nr_bins) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int bin_index = A[id];
	//clear the scratch bin
	if (lid < nr_bins) {
		LH[lid] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	atomic_inc(&LH[bin_index]);

	barrier(CLK_LOCAL_MEM_FENCE);

	//copy LH from local to global
	if (lid < nr_bins) {
		atomic_add(&H[lid], LH[lid]);
	}
}

kernel void cumulativeHistogram(global const int* H, global int* CH) {
	int id = get_global_id(0);
	
	int CF = 0;
	for (int i = 0; i <= id; i++) {
		CF += H[i];
	}
	CH[id] = CF;
}
kernel void NormaliseAndScale(global const int* CH, global int* NH, const float SF) {
	int id = get_global_id(0);

	NH[id] = CH[id] * SF;//multiplies each value by a scale factor which is input, which then normalises and scales the data within the range of either 0-255 or 0-65535
}
kernel void backProjection(global const uchar* A, global const int* NH, global uchar* B) {
	int id = get_global_id(0);
	B[id] = NH[A[id]];
}