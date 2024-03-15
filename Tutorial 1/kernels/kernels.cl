//a simple OpenCL kernel which adds two vectors A and B together into a third vector C
kernel void add(global const int* A, global const int* B, global int* C) {
	int id = get_global_id(0);
	if (id == 0) { // perform this part only once i.e. for work item 0
		printf("work group size %d\n", get_local_size(0));
	}
	int loc_id = get_local_id(0);
	printf("global id = %d, local id = %d\n", id, loc_id); // do it for each work item
	C[id] = A[id] + B[id];
}

//a simple smoothing kernel averaging values in a local window (radius 1)
kernel void avg_filter(global const int* A, global int* B) {
	int id = get_global_id(0);
	int arr[5] = { A[id - 2], A[id - 1], A[id], A[id + 1], A[id + 2] };
	int total = 0;
	for (int i = 0; i < 4; i++) {
		if (arr[i] != NULL) {
			total += arr[i];
		}
	}
	B[id] = (total)/5;
	
}

//a simple 2D kernel
kernel void add2D(global const int* A, global const int* B, global int* C) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	int id = x + y*width;

	printf("id = %d x = %d y = %d w = %d h = %d\n", id, x, y, width, height);

	C[id]= A[id]+ B[id];
}
