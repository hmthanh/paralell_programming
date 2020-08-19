// extern __shared__ uchar3 s_inPixels[];

	// int idxR = blockIdx.y * blockDim.y + threadIdx.y;
	// int idxC = blockIdx.x * blockDim.x + threadIdx.x;

	// int filterPadding = filterWidth/2;
	// int shareBlockWidth = blockDim.x + filterPadding;

	// int inR = idxR - filterPadding;
	// int inC = idxC - filterPadding;
	// inR = min(height - 1, max(0, inR));
	// inC = min(width - 1, max(0, inC));
	// s_inPixels[threadIdx.y * shareBlockWidth + threadIdx.x] = inPixels[inR * width + inC];

	// if (floorf(threadIdx.x / filterWidth) == 0) {
	// 	inR = idxR;
	// 	inC = idxC + blockDim.x - filterPadding;
	// 	inR = min(height - 1, max(0, inR));
	// inC = min(width - 1, max(0, inC));
	// 	s_inPixels[(threadIdx.y + blockDim.y) * shareBlockWidth + (threadIdx.x + blockDim.x)] = inPixels[inR * width + inC];
	// }

	// if (floorf(threadIdx.y / filterWidth) == 0) {
	// 	inR = idxR + blockDim.y - filterPadding;
	// 	inC = idxC;
	// 	inR = min(height - 1, max(0, inR));
	// inC = min(width - 1, max(0, inC));
	// 	s_inPixels[(threadIdx.y + blockDim.y) * shareBlockWidth + threadIdx.x] = inPixels[inR * width + inC];
	// }

	// if (floorf(threadIdx.y / filterWidth) == 0 && floorf(threadIdx.x / filterWidth) == 0) {
	// 	inR = idxR + blockDim.y - filterPadding;
	// 	inC = idxC + blockDim.x - filterPadding;
	// 	inR = min(height - 1, max(0, inR));
	// 	inC = min(width - 1, max(0, inC));
	// 	s_inPixels[(threadIdx.y + blockDim.y) * shareBlockWidth + (threadIdx.x + blockDim.x)] = inPixels[inR * width + inC];
	// }
	// __syncthreads();
	// 	if (idxR < height && idxC < width)
	// 	{
	// 	float3 outPixel = make_float3(0, 0, 0);
	// 	for (int fR = 0; fR < filterWidth; fR++){
	// 		for (int fC = 0; fC < filterWidth; fC++){
	// 			float filterVal = filter[fR * filterWidth + fC];

	// 			int inPixelR = threadIdx.y + fR;
	// 			int inPixelC = threadIdx.x + fC;
	// 			uchar3 inPixel = s_inPixels[inPixelR * shareBlockWidth + inPixelC];

	// 			outPixel.x += filterVal * inPixel.x;
	// 			outPixel.y += filterVal * inPixel.y;
	// 			outPixel.z += filterVal * inPixel.z;
	// 		}
	// 	}
	// 	int idx = idxR * width + idxC;
	// 	outPixels[idx] = make_uchar3(outPixel.x, outPixel.y, outPixel.z);
	// }