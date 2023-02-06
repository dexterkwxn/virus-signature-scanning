/*
        CS3210 Assignment 2
        CUDA Virus Scanning

        Most of your CUDA code should go in here.

        Feel free to change any code in the skeleton, as long as you conform
        to the input and output formats specified in the assignment pdf.

        If you rename this file or add new files, remember to modify the
        Makefile! Just make sure (haha) that the default target still builds
        your program, and you don't rename the program (`scanner`).

        The skeleton demonstrates how asnychronous kernel launches can be
        done; it is up to you to decide (and implement!) the parallelisation
        paradigm for the kernel. The provided implementation is not great,
        since it launches one kernel per file+signature combination (a lot!).
        You should try to do more work per kernel in your implementation.

        You can launch as many kernels as you want; if any preprocessing is
        needed for your algorithm of choice, you can also do that on the GPU
        by running different kernels.

        'defs.h' contains the definitions of the structs containing the input
        and signature data parsed by the provided skeleton code; there should
        be no need to change it, but you can if you want to.

        'common.cpp' contains the aforementioned parsing for the input files.
        The input files are already efficiently read with mmap(), so there
        should be little to no gain trying to optimise that portion of the
        skeleton.

        Remember: print any debugging statements to STDERR!
*/

#include <vector>

#include "defs.h"

constexpr int N = 64;

//__device__ int global_counter = 0;

__global__ void matchFile(const char *file_name, const uint8_t *file_data,
                          size_t file_len, char **sig_names,
                          const char *sigs_buf, int32_t *sig_offsets) {

  int block_offset =
      gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;

  int thread_offset = blockDim.x * blockDim.y * threadIdx.z +
                      blockDim.x * threadIdx.y + threadIdx.x;

  int len = sig_offsets[block_offset + 1] - sig_offsets[block_offset];
  const char *signature = sigs_buf + sig_offsets[block_offset];

  int file_blk_sz = (file_len + N - 1) / N;
  int start_idx = thread_offset * file_blk_sz;
  int end_idx = (thread_offset + 1) * file_blk_sz;
  for (size_t i = start_idx; i < end_idx; ++i) {
    if (i >= file_len) {
      // exceed file size, stop
      return;
    }
    bool match = true;
    int file_data1, file_data2;
    int first_value, second_value;
    for (int j = 0; j < len / 2; ++j) {

      if (!match)
        break;
      if (i + 2 * j + 1 >= file_len) {
        match = 0;
        break;
      }
      // convert byte that we are comparing
      char first_half = signature[2 * j];
      char second_half = signature[2 * j + 1];

      file_data1 = file_data[i + j] / 16;
      file_data2 = file_data[i + j] % 16;

      if (first_half != '?') {
        if (first_half > 96) {
          first_value = first_half - 'a' + 10;
        } else if (first_half > 47) {
          first_value = first_half - '0';
        }

        match = match && (file_data1 == first_value);
        // if (match) printf("file_data1: %d, first_value: %d, i: %ld, j: %d\n",
        // file_data1, first_value, i, j);
      }

      if (second_half != '?') {
        if (second_half > 96) {
          second_value = second_half - 'a' + 10;
        } else if (second_half > 47) {
          second_value = second_half - '0';
        }

        match = match && (file_data2 == second_value);
        // if (match) printf("file_data2: %d, second_value: %d, i: %ld, j:
        // %d\n", file_data2, second_value, i, j);
      }
    }
    // if match, store it
    if (match) {
      //      printf("-------------------------MATCH----------------------- i:
      //      %ld, "
      //             "sig_idx: %d \n",
      //             i, block_offset);
      printf("%s: %s\n", file_name, sig_names[block_offset]);
    }
  }

  __syncthreads();
  // we assume the same substring cannot represent 2 viruses
}

void runScanner(std::vector<Signature> &signatures,
                std::vector<InputFile> &inputs) {
  {
    cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));

    fprintf(stderr, "cuda stats:\n");
    fprintf(stderr, "  # of SMs: %d\n", prop.multiProcessorCount);
    fprintf(stderr, "  global memory: %.2f MB\n",
            prop.totalGlobalMem / 1024.0 / 1024.0);
    fprintf(stderr, "  shared mem per block: %zu bytes\n",
            prop.sharedMemPerBlock);
    fprintf(stderr, "  constant memory: %zu bytes\n", prop.totalConstMem);
  }

  std::vector<cudaStream_t> streams{};
  streams.resize(inputs.size());

  std::vector<uint8_t *> file_bufs{};

  for (size_t i = 0; i < inputs.size(); i++) {
    cudaStreamCreate(&streams[i]);

    // allocate memory on the device for the file
    uint8_t *ptr = 0;
    check_cuda_error(cudaMalloc(&ptr, inputs[i].size));
    file_bufs.push_back(ptr);
  }

  // allocate memory for the signatures
  char *sigs_buf;
  int32_t *sig_offsets;
  int32_t cur_offset = 0;
  char **sig_names;
  check_cuda_error(cudaMallocManaged(&sig_offsets, (signatures.size() + 1) *
                                                       sizeof(int32_t)));
  check_cuda_error(cudaMallocManaged(&sig_names, signatures.size()));
  for (size_t i = 0; i < signatures.size(); i++) {
    cur_offset += signatures[i].size;
  }
  check_cuda_error(cudaMalloc(&sigs_buf, cur_offset));
  sig_offsets[signatures.size()] = cur_offset;
  cur_offset = 0;
  for (size_t i = 0; i < signatures.size(); i++) {
    sig_offsets[i] = cur_offset;
    cudaMemcpy(sigs_buf + cur_offset, signatures[i].data, signatures[i].size,
               cudaMemcpyHostToDevice);
    check_cuda_error(cudaMalloc(&sig_names[i], signatures[i].name.size()));
    cudaMemcpy(sig_names[i], signatures[i].name.data(),
               signatures[i].name.size(), cudaMemcpyHostToDevice);
    cur_offset += signatures[i].size;
  }

  std::vector<char *> file_names(inputs.size());
  dim3 blockDimensions(1, 1, N);
  for (size_t file_idx = 0; file_idx < inputs.size(); file_idx++) {
    // asynchronously copy the file contents from host memory
    // (the `inputs`) to device memory (file_bufs, which we allocated above)
    cudaMemcpyAsync(file_bufs[file_idx], inputs[file_idx].data,
                    inputs[file_idx].size, cudaMemcpyHostToDevice,
                    streams[file_idx]);
    check_cuda_error(
        cudaMalloc(&file_names[file_idx], inputs[file_idx].name.size()));
    cudaMemcpyAsync(file_names[file_idx], inputs[file_idx].name.data(),
                    inputs[file_idx].name.size(), cudaMemcpyHostToDevice,
                    streams[file_idx]);
    // pass in the stream here to do this async

    dim3 gridDimensions(1, 1, signatures.size());
    matchFile<<<gridDimensions, blockDimensions, 0, streams[file_idx]>>>(
        file_names[file_idx], file_bufs[file_idx], inputs[file_idx].size,
        sig_names, sigs_buf, sig_offsets);
  }

  cudaFree(sigs_buf);
  cudaFree(sig_offsets);

  // free the device memory, though this is not strictly necessary
  // (the CUDA driver will clean up when your program exits)
  for (auto buf : file_bufs)
    cudaFree(buf);

  // clean up streams (again, not strictly necessary)
  for (auto &s : streams)
    cudaStreamDestroy(s);
}
