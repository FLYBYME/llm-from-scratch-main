# LLM from Scratch - GPU Support

This project uses TensorFlow.js for training and inference. To utilize your NVIDIA GPU, follow these steps:

## Prerequisites

To use `@tensorflow/tfjs-node-gpu`, you must install the specific versions of CUDA and cuDNN that match the TensorFlow C++ library version (v4.22.0 uses TensorFlow 2.x).

1.  **NVIDIA Driver:** Ensure your driver is up to date (version > 450.x).
2.  **CUDA Toolkit 11.2:** Download from the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11.2.0-download-archive).
3.  **cuDNN 8.1.0:** 
    *   **Note:** You will need to create a free [NVIDIA Developer account](https://developer.nvidia.com/login) to download this.
    *   Download the zip for CUDA 11.x from the [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive).
    *   Extract and copy the contents of `bin`, `include`, and `lib` into your CUDA installation directory (usually `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`).
4.  **Environment Variables:** Add the following to your `Path`:
    *   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
    *   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp`

## Verification

After installation, you can verify if the GPU is being used by running:

```bash
npm run train
```

If successful, you should see logs indicating the GPU device is initialized.

## Troubleshooting

If you see `ERR_DLOPEN_FAILED`, it likely means `cudart64_110.dll` or `cudnn64_8.dll` is not in your `Path`.
