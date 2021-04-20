/**
 * Run TensorFlow Lite model on EdgeTPU
 * Source: https://github.com/google-coral/tflite/blob/master/cpp/examples/classification/classify.cc
 */

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "edgetpu_c.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#define DEBUG 10

namespace util {

constexpr size_t kBmpFileHeaderSize = 14;
constexpr size_t kBmpInfoHeaderSize = 40;
constexpr size_t kBmpHeaderSize = kBmpFileHeaderSize + kBmpInfoHeaderSize;

int32_t ToInt32(const char p[4]) {
    return (p[3] << 24) | (p[2] << 16) | (p[1] << 8) | p[0];
}

std::vector<uint8_t> ReadBmpImage(const char *filename,
                                int *out_width = nullptr,
                                int *out_height = nullptr,
                                int *out_channels = nullptr)
{
    assert(filename);

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        #if DEBUG >= 1
            std::cerr << "ERROR: Could not open input image file" << std::endl;
        #endif
        return {};
    }

    char header[kBmpHeaderSize];
    if (!file.read(header, sizeof(header))) {
        #if DEBUG >= 1
            std::cerr << "ERROR: Could not read input image file" << std::endl;
        #endif
        return {};
    }

    const char *file_header = header;
    const char *info_header = header + kBmpFileHeaderSize;

    if (file_header[0] != 'B' || file_header[1] != 'M') {
        #if DEBUG >= 1
            std::cerr << "ERROR: Invalid input image file type" << std::endl;
        #endif
        return {};
    }

    const int channels = info_header[14] / 8;
    if (channels != 1 && channels != 3) {
        #if DEBUG >= 1
            std::cerr << "ERROR: Unsupported bits per pixel in input image" << std::endl;
        #endif
        return {};
    }

    if (ToInt32(&info_header[16]) != 0) {
        #if DEBUG >= 1
            std::cerr << "ERROR: Unsupported compression for input image" << std::endl;
        #endif
        return {};
    }

    const uint32_t offset = ToInt32(&file_header[10]);
    if (offset > kBmpHeaderSize && !file.seekg(offset - kBmpHeaderSize, std::ios::cur)) {
        #if DEBUG >= 1
            std::cerr << "ERROR: Seek failed while reading input image" << std::endl;
        #endif
        return {};
    }

    int width = ToInt32(&info_header[4]);
    if (width < 0) {
        #if DEBUG >= 1
            std::cerr << "ERROR: Invalid input image width" << std::endl;
        #endif
        return {};
    }

    int height = ToInt32(&info_header[8]);
    const bool top_down = height < 0;
    if (top_down) height = -height;

    const int line_bytes = width * channels;
    const int line_padding_bytes = 4 * ((8 * channels * width + 31) / 32) - line_bytes;
    std::vector<uint8_t> image(line_bytes * height);
    for (int i = 0; i < height; ++i) {
        uint8_t *line = &image[(top_down ? i : (height - 1 - i)) * line_bytes];
        if (!file.read(reinterpret_cast<char *>(line), line_bytes)) {
            #if DEBUG >= 1
                std::cerr << "ERROR: Failed to read input image" << std::endl;
            #endif
            return {};
        }
        if (!file.seekg(line_padding_bytes, std::ios::cur)) {
            #if DEBUG >= 1
                std::cerr << "ERROR: Seek failed while reading input image" << std::endl;
            #endif
            return {};
        }
        if (channels == 3) {
            for (int j = 0; j < width; ++j)
                std::swap(line[3 * j], line[3 * j + 2]);
        }
    }

    if (out_width)
        *out_width = width;
    if (out_height)
        *out_height = height;
    if (out_channels)
        *out_channels = channels;
    return image;
}

std::string GetGoldenFilenameFromModelFilename(std::string model_filename) {    
    auto slash_pos = model_filename.find_last_of("/");
    std::string model_name = model_filename.substr(slash_pos+1);
    
    auto ext_pos = model_name.find_last_of(".");
    model_name =  model_name.substr(0, ext_pos);

    std::string golden_filename = "golden_" + model_name + ".out";
    return golden_filename;
}

bool SaveGoldenOutput(const TfLiteTensor *out_tensor, std::string golden_filename) {
    std::ofstream golden_file(golden_filename, std::ios::binary|std::ios::out);
    if (!golden_file) {
        std::cerr << "ERROR: Could not write golden output file" << std::endl;
        return false;
    }

    TfLiteIntArray *out_dims = out_tensor->dims;

    // Write output data dimensions size
    golden_file.write((char*)&out_dims->size, sizeof(int));

    // Write output data dimensions
    golden_file.write((char*)out_dims->data, out_dims->size*sizeof(int));

    // Write output data size
    golden_file.write((char*)&out_tensor->bytes, sizeof(size_t));

    // Write output data
    const uint8_t *out_data = reinterpret_cast<const uint8_t*>(out_tensor->data.data);
    golden_file.write((char*)out_data, out_tensor->bytes);

    golden_file.close();

    return true;
}

#define CHECK_OUTPUT_DIMS_MISMATCH -1

int CheckOutputAgainsGolden(const TfLiteTensor *out_tensor, std::string golden_filename) {
    std::ifstream golden_file(golden_filename, std::ios::binary);
    if (!golden_file) {
        throw std::runtime_error("Could not open golden output file `" + golden_filename + "`");
    }

    // Read output data dimensions size
    int g_out_dims_size;
    golden_file.read((char*)&g_out_dims_size, sizeof(int));

    // Read output data dimensions
    int *g_out_dims = (int*)malloc(g_out_dims_size * sizeof(int));
    golden_file.read((char*)g_out_dims, g_out_dims_size*sizeof(int));

    // Read output data size
    size_t g_out_bytes;
    golden_file.read((char*)&g_out_bytes, sizeof(size_t));

    // Read output data
    uint8_t *g_out_data = (uint8_t*)malloc(g_out_bytes);
    golden_file.read((char*)g_out_data, g_out_bytes);

    #if DEBUG >= 2
        std::cout << "INFO: Data from golden file was successfully read" << std:: endl;
        std::cout << "      - Dims: ("; 
        for (int i = 0; i < g_out_dims_size; i++) 
            std::cout << g_out_dims[i] << (i+1 < g_out_dims_size ? ", " : ")");
        std::cout << std::endl;
    #endif

    if ((out_tensor->dims->size != g_out_dims_size) ||
        (out_tensor->bytes != g_out_bytes)) {
        return CHECK_OUTPUT_DIMS_MISMATCH;
    }

    for (int i = 0; i < g_out_dims_size; i++) {
        if (out_tensor->dims->data[i] != g_out_dims[i])
            return CHECK_OUTPUT_DIMS_MISMATCH;
    }

    const uint8_t *out_data = reinterpret_cast<const uint8_t*>(out_tensor->data.data);
    int errors = 0;
    for (int i = 0; i < g_out_bytes; i++) {
        if (out_data[i] != g_out_data[i])
            errors++;
    }

    golden_file.close();
    free(g_out_dims);
    free(g_out_data);

    return errors;
}

} // namespace util

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << argv[0] << " <model_file> <image_file>" << std::endl;
        return 1;
    }

    // Parameters:
    const std::string model_file = argv[1];
    const std::string image_file = argv[2];

    // Find TPU device
    size_t num_devices;
    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
        edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

    if (num_devices == 0) {
        std::cerr << "ERROR: No connected TPU found" << std::endl;
        return 1;
    } else {
        #if DEBUG >= 1
            std::cerr << "INFO: " << num_devices << " EdgeTPU(s) found" << std::endl;
        #endif
    }
    const auto &device = devices.get()[0];

    // Load input image
    int image_width, image_height, image_channels;
    auto image = util::ReadBmpImage(image_file.c_str(), &image_width, &image_height, &image_channels);
    if (image.empty()) {
        std::cerr << "ERROR: Could not read image from `" << image_file << "`" << std::endl;
        return 1;
    }

    #if DEBUG >= 2
        std::cout << "INFO: Dimensions of input image: " << 
                  "(" << image_width << ", " << image_height << ", " << image_channels << ")" <<
                  std::endl;
    #endif

    // Load model
    auto model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
    if (!model) {
        std::cerr << "ERROR: Could not read model from `" << model_file << "`" << std::endl;
        return 1;
    }

    #if DEBUG >= 2
        std::cout << "INFO: Model file loaded successfully" << std::endl;
    #endif

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
        std::cerr << "ERROR: Could not create interpreter" << std::endl;
        return 1;
    }

    #if DEBUG >= 2
        std::cout << "INFO: Interpreter created successfully" << std::endl;
    #endif

    // std::unique_ptr<TfLiteDelegate, decltype(&edgetpu_free_delegate)> delegate(
    //     edgetpu_create_delegate(device.type, device.path, nullptr, 0), &edgetpu_free_delegate);

    TfLiteDelegate *delegate = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
    interpreter->ModifyGraphWithDelegate(delegate);

    #if DEBUG >= 2
        std::cout << "INFO: Edge TPU delegate loaded successfully" << std::endl;
    #endif

    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "ERROR: Could not allocate interpreter tensors" << std::endl;
        return 1;
    }

    #if DEBUG >= 2
        std::cout << "INFO: Interpreter tensors allocated successfully" << std::endl;
    #endif

    // Set interpreter input
    const auto* input_tensor = interpreter->input_tensor(0);
    if (input_tensor->type != kTfLiteUInt8 ||           //
        input_tensor->dims->data[0] != 1 ||             //
        input_tensor->dims->data[1] != image_height ||  //
        input_tensor->dims->data[2] != image_width ||   //
        input_tensor->dims->data[3] != image_channels) {
        std::cerr << "ERROR: Input tensor shape does not match input image" << std::endl;
        return 1;
    }

    std::copy(image.begin(), image.end(), interpreter->typed_input_tensor<uint8_t>(0));

    #if DEBUG >= 2
        std::cout << "INFO: Interpreter input set successfully" << std::endl;
    #endif

    // Run model
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "ERROR: Cannot invoke interpreter" << std::endl;
        return 1;
    }

    #if DEBUG >= 2
        std::cout << "INFO: Model execution completed successfully" << std::endl;
    #endif

    // Get output tensor
    const TfLiteTensor *out_tensor = interpreter->output_tensor(0);

    #if DEBUG >= 2
        size_t out_vals_count = out_tensor->bytes / sizeof(uint8_t);
        TfLiteIntArray *out_dims = out_tensor->dims;
        std::cout << "INFO: Output tensor has " << out_vals_count << " values (";
        for (int i = 0; i < out_dims->size; i++) 
            std::cout << out_dims->data[i] << (i+1 < out_dims->size ? ", " : ")");
        std::cout << std::endl;
    #endif

    std::string golden_filename = util::GetGoldenFilenameFromModelFilename(model_file);

    // Check output
    try {
        int errors = util::CheckOutputAgainsGolden(out_tensor, golden_filename);
        if (errors < 0) {
            std::cerr << "ERROR: Output dimensions don't match golden output" << std::endl;
        } else if (errors > 0) {
            std::cout << "INFO: " << errors << " error(s) found in the output" << std::endl;
        } else {
            std::cout << "INFO: Output matches golden output" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
    }

    // Save golden output
    bool golden_saved = util::SaveGoldenOutput(out_tensor, golden_filename);
    #if DEBUG >= 2
        if (golden_saved) {
            std::cout << "INFO: Golden output saved to `" << golden_filename << "`" << std::endl;
        }
    #endif
}