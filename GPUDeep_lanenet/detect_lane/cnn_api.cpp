/* Copyright 2016-2020 The MathWorks, Inc. */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdexcept>
#include <string>
#include <stdarg.h>
#include <cassert>
#include <vector>
#include "cnn_api.hpp"
#include "MWCNNLayerImpl.hpp"
#include "MWTargetNetworkImpl.hpp"
// Get string value from the macro
#define XSTR(x) #x
#define STR(x) XSTR(x)

#if defined(MW_TARGET_TYPE_CUDNN) || defined(MW_TARGET_TYPE_ARMNEON) || defined(MW_TARGET_TYPE_MKLDNN)
#define DECOUPLE 1
#else
#define DECOUPLE 0
#endif


MWCNNLayer::MWCNNLayer()
        : m_impl(NULL), m_isCustomLayer(false) {
}

MWCNNLayer::~MWCNNLayer() {
}

void MWCNNLayer::predict() {
    if (m_impl) {
        m_impl->predict();
    }
}

/**
 * everything instantiated during setup() should be cleaned up here
 */
void MWCNNLayer::cleanup() {
    if (m_impl) {
        m_impl->cleanup();
        delete m_impl;
        m_impl = 0;
    }

    for (int idx = 0; idx < getNumOutputs(); idx++) {
        MWTensorBase *op = getOutputTensor(idx);
        delete op;
        op = 0;
        m_output[idx] = 0;
    }

    // free up tensor maps
    m_input.clear();
    m_output.clear();
}


void MWCNNLayer::allocate() {
    if (m_impl) {
#if defined(MW_TARGET_TYPE_ARMNEON)
        // Call to ARM Neon allocateInputData to update right
        // unpadded buffer address to m_unpaddedIpData, input ARM tensors
        // in Hand-coded and ACL layer respectively.
        for (int i = 0; i < getNumInputs(); ++i) {
            m_impl->allocateInputData(i);
        }
#endif
        for (int i = 0; i < getNumOutputs(); ++i) {
            // allocate memory for each output tensor
            if (getOutputTensor(i)->isFloat()) {
                allocateOutputData<float>(i);
            } else {
                assert(getOutputTensor(i)->isInt8());
                allocateOutputData<signed char>(i);
            }
        }
        m_impl->allocate();
    }
}

template<class T>
void MWCNNLayer::allocateOutputData(int i) {
    m_impl->allocateOutputData<T>(i);
}

void MWCNNLayer::deallocate() {
    if (m_impl) {
#if defined(MW_TARGET_TYPE_ARMNEON)
        for (int i = 0; i < getNumInputs(); ++i) {
            m_impl->deallocateInputData(i);
        }
#endif
        // deallocate memory for each output tensor
        for (int i = 0; i < getNumOutputs(); ++i) {
            if (getOutputTensor(i)->isFloat()) {
                deallocateOutputData<float>(i);
            } else {
                assert(getOutputTensor(i)->isInt8());
                deallocateOutputData<signed char>(i);
            }
        }
        m_impl->deallocate();
    }
    resetTensorDataPointers();
}

// remove dangling pointers
void MWCNNLayer::resetTensorDataPointers() {
    for (int i = 0; i < getNumOutputs(); ++i) {
        MWTensorBase *opTensorBase = getOutputTensor(i);

        if (opTensorBase->isFloat()) {
            static_cast<MWTensor<float> *>(opTensorBase)->setData((float *) NULL);
        } else {
            assert(opTensorBase->isInt8());
            static_cast<MWTensor<signed char> *>(opTensorBase)->setData((signed char *) NULL);
        }
    }
}

template<class T>
void MWCNNLayer::deallocateOutputData(int i) {
    MWTensorBase *opTensorBase = getOutputTensor(i);

    bool bufferReuse = opTensorBase->getopBufIndex() >= 0;
    if (!bufferReuse) {
        T *data = static_cast<MWTensor<T> *>(opTensorBase)->getData();
        if (data) {
            m_impl->deallocateOutputData<T>(i);
        }
    }
}

void MWCNNLayer::postSetup() {
    if (m_impl) {
        m_impl->postSetup();
    }
}

std::string getFilePath(std::string fileS) {
    char *usrDataPath = NULL;
    std::string path;

    // Get file path from user environment
    usrDataPath = getenv("USER_DL_DATA_PATH");

    if (usrDataPath != NULL) {
        path = usrDataPath;
    } else {
        // Get file path from macro
        path = STR(MW_DL_DATA_PATH);
    }

    // Find name of file
    size_t fNamePos = fileS.find_last_of("/\\");

    if (fNamePos != std::string::npos) {
        std::string fileN(fileS.substr(fNamePos));
        fileS = path + fileN;
    } else {
        fileS = path + fileS;
    }

    return fileS;
}

// open filename
// if file is not found, look in current directory
FILE *MWCNNLayer::openBinaryFile(const char *fileName) {
    FILE *fp = fopen(fileName, "rb");
    if (!fp) {
        std::string fileS(fileName);
        std::string codegenFolder;

        /* Code to extract the directory path */
        size_t fNamePos = fileS.find_last_of("/\\");

        if (fNamePos != std::string::npos) {
            codegenFolder = fileS.substr(0, fNamePos);
        } else {
            /* Default path when there is no directory*/
            codegenFolder = std::string(".");
        }

        size_t pos = 0;
#ifdef MW_DL_DATA_PATH
        fileS = getFilePath(fileS);
#else
#if defined(_WIN32) || defined(_WIN64)

        char delim_unix[] = "/";
        char delim_win[] = "\\";

        while (((pos = fileS.find(delim_unix)) != std::string::npos) ||
               ((pos = fileS.find(delim_win)) != std::string::npos))
#else
        char delim_unix[] = "/";

        while ((pos = fileS.find(delim_unix)) != std::string::npos)
#endif
        {
            if (pos == (fileS.size() - 1)) {
                fileS = "";
                break;
            }
            fileS = fileS.substr(pos + 1);
        }
#endif

        if (!fileS.empty()) {
            fp = fopen(fileS.c_str(), "rb");
        }

        if (!fp) {
            std::string errmsg = std::string("Unable to find the ") + fileS +
                                 std::string(" binary file in ") + codegenFolder +
                                 std::string(" .");
            throw std::runtime_error(errmsg.c_str());
        }
    }

    return fp;
}

std::runtime_error MWCNNLayer::getFileOpenError(const char *filename) {
    const std::string message = std::string("Error! Unable to open file ") + std::string(filename);
    return std::runtime_error(message);
}

void MWCNNLayer::setName(const std::string &n) {
    m_name = n;
    return;
}

MWTensorBase *MWCNNLayer::getInputTensor(int index) {
    std::map<int, MWTensorBase *>::iterator it = m_input.find(index);
    assert(it != m_input.end());
    return it->second;
}

MWTensorBase *MWCNNLayer::getOutputTensor(size_t index) {
    std::map<int, MWTensorBase *>::iterator it = m_output.find(static_cast<const int>(index));
    assert(it != m_output.end());
    return it->second;
}

void MWCNNLayer::setInputTensor(MWTensorBase *other, int index) {
    m_input[index] = other;
}

int MWCNNLayer::getHeight(int index) {
    return getOutputTensor(index)->getHeight();
}

int MWCNNLayer::getBatchSize() {
    // return batch size from the output tensor
    return getOutputTensor(0)->getBatchSize();
}

int MWCNNLayer::getWidth(int index) {
    return getOutputTensor(index)->getWidth();
}

int MWCNNLayer::getNumInputFeatures(int index) {
    return getInputTensor(index)->getChannels();
}

int MWCNNLayer::getNumOutputFeatures(int index) {
    return getOutputTensor(index)->getChannels();
}

float *MWCNNLayer::getLayerOutput(int index) {
    // assumes layer output is float type
    return static_cast<MWTensor<float> *>(getOutputTensor(index))->getData();
}

void MWCNNLayer::resizeOutputTensor(int numHeight,
                                    int numWidth,
                                    int numChannels,
                                    int batchSize,
                                    int sequenceLength,
                                    int index) {
    std::map<int, MWTensorBase *>::iterator it = m_output.find(index);
    assert(it != m_output.end());
    it->second->setHeight(numHeight);
    it->second->setWidth(numWidth);
    it->second->setChannels(numChannels);
    it->second->setBatchSize(batchSize);
    it->second->setSequenceLength(sequenceLength);
}

void MWCNNLayer::setupTensors(int numInputs, int numOutputs, ...) {

    // initialize the variable argument list
    va_list args;
    va_start(args, numOutputs);

    // set all the input tensors
    for (int iTensor = 0; iTensor < numInputs; iTensor++) {
        MWTensorBase *inputTensor = va_arg(args, MWTensorBase*);
        setInputTensor(inputTensor, iTensor);
    }

    // set buffer index for all the output tensors
    for (int oTensor = 0; oTensor < numOutputs; oTensor++) {
        // allocate the tensor
        allocateOutputTensor<float>(-1, -1, -1, -1, -1, NULL, oTensor);

        int outbufIdx = va_arg(args, int);
        getOutputTensor(oTensor)->setopBufIndex(outbufIdx);
    }

    va_end(args);
}


MWTensorBase::MWTensorBase(int height,
                           int width,
                           int channels,
                           int batchsize,
                           int sequencelength,
                           MWCNNLayer *owner,
                           int srcport)
        : m_owner(owner), m_srcport(srcport), opBufIndex(-1), m_size(5) {
    m_shape[H] = height;
    m_shape[W] = width;
    m_shape[C] = channels;
    m_shape[N] = batchsize;
    m_shape[S] = sequencelength;

    m_dimsLayout[0] = S;
    m_dimsLayout[1] = N;
    m_dimsLayout[2] = C;
    m_dimsLayout[3] = H;
    m_dimsLayout[4] = W;
}


void MWTensorBase::getShape(int tensorShape[5]) const {
    tensorShape[H] = m_shape[H];
    tensorShape[W] = m_shape[W];
    tensorShape[C] = m_shape[C];
    tensorShape[N] = m_shape[N];
    tensorShape[S] = m_shape[S];
}

void MWTensorBase::getLayout(DIMSLABEL dimsLayout[]) {
    for (int idx = 0; idx < m_size; idx++) {
        dimsLayout[idx] = m_dimsLayout[idx];
    }
}

void MWTensorBase::setLayout(DIMSLABEL dimsLayout[]) {
    for (int idx = 0; idx < m_size; idx++) {
        m_dimsLayout[idx] = dimsLayout[idx];
    }
}

void MWTensorBase::getDimsByLayout(DIMSLABEL layout[], int size, int dims[]) {
    // size is size of layout
    // for TensorRT layout size < m_size
    for (int idx = 0; idx < size; idx++) {
        dims[idx] = m_shape[layout[idx]];
    }
}

/********************************************************************
                         Static Methods
*********************************************************************/

// Inputs
//   dims: array of dimensions, sorted from most-slowly varying dimension to fastest-varying
//   dimension left-to-right size: number of dimensions
// Output:
//   stride: an array of the same size as dimensions specifying the stride of the corresponding
//   dimension
//
//   The right-most stride should always be 1. The stride of any other dimension will be the
//   cumulative product of all more-slowly-varying dimensions. Thus, if dims is passed in as
//   {seqLength, batchSize, channels, height, width}, the output strides will be SNCHW.
void MWTensorBase::getStrides(const int dims[], int size, int stride[]) {
    int totalStride = 1;
    for (int i = 0; i < size; ++i) {
        stride[size - 1 - i] = totalStride;
        totalStride *= dims[size - 1 - i];
    }
}


// Inputs
//   srcLayout : array of DIMSLABEL for the source
//   destLayout : array of DIMSLABEL for the destination
//   dims: array of dimensions, sorted from most-slowly varying dimension to fastest-varying
//   dimension left-to-right
//   size: number of dimensions
// Output:
//   stride: an array of the same size as dimensions specifying the stride of the DIMSLABEL
//      dimension corresponding to the source
//
// If source is {H, W, C} and dest is {C, W, H} the strides for transforming data from source to
// dest is {1, H, WH} The stride {1, H, WH} correspond to the source layout of {H, W, C}. Consumer
// of this stride is cudnn/mkldnn API's where "dims" correspond to the source layout

void MWTensorBase::getTransformStrides(const DIMSLABEL srcLayout[],
                                       const DIMSLABEL destLayout[],
                                       const int dims[],
                                       int size,
                                       int strides[]) {
    // conversion from src layout to dest layout

    for (int idxSrc = 0; idxSrc < size; idxSrc++) {
        int totalStride = 1;
        bool foundDim = false;
        for (int idxDest = 0; idxDest < size; idxDest++) {
            if (srcLayout[idxSrc] == destLayout[idxDest]) {
                foundDim = true;
            }
            if (foundDim && idxDest < (size - 1)) {
                // loop over dimensions to the right to find total stride for current dimension
                totalStride *= dims[idxDest + 1];
            }
        }
        strides[idxSrc] = totalStride;
    }
}

// Inputs
//   srcLayout : array of DIMSLABEL for the source
//   destLayout : array of DIMSLABEL for the destination
//   size: number of dimensions of the source/dest layout
// Output:
//   order: an array of the same size as layout specifying the order by which the source needs
//      needs to be permuted to obtain dest layout
//
// If source is {H, W, C} and dest is {C, W, H}, then order is {2, 1, 0}
void MWTensorBase::getPermutationOrder(const DIMSLABEL srcLayout[],
                                       const DIMSLABEL destLayout[],
                                       int size,
                                       int order[]) {

    // conversion from src layout to dest layout
    for (int idxDest = 0; idxDest < size; idxDest++) {
        for (int idxSrc = 0; idxSrc < size; idxSrc++) {
            if (srcLayout[idxSrc] == destLayout[idxDest]) {
                order[idxDest] = idxSrc;
                break;
            }
        }
    }
}


#if DECOUPLE

// Creating the ImageInputLayer
void MWInputLayer::createInputLayer(MWTargetNetworkImpl *ntwk_impl,
                                    MWTensorBase *m_in,
                                    int /*m_height*/,
                                    int /*m_width*/,
                                    int /*m_channels*/,
                                    int /*m_withAvg*/,
                                    const char * /*avg_file_name*/,
                                    int outbufIdx) {
    setInputTensor(m_in);
    allocateOutputTensor<float>(-1, -1, -1, -1, -1, NULL);

    getOutputTensor(0)->setopBufIndex(outbufIdx);

    m_impl = new MWInputLayerImpl(this, ntwk_impl);
}

#elif defined(MW_TARGET_TYPE_TENSORRT) || defined(MW_TARGET_TYPE_ARMMALI)
void MWInputLayer::createInputLayer(MWTargetNetworkImpl* ntwk_impl,
                                    MWTensorBase* m_in,
                                    int  m_height,
                                    int m_width,
                                    int m_channels,
                                    int /*m_withAvg*/,
                                    const char* /*avg_file_name*/,
                                    int outbufIdx) {
    // populate output tensor
    allocateOutputTensor<float>(m_height, m_width, m_channels, m_in->getBatchSize(), 1, NULL);

    assert(m_in->getSequenceLength() == 1);

    getOutputTensor(0)->setopBufIndex(outbufIdx);

    m_impl = new MWInputLayerImpl(this, ntwk_impl);

    assert(getOutputTensor()->getSequenceLength() == 1);
}

#else
void MWInputLayer::createInputLayer(MWTargetNetworkImpl* ntwk_impl,
                                    MWTensorBase* m_in,
                                    int  m_height,
                                    int m_width,
                                    int m_channels,
                                    int m_withAvg,
                                    const char* avg_file_name,
                                    int outbufIdx) {
    
    // This branch is to maintain compatibility with cmdnn
     // populate output tensor
    allocateOutputTensor<float>(m_height, m_width, m_channels, m_in->getBatchSize(), 1, NULL);

    assert(m_in->getSequenceLength() == 1);

    m_impl = new MWInputLayerImpl(this,
                                  ntwk_impl,
                                  m_in->getBatchSize(),
                                  m_height,
                                  m_width,
                                  m_channels,
                                  m_withAvg,
                                  avg_file_name,
                                  outbufIdx);

    assert(getOutputTensor()->getSequenceLength() == 1);
}

#endif

void MWInputLayer::propagateSize() {
#if DECOUPLE
    resizeOutputTensor(getInputTensor(0)->getHeight(), getInputTensor(0)->getWidth(),
                       getInputTensor(0)->getChannels(), getInputTensor(0)->getBatchSize(),
                       getInputTensor(0)->getSequenceLength());

    assert(getOutputTensor()->getSequenceLength() == 1);

    m_impl->propagateSize();
#endif
}

// Create ReLULayer
void MWReLULayer::createReLULayer(MWTargetNetworkImpl *ntwk_impl,
                                  MWTensorBase *m_in,
                                  int outbufIdx) {
#if DECOUPLE
    setInputTensor(m_in);
    allocateOutputTensor<float>(-1, -1, -1, -1, -1, NULL);

    getOutputTensor(0)->setopBufIndex(outbufIdx);

    m_impl = new MWReLULayerImpl(this, ntwk_impl);

#else
    setInputTensor(m_in);

    // allocate output, reusing input tensor's data buffer
    int numOutputFeatures = getInputTensor()->getChannels();
    allocateOutputTensor<float>(getInputTensor()->getHeight(), getInputTensor()->getWidth(),
                                numOutputFeatures, getInputTensor()->getBatchSize(),
                                getInputTensor()->getSequenceLength(), NULL);

    m_impl = new MWReLULayerImpl(this, ntwk_impl, outbufIdx);

    return;
#endif
}

void MWReLULayer::propagateSize() {
#if DECOUPLE
    resizeOutputTensor(getInputTensor()->getHeight(), getInputTensor()->getWidth(),
                       getInputTensor()->getChannels(), getInputTensor()->getBatchSize(),
                       getInputTensor()->getSequenceLength());

    m_impl->propagateSize();
#endif
}

// Create CrossChannelNormalizationLayer
// Parameters here are the same naming as NNT.
void MWNormLayer::createNormLayer(MWTargetNetworkImpl *ntwk_impl,
                                  MWTensorBase *m_in,
                                  unsigned m_WindowChannelSize,
                                  double m_Alpha,
                                  double m_Beta,
                                  double m_K,
                                  int outbufIdx) {
#if DECOUPLE
    setInputTensor(m_in);
    allocateOutputTensor<float>(-1, -1, -1, -1, -1, NULL);

    getOutputTensor(0)->setopBufIndex(outbufIdx);

    m_impl = new MWNormLayerImpl(this, ntwk_impl, m_WindowChannelSize, m_Alpha, m_Beta, m_K);

#else
    setInputTensor(m_in);

    int numOutputFeatures = getInputTensor()->getChannels();
    allocateOutputTensor<float>(getInputTensor()->getHeight(), getInputTensor()->getWidth(),
                                numOutputFeatures, getInputTensor()->getBatchSize(),
                                getInputTensor()->getSequenceLength(), NULL);
    assert(getOutputTensor()->getSequenceLength() == 1);

    m_impl =
        new MWNormLayerImpl(this, ntwk_impl, m_WindowChannelSize, m_Alpha, m_Beta, m_K, outbufIdx);
#endif
}

void MWNormLayer::propagateSize() {
#if DECOUPLE
    assert(getInputTensor()->getSequenceLength() == 1);

    resizeOutputTensor(getInputTensor()->getHeight(), getInputTensor()->getWidth(),
                       getInputTensor()->getChannels(), getInputTensor()->getBatchSize(),
                       getInputTensor()->getSequenceLength());

    m_impl->propagateSize();
#endif
}

// Create MaxPooling2DLayer with PoolSize = [ PoolH PoolW ]
//                                Stride = [ StrideH StrideW ]
//                               Padding = [ PaddingH_T PaddingH_B PaddingW_L PaddingW_R ]
void MWMaxPoolingLayer::createMaxPoolingLayer(MWTargetNetworkImpl *ntwk_impl,
                                              MWTensorBase *m_in,
                                              int m_PoolH,
                                              int m_PoolW,
                                              int m_StrideH,
                                              int m_StrideW,
                                              int m_PaddingH_T,
                                              int m_PaddingH_B,
                                              int m_PaddingW_L,
                                              int m_PaddingW_R,
                                              bool m_hasIndices,
                                              int numOutputs,
                                              ...) {
#if DECOUPLE
    setInputTensor(m_in);
    allocateOutputTensor<float>(-1, -1, -1, -1, -1, NULL, 0);

    if (m_hasIndices) {
        // allocate index tensor
        allocateOutputTensor<float>(-1, -1, -1, -1, -1, NULL, 1);
    }

    strideH = m_StrideH;
    strideW = m_StrideW;

    poolH = m_PoolH;
    poolW = m_PoolW;

    isGlobalAveragePooling = (poolH == -1) && (poolW == -1);

    paddingH_T = m_PaddingH_T;
    paddingH_B = m_PaddingH_B;
    paddingW_L = m_PaddingW_L;
    paddingW_R = m_PaddingW_R;

    hasIndices = m_hasIndices;

    {
        va_list args;
        va_start(args, numOutputs);
        std::vector<int> bufIndices(numOutputs, -1);

        for (int i = 0; i < numOutputs; i++) {
            bufIndices[i] = va_arg(args, int);
        }

        for (int i = 0; i < getNumOutputs(); ++i) {
            getOutputTensor(i)->setopBufIndex(bufIndices[i]);
        }

        m_impl = new MWMaxPoolingLayerImpl(this, ntwk_impl, m_PoolH, m_PoolW, m_StrideH, m_StrideW,
                                           m_PaddingH_T, m_PaddingH_B, m_PaddingW_L, m_PaddingW_R,
                                           m_hasIndices, numOutputs);
    }
#else
    setInputTensor(m_in);

    // Global Max Pooling case
    if ((m_PoolH == -1) && (m_PoolW == -1)) {
        m_PoolH = getInputTensor()->getHeight();
        m_PoolW = getInputTensor()->getWidth();
    }

    int outputH =
        ((getInputTensor()->getHeight() - m_PoolH + m_PaddingH_T + m_PaddingH_B) / m_StrideH) + 1;
    int outputW =
        ((getInputTensor()->getWidth() - m_PoolW + m_PaddingW_L + m_PaddingW_R) / m_StrideW) + 1;

    int numOutputFeatures = getInputTensor()->getChannels();
    allocateOutputTensor<float>(outputH, outputW, numOutputFeatures,
                                getInputTensor()->getBatchSize(),
                                getInputTensor()->getSequenceLength(), NULL, 0);
    assert(getOutputTensor()->getSequenceLength() == 1);

    if (m_hasIndices) {
        // allocate index tensor
        allocateOutputTensor<float>(outputH, outputW, numOutputFeatures,
                                    getInputTensor()->getBatchSize(),
                                    getInputTensor()->getSequenceLength(), NULL, 1);
        MWTensorBase* indexOpTensor = getOutputTensor(1);
        assert(indexOpTensor->getSequenceLength() == 1);
    }


    {
        va_list args;
        va_start(args, numOutputs);
        std::vector<int> bufIndices(numOutputs, -1);

        for (int i = 0; i < numOutputs; i++) {
            bufIndices[i] = va_arg(args, int);
        }

        m_impl = new MWMaxPoolingLayerImpl(this, ntwk_impl, m_PoolH, m_PoolW, m_StrideH, m_StrideW,
                                           m_PaddingH_T, m_PaddingH_B, m_PaddingW_L, m_PaddingW_R,
                                           m_hasIndices, numOutputs, bufIndices);
    }
#endif
}

void MWMaxPoolingLayer::propagateSize() {
#if DECOUPLE
    // Global Average Pooling case
    if (isGlobalAveragePooling) {
        poolH = getInputTensor()->getHeight();
        poolW = getInputTensor()->getWidth();
    }

    int outputH = ((getInputTensor()->getHeight() - poolH + paddingH_T + paddingH_B) / strideH) + 1;
    int outputW = ((getInputTensor()->getWidth() - poolW + paddingW_L + paddingW_R) / strideW) + 1;

    assert(getInputTensor()->getSequenceLength() == 1);

    resizeOutputTensor(outputH, outputW, getInputTensor()->getChannels(),
                       getInputTensor()->getBatchSize(), getInputTensor()->getSequenceLength());

    if (hasIndices) {
        // allocate index tensor
        resizeOutputTensor(outputH, outputW, getInputTensor()->getChannels(),
                           getInputTensor()->getBatchSize(), getInputTensor()->getSequenceLength(),
                           1);
    }

    m_impl->propagateSize();
#endif
}

// Create FullyConnectedLayer with corresponding InputSize and OutputSize.
// This implementation uses SGEMV for m_BatchSize = 1 and SGEMM for others.
void MWFCLayer::createFCLayer(MWTargetNetworkImpl *ntwk_impl,
                              MWTensorBase *m_in,
                              int m_InputSize,
                              int m_OutputSize,
                              const char *m_weights_file,
                              const char *m_bias_file,
                              int outbufIdx) {
#if DECOUPLE
    setInputTensor(m_in);
    allocateOutputTensor<float>(-1, -1, -1, -1, -1, NULL);

    getOutputTensor(0)->setopBufIndex(outbufIdx);

    numInputFeatures = m_InputSize;
    numOutputFeatures = m_OutputSize;

    m_impl =
            new MWFCLayerImpl(this, ntwk_impl, m_InputSize, m_OutputSize, m_weights_file, m_bias_file);

#else
    setInputTensor(m_in);
    allocateOutputTensor<float>(1, 1, m_OutputSize, getInputTensor()->getBatchSize(),
                                getInputTensor()->getSequenceLength(), NULL);

    m_impl =
        new MWFCLayerImpl(this, ntwk_impl, m_InputSize, m_weights_file, m_bias_file, outbufIdx);

    return;
#endif
}

void MWFCLayer::propagateSize() {
#if DECOUPLE
    resizeOutputTensor(1, 1, numOutputFeatures, getInputTensor()->getBatchSize(),
                       getInputTensor()->getSequenceLength());

    m_impl->propagateSize();
#endif
}

// Create SoftmaxLayer
void MWSoftmaxLayer::createSoftmaxLayer(MWTargetNetworkImpl *ntwk_impl,
                                        MWTensorBase *m_in,
                                        int outbufIdx) {
#if DECOUPLE
    setInputTensor(m_in);
    allocateOutputTensor<float>(-1, -1, -1, -1, -1, NULL);

    getOutputTensor(0)->setopBufIndex(outbufIdx);

    m_impl = new MWSoftmaxLayerImpl(this, ntwk_impl);

#else
    setInputTensor(m_in);

    allocateOutputTensor<float>(getInputTensor()->getHeight(), getInputTensor()->getWidth(),
                                getInputTensor()->getChannels(), getInputTensor()->getBatchSize(),
                                getInputTensor()->getSequenceLength(), NULL);

    m_impl = new MWSoftmaxLayerImpl(this, ntwk_impl, outbufIdx);

    return;
#endif
}

void MWSoftmaxLayer::propagateSize() {
#if DECOUPLE
    resizeOutputTensor(getInputTensor()->getHeight(), getInputTensor()->getWidth(),
                       getInputTensor()->getChannels(), getInputTensor()->getBatchSize(),
                       getInputTensor()->getSequenceLength());

    m_impl->propagateSize();
#endif
}

// Create ClassificationOutputLayer
// We are doing inference only so LossFunction is not needed.
// This layer would do nothing but point the data to previous layer.
void MWOutputLayer::createOutputLayer(MWTargetNetworkImpl *ntwk_impl,
                                      MWTensorBase *m_in,
                                      int outbufIdx) {
#if DECOUPLE
    setInputTensor(m_in);
    allocateOutputTensor<float>(-1, -1, -1, -1, -1, NULL);

    getOutputTensor(0)->setopBufIndex(outbufIdx);

    m_impl = new MWOutputLayerImpl(this, ntwk_impl);

#else
    setInputTensor(m_in);

    assert(getInputTensor()->isFloat());
    allocateOutputTensor<float>(getInputTensor()->getHeight(), getInputTensor()->getWidth(),
                                getInputTensor()->getChannels(), getInputTensor()->getBatchSize(),
                                getInputTensor()->getSequenceLength(),
                                static_cast<MWTensor<float>*>(getInputTensor())->getData());

    m_impl = new MWOutputLayerImpl(this, ntwk_impl, outbufIdx);
    return;
#endif
}

void MWOutputLayer::propagateSize() {
#if DECOUPLE
    resizeOutputTensor(getInputTensor()->getHeight(), getInputTensor()->getWidth(),
                       getInputTensor()->getChannels(), getInputTensor()->getBatchSize(),
                       getInputTensor()->getSequenceLength());
    m_impl->propagateSize();
#endif
}

void MWOutputLayer::predict() {
    m_impl->predict();
}


// Create pass through
// This layer would do nothing but point the data to previous layer.
void MWPassthroughLayer::createPassthroughLayer(MWTargetNetworkImpl * /* ntwk_impl */,
                                                MWTensorBase *m_in,
                                                int /* outbufIdx */) {
#if DECOUPLE
    setInputTensor(m_in);
    allocateOutputTensor<float>(-1, -1, -1, -1, -1, NULL);

#else

    setInputTensor(m_in);

    assert(getInputTensor()->isFloat());
    allocateOutputTensor<float>(getInputTensor()->getHeight(), getInputTensor()->getWidth(),
                                getInputTensor()->getChannels(), getInputTensor()->getBatchSize(),
                                getInputTensor()->getSequenceLength(),
                                static_cast<MWTensor<float>*>(getInputTensor())->getData());

    return;
#endif
}

void MWPassthroughLayer::propagateSize() {
#if DECOUPLE
    resizeOutputTensor(getInputTensor(0)->getHeight(), getInputTensor(0)->getWidth(),
                       getInputTensor(0)->getChannels(), getInputTensor(0)->getBatchSize(),
                       getInputTensor(0)->getSequenceLength());
#endif
}

void MWPassthroughLayer::allocate() {
#if DECOUPLE
    assert(getInPlaceIndex(0) == 0);

    MWTensorBase *ipTensorBase = getInputTensor(0);
    MWTensorBase *opTensorBase = getOutputTensor(0);

    static_cast<MWTensor<float> *>(opTensorBase)
            ->setData(static_cast<MWTensor<float> *>(ipTensorBase)->getData());

#endif
}

// Create AvgPooling2DLayer with PoolSize = [ PoolH PoolW ]
//                                Stride = [ StrideH StrideW ]
//                               Padding = [ PaddingH_T PaddingH_T_B PaddingW_L PaddingW_R ]
void MWAvgPoolingLayer::createAvgPoolingLayer(MWTargetNetworkImpl *ntwk_impl,
                                              MWTensorBase *m_in,
                                              int m_PoolH,
                                              int m_PoolW,
                                              int m_StrideH,
                                              int m_StrideW,
                                              int m_PaddingH_T,
                                              int m_PaddingH_B,
                                              int m_PaddingW_L,
                                              int m_PaddingW_R,
                                              int outbufIdx) {
#if DECOUPLE
    setInputTensor(m_in);
    allocateOutputTensor<float>(-1, -1, -1, -1, -1, NULL);

    getOutputTensor(0)->setopBufIndex(outbufIdx);

    strideH = m_StrideH;
    strideW = m_StrideW;

    poolH = m_PoolH;
    poolW = m_PoolW;

    isGlobalAveragePooling = (poolH == -1) && (poolW == -1);

    paddingH_T = m_PaddingH_T;
    paddingH_B = m_PaddingH_B;
    paddingW_L = m_PaddingW_L;
    paddingW_R = m_PaddingW_R;

    m_impl = new MWAvgPoolingLayerImpl(this, ntwk_impl, m_PoolH, m_PoolW, m_StrideH, m_StrideW,
                                       m_PaddingH_T, m_PaddingH_B, m_PaddingW_L, m_PaddingW_R);

#else
    setInputTensor(m_in);

    // Global Average Pooling case
    if ((m_PoolH == -1) && (m_PoolW == -1)) {
        m_PoolH = getInputTensor()->getHeight();
        m_PoolW = getInputTensor()->getWidth();
    }

    int outputH =
        ((getInputTensor()->getHeight() - m_PoolH + m_PaddingH_T + m_PaddingH_B) / m_StrideH) + 1;
    int outputW =
        ((getInputTensor()->getWidth() - m_PoolW + m_PaddingW_L + m_PaddingW_R) / m_StrideW) + 1;

    int numOutputFeatures = getInputTensor()->getChannels();
    allocateOutputTensor<float>(outputH, outputW, numOutputFeatures,
                                getInputTensor()->getBatchSize(),
                                getInputTensor()->getSequenceLength(), NULL);
    assert(getOutputTensor()->getSequenceLength() == 1);

    m_impl = new MWAvgPoolingLayerImpl(this, ntwk_impl, m_PoolH, m_PoolW, m_StrideH, m_StrideW,
                                       m_PaddingH_T, m_PaddingH_B, m_PaddingW_L, m_PaddingW_R,
                                       outbufIdx);
#endif
}

void MWAvgPoolingLayer::propagateSize() {
#if DECOUPLE
    // Global Average Pooling case
    if (isGlobalAveragePooling) {
        poolH = getInputTensor()->getHeight();
        poolW = getInputTensor()->getWidth();
    }

    int outputH = ((getInputTensor()->getHeight() - poolH + paddingH_T + paddingH_B) / strideH) + 1;
    int outputW = ((getInputTensor()->getWidth() - poolW + paddingW_L + paddingW_R) / strideW) + 1;

    assert(getInputTensor()->getSequenceLength() == 1);

    resizeOutputTensor(outputH, outputW, getInputTensor()->getChannels(),
                       getInputTensor()->getBatchSize(), getInputTensor()->getSequenceLength());

    m_impl->propagateSize();
#endif
}
