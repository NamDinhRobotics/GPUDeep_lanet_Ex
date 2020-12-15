/* Copyright 2016-2020 The MathWorks, Inc. */
#ifndef CNN_API_HPP
#define CNN_API_HPP

#include <cstdio>
#include <string>
#include <stdexcept>
#include <map>
#include <cassert>
#include <vector>

template<typename T>
void call_fread(T *m_buffer,
                size_t m_size,
                size_t m_num,
                FILE *m_fp,
                char const *const m_fileName) {
    if (fread(m_buffer, m_size, m_num, m_fp) != m_num) {
        throw std::runtime_error("Unexpected number of bytes read from " + std::string(m_fileName));
    }
}

class MWTensorBase;

template<class T>
class MWTensor;

class MWTargetNetworkImpl;

class MWCNNLayerImpl;

class MWCNNLayer {
protected:
    std::string m_name;                    // Name of the layer
    std::map<int, MWTensorBase *> m_input;  // inputs
    std::map<int, MWTensorBase *> m_output; // outputs

    MWCNNLayerImpl *m_impl; // layer impl

    bool m_isCustomLayer; // flag that returns true if the layer is a custom layer

public:
    MWCNNLayer();

    virtual ~MWCNNLayer();

    virtual void predict();

    virtual void cleanup();

    virtual void propagateSize() = 0;

    virtual void allocate();

    template<class T>
    void allocateOutputData(int i);

    void deallocate();

    template<class T>
    void deallocateOutputData(int i);

    void resetTensorDataPointers();

    void postSetup();

    MWCNNLayerImpl *getImpl() {
        return m_impl;
    }

    size_t getNumOutputs() {
        return m_output.size();
    }

    size_t getNumInputs() {
        return m_input.size();
    }

    MWTensorBase *getInputTensor(int index = 0);

    float *getLayerOutput(int index);

    MWTensorBase *getOutputTensor(size_t index = 0);

    void setName(const std::string &); // Set the name for this layer
    std::string getName() const {
        return m_name;
    }; // Get the name for this layer

    int getInPlaceIndex(int outIdx) {
        std::map<int, int>::iterator it = m_inPlaceIndexMap.find(outIdx);
        if (it == m_inPlaceIndexMap.end()) {
            return -1;
        }

        int inIdx = it->second;

        assert(inIdx != -1);

        return inIdx;
    }

    void setInPlaceIndex(int outIdx, int inIdx) {
        std::map<int, int>::iterator it = m_inPlaceIndexMap.find(outIdx);
        assert(it == m_inPlaceIndexMap.end());
        m_inPlaceIndexMap[outIdx] = inIdx;
    }

    static bool isDebuggingEnabled() {
        return m_enableDebugging;
    };

    bool isCustomLayer() const {
        return m_isCustomLayer;
    }

protected:
    int getBatchSize();                      // Get the batch size
    int getHeight(int index = 0);            // Get the height of output y
    int getWidth(int index = 0);             // Get the width of output y
    int getNumInputFeatures(int index = 0);  // Get the number of channels of the input
    int getNumOutputFeatures(int index = 0); // Get the number of channels of the output

    void setInputTensor(MWTensorBase *other, int index = 0); // shallow copy tensor from other

    template<class T>
    void allocateOutputTensor(int numHeight,
                              int numWidth,
                              int numChannels,
                              int batchsize,
                              int sequencelength,
                              T *data,
                              int index = 0); // allocate output tensor

    void resizeOutputTensor(int numHeight,
                            int numWidth,
                            int numChannels,
                            int batchsize,
                            int sequencelength,
                            int index = 0); // resize output tensor

    // load non-scalar numeric parameters for custom layers
    template<typename T>
    void loadParams(const char *MW_mangled_fileName,
                    const int MW_mangled_num,
                    T *MW_mangled_buffer) {
        // MW_mangled_buffer is float buffer of size MW_mangled_num
        FILE *MW_mangled_fp = MWCNNLayer::openBinaryFile(MW_mangled_fileName);
        assert(MW_mangled_fp);
        call_fread(MW_mangled_buffer, sizeof(T), MW_mangled_num, MW_mangled_fp,
                   MW_mangled_fileName);
        fclose(MW_mangled_fp);
    }

    void setupTensors(int numInputs, int numOutputs, ...); // setup method for input-output tensors


public:
    static FILE *openBinaryFile(const char *filename);

    static std::runtime_error getFileOpenError(const char *filename);

private:
    std::map<int, int> m_inPlaceIndexMap;

#if DEBUG

    const static bool m_enableDebugging = true;

#else

    const static bool m_enableDebugging = false;

#endif
};

template<class T>
void MWCNNLayer::allocateOutputTensor(int numHeight,
                                      int numWidth,
                                      int numChannels,
                                      int batchSize,
                                      int sequenceLength,
                                      T *data,
                                      int index) {
    MWTensorBase *op = new MWTensor<T>(numHeight, numWidth, numChannels, batchSize, sequenceLength,
                                       data, this, index);
    assert(op != NULL);
    std::map<int, MWTensorBase *>::iterator it = m_output.find(index);
    assert(it == m_output.end());
    m_output[index] = op;
}

class MWTensorBase {
public:
    MWTensorBase()
            : m_owner(NULL), m_srcport(-1), opBufIndex(-1), m_size(5) {

        m_shape[H] = -1;
        m_shape[W] = -1;
        m_shape[C] = -1;
        m_shape[N] = -1;
        m_shape[S] = -1;

        m_dimsLayout[0] = S;
        m_dimsLayout[1] = N;
        m_dimsLayout[2] = C;
        m_dimsLayout[3] = H;
        m_dimsLayout[4] = W;
    };

    MWTensorBase(int height,
                 int width,
                 int channels,
                 int batchsize,
                 int sequencelength,
                 MWCNNLayer *owner,
                 int srcport);

    virtual ~MWTensorBase() {
    }

    enum DIMSLABEL {
        H = 0, W, C, N, S
    };

    int getHeight() const {
        return m_shape[H];
    }

    int getWidth() const {
        return m_shape[W];
    }

    int getChannels() const {
        return m_shape[C];
    }

    int getBatchSize() const {
        return m_shape[N];
    }

    int getSequenceLength() const {
        return m_shape[S];
    }

    int getNumElements() const {
        return getHeight() * getWidth() * getChannels() * getBatchSize() * getSequenceLength();
    }

    void setHeight(int height) {
        m_shape[H] = height;
    }

    void setWidth(int width) {
        m_shape[W] = width;
    }

    void setChannels(int channels) {
        m_shape[C] = channels;
    }

    void setBatchSize(int batchSize) {
        m_shape[N] = batchSize;
    }

    void setSequenceLength(int sequenceLength) {
        m_shape[S] = sequenceLength;
    }

    MWCNNLayer *getOwner() const {
        return m_owner;
    }

    int getSourcePortIndex() const {
        return m_srcport;
    }

    void setopBufIndex(int bufIndex) {
        opBufIndex = bufIndex;
    }

    int getopBufIndex() {
        return opBufIndex;
    };

    int getSize() {
        return m_size;
    }

    void getShape(int tensorShape[5]) const;

    void getLayout(DIMSLABEL[]);

    void setLayout(DIMSLABEL[]);

    void getDimsByLayout(DIMSLABEL[], int, int[]);

    // get strides based on dims
    static void getStrides(const int dims[], int size, int stride[]);

    // get strides for transforming from srcLayout to destLayout
    static void getTransformStrides(const DIMSLABEL srcLayout[],
                                    const DIMSLABEL destLayout[],
                                    const int dims[],
                                    int size,
                                    int strides[]);

    // get permutation order for permuting from srcLayout to destLayout
    static void getPermutationOrder(const DIMSLABEL srcLayout[],
                                    const DIMSLABEL destLayout[],
                                    int size,
                                    int order[]);

    virtual bool isFloat() const = 0;

    virtual bool isInt8() const = 0;

private:
    MWCNNLayer *m_owner;
    int m_srcport;
    int opBufIndex;

    int m_size;
    DIMSLABEL m_dimsLayout[5];
    int m_shape[5];
};

// Helper functions used by the type-checking functions in MWTensor<T>.
// These are used in lieu of std::is_same<T,U>().
// We cannot use std::is_same<T,U>() since it is a C++11 language feature.
template<class T>
bool is_float() {
    return false;
}

template<>
inline bool is_float<float>() {
    return true;
}

template<class T>
bool is_int8() {
    return false;
}

template<>
inline bool is_int8<signed char>() {
    return true;
}


template<class T>
class MWTensor : public MWTensorBase {
public:
    MWTensor()
            : MWTensorBase(), m_data(NULL) {};

    MWTensor(int height,
             int width,
             int channels,
             int batchsize,
             int sequencelength,
             T *data,
             MWCNNLayer *owner,
             int srcport)
            : MWTensorBase(height, width, channels, batchsize, sequencelength, owner, srcport), m_data(data) {};

    ~MWTensor() {
    }

    void setData(T *data) {
        m_data = data;
    }

    T *getData() const {
        return m_data;
    }

    virtual bool isFloat() const {
        return is_float<T>();
    }

    virtual bool isInt8() const {
        return is_int8<T>();
    }

private:
    T *m_data;
};

// ImageInputLayer
class MWInputLayer : public MWCNNLayer {
public:
    MWInputLayer() {
    }

    ~MWInputLayer() {
    }

    void createInputLayer(MWTargetNetworkImpl *ntwk_impl,
                          MWTensorBase *m_in,
                          int,
                          int,
                          int,
                          int,
                          const char *avg_file_name,
                          int);

    void propagateSize();
};

// ReLULayer
class MWReLULayer : public MWCNNLayer {
public:
    MWReLULayer() {
    }

    ~MWReLULayer() {
    }

    void createReLULayer(MWTargetNetworkImpl *, MWTensorBase *, int);

    void propagateSize();
};

// CrossChannelNormalizationLayer
class MWNormLayer : public MWCNNLayer {
public:
    MWNormLayer() {
    }

    ~MWNormLayer() {
    }

    void
    createNormLayer(MWTargetNetworkImpl *, MWTensorBase *, unsigned, double, double, double, int);

    void propagateSize();
};

// AvgPooling2DLayer
class MWAvgPoolingLayer : public MWCNNLayer {
public:
    MWAvgPoolingLayer()
            : isGlobalAveragePooling(false) {
    }

    ~MWAvgPoolingLayer() {
    }

    void createAvgPoolingLayer(MWTargetNetworkImpl *,
                               MWTensorBase *,
                               int,
                               int,
                               int,
                               int,
                               int,
                               int,
                               int,
                               int,
                               int);

    void propagateSize();

    int strideH;
    int strideW;

    int poolH;
    int poolW;

    int paddingH_T;
    int paddingH_B;
    int paddingW_L;
    int paddingW_R;

    bool isGlobalAveragePooling;
};

// MaxPooling2DLayer
class MWMaxPoolingLayer : public MWCNNLayer {
public:
    MWMaxPoolingLayer()
            : isGlobalAveragePooling(false) {
    }

    ~MWMaxPoolingLayer() {
    }

    // Create MaxPooling2DLayer with PoolSize = [ PoolH PoolW ]
    //                                Stride = [ StrideH StrideW ]
    //                               Padding = [ PaddingH_T PaddingH_B PaddingW_L PaddingW_R ]
    void createMaxPoolingLayer(MWTargetNetworkImpl *,
                               MWTensorBase *,
                               int,
                               int,
                               int,
                               int,
                               int,
                               int,
                               int,
                               int,
                               bool,
                               int,
                               ...);

    void propagateSize();

private:
    int strideH;
    int strideW;

    int poolH;
    int poolW;

    int paddingH_T;
    int paddingH_B;
    int paddingW_L;
    int paddingW_R;

    bool isGlobalAveragePooling;

    bool hasIndices;
};

// FullyConnectedLayer
class MWFCLayer : public MWCNNLayer {
public:
    MWFCLayer() {
    }

    ~MWFCLayer() {
    }

    void
    createFCLayer(MWTargetNetworkImpl *, MWTensorBase *, int, int, const char *, const char *, int);

    void propagateSize();

private:
    int numInputFeatures;
    int numOutputFeatures;
};

// SoftmaxLayer
class MWSoftmaxLayer : public MWCNNLayer {
public:
    MWSoftmaxLayer() {
    }

    ~MWSoftmaxLayer() {
    }

    void createSoftmaxLayer(MWTargetNetworkImpl *, MWTensorBase *, int);

    void propagateSize();
};

// ClassificationOutputLayer
class MWOutputLayer : public MWCNNLayer {
public:
    MWOutputLayer() {
    }

    ~MWOutputLayer() {
    }

    void createOutputLayer(MWTargetNetworkImpl *, MWTensorBase *, int);

    void predict();

    void propagateSize();
};

// pass through
class MWPassthroughLayer : public MWCNNLayer {
public:
    MWPassthroughLayer() {
    }

    ~MWPassthroughLayer() {
    }

    void createPassthroughLayer(MWTargetNetworkImpl *, MWTensorBase *, int);

    void propagateSize();

    void allocate();
};

// Onnx FlattenLayer
class MWRowMajorFlattenLayer : public MWCNNLayer {
public:
    MWRowMajorFlattenLayer() {
    }

    ~MWRowMajorFlattenLayer() {
    }

    void createRowMajorFlattenLayer(MWTargetNetworkImpl *, MWTensorBase *, int);

    void propagateSize();

    void allocate();
};

// FlattenLayer
class MWFlattenLayer : public MWCNNLayer {
public:
    MWFlattenLayer() {
    }

    ~MWFlattenLayer() {
    }

    void createFlattenLayer(MWTargetNetworkImpl *, MWTensorBase *, int);

    void propagateSize();
};

#endif
