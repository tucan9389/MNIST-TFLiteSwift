/*
* Copyright Doyoung Gwak 2020
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

//
//  TFLiteImageInterpreter.swift
//  PoseEstimation-TFLiteSwift
//
//  Created by Doyoung Gwak on 2020/03/14.
//  Copyright © 2020 Doyoung Gwak. All rights reserved.
//

import TensorFlowLite

struct TFLiteResult {
    let outputTensors: [Tensor]
}

class TFLiteImageInterpreter {
    let interpreter: Interpreter
    let options: Options
    var inputTensor: Tensor?
    var outputTensors: [Tensor] = []
    
    init(options: Options) {
        guard let modelPath = Bundle.main.path(forResource: options.modelName, ofType: "tflite") else {
            fatalError("Failed to load the model file with name: \(options.modelName).")
        }
        
        // Specify the options for the `Interpreter`.
        var interpreterOptions = Interpreter.Options()
        interpreterOptions.threadCount = options.threadCount
        
        // Specify the delegates for the `Interpreter`.
        var delegates: [Delegate]?
        switch options.accelerator {
        case .metal:
            delegates = [MetalDelegate()]
        default:
            delegates = nil
        }
        
        guard let interpreter = try? Interpreter(modelPath: modelPath, options: interpreterOptions, delegates: delegates) else {
            fatalError("Failed to craete interpreter")
        }
        
        self.interpreter = interpreter
        self.options = options
        
        do {
            try setupTensor(with: interpreter, options: options)
        } catch {
            fatalError("Failed to setup tensor: \(error.localizedDescription)")
        }
    }
    
    private func setupTensor(with interpreter: Interpreter, options: Options) throws {
        // Initialize input and output `Tensor`s.
        // Allocate memory for the model's input `Tensor`s.
        try interpreter.allocateTensors()
        
        // input tensor
        let inputTensor = try interpreter.input(at: 0)
        // check input tensor dimension
        guard inputTensor.shape.dimensions[0] == 1,
            inputTensor.shape.dimensions[1] == options.inputHeight,
            inputTensor.shape.dimensions[2] == options.inputWidth,
            inputTensor.shape.dimensions[3] == options.inputChannel
        else {
            fatalError("Unexpected Model: input shape \n\(inputTensor.shape) != [\(1), \(options.inputHeight), \(options.inputWidth), \(options.inputChannel)]")
        }
        self.inputTensor = inputTensor
        
        // output tensor
        let outputTensors = try (0..<interpreter.outputTensorCount).map { outputTensorIndex -> Tensor in
            let outputTensor = try interpreter.output(at: outputTensorIndex)
            return outputTensor
        }
        // check output tensors dimension
        outputTensors.enumerated().forEach { (offset, outputTensor) in
            // <#TODO#>
        }
        self.outputTensors = outputTensors
        
        // <#TODO#> - check quantization or not
    }
    
    func preprocess(with input: TFLiteVisionInput) -> Data? {
        let modelInputSize = CGSize(width: options.inputWidth, height: options.inputHeight)
        guard let thumbnail = input.croppedPixelBuffer(with: modelInputSize) else { return nil }
        
        // Remove the alpha component from the image buffer to get the initialized `Data`.
        let byteCount = 1 * options.inputHeight * options.inputWidth * options.inputChannel
        guard let inputData = thumbnail.rgbData(byteCount: byteCount,
                                                isNormalized: options.isNormalized,
                                                isModelQuantized: options.isQuantized) else {
            print("Failed to convert the image buffer to RGB data.")
            return nil
        }
        
        return inputData
    }
    
    func preprocess(with pixelBuffer: CVPixelBuffer, from targetSquare: CGRect) -> Data? {
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32BGRA)
        
        // Resize `targetSquare` of input image to `modelSize`.
        let modelSize = CGSize(width: options.inputWidth, height: options.inputHeight)
        guard let thumbnail = pixelBuffer.resize(from: targetSquare, to: modelSize) else { return nil }
        
        // Remove the alpha component from the image buffer to get the initialized `Data`.
        let byteCount = 1 * options.inputHeight * options.inputWidth * options.inputChannel
        guard let inputData = thumbnail.rgbData(byteCount: byteCount,
                                                isNormalized: options.isNormalized,
                                                isModelQuantized: options.isQuantized) else {
            print("Failed to convert the image buffer to RGB data.")
            return nil
        }
        
        return inputData
    }
    
    func inference(with inputData: Data) -> [TFLiteFlatArray<Float32>]? {
        // Copy the initialized `Data` to the input `Tensor`.
        do {
            // Copy input into interpreter's 0th `Tensor`.
            try interpreter.copy(inputData, toInputAt: 0)
            
            // Run inference by invoking the `Interpreter`.
            try interpreter.invoke()
            
            // Get the output `Tensor` to process the inference results.
            for (index) in 0..<outputTensors.count {
                outputTensors[index] = try interpreter.output(at: index)
            }
        } catch /*let error*/ {
            // fatalError("Failed to invoke the interpreter with error:" + error.localizedDescription)
            return nil
        }
        
        return outputTensors.map { TFLiteFlatArray(tensor: $0) }
    }
}

extension TFLiteImageInterpreter {
    struct Options {
        let modelName: String
        let threadCount: Int
        let accelerator: Accelerator
        let isQuantized: Bool
        let inputWidth: Int
        let inputHeight: Int
        let isGrayScale: Bool
        var inputChannel: Int { return isGrayScale ? 1 : 3 }
        let isNormalized: Bool // true: 0.0~1.0, false: 0.0~255.0
        
        init(modelName: String, threadCount: Int = 1, accelerator: Accelerator = .metal, isQuantized: Bool = false, inputWidth: Int, inputHeight: Int, isGrayScale: Bool = false, isNormalized: Bool = false) {
            self.modelName = modelName
            self.threadCount = threadCount
            #if targetEnvironment(simulator)
            self.accelerator = .cpu
            #else
            self.accelerator = accelerator
            #endif
            self.isQuantized = isQuantized
            self.inputWidth = inputWidth
            self.inputHeight = inputHeight
            self.isGrayScale = isGrayScale
            self.isNormalized = isNormalized
        }
    }
}

extension TFLiteImageInterpreter {
    enum Accelerator {
        case cpu
        case metal
    }
}

extension TFLiteImageInterpreter {
    struct ModelInput {
        static let batchSize = 1
        static let channel = 3 // rgb: 3, gray: 1
    }
}

enum TFLiteVisionInput {
    case pixelBuffer(pixelBuffer: CVPixelBuffer, preprocessOptions: PreprocessOptions)
    case uiImage(uiImage: UIImage, preprocessOptions: PreprocessOptions)
    
    var pixelBuffer: CVPixelBuffer? {
        switch self {
        case .pixelBuffer(let pixelBuffer, _):
            return pixelBuffer
        case .uiImage(let uiImage, _):
            return uiImage.pixelBufferFromImage()
        }
    }
    
    var cropArea: PreprocessOptions.CropArea {
        switch self {
        case .pixelBuffer(_, let preprocessOptions):
            return preprocessOptions.cropArea
        case .uiImage(_, let preprocessOptions):
            return preprocessOptions.cropArea
        }
    }
    
    var imageSize: CGSize {
        switch self {
        case .pixelBuffer(let pixelBuffer, _):
            return pixelBuffer.size
        case .uiImage(let uiImage, _):
            return uiImage.size
        }
    }
    
    var targetSquare: CGRect {
        switch cropArea {
        case .customAspectFill(let rect):
            return rect
        case .squareAspectFill:
            let size = imageSize
            let minLength = min(size.width, size.height)
            return CGRect(x: (size.width - minLength) / 2,
                          y: (size.height - minLength) / 2,
                          width: minLength, height: minLength)
        }
    }
    
    func croppedPixelBuffer(with inputModelSize: CGSize) -> CVPixelBuffer? {
        guard let pixelBuffer = pixelBuffer else { return nil }
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32BGRA)
        
        // Resize `targetSquare` of input image to `modelSize`.
        return pixelBuffer.resize(from: targetSquare, to: inputModelSize)
    }
}

struct PreprocessOptions {
    let cropArea: CropArea
    
    enum CropArea {
        case customAspectFill(rect: CGRect)
        case squareAspectFill
    }
}
