//
//  MNISTImageClassifier.swift
//  MNIST-TFLiteSwift
//
//  Created by Doyoung Gwak on 2020/06/19.
//  Copyright © 2020 Doyoung Gwak. All rights reserved.
//

import Foundation

class MNISTImageClassifier: ImageClassifier {
    typealias MNISTClassificationResult = Result<ImageClassificationOutput, ImageClassificationError>
    
    lazy var imageInterpreter: TFLiteImageInterpreter = {
        let options = TFLiteImageInterpreter.Options(
            modelName: "mnistCNN",
            inputWidth: Input.width,
            inputHeight: Input.height,
            isGrayScale: Input.isGrayScale,
            isNormalized: Input.isNormalized
        )
        let imageInterpreter = TFLiteImageInterpreter(options: options)
        return imageInterpreter
    }()
    
    var modelOutput: [TFLiteFlatArray<Float32>]?
    
    func inference(_ input: ClassificationInput) -> MNISTClassificationResult {
        
        // initialize
        modelOutput = nil
        
        // preprocss
//        guard let inputData = imageInterpreter.preprocess(with: input.input)
//            else { return .failure(.failToCreateInputData) }
        guard let pixelData = input.input.pixelData
            else { return .failure(.failToCreateInputData) }
        let inputData = Data(copyingBufferOf: pixelData)
        
        // inference
        guard let outputs = imageInterpreter.inference(with: inputData)
            else { return .failure(.failToInference) }
        
        // postprocess
        let result = MNISTClassificationResult.success(postprocess(with: outputs))
        
        return result
    }
        
    private func postprocess(with outputs: [TFLiteFlatArray<Float32>]) -> ImageClassificationOutput {
        return ImageClassificationOutput(outputs: outputs)
    }
    
    func postprocessOnLastOutput(options: PostprocessOptions) -> ImageClassificationOutput? {
        guard let outputs = modelOutput else { return nil }
        return postprocess(with: outputs)
    }
}

private extension MNISTImageClassifier {
    struct Input {
        static let width = 28
        static let height = 28
        static let isGrayScale = true
        static let isNormalized = true
    }
    struct Output {
        static let numberOfCategories = 10
    }
}

private extension ImageClassificationOutput {
    init(outputs: [TFLiteFlatArray<Float32>]) {
        self.outputs = outputs
        self.number = outputs[0].argmax()
    }
}

extension TFLiteFlatArray where Element == Float32 {
    func argmax() -> Int {
        let numberOfElements = dimensions.reduce(1) { $0 * $1 }
        
        var maxValue: Float32 = 0
        var maxIndex: Int = 0
        
        for i in 0..<numberOfElements {
            print(self.element(at: [0, i]))
            if maxValue < element(at: [0, i]) {
                maxValue = element(at: [0, i])
                maxIndex = i
            }
        }
        return maxIndex
    }
}
