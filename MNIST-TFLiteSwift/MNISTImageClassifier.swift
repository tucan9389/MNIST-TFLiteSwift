//
//  MNISTImageClassifier.swift
//  MNIST-TFLiteSwift
//
//  Created by Doyoung Gwak on 2020/06/19.
//  Copyright © 2020 Doyoung Gwak. All rights reserved.
//

import Foundation
import TFLiteSwift_Vision

class MNISTImageClassifier: ImageClassifier {
    
    lazy var imageInterpreter: TFLiteVisionInterpreter = {
        let interpreterOptions = TFLiteVisionInterpreter.Options(
            modelName: "mnistCNN",
            inputRankType: .bhwc,
            normalization: .scaled(from: 0.0, to: 1.0)
        )
        let imageInterpreter = TFLiteVisionInterpreter(options: interpreterOptions)
        return imageInterpreter
    }()
    
    var modelOutput: [TFLiteFlatArray<Float32>]?
    
    func inference(_ uiImage: UIImage) -> Result<ImageClassificationOutput, ImageClassificationError> {
        
        // initialize
        modelOutput = nil
        
        // preprocss and inference
        guard let outputs = imageInterpreter.inference(with: uiImage)
            else { return .failure(.failToInference) }
        
        // postprocess
        let result:  Result<ImageClassificationOutput, ImageClassificationError> = Result.success(postprocess(with: outputs))
        
        return result
    }
    
    func inference(_ pixelBuffer: CVPixelBuffer) -> Result<ImageClassificationOutput, ImageClassificationError> {
        
        // initialize
        modelOutput = nil
        
        // preprocss and inference
        guard let outputs = imageInterpreter.inference(with: pixelBuffer)
            else { return .failure(.failToInference) }
        
        // postprocess
        let result:  Result<ImageClassificationOutput, ImageClassificationError> = Result.success(postprocess(with: outputs))
        
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
