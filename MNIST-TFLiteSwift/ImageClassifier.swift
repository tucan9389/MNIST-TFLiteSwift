//
//  ImageClassifier.swift
//  MNIST-TFLiteSwift
//
//  Created by Doyoung Gwak on 2020/06/19.
//  Copyright © 2020 Doyoung Gwak. All rights reserved.
//

import Foundation

struct PostprocessOptions {
    let numberOfCategories: Int
}

struct ClassificationInput {
    let input: TFLiteVisionInput
    let postprocessOptions: PostprocessOptions
}

struct ImageClassificationOutput {
    
    var outputs: [TFLiteFlatArray<Float32>]
    var number: Int // 0...9
}

enum ImageClassificationError: Error {
    case failToCreateInputData
    case failToInference
}

protocol ImageClassifier {
    func inference(_ input: ClassificationInput) -> Result<ImageClassificationOutput, ImageClassificationError>
    func postprocessOnLastOutput(options: PostprocessOptions) -> ImageClassificationOutput?
}
