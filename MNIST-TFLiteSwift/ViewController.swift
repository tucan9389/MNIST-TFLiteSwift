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
//  ViewController.swift
//  MNIST-TFLiteSwift
//
//  Created by Doyoung Gwak on 2020/06/19.
//  Copyright © 2020 Doyoung Gwak. All rights reserved.
//

import UIKit
import CoreImage
import CoreVideo

class ViewController: UIViewController {
    
    // MARK: - UI
    @IBOutlet weak var drawView: DrawView?
    @IBOutlet weak var predictLabel: UILabel?
    
    // MARK: - ML
    let classifier: ImageClassifier = MNISTImageClassifier()
    
    let context = CIContext()
    var pixelBuffer: CVPixelBuffer?

    // MARK: - VC Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        
        predictLabel?.text = ""
        
        setupPixelBuffer()
    }
    
    func setupPixelBuffer() {
        // Set the pixel buffer dimensions - Remember it's grayscale
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        CVPixelBufferCreate(kCFAllocatorDefault, 28, 28, kCVPixelFormatType_OneComponent8, attrs, &pixelBuffer)
    }
    
    @IBAction func clear(_ sender: Any) {
        drawView?.lines = []
        drawView?.setNeedsDisplay()
        predictLabel?.text = ""
    }
    
    @IBAction func classify(_ sender: Any) {
        guard let drawView = drawView, let predictLabel = predictLabel else { return }
        // guard !drawView.lines.isEmpty else { return }
        // guard let pixelBuffer = pixelBuffer else { return }
        
        // get drawn image
        let viewContext = drawView.getViewContext()
        
        guard let cgImage = viewContext?.makeImage() else { return }
        let uiImage = UIImage(cgImage: cgImage)

        // prediction
        let result: Result<ImageClassificationOutput, ImageClassificationError> = classifier.inference(uiImage)

        // show the result of the prediction
        switch (result) {
        case .success(let output):
            predictLabel.text = "\(output.number)"
        case .failure(_):
            break
        }
    }
}
