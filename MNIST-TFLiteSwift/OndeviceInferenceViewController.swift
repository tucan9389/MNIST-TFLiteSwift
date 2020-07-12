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
//  OndeviceInferenceViewController.swift
//  MNIST-TFLiteSwift
//
//  Created by Doyoung Gwak on 2020/06/19.
//  Copyright © 2020 Doyoung Gwak. All rights reserved.
//

import UIKit
import CoreImage
import CoreVideo

class OndeviceInferenceViewController: UIViewController {
    
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
    }
    
    @IBAction func clear(_ sender: Any) {
        drawView?.lines = []
        drawView?.setNeedsDisplay()
        predictLabel?.text = ""
    }
    
    var pixelData = [UInt8](repeating: 0, count: Int(28 * 28))
    
    @IBAction func classify(_ sender: Any) {
        guard let drawView = drawView, let predictLabel = predictLabel else { return }
        // guard !drawView.lines.isEmpty else { return }
        // guard let pixelBuffer = pixelBuffer else { return }
        
        // get drawn image
        let viewContext = drawView.getViewContext()
        
        guard let cgImage = viewContext?.makeImage() else { return }
        
        let uiImage = UIImage(cgImage: cgImage)
        
        guard let inputData = uiImage.grayScaled() else { return }
        
        // create input with the above image
        let input = ClassificationInput(input: .pixelData(pixelData: inputData,
                                                          preprocessOptions: PreprocessOptions(cropArea: .none)),
                                        postprocessOptions: PostprocessOptions(numberOfCategories: 10))

        // prediction
        let result: Result<ImageClassificationOutput, ImageClassificationError> = classifier.inference(input)

        // show the result of the prediction
        switch (result) {
        case .success(let output):
            predictLabel.text = "\(output.number)"
        case .failure(_):
            break
        }
    }
}

extension UInt8 {
    func str(digitNumber: Int, emptyStr: String) -> String {
        var tmp = "\(self)"
        while tmp.count < digitNumber {
            tmp = emptyStr + tmp
        }
        return tmp
    }
}

// https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
// syj 
extension UIImage {
    func grayScaled(targetSize: CGSize = CGSize(width: 28, height: 28)) -> [Float32]? {
        UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
        self.draw(in: CGRect(origin: .zero, size: targetSize))
        let toConvertImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()

        let size = toConvertImage.size
        let width = Int(size.width)
        let height = Int(size.height)

        let pixels = UnsafeMutablePointer<UInt32>.allocate(capacity: width * height * MemoryLayout<UInt32>.size)
        defer {
            pixels.deallocate()
        }
        memset(pixels, 0, width * height * MemoryLayout<UInt32>.size)
        
        let bitmapInfo: CGBitmapInfo = [
            .byteOrder32Little,
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        ]
        let colorSpace = CGColorSpaceCreateDeviceRGB()

        let context = CGContext(data: pixels, width: width, height: height,
                                bitsPerComponent: 8,
                                bytesPerRow: width * MemoryLayout<UInt32>.size,
                                space: colorSpace,
                                bitmapInfo: bitmapInfo.rawValue)

        context?.draw(toConvertImage.cgImage!, in: CGRect(x: 0, y: 0, width: targetSize.width, height: targetSize.height))

        var array = Array<Float32>()
        array.reserveCapacity(height * width)
        
        for y in 0..<height {
            for x in 0..<width {
                let pixel = pixels[y * width + x].toUInt8s()
                let grayed = Float(pixel[0]) * 0.3 + Float(pixel[1]) * 0.59 + Float(pixel[2]) * 0.11
                array.append(grayed / 255)
            }
        }
        
        return array
    }
}


extension UInt32 {
    func toUInt8s() -> [UInt8] {
        var bigEndian = self.bigEndian
        let count = MemoryLayout<UInt32>.size
        let bytePtr = withUnsafePointer(to: &bigEndian) {
            $0.withMemoryRebound(to: UInt8.self, capacity: count) {
                UnsafeBufferPointer(start: $0, count: count)
            }
        }
        return Array(bytePtr)
    }
}
