#!/usr/bin/env swift

import AppKit
import CryptoKit
import Foundation

struct SourceInstrument: Decodable { let path: String; let sha256: String }
struct Renderer: Decodable {
    let font_name: String
    let heading_point_size: CGFloat
    let body_point_size: CGFloat
    let canvas_width_pixels: Int
    let horizontal_margin_pixels: Int
    let vertical_margin_pixels: Int
    let paragraph_spacing_pixels: Int
    let chart_spacing_pixels: Int
}
struct Placement: Decodable { let x: Int; let y: Int; let width: Int; let height: Int }
struct Arm: Decodable {
    let arm_id: String
    let heading: String
    let paragraphs: [String]
    let source_chart_member: String?
    let source_chart_sha256: String?
    let chart_placement: Placement?
    let output_path: String
}
struct Recipe: Decodable {
    let source_instrument: SourceInstrument
    let renderer: Renderer
    let arms: [Arm]
}

func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func unzipMember(archive: URL, member: String) throws -> Data {
    let process = Process()
    let pipe = Pipe()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
    process.arguments = ["-p", archive.path, member]
    process.standardOutput = pipe
    process.standardError = Pipe()
    try process.run()
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        throw NSError(domain: "6wbd7", code: 2, userInfo: [NSLocalizedDescriptionKey: "could not extract \(member)"])
    }
    return data
}

func textHeight(_ text: String, width: CGFloat, font: NSFont, paragraphSpacing: CGFloat) -> CGFloat {
    let style = NSMutableParagraphStyle()
    style.lineBreakMode = .byWordWrapping
    style.lineSpacing = 7
    style.paragraphSpacing = paragraphSpacing
    return ceil((text as NSString).boundingRect(
        with: NSSize(width: width, height: .greatestFiniteMagnitude),
        options: [.usesLineFragmentOrigin, .usesFontLeading],
        attributes: [.font: font, .paragraphStyle: style]
    ).height)
}

func drawText(_ text: String, rect: NSRect, font: NSFont, paragraphSpacing: CGFloat) {
    let style = NSMutableParagraphStyle()
    style.lineBreakMode = .byWordWrapping
    style.lineSpacing = 7
    style.paragraphSpacing = paragraphSpacing
    (text as NSString).draw(
        in: rect,
        withAttributes: [.font: font, .foregroundColor: NSColor.black, .paragraphStyle: style]
    )
}

let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let recipeURL = root.appendingPathComponent("data/manifests/stimuli/6wbd7_derived_composite_v1.json")
let recipe = try JSONDecoder().decode(Recipe.self, from: Data(contentsOf: recipeURL))
let sourceURL = root.appendingPathComponent(recipe.source_instrument.path)
let sourceData = try Data(contentsOf: sourceURL)
guard sha256(sourceData) == recipe.source_instrument.sha256 else {
    fatalError("source instrument hash mismatch")
}

let width = CGFloat(recipe.renderer.canvas_width_pixels)
let margin = CGFloat(recipe.renderer.horizontal_margin_pixels)
let vertical = CGFloat(recipe.renderer.vertical_margin_pixels)
let contentWidth = width - 2 * margin
guard let headingFont = NSFont(name: recipe.renderer.font_name, size: recipe.renderer.heading_point_size),
      let bodyFont = NSFont(name: recipe.renderer.font_name, size: recipe.renderer.body_point_size) else {
    fatalError("required renderer font is unavailable")
}

for arm in recipe.arms {
    let headingHeight = textHeight(arm.heading, width: contentWidth, font: headingFont, paragraphSpacing: 0)
    let body = arm.paragraphs.joined(separator: "\n\n")
    let bodyHeight = textHeight(body, width: contentWidth, font: bodyFont, paragraphSpacing: CGFloat(recipe.renderer.paragraph_spacing_pixels))
    var chartImage: NSImage? = nil
    var chartHeight: CGFloat = 0
    if let member = arm.source_chart_member, let expected = arm.source_chart_sha256, let placement = arm.chart_placement {
        let chartData = try unzipMember(archive: sourceURL, member: member)
        guard sha256(chartData) == expected else { fatalError("chart member hash mismatch") }
        guard let image = NSImage(data: chartData) else { fatalError("chart image is invalid") }
        guard Int(image.size.width) == placement.width, Int(image.size.height) == placement.height else {
            fatalError("chart member dimensions drifted")
        }
        chartImage = image
        chartHeight = CGFloat(placement.height) + CGFloat(recipe.renderer.chart_spacing_pixels)
    }
    let totalHeight = Int(ceil(vertical + headingHeight + CGFloat(recipe.renderer.chart_spacing_pixels) + chartHeight + bodyHeight + vertical))
    guard let bitmap = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: Int(width),
        pixelsHigh: totalHeight,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    ) else { fatalError("could not allocate output bitmap") }
    let context = NSGraphicsContext(bitmapImageRep: bitmap)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = context
    context.imageInterpolation = .none
    NSColor.white.setFill()
    NSRect(x: 0, y: 0, width: width, height: CGFloat(totalHeight)).fill()

    var top = vertical
    drawText(arm.heading, rect: NSRect(x: margin, y: CGFloat(totalHeight) - top - headingHeight, width: contentWidth, height: headingHeight), font: headingFont, paragraphSpacing: 0)
    top += headingHeight + CGFloat(recipe.renderer.chart_spacing_pixels)
    if let image = chartImage, let placement = arm.chart_placement {
        guard Int(top) == placement.y else { fatalError("chart vertical placement drifted") }
        let chartRect = NSRect(x: CGFloat(placement.x), y: CGFloat(totalHeight) - top - CGFloat(placement.height), width: CGFloat(placement.width), height: CGFloat(placement.height))
        image.draw(in: chartRect, from: .zero, operation: .copy, fraction: 1.0, respectFlipped: false, hints: [.interpolation: NSImageInterpolation.none])
        top += CGFloat(placement.height) + CGFloat(recipe.renderer.chart_spacing_pixels)
    }
    drawText(body, rect: NSRect(x: margin, y: CGFloat(totalHeight) - top - bodyHeight, width: contentWidth, height: bodyHeight), font: bodyFont, paragraphSpacing: CGFloat(recipe.renderer.paragraph_spacing_pixels))
    NSGraphicsContext.restoreGraphicsState()
    guard let png = bitmap.representation(using: .png, properties: [:]) else { fatalError("could not encode PNG") }
    let output = root.appendingPathComponent(arm.output_path)
    try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
    try png.write(to: output, options: .atomic)
    print("\(arm.arm_id)\t\(sha256(png))\t\(Int(width))x\(totalHeight)")
}
