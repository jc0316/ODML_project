//
//  Bot.swift
//  Vision + CoreML
//
//  Created by Johnny Chen on 2024/10/28.
//  Copyright © 2024 Apple. All rights reserved.
//


import SwiftUI
import LLM

class Bot: LLM {
    convenience init?(_ update: @escaping (Double) -> Void) async {
        let systemPrompt = "Your role is to act as a text corrector and enhancer. The input you receive is a transcription from an ASL recognition model, which may contain repeated characters, slight spelling errors, and occasional missed words. Your task is to output a clean, grammatically correct sentence that closely resembles natural English."
        
        let model = HuggingFaceModel("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", .Q2_K, template: .chatML(systemPrompt))
        try? await self.init(from: model) { progress in update(progress) }
    }
}

struct BotView: View {
    @ObservedObject var bot: Bot
    @State var input = "Enter prompt here"
    init(_ bot: Bot) { self.bot = bot }
    func respond() { Task { await bot.respond(to: input) } }
    func stop() { bot.stop() }
    var body: some View {
        VStack(alignment: .leading) {
            ScrollView { Text(bot.output).monospaced() }
            Spacer()
            HStack {
                ZStack {
                    RoundedRectangle(cornerRadius: 8).foregroundStyle(.thinMaterial).frame(height: 40)
                    TextField("input", text: $input).padding(8)
                }
                Button(action: respond) { Image(systemName: "paperplane.fill") }
                Button(action: stop) { Image(systemName: "xmark") }
            }
        }.frame(maxWidth: .infinity).padding()
    }
}

struct ContentView: View {
    @State var bot: Bot? = nil
    @State var progress: CGFloat = 0
    func updateProgress(_ progress: Double) {
        self.progress = CGFloat(progress)
    }
    var body: some View {
        if let bot {
            BotView(bot)
        } else {
            ProgressView(value: progress) {
                Text("loading huggingface model...")
            } currentValueLabel: {
                Text(String(format: "%.2f%%", progress * 100))
            }
            .padding()
            .onAppear() { Task {
                let bot = await Bot(updateProgress)
                await MainActor.run { self.bot = bot }
            } }
        }
    }
}
