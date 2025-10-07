//
//  DataStructure.swift
//  MusicTest
//
//  Created by Andy Lyu on 10/7/25.
//

import Foundation

struct ChatMessage: Codable {
    let role: String
    let content: String
}

struct ChatRequest: Codable {
    let model: String
    let messages: [ChatMessage]
    let temperature: Double?
    let stream: Bool?
}

struct ChatResponse: Codable {
    struct Choice: Codable {
        struct Message: Codable {
            let role: String
            let content: String
        }
        let message: Message
    }
    struct Error: Codable {
        let message: String
        let type: String?
    }
    let choices: [Choice]?
    let error: Error?
}
