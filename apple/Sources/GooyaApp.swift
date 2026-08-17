import SwiftUI

@main
struct GooyaApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.layoutDirection, .rightToLeft)
        }
#if os(macOS)
        .defaultSize(width: 620, height: 720)
#endif
    }
}

